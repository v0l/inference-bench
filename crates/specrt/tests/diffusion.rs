use serde_json::Value;
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn json(name: &str, file: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture(name).join(file)).unwrap()).unwrap()
}

fn meta(name: &str) -> Value {
    json(name, &format!("reference/{name}.json"))
}

fn floats(name: &str, key: &str) -> Vec<f32> {
    let m = meta(name);
    let entry = &m["files"][key];
    let bytes = std::fs::read(fixture(name).join("reference").join(entry["file"].as_str().unwrap())).unwrap();
    let values: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let shape: usize = entry["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).product();
    assert_eq!(values.len(), shape, "{name} {key}");
    values
}

fn shape(name: &str, key: &str) -> Vec<u64> {
    meta(name)["files"][key]["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect()
}

fn tensor_shape(name: &str, file: &str, tensor: &str) -> Vec<usize> {
    let bytes = std::fs::read(fixture(name).join(file)).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    st.tensor(tensor).unwrap_or_else(|_| panic!("{tensor} missing from {file}")).shape().to_vec()
}

fn finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite()) && v.iter().any(|x| *x != 0.0)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn size(dir: &Path) -> u64 {
    std::fs::read_dir(dir).unwrap().flatten().map(|e| if e.path().is_dir() { size(&e.path()) } else { e.metadata().unwrap().len() }).sum()
}

#[test]
fn the_tiny_flux2_klein_pipeline_names_its_real_components() {
    let index = json("tiny-flux2-klein", "model_index.json");
    assert_eq!(index["_class_name"], "Flux2KleinPipeline");
    assert_eq!(index["is_distilled"], true);
    assert_eq!(index["text_encoder"][1], "Qwen3ForCausalLM");
    assert_eq!(index["transformer"][1], "Flux2Transformer2DModel");
    assert_eq!(index["vae"][1], "AutoencoderKLFlux2");
    assert_eq!(index["scheduler"][1], "FlowMatchEulerDiscreteScheduler");
    let t = json("tiny-flux2-klein", "transformer/config.json");
    assert_eq!((t["num_layers"].as_u64(), t["num_single_layers"].as_u64()), (Some(1), Some(1)));
    assert_eq!(t["axes_dims_rope"], serde_json::json!([4, 4, 4, 4]));
    assert_eq!(t["joint_attention_dim"].as_u64(), Some(3 * 32));
    let s = json("tiny-flux2-klein", "scheduler/scheduler_config.json");
    assert_eq!(s["use_dynamic_shifting"], true);
    assert_eq!(s["time_shift_type"], "exponential");
    let v = json("tiny-flux2-klein", "vae/config.json");
    assert_eq!((v["latent_channels"].as_u64(), v["patch_size"].clone()), (Some(4), serde_json::json!([2, 2])));
}

#[test]
fn the_flux2_text_encoder_is_a_qwen3_the_engine_already_runs() {
    let config = std::fs::read_to_string(fixture("tiny-flux2-klein").join("text_encoder/config.json")).unwrap();
    let spec = Spec::from_hf_config(&config).unwrap();
    assert_eq!((spec.dims.hidden, spec.layers.count), (32, 4));
    let layers: Vec<u64> = meta("tiny-flux2-klein")["text_encoder_out_layers"].as_array().unwrap().iter().map(|l| l.as_u64().unwrap()).collect();
    assert!(layers.iter().all(|&l| (l as usize) < spec.layers.count), "{layers:?}");
}

#[test]
fn the_flux2_checkpoints_use_the_real_tensor_names() {
    let tr = "transformer/diffusion_pytorch_model.safetensors";
    let n = "tiny-flux2-klein";
    assert_eq!(tensor_shape(n, tr, "x_embedder.weight"), vec![32, 16]);
    assert_eq!(tensor_shape(n, tr, "context_embedder.weight"), vec![32, 96]);
    assert_eq!(tensor_shape(n, tr, "double_stream_modulation_img.linear.weight"), vec![2 * 3 * 32, 32]);
    assert_eq!(tensor_shape(n, tr, "single_stream_modulation.linear.weight"), vec![3 * 32, 32]);
    assert_eq!(tensor_shape(n, tr, "transformer_blocks.0.attn.norm_added_q.weight"), vec![16]);
    assert_eq!(tensor_shape(n, tr, "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"), vec![3 * 32 + 2 * 96, 32]);
    assert_eq!(tensor_shape(n, tr, "single_transformer_blocks.0.attn.to_out.weight"), vec![32, 32 + 96]);
    assert_eq!(tensor_shape(n, tr, "time_guidance_embed.timestep_embedder.linear_1.weight"), vec![32, 32]);
    assert_eq!(tensor_shape(n, tr, "norm_out.linear.weight"), vec![64, 32]);
    let vae = "vae/diffusion_pytorch_model.safetensors";
    assert_eq!(tensor_shape(n, vae, "bn.running_var"), vec![16]);
    assert_eq!(tensor_shape(n, vae, "post_quant_conv.weight"), vec![4, 4, 1, 1]);
    assert_eq!(tensor_shape(n, vae, "decoder.conv_in.weight"), vec![32, 4, 3, 3]);
    assert_eq!(tensor_shape(n, vae, "decoder.up_blocks.0.upsamplers.0.conv.weight"), vec![32, 32, 3, 3]);
    assert_eq!(tensor_shape(n, vae, "decoder.up_blocks.1.resnets.0.conv_shortcut.weight"), vec![16, 32, 1, 1]);
    assert_eq!(tensor_shape(n, vae, "decoder.mid_block.attentions.0.group_norm.weight"), vec![32]);
    assert_eq!(tensor_shape(n, "text_encoder/model.safetensors", "model.layers.3.mlp.down_proj.weight"), vec![32, 64]);
}

#[test]
fn the_flux2_reference_has_three_steps_and_a_decoded_image() {
    let n = "tiny-flux2-klein";
    let m = meta(n);
    assert_eq!((m["steps"].as_u64(), m["height"].as_u64(), m["width"].as_u64()), (Some(3), Some(32), Some(32)));
    assert_eq!(m["latent"], serde_json::json!([16, 16]));
    let sigmas: Vec<f64> = m["sigmas"].as_array().unwrap().iter().map(|s| s.as_f64().unwrap()).collect();
    assert_eq!(sigmas.len(), 4);
    assert!(sigmas.windows(2).all(|w| w[0] > w[1]) && sigmas[0] == 1.0 && sigmas[3] == 0.0, "{sigmas:?}");
    assert_eq!(shape(n, "text.hidden"), vec![1, 3, 16, 32]);
    assert_eq!(shape(n, "text.embeds"), vec![1, 16, 96]);
    assert_eq!(shape(n, "noise"), vec![1, 64, 16]);
    assert!(finite(&floats(n, "text.embeds")));
    let mut latent = floats(n, "noise");
    for i in 0..3 {
        let velocity = floats(n, &format!("step{i}.velocity"));
        let next = floats(n, &format!("step{i}.latent"));
        assert!(finite(&velocity) && finite(&next));
        let dt = (sigmas[i + 1] - sigmas[i]) as f32;
        let euler: Vec<f32> = latent.iter().zip(&velocity).map(|(x, v)| x + dt * v).collect();
        assert!(max_diff(&euler, &next) < 1e-5, "step {i}: the flow Euler update x + (s1 - s0) v is the scheduler step");
        latent = next;
    }
    assert_eq!(shape(n, "vae.latent"), vec![1, 4, 16, 16]);
    assert_eq!(shape(n, "vae.decoded"), vec![1, 3, 32, 32]);
    let decoded = floats(n, "vae.decoded");
    let image = floats(n, "image");
    let post: Vec<f32> = decoded.iter().map(|x| (x / 2.0 + 0.5).clamp(0.0, 1.0)).collect();
    assert!(max_diff(&post, &image) < 1e-5);
    assert!(finite(&floats(n, "cfg.image")));
    assert!(max_diff(&floats(n, "cfg.step2.latent"), &latent) > 1e-3, "guidance changes the result");
    assert!(size(&fixture(n)) < 20 * 1024 * 1024);
}

#[test]
fn the_tiny_wan21_pipeline_is_a_t5_encoder_a_space_time_dit_and_a_causal_3d_vae() {
    let n = "tiny-wan21";
    let index = json(n, "model_index.json");
    assert_eq!(index["_class_name"], "WanPipeline");
    assert_eq!(index["text_encoder"][1], "UMT5EncoderModel");
    assert_eq!(index["transformer"][1], "WanTransformer3DModel");
    assert_eq!(index["vae"][1], "AutoencoderKLWan");
    assert_eq!(index["scheduler"][1], "UniPCMultistepScheduler");
    let t = json(n, "transformer/config.json");
    assert_eq!(t["patch_size"], serde_json::json!([1, 2, 2]));
    assert_eq!(t["qk_norm"], "rms_norm_across_heads");
    let te = std::fs::read_to_string(fixture(n).join("text_encoder/config.json")).unwrap();
    assert!(Spec::from_hf_config(&te).is_err(), "umt5 has no family yet");
    let tr = "transformer/diffusion_pytorch_model.safetensors";
    assert_eq!(tensor_shape(n, tr, "patch_embedding.weight"), vec![48, 4, 1, 2, 2]);
    assert_eq!(tensor_shape(n, tr, "blocks.0.scale_shift_table"), vec![1, 6, 48]);
    assert_eq!(tensor_shape(n, tr, "condition_embedder.time_proj.weight"), vec![6 * 48, 48]);
    assert_eq!(tensor_shape(n, tr, "blocks.1.attn2.to_k.weight"), vec![48, 48]);
    let vae = "vae/diffusion_pytorch_model.safetensors";
    assert_eq!(tensor_shape(n, vae, "decoder.conv_in.weight"), vec![16, 4, 3, 3, 3]);
    assert_eq!(tensor_shape(n, vae, "decoder.up_blocks.0.upsamplers.0.time_conv.weight"), vec![32, 16, 3, 1, 1]);
    assert_eq!(tensor_shape(n, vae, "decoder.mid_block.resnets.0.norm1.gamma"), vec![16, 1, 1, 1]);
    assert_eq!(tensor_shape(n, "text_encoder/model.safetensors", "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"), vec![32, 4]);
}

#[test]
fn the_wan21_reference_has_three_guided_steps_and_five_frames() {
    let n = "tiny-wan21";
    let m = meta(n);
    assert_eq!((m["steps"].as_u64(), m["frames"].as_u64(), m["decoded_frames"].as_u64()), (Some(3), Some(5), Some(5)));
    assert_eq!(m["latent"], serde_json::json!([3, 8, 8]));
    assert_eq!(shape(n, "noise"), vec![1, 4, 3, 8, 8]);
    for i in 0..3 {
        for key in ["velocity.cond", "velocity.uncond", "latent"] {
            assert!(finite(&floats(n, &format!("step{i}.{key}"))), "step {i} {key}");
        }
    }
    assert_eq!(shape(n, "vae.decoded"), vec![1, 3, 5, 32, 32]);
    assert_eq!(shape(n, "video"), vec![1, 5, 3, 32, 32]);
    assert!(finite(&floats(n, "video")));
    assert!(size(&fixture(n)) < 20 * 1024 * 1024);
}

#[test]
fn the_engine_refuses_diffusion_pipelines_until_their_pieces_exist() {
    for (n, files) in [
        ("tiny-flux2-klein", &["model_index.json", "transformer/config.json", "vae/config.json"][..]),
        ("tiny-wan21", &["model_index.json", "transformer/config.json", "vae/config.json", "text_encoder/config.json"][..]),
    ] {
        for f in files {
            let config = std::fs::read_to_string(fixture(n).join(f)).unwrap();
            assert!(Spec::from_hf_config(&config).is_err(), "{n} {f} must be refused");
            assert!(Spec::from_hf_config_serving(&config).is_err(), "{n} {f} must be refused for serving");
        }
    }
}
