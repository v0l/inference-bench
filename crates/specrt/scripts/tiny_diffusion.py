import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests/fixtures"
SEED = 0
STEPS = 3
TEXT_LEN = 16
PROMPT_LEN = 10
MAX_BYTES = 20 * 1024 * 1024


def floats(t):
    return t.detach().float().contiguous().numpy().astype(np.float32)


def engine_noise(seed, shape):
    n = int(np.prod(shape))
    pairs = (n + 1) // 2
    with np.errstate(over="ignore"):
        i = np.arange(1, 2 * pairs + 1, dtype=np.uint64)
        z = np.uint64(seed) + i * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))
    r = np.sqrt(-2.0 * np.log1p(-u[0::2]))
    angle = 2.0 * np.pi * u[1::2]
    out = np.empty(2 * pairs, dtype=np.float64)
    out[0::2] = r * np.cos(angle)
    out[1::2] = r * np.sin(angle)
    return torch.from_numpy(out[:n].astype(np.float32).reshape(shape))


def round_bf16(module):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(p.to(torch.bfloat16).float())


class Reference:
    def __init__(self, fixture, name):
        self.out = fixture / "reference"
        self.out.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.meta = {"files": {}}

    def put(self, key, tensor):
        array = floats(tensor)
        file = f"{self.name}.{key}.bin"
        array.tofile(self.out / file)
        self.meta["files"][key] = {"file": file, "shape": list(array.shape)}

    def finish(self, **meta):
        self.meta.update(meta)
        (self.out / f"{self.name}.json").write_text(json.dumps(self.meta, indent=1) + "\n")


def randomise(module, g, scale=0.08):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("weight") and p.ndim == 1:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(scale * torch.randn(p.shape, generator=g))


def token_ids(vocab, g, real, pad_id):
    ids = torch.full((1, TEXT_LEN), pad_id, dtype=torch.long)
    ids[0, :real] = torch.randint(2, vocab, (real,), generator=g)
    mask = torch.zeros((1, TEXT_LEN), dtype=torch.long)
    mask[0, :real] = 1
    return ids, mask


def check_size(fixture):
    size = sum(f.stat().st_size for f in fixture.rglob("*") if f.is_file())
    if size > MAX_BYTES:
        sys.exit(f"{fixture.name} is {size} bytes, over {MAX_BYTES}")
    print(f"{fixture.name}: {size} bytes")


def flux2_klein():
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
    from transformers import Qwen3Config, Qwen3ForCausalLM

    name = "tiny-flux2-klein"
    fixture = FIXTURES / name
    g = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    layers = [1, 2, 3]
    te_config = Qwen3Config(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=256,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        torch_dtype="float32",
    )
    text_encoder = Qwen3ForCausalLM(te_config).eval()
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=len(layers) * te_config.hidden_size,
        timestep_guidance_channels=32,
        mlp_ratio=3.0,
        axes_dims_rope=(4, 4, 4, 4),
        rope_theta=2000,
        eps=1e-6,
        guidance_embeds=False,
    ).eval()
    vae = AutoencoderKLFlux2(
        block_out_channels=(16, 32),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=32,
        patch_size=(2, 2),
    ).eval()
    for m in (text_encoder, transformer, vae):
        randomise(m, g)
    round_bf16(text_encoder)
    with torch.no_grad():
        vae.bn.running_mean.copy_(0.1 * torch.randn(vae.bn.running_mean.shape, generator=g))
        vae.bn.running_var.copy_(0.5 + torch.rand(vae.bn.running_var.shape, generator=g))
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=4096,
        max_shift=1.15,
        num_train_timesteps=1000,
        shift=3.0,
        time_shift_type="exponential",
        use_dynamic_shifting=True,
    )
    pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=None, transformer=transformer, is_distilled=True)
    pipe.save_pretrained(fixture, safe_serialization=True)
    reload = Flux2KleinPipeline.from_pretrained(fixture, tokenizer=None)

    ids, mask = token_ids(te_config.vocab_size, g, PROMPT_LEN, 0)
    neg_ids, neg_mask = token_ids(te_config.vocab_size, g, 4, 0)

    def embed(ids, mask):
        with torch.no_grad():
            out = reload.text_encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        stacked = torch.stack([out.hidden_states[k] for k in layers], dim=1)
        b, c, s, h = stacked.shape
        return stacked, stacked.permute(0, 2, 1, 3).reshape(b, s, c * h)

    hidden, prompt_embeds = embed(ids, mask)
    _, negative_embeds = embed(neg_ids, neg_mask)
    height = width = 32
    latents = engine_noise(SEED, (1, 16, 8, 8))

    def run(guidance, negative):
        steps = []
        velocities = []
        handle = reload.transformer.register_forward_hook(lambda m, i, o: velocities.append(o[0].clone() if isinstance(o, tuple) else o.sample.clone()))

        def on_step(p, i, t, kw):
            steps.append(kw["latents"].clone())
            return {}

        with torch.no_grad():
            image = reload(
                prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative,
                height=height,
                width=width,
                num_inference_steps=STEPS,
                guidance_scale=guidance,
                latents=latents.clone(),
                output_type="pt",
                callback_on_step_end=on_step,
            ).images
        handle.remove()
        return steps, velocities, image

    ref = Reference(fixture, name)
    ref.put("text.ids", ids.float())
    ref.put("text.mask", mask.float())
    ref.put("text.hidden", hidden)
    ref.put("text.embeds", prompt_embeds)
    ref.put("negative.ids", neg_ids.float())
    ref.put("negative.mask", neg_mask.float())
    ref.put("negative.embeds", negative_embeds)
    ref.put("noise", latents.reshape(1, 16, 64).permute(0, 2, 1))
    steps, velocities, image = run(1.0, None)
    sigmas = reload.scheduler.sigmas.tolist()
    timesteps = reload.scheduler.timesteps.tolist()
    if len(steps) != STEPS or len(velocities) != STEPS:
        sys.exit(f"captured {len(steps)} latents and {len(velocities)} velocities")
    for i in range(STEPS):
        ref.put(f"step{i}.velocity", velocities[i])
        ref.put(f"step{i}.latent", steps[i])
    with torch.no_grad():
        packed = steps[-1]
        unpacked = reload._unpack_latents_with_ids(packed, reload._prepare_latent_ids(latents), 8, 8)
        std = torch.sqrt(reload.vae.bn.running_var.view(1, -1, 1, 1) + reload.vae.config.batch_norm_eps)
        denorm = unpacked * std + reload.vae.bn.running_mean.view(1, -1, 1, 1)
        vae_latent = reload._unpatchify_latents(denorm)
        decoded = reload.vae.decode(vae_latent, return_dict=False)[0]
    ref.put("vae.latent", vae_latent)
    ref.put("vae.decoded", decoded)
    ref.put("image", image)
    check = (decoded / 2 + 0.5).clamp(0, 1) - image
    if check.abs().max() > 1e-5:
        sys.exit(f"decoded image differs from the pipeline output by {check.abs().max()}")
    reload.register_to_config(is_distilled=False)
    cfg_steps, cfg_velocities, cfg_image = run(4.0, negative_embeds)
    reload.register_to_config(is_distilled=True)
    if len(cfg_velocities) != 2 * STEPS:
        sys.exit(f"the base variant ran {len(cfg_velocities)} forwards, not {2 * STEPS}")
    ref.put("cfg.image", cfg_image)
    for i in range(STEPS):
        ref.put(f"cfg.step{i}.latent", cfg_steps[i])
    round_bf16(reload.transformer)
    bf16_steps, _, bf16_image = run(1.0, None)
    reload.register_to_config(is_distilled=False)
    bf16_cfg_steps, _, bf16_cfg_image = run(4.0, negative_embeds)
    reload.register_to_config(is_distilled=True)
    ref.put("bf16.image", bf16_image)
    ref.put("bf16.cfg.image", bf16_cfg_image)
    for i in range(STEPS):
        ref.put(f"bf16.step{i}.latent", bf16_steps[i])
        ref.put(f"bf16.cfg.step{i}.latent", bf16_cfg_steps[i])
    ref.finish(
        pipeline="Flux2KleinPipeline",
        seed=SEED,
        noise="splitmix64 counter, Box-Muller pairs",
        steps=STEPS,
        height=height,
        width=width,
        latent=[16, 16],
        image_tokens=64,
        text_tokens=TEXT_LEN,
        prompt_tokens=PROMPT_LEN,
        text_encoder_out_layers=layers,
        sigmas=sigmas,
        timesteps=timesteps,
        guidance=1.0,
        cfg_guidance=4.0,
        cfg_velocities_per_step=len(cfg_velocities) // STEPS,
        diffusers=__import__("diffusers").__version__,
        torch=torch.__version__,
    )
    check_size(fixture)


def tiny_tokenizer(fixture, vocab_size, pad):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<pad>": pad, "<unk>": 1}
    vocab.update({f"w{i}": i for i in range(2, vocab_size)})
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    out = fixture / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": TEXT_LEN,
        "chat_template": "{{ messages[0]['content'] }}",
    }
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=1) + "\n")


def flux2_tokenizer():
    fixture = FIXTURES / "tiny-flux2-klein"
    vocab = json.loads((fixture / "text_encoder/config.json").read_text())["vocab_size"]
    tiny_tokenizer(fixture, vocab, 0)


def wan21():
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
    from transformers import UMT5Config, UMT5EncoderModel

    name = "tiny-wan21"
    fixture = FIXTURES / name
    g = torch.Generator().manual_seed(SEED + 1)
    torch.manual_seed(SEED + 1)
    te_config = UMT5Config(
        vocab_size=512,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        feed_forward_proj="gated-gelu",
        dense_act_fn="gelu_new",
        is_gated_act=True,
        layer_norm_epsilon=1e-6,
        scalable_attention=True,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        torch_dtype="float32",
    )
    text_encoder = UMT5EncoderModel(te_config).eval()
    transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=24,
        in_channels=4,
        out_channels=4,
        text_dim=te_config.d_model,
        freq_dim=32,
        ffn_dim=96,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=64,
    ).eval()
    z = 4
    vae = AutoencoderKLWan(
        base_dim=8,
        z_dim=z,
        dim_mult=[1, 2, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True],
        latents_mean=[0.1, -0.2, 0.05, 0.3],
        latents_std=[1.5, 0.8, 1.2, 2.0],
        scale_factor_temporal=2,
        scale_factor_spatial=4,
    ).eval()
    for m in (text_encoder, transformer, vae):
        randomise(m, g)
    scheduler = UniPCMultistepScheduler(
        beta_end=0.02,
        beta_schedule="linear",
        beta_start=0.0001,
        final_sigmas_type="zero",
        flow_shift=3.0,
        lower_order_final=True,
        num_train_timesteps=1000,
        predict_x0=True,
        prediction_type="flow_prediction",
        solver_order=2,
        solver_type="bh2",
        timestep_spacing="linspace",
        use_flow_sigmas=True,
    )
    pipe = WanPipeline(tokenizer=None, text_encoder=text_encoder, vae=vae, scheduler=scheduler, transformer=transformer)
    pipe.save_pretrained(fixture, safe_serialization=True)
    reload = WanPipeline.from_pretrained(fixture, tokenizer=None)

    ids, mask = token_ids(te_config.vocab_size, g, PROMPT_LEN, 0)
    neg_ids, neg_mask = token_ids(te_config.vocab_size, g, 4, 0)

    def embed(ids, mask):
        with torch.no_grad():
            out = reload.text_encoder(ids, mask).last_hidden_state
        n = int(mask.sum())
        return torch.cat([out[:, :n], out.new_zeros(1, TEXT_LEN - n, out.shape[-1])], dim=1)

    prompt_embeds = embed(ids, mask)
    negative_embeds = embed(neg_ids, neg_mask)
    frames, height, width = 5, 32, 32
    latent_frames = (frames - 1) // 2 + 1
    latents = torch.randn((1, z, latent_frames, height // 4, width // 4), generator=torch.Generator().manual_seed(SEED + 1))
    steps = []
    velocities = []
    handle = reload.transformer.register_forward_hook(lambda m, i, o: velocities.append(o[0].clone() if isinstance(o, tuple) else o.sample.clone()))

    def on_step(p, i, t, kw):
        steps.append(kw["latents"].clone())
        return {}

    guidance = 5.0
    with torch.no_grad():
        video = reload(
            prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=STEPS,
            guidance_scale=guidance,
            latents=latents.clone(),
            output_type="pt",
            callback_on_step_end=on_step,
        ).frames
    handle.remove()
    if len(steps) != STEPS or len(velocities) != 2 * STEPS:
        sys.exit(f"captured {len(steps)} latents and {len(velocities)} velocities")
    ref = Reference(fixture, name)
    ref.put("text.ids", ids.float())
    ref.put("text.mask", mask.float())
    ref.put("text.embeds", prompt_embeds)
    ref.put("negative.ids", neg_ids.float())
    ref.put("negative.mask", neg_mask.float())
    ref.put("negative.embeds", negative_embeds)
    ref.put("noise", latents)
    for i in range(STEPS):
        ref.put(f"step{i}.velocity.cond", velocities[2 * i])
        ref.put(f"step{i}.velocity.uncond", velocities[2 * i + 1])
        ref.put(f"step{i}.latent", steps[i])
    with torch.no_grad():
        mean = torch.tensor(reload.vae.config.latents_mean).view(1, z, 1, 1, 1)
        std = torch.tensor(reload.vae.config.latents_std).view(1, z, 1, 1, 1)
        vae_latent = steps[-1] * std + mean
        decoded = reload.vae.decode(vae_latent, return_dict=False)[0]
    ref.put("vae.latent", vae_latent)
    ref.put("vae.decoded", decoded)
    ref.put("video", video)
    ref.finish(
        pipeline="WanPipeline",
        seed=SEED + 1,
        steps=STEPS,
        frames=frames,
        height=height,
        width=width,
        latent=[latent_frames, height // 4, width // 4],
        video_tokens=latent_frames * (height // 8) * (width // 8),
        text_tokens=TEXT_LEN,
        prompt_tokens=PROMPT_LEN,
        guidance=guidance,
        sigmas=reload.scheduler.sigmas.tolist(),
        timesteps=reload.scheduler.timesteps.tolist(),
        decoded_frames=int(decoded.shape[2]),
        diffusers=__import__("diffusers").__version__,
        torch=torch.__version__,
    )
    check_size(fixture)


if __name__ == "__main__":
    which = sys.argv[1:] or ["flux2", "wan"]
    if "flux2" in which:
        flux2_klein()
    if "flux2" in which or "tokenizer" in which:
        flux2_tokenizer()
    if "wan" in which:
        wan21()
