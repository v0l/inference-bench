use super::cuda::CudaSpeech;
use super::decoder::HostDecoder;
use super::host::{HostEncoder, HostSpeech};
use super::transcribe::{Engine, Transcriber, log_mel_clip, rules_for};
use super::graph::SpeechPlan;
use crate::codegen::Module;
use crate::jit::Jit;
use crate::plan::{Buf, DitOp, NormKind, Op, Weight};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, sys};
use kir::conv::{ConvKernel, ConvShape, Epilogue, Prologue};
use kir::{Access, Param, Role, Scalar};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Epi {
    Store,
    Residual,
    Gelu,
}

struct Lin {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    n: usize,
    k: usize,
}

type LineKey = (usize, usize, Epi, bool);

enum EncStep {
    Add { table: CudaSlice<f32>, from: &'static str, into: &'static str },
    Norm { w: CudaSlice<f32>, b: CudaSlice<f32>, from: &'static str, into: &'static str },
    Linear { lin: Lin, input: &'static str, output: &'static str, epi: Epi },
    Pack { from: &'static str },
    Attend { into: &'static str },
}

enum DecStep {
    Embed,
    Position,
    Norm { w: CudaSlice<f32>, b: CudaSlice<f32>, into: Buf },
    Linear { lin: Lin, input: Buf, output: Buf, epi: Epi },
    KvWrite { layer: usize },
    SelfAttend { layer: usize, write: bool },
    CrossAttend { layer: usize },
}

pub struct GpuSeq {
    cross: Vec<CudaSlice<f32>>,
    kv: Vec<CudaSlice<f32>>,
    len: usize,
}

pub struct GpuEngine {
    s: Arc<CudaStream>,
    speech: CudaSpeech,
    mel_samples: usize,
    enc: Vec<EncStep>,
    enc_bufs: BTreeMap<&'static str, CudaSlice<f32>>,
    enc_q: CudaSlice<f32>,
    enc_kv: CudaSlice<f32>,
    enc_rows: usize,
    dec: Vec<DecStep>,
    cross_kv: Vec<Lin>,
    dec_bufs: BTreeMap<Buf, CudaSlice<f32>>,
    embed: CudaSlice<f32>,
    positions: CudaSlice<f32>,
    max_positions: usize,
    cross_rows: usize,
    layers: usize,
    hidden: usize,
    q_dim: usize,
    vocab: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
    enc_heads: usize,
    enc_head_dim: usize,
    enc_eps: u32,
    eps: u32,
    line: BTreeMap<LineKey, (ConvKernel, CudaFunction)>,
    norms: BTreeMap<(usize, u32), CudaFunction>,
    adds: BTreeMap<usize, CudaFunction>,
    norm_threads: u32,
    attend: CudaFunction,
    ones: CudaSlice<f32>,
    zero: CudaSlice<f32>,
    pub weight_bytes: usize,
}

pub type GpuTranscriber = Transcriber<GpuEngine>;

fn rows_family(n: usize, k: usize, rows: usize) -> bool {
    kir::conv::rows_fit(&ConvShape::line(k, n, 1, 1, 1), rows)
}

fn line_kernel(t: &kir::Target, (n, k, epi, rows): LineKey) -> Result<ConvKernel> {
    let shape = ConvShape::line(k, n, 1, 1, 1);
    let e = Epilogue { residual: epi == Epi::Residual, stats: None, gelu: epi == Epi::Gelu };
    let k = if rows { ConvKernel::rows(t, shape, e) } else { ConvKernel::new(t, shape, Prologue::None, e) };
    k.map_err(anyhow::Error::msg)
}

fn mapped_epi(spec: &crate::spec::Spec) -> Result<Epi> {
    let m = spec.mlp.mapped.as_ref().context("a Map epilogue needs a mapped activation")?;
    let e = kir::expr::Expr::parse(&m.expr).map_err(anyhow::Error::msg)?;
    ensure!(e == kir::expr::Expr::parse("gelu(x)").map_err(anyhow::Error::msg)?, "the line conv epilogue lowers gelu(x), not {}", m.expr);
    Ok(Epi::Gelu)
}

fn epi_of(spec: &crate::spec::Spec, weight: Weight) -> Result<Epi> {
    Ok(match crate::gemm::linear_shape(spec, weight.name()).epi {
        kir::gemm::Epilogue::Store => Epi::Store,
        kir::gemm::Epilogue::Residual => Epi::Residual,
        kir::gemm::Epilogue::Map => mapped_epi(spec)?,
        other => bail!("the speech engine has no {other:?} epilogue for {}", weight.name()),
    })
}

impl GpuEngine {
    pub fn open(stream: &Arc<CudaStream>, jit: &Jit, dir: &Path) -> Result<(GpuEngine, HostDecoder, SpeechPlan)> {
        let s = stream.clone();
        let host = HostSpeech::open(dir)?;
        let speech = CudaSpeech::load(stream, jit, &host)?;
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let encoder = HostEncoder::new(&config)?;
        let decoder = HostDecoder::open(dir, false)?;
        let ck = crate::weights::Checkpoint::open(dir)?;
        let target = crate::codegen::cuda_target(stream.context())?;
        let mut weight_bytes = 0usize;
        let mut up = |v: &[f32]| -> Result<CudaSlice<f32>> {
            weight_bytes += 4 * v.len();
            Ok(s.clone_htod(v)?)
        };
        let mut wants: Vec<LineKey> = Vec::new();

        let (eg, es, ep) = (&encoder.graph, &encoder.spec, &encoder.plan);
        let enc_rows = encoder.rows;
        let width = |name: &str| -> Result<usize> { es.size(&eg.decl(name).with_context(|| format!("no encoder buffer {name}"))?.cols) };
        let mut enc = Vec::new();
        let mut lower = |ops: &[Op], block: Option<usize>, enc: &mut Vec<EncStep>, wants: &mut Vec<LineKey>| -> Result<()> {
            for op in ops {
                let Op::Dit(d) = op else { bail!("{} is not an encoder graph op", op.label()) };
                let at = |w: &str| w.replace("{i}", &block.unwrap_or(0).to_string());
                enc.push(match *d {
                    DitOp::TableAdd { table, from, into } => {
                        ensure!(width(from)? == width(into)?, "table_add {table} widens {from}");
                        let name = at(table);
                        let t = ck.f32_values(&name, &ck.shape_of(&name)?)?;
                        ensure!(t.len() == enc_rows * width(into)?, "{name}: {} values for {enc_rows} rows of {}", t.len(), width(into)?);
                        EncStep::Add { table: up(&t)?, from, into }
                    }
                    DitOp::LayerNorm { from, into, weight, .. } => {
                        let w = at(weight);
                        let g = ck.f32_values(&format!("{w}.weight"), &ck.shape_of(&format!("{w}.weight"))?)?;
                        let b = ck.f32_values(&format!("{w}.bias"), &ck.shape_of(&format!("{w}.bias"))?)?;
                        EncStep::Norm { w: up(&g)?, b: up(&b)?, from, into }
                    }
                    DitOp::Linear { weight, input, output, add, bias, act, .. } => {
                        let (mut w, mut bv) = (Vec::new(), Vec::new());
                        for ((name, range), biased) in crate::denoiser::weight_names(weight, block).into_iter().zip(crate::denoiser::biased_parts(weight)) {
                            ensure!(range.is_none(), "{name}: the speech engine reads whole linear weights");
                            let shape = ck.shape_of(&name)?;
                            let rows = shape[0];
                            w.extend(ck.f32_values(&name, &shape)?);
                            let bn = format!("{}.bias", name.strip_suffix(".weight").unwrap_or(&name));
                            if bias && biased {
                                bv.extend(ck.f32_values(&bn, &ck.shape_of(&bn)?)?);
                            } else {
                                bv.extend(std::iter::repeat_n(0.0, rows));
                            }
                        }
                        let (n, k) = (bv.len(), width(input)?);
                        ensure!(w.len() == n * k && n == width(output)?, "{weight}: {} weights for {n} x {k}", w.len());
                        let epi = match (add, act) {
                            (true, 0) => Epi::Residual,
                            (false, 0) => Epi::Store,
                            (false, 2) => Epi::Gelu,
                            _ => bail!("{weight}: the line conv epilogue has no add {add} with act {act}"),
                        };
                        wants.push((n, k, epi, rows_family(n, k, enc_rows)));
                        EncStep::Linear { lin: Lin { w: up(&w)?, b: up(&bv)?, n, k }, input, output, epi }
                    }
                    DitOp::Pack { from, norms, rope, .. } => {
                        ensure!(!rope && crate::denoiser::pack_parts(norms).iter().all(|n| n.is_empty()), "the speech engine packs q, k and v without norms or rope");
                        EncStep::Pack { from }
                    }
                    DitOp::JointAttention { into, .. } => EncStep::Attend { into },
                    _ => bail!("the speech engine has no {} encoder op", d.label()),
                });
            }
            Ok(())
        };
        lower(&ep.head, None, &mut enc, &mut wants)?;
        for (b, g) in ep.blocks.iter().zip(&eg.block) {
            for i in 0..es.size(&g.count)? {
                lower(&b.ops, Some(i), &mut enc, &mut wants)?;
            }
        }
        lower(&ep.tail, None, &mut enc, &mut wants)?;
        let mut enc_bufs = BTreeMap::new();
        for d in eg.buffer.iter().chain(&eg.input).chain(&eg.output) {
            enc_bufs.insert(crate::denoiser::intern_name(&d.name), s.alloc_zeros::<f32>(enc_rows * es.size(&d.cols)?)?);
        }
        let eh = es.hidden();
        let enc_head_dim = es.size("head_dim")?;

        let spec = &decoder.spec;
        let (h, qd, vocab) = (spec.dims.hidden, spec.q_dim(), spec.dims.vocab);
        ensure!(spec.norm.kind == crate::spec::NormKind::LayerNorm && !spec.norm.unit_offset, "the speech engine runs a LayerNorm decoder");
        ensure!(spec.attention.kv_heads == spec.attention.heads, "the speech engine runs equal query and key heads");
        let cross = spec.cross.as_ref().context("the spec has no [cross] section")?;
        let mut dec = Vec::new();
        let mut sizes: BTreeMap<Buf, usize> = BTreeMap::from([(Buf::X, h)]);
        let lin_of = |weight: Weight, layer: Option<usize>, rows: usize, epi: Epi, wants: &mut Vec<LineKey>, up: &mut dyn FnMut(&[f32]) -> Result<CudaSlice<f32>>| -> Result<Lin> {
            let (w, b, n, k) = decoder.linear_weights(weight, layer)?;
            wants.push((n, k, epi, rows_family(n, k, rows)));
            Ok(Lin { w: up(&w)?, b: up(&b)?, n, k })
        };
        let lower_dec = |ops: &[Op], layer: Option<usize>, dec: &mut Vec<DecStep>, sizes: &mut BTreeMap<Buf, usize>, wants: &mut Vec<LineKey>, up: &mut dyn FnMut(&[f32]) -> Result<CudaSlice<f32>>| -> Result<()> {
            for op in ops {
                dec.push(match *op {
                    Op::Embed => DecStep::Embed,
                    Op::PositionAdd => DecStep::Position,
                    Op::Norm { which, into } => {
                        let (g, b) = decoder.norm_weights(which, layer.filter(|_| which != NormKind::Final))?;
                        sizes.insert(into, h);
                        DecStep::Norm { w: up(&g)?, b: up(&b)?, into }
                    }
                    Op::Linear { weight, input, output, .. } => {
                        let epi = epi_of(spec, weight)?;
                        let lin = lin_of(weight, layer, 1, epi, wants, up)?;
                        ensure!(sizes.get(&input).is_none_or(|&c| c >= lin.k), "{}: the input buffer is narrower than {}", weight.name(), lin.k);
                        let e = sizes.entry(output).or_insert(lin.n);
                        *e = (*e).max(lin.n);
                        DecStep::Linear { lin, input, output, epi }
                    }
                    Op::KvRope | Op::DecodeRope => DecStep::KvWrite { layer: layer.context("the kv write runs inside a layer")? },
                    Op::Flash | Op::DecodeAttention => {
                        sizes.insert(Buf::AttnPlanes, qd);
                        DecStep::SelfAttend { layer: layer.context("attention runs inside a layer")?, write: *op == Op::DecodeAttention }
                    }
                    Op::CrossAttention { .. } => {
                        sizes.insert(Buf::AttnPlanes, qd);
                        DecStep::CrossAttend { layer: layer.context("cross attention runs inside a layer")? }
                    }
                    ref other => bail!("the speech engine does not run decoder op {}", other.label()),
                });
            }
            Ok(())
        };
        let plan = &decoder.decode;
        lower_dec(&plan.head, None, &mut dec, &mut sizes, &mut wants, &mut up)?;
        for i in 0..spec.layers.count {
            lower_dec(plan.layer(spec.block(i)), Some(i), &mut dec, &mut sizes, &mut wants, &mut up)?;
        }
        lower_dec(&plan.tail, None, &mut dec, &mut sizes, &mut wants, &mut up)?;
        ensure!(decoder.prefill.window == vec![Op::CrossKv], "the speech engine runs a window of one cross_kv op, not {:?}", decoder.prefill.window);
        let mut cross_kv = Vec::new();
        for i in 0..spec.layers.count {
            cross_kv.push(lin_of(Weight::CrossKv, Some(i), cross.rows, Epi::Store, &mut wants, &mut up)?);
        }
        ensure!(sizes.get(&Buf::Logits) == Some(&vocab), "the decode plan writes no logits of {vocab}");
        let mut dec_bufs = BTreeMap::new();
        for (b, n) in &sizes {
            dec_bufs.insert(*b, s.alloc_zeros::<f32>(*n)?);
        }
        let embed = up(decoder.tensor("model.embed_tokens.weight")?)?;
        let pos_name = spec.dims.positions.clone().context("no [embedding] positions")?;
        let pos = decoder.tensor(&format!("model.{pos_name}"))?;
        let max_positions = pos.len() / h;
        let positions = up(pos)?;

        let mut convs: BTreeMap<String, (LineKey, ConvKernel)> = BTreeMap::new();
        for w in &wants {
            let k = line_kernel(&target, *w)?;
            convs.entry(k.name()).or_insert((*w, k));
        }
        let list: Vec<ConvKernel> = convs.values().map(|(_, k)| k.clone()).collect();
        let names: Vec<&'static str> = list.iter().map(|k| &*Box::leak(k.name().into_boxed_str())).collect();
        let module = jit.load(&Module { source: kir::cuda::emit_line(&list).map_err(anyhow::Error::msg)?, kernels: names, smem: BTreeMap::new(), arch_features: false, kir_hash: kir::fold_hashes(list.iter().map(ConvKernel::hash)) })?;
        let mut line = BTreeMap::new();
        for (key, k) in convs.into_values() {
            let f = module.function(&k.name())?;
            use cudarc::driver::sys::CUfunction_attribute as A;
            f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k.smem_bytes() as i32)?;
            line.insert(key, (k, f));
        }

        let enc_eps = es.get("norm_eps")? as f32;
        let eps = spec.norm.eps as f32;
        let threads = 256;
        let mut rows_kernels = Vec::new();
        for (wd, e) in [(eh, enc_eps), (h, eps)] {
            let name: &'static str = Box::leak(format!("speech_layernorm_{wd}_{}", e.to_bits()).into_boxed_str());
            let params = vec![Param { name: "x", ty: Scalar::F32, role: Role::Activation, access: Access::Write }, Param { name: "w", ty: Scalar::F32, role: Role::Weight, access: Access::Read }, Param { name: "b", ty: Scalar::F32, role: Role::Weight, access: Access::Read }];
            rows_kernels.push((wd, e, kir::Kernel::rows(&target, name, threads, params, kir::rows::RowBody::LayerNorm { width: wd as u32, row: wd as u32, head: 0, offset: 0, eps: e }).map_err(anyhow::Error::msg)?));
        }
        let mut add_kernels = Vec::new();
        for wd in [eh, h] {
            let name: &'static str = Box::leak(format!("speech_add_{wd}").into_boxed_str());
            add_kernels.push((wd, kir::Kernel::add(&target, name, wd as u32, threads).map_err(anyhow::Error::msg)?));
        }
        let mut kernels: Vec<kir::Kernel> = rows_kernels.iter().map(|r| r.2.clone()).chain(add_kernels.iter().map(|a| a.1.clone())).collect();
        kernels.sort_by_key(|k| k.name);
        kernels.dedup_by_key(|k| k.name);
        let names: Vec<&'static str> = kernels.iter().map(|k| k.name).collect();
        let rows_mod = jit.load(&Module { source: kir::cuda::emit_module(&kernels).map_err(anyhow::Error::msg)?, kernels: names, smem: BTreeMap::new(), arch_features: false, kir_hash: kir::module_hash(&kernels) })?;
        let mut norms = BTreeMap::new();
        for (wd, e, k) in &rows_kernels {
            norms.insert((*wd, e.to_bits()), rows_mod.function(k.name)?);
        }
        let mut adds = BTreeMap::new();
        for (wd, k) in &add_kernels {
            adds.insert(*wd, rows_mod.function(k.name)?);
        }
        let norm_threads = rows_kernels[0].2.threads;

        ensure!(enc_head_dim == spec.attention.head_dim, "the encoder and decoder heads differ ({enc_head_dim} and {})", spec.attention.head_dim);
        let vk = kir::vision::VisionKernel::new(&target, enc_head_dim, true).map_err(anyhow::Error::msg)?.joint();
        let mut vnames = kir::vision::KERNELS.to_vec();
        vnames.extend_from_slice(kir::vision::JOINT_KERNELS);
        let vision = jit.load(&Module { source: kir::cuda::emit_vision(&vk).map_err(anyhow::Error::msg)?, kernels: vnames, smem: BTreeMap::new(), arch_features: false, kir_hash: kir::fnv_hash(&vk.hash()) })?;
        let attend = vision.function("vis_flash_kv_rows")?;
        {
            use cudarc::driver::sys::CUfunction_attribute as A;
            let most = enc_rows.max(cross.rows).max(max_positions);
            attend.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kir::vision::cross_rows_smem(most) as i32)?;
        }
        let widest = wants.iter().map(|w| w.0).max().unwrap_or(1);
        let a = &spec.attention;
        let scale = a.query_scalar.map_or(1.0 / (a.head_dim as f32).sqrt(), |q| (1.0 / q.sqrt()) as f32);
        let engine = GpuEngine {
            speech,
            mel_samples: host.plan.mel.samples,
            enc,
            enc_bufs,
            enc_q: s.alloc_zeros::<f32>(enc_rows * eh)?,
            enc_kv: s.alloc_zeros::<f32>(enc_rows * 2 * eh)?,
            enc_rows,
            dec,
            cross_kv,
            dec_bufs,
            embed,
            positions,
            max_positions,
            cross_rows: cross.rows,
            layers: spec.layers.count,
            hidden: h,
            q_dim: qd,
            vocab,
            heads: a.heads,
            head_dim: a.head_dim,
            scale,
            enc_heads: eh / enc_head_dim,
            enc_head_dim,
            enc_eps: enc_eps.to_bits(),
            eps: eps.to_bits(),
            line,
            norms,
            adds,
            norm_threads,
            attend,
            ones: s.clone_htod(&vec![1f32; widest])?,
            zero: s.alloc_zeros::<f32>(1)?,
            weight_bytes,
            s,
        };
        engine.s.synchronize()?;
        let plan = host.plan.clone();
        Ok((engine, decoder, plan))
    }

    fn ptr(&self, v: &CudaSlice<f32>) -> u64 {
        v.device_ptr(&self.s).0
    }

    fn linear(&self, lin: &Lin, x: u64, y: u64, rows: usize, epi: Epi) -> Result<()> {
        let key = (lin.n, lin.k, epi, rows_family(lin.n, lin.k, rows));
        let (k, f) = self.line.get(&key).with_context(|| format!("no line kernel for {key:?}"))?;
        let (st, w, b, sc) = (self.ptr(&self.zero), self.ptr(&lin.w), self.ptr(&lin.b), self.ptr(&self.ones));
        let (bb, t) = (1i32, rows as i32);
        let (gx, gy, gz) = k.line_grid(rows);
        let mut lb = self.s.launch_builder(f);
        lb.arg(&x).arg(&st).arg(&w).arg(&b).arg(&sc).arg(&y).arg(&bb).arg(&t).arg(&t);
        unsafe { lb.launch(LaunchConfig { grid_dim: (gx, gy, gz), block_dim: (kir::conv::THREADS, 1, 1), shared_mem_bytes: k.smem_bytes() }) }?;
        Ok(())
    }

    fn layer_norm(&self, width: usize, eps: u32, x: u64, w: &CudaSlice<f32>, b: &CudaSlice<f32>, rows: usize) -> Result<()> {
        let f = self.norms.get(&(width, eps)).with_context(|| format!("no layer norm of width {width}"))?;
        let (wp, bp) = (self.ptr(w), self.ptr(b));
        let mut lb = self.s.launch_builder(f);
        lb.arg(&x).arg(&wp).arg(&bp);
        unsafe { lb.launch(LaunchConfig { grid_dim: (1, rows as u32, 1), block_dim: (self.norm_threads, 1, 1), shared_mem_bytes: 0 }) }?;
        Ok(())
    }

    fn add(&self, width: usize, x: u64, delta: u64, rows: usize) -> Result<()> {
        let f = self.adds.get(&width).with_context(|| format!("no add of width {width}"))?;
        let mut lb = self.s.launch_builder(f);
        lb.arg(&x).arg(&delta).arg(&x);
        unsafe { lb.launch(LaunchConfig { grid_dim: (width.div_ceil(256) as u32, rows as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attend(&self, q: u64, kv: u64, out: u64, kv_rows: usize, seqs: usize, seq_stride: i64, heads: usize, head_dim: usize, scale: f32) -> Result<()> {
        let (r, hh, hd) = (kv_rows as i32, heads as i32, head_dim as i32);
        let mut lb = self.s.launch_builder(&self.attend);
        lb.arg(&q).arg(&kv).arg(&out).arg(&r).arg(&hh).arg(&hd).arg(&seq_stride).arg(&scale);
        unsafe { lb.launch(LaunchConfig { grid_dim: (heads as u32, seqs as u32, 1), block_dim: (kir::vision::ROWS_THREADS, 1, 1), shared_mem_bytes: kir::vision::cross_rows_smem(kv_rows) }) }?;
        Ok(())
    }

    fn copy(&self, dst: u64, src: u64, floats: usize) -> Result<()> {
        unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, floats * 4, self.s.cu_stream()) }?;
        Ok(())
    }

    fn copy_2d(&self, dst: u64, dst_pitch: usize, src: u64, src_pitch: usize, width: usize, height: usize) -> Result<()> {
        let c = sys::CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: src,
            srcArray: std::ptr::null_mut(),
            srcPitch: src_pitch * 4,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst,
            dstArray: std::ptr::null_mut(),
            dstPitch: dst_pitch * 4,
            WidthInBytes: width * 4,
            Height: height,
        };
        unsafe { sys::cuMemcpy2DAsync_v2(&c, self.s.cu_stream()) }.result()?;
        Ok(())
    }

    fn enc_ptr(&self, name: &str) -> Result<u64> {
        Ok(self.ptr(self.enc_bufs.get(name).with_context(|| format!("no encoder buffer {name}"))?))
    }

    fn dec_ptr(&self, b: Buf) -> Result<u64> {
        Ok(self.ptr(self.dec_bufs.get(&b).with_context(|| format!("no decoder buffer {}", b.label()))?))
    }

    fn run_decoder(&self, seq: &mut GpuSeq, token: u32) -> Result<()> {
        ensure!(seq.len < self.max_positions, "position {} is past the learned table", seq.len);
        ensure!((token as usize) < self.vocab, "token {token} is past the vocabulary");
        let (h, qd) = (self.hidden, self.q_dim);
        let x = self.dec_ptr(Buf::X)?;
        for step in &self.dec {
            match step {
                DecStep::Embed => self.copy(x, self.ptr(&self.embed) + (token as usize * h * 4) as u64, h)?,
                DecStep::Position => self.add(h, x, self.ptr(&self.positions) + (seq.len * h * 4) as u64, 1)?,
                DecStep::Norm { w, b, into } => {
                    let y = self.dec_ptr(*into)?;
                    self.copy(y, x, h)?;
                    self.layer_norm(h, self.eps, y, w, b, 1)?;
                }
                DecStep::Linear { lin, input, output, epi } => self.linear(lin, self.dec_ptr(*input)?, self.dec_ptr(*output)?, 1, *epi)?,
                DecStep::KvWrite { layer } => self.copy(self.ptr(&seq.kv[*layer]) + (seq.len * 2 * qd * 4) as u64, self.dec_ptr(Buf::Qkv)? + (qd * 4) as u64, 2 * qd)?,
                DecStep::SelfAttend { layer, write } => {
                    let qkv = self.dec_ptr(Buf::Qkv)?;
                    let kv = self.ptr(&seq.kv[*layer]);
                    if *write {
                        self.copy(kv + (seq.len * 2 * qd * 4) as u64, qkv + (qd * 4) as u64, 2 * qd)?;
                    }
                    self.attend(qkv, kv, self.dec_ptr(Buf::AttnPlanes)?, seq.len + 1, 1, 0, self.heads, self.head_dim, self.scale)?;
                }
                DecStep::CrossAttend { layer } => {
                    self.attend(self.dec_ptr(Buf::Qkv)?, self.ptr(&seq.cross[*layer]), self.dec_ptr(Buf::AttnPlanes)?, self.cross_rows, 1, 0, self.heads, self.head_dim, self.scale)?;
                }
            }
        }
        seq.len += 1;
        Ok(())
    }

    fn logits(&self) -> Result<Vec<f32>> {
        Ok(self.s.clone_dtoh(self.dec_bufs.get(&Buf::Logits).context("no logits buffer")?)?)
    }
}

impl Engine for GpuEngine {
    type States = CudaSlice<f32>;
    type Seq = GpuSeq;

    fn log_mel(&self, plan: &SpeechPlan, pcm: &[f32]) -> Result<Vec<f32>> {
        if pcm.len() <= self.mel_samples { self.speech.mel(pcm) } else { log_mel_clip(plan, pcm) }
    }

    fn encode(&self, window: &[f32]) -> Result<CudaSlice<f32>> {
        let stem = self.speech.stem(window)?;
        let rows = self.enc_rows;
        let h = self.enc_heads * self.enc_head_dim;
        let stem_buf = self.enc_ptr("stem")?;
        ensure!(stem.len() == rows * h, "{} stem values, the encoder reads {rows} rows of {h}", stem.len());
        self.copy(stem_buf, self.ptr(&stem), rows * h)?;
        for step in &self.enc {
            match step {
                EncStep::Add { table, from, into } => {
                    let (f, t) = (self.enc_ptr(from)?, self.enc_ptr(into)?);
                    if f != t {
                        self.copy(t, f, rows * h)?;
                    }
                    self.add(h, t, self.ptr(table), rows)?;
                }
                EncStep::Norm { w, b, from, into } => {
                    let (f, t) = (self.enc_ptr(from)?, self.enc_ptr(into)?);
                    self.copy(t, f, rows * h)?;
                    self.layer_norm(h, self.enc_eps, t, w, b, rows)?;
                }
                EncStep::Linear { lin, input, output, epi } => self.linear(lin, self.enc_ptr(input)?, self.enc_ptr(output)?, rows, *epi)?,
                EncStep::Pack { from } => {
                    let src = self.enc_ptr(from)?;
                    self.copy_2d(self.ptr(&self.enc_q), h, src, 3 * h, h, rows)?;
                    self.copy_2d(self.ptr(&self.enc_kv), 2 * h, src + (h * 4) as u64, 3 * h, 2 * h, rows)?;
                }
                EncStep::Attend { into } => {
                    let scale = 1.0 / (self.enc_head_dim as f32).sqrt();
                    self.attend(self.ptr(&self.enc_q), self.ptr(&self.enc_kv), self.enc_ptr(into)?, rows, rows, 0, self.enc_heads, self.enc_head_dim, scale)?;
                }
            }
        }
        let out = self.s.alloc_zeros::<f32>(rows * h)?;
        self.copy(self.ptr(&out), self.enc_ptr("states")?, rows * h)?;
        Ok(out)
    }

    fn start(&self, states: &CudaSlice<f32>) -> Result<GpuSeq> {
        ensure!(states.len() == self.cross_rows * self.hidden, "{} encoder values, the [cross] section reads {} rows of {}", states.len(), self.cross_rows, self.hidden);
        let mut cross = Vec::with_capacity(self.layers);
        for lin in &self.cross_kv {
            let out = self.s.alloc_zeros::<f32>(self.cross_rows * lin.n)?;
            self.linear(lin, self.ptr(states), self.ptr(&out), self.cross_rows, Epi::Store)?;
            cross.push(out);
        }
        let kv = (0..self.layers).map(|_| self.s.alloc_zeros::<f32>(self.max_positions * 2 * self.q_dim)).collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(GpuSeq { cross, kv, len: 0 })
    }

    fn prefill(&self, seq: &mut GpuSeq, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(tokens.len() * self.vocab);
        for &t in tokens {
            self.run_decoder(seq, t)?;
            out.extend(self.logits()?);
        }
        Ok(out)
    }

    fn step(&self, seq: &mut GpuSeq, token: u32) -> Result<Vec<f32>> {
        self.run_decoder(seq, token)?;
        self.logits()
    }
}

impl GpuTranscriber {
    pub fn open(stream: &Arc<CudaStream>, jit: &Jit, dir: &Path) -> Result<GpuTranscriber> {
        let (engine, decoder, plan) = GpuEngine::open(stream, jit, dir)?;
        let rules = rules_for(dir, &decoder)?;
        Ok(Transcriber { plan, rules, engine })
    }
}
