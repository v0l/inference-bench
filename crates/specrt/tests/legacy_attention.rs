use cudarc::driver::CudaContext;
use specrt::codegen::{Module, cuda_target};
use specrt::jit::Jit;
use specrt::spec::Spec;
use std::path::Path;

#[test]
#[ignore = "needs a CUDA GPU"]
fn legacy_attention_kernels_load_at_gqa_32_over_2_with_head_dim_256() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5");
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.attention.heads = 32;
    spec.attention.kv_heads = 2;
    spec.attention.head_dim = 256;
    spec.runtime.gemm_fp32 = true;
    let ctx = CudaContext::new(0).unwrap();
    let target = cuda_target(&ctx).unwrap();
    let module = Module::generate_kv(&spec, 256, 2, &target).unwrap();
    let loaded = Jit::new(ctx.clone(), specrt::jit::default_cache_dir()).unwrap().load(&module).unwrap();
    use cudarc::driver::sys::CUfunction_attribute as A;
    for name in ["attention_split", "flash_attention"] {
        let k = loaded.function(name).unwrap();
        let dynamic = module.smem.get(name).copied().unwrap_or(0);
        k.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic as i32).unwrap();
        let fixed = k.get_attribute(A::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES).unwrap() as u32;
        assert!(fixed + dynamic <= target.smem_limit, "{name} needs {fixed} + {dynamic} bytes of shared memory");
    }
}
