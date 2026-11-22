# Measured split counts for the batched fp4/fp8 GEMV (#408)

Result: the tuner was built, wired and measured, and dropped. No shape gains 5% on a repeatable basis, and the one that does (gate_up batch 16, 5.7%) is within the run to run spread of the same kernel (formula 99.8 us in one pass and 100.8 us in the other).

Step 1 (env knobs to config): `SPECRT_TINY_WN`, `SPECRT_FP8_BK`, `SPECRT_BK` and `SPECRT_STAGES` are no longer read anywhere in `crates/specrt/src` or `crates/kir/src`. The tiny tile width is the constant `TINY_WN` in `crates/kir/src/gemm.rs`, and the stages are `[runtime] fp4_stages`. Remaining env reads in src: `HF_HUB_CACHE`, `HF_HOME` (weights.rs, Hugging Face standard), `SPECRT_NO_MADVISE` (gguf.rs), `SPECRT_NO_COOP`, `SPECRT_COUNT`, `SPECRT_CHECKED` (wgpu backend debugging), `COMPUTERNAME` and `HOSTNAME` (hardware.rs, host name). Tests still read `SPECRT_GEMM_TOKENS`, `SPECRT_BK`, `SPECRT_FP8_BK` (docs/testing.md).

Step 2 measurement: every candidate split count (formula/2, formula-1, formula, formula+1, formula*2, within the partial buffer) of the batched kernels on the real 27B shapes, one GPU, 10 launches per sample, best of 20 rounds, tokens 1, 4, 8, 16 (32 for down and gate_up). All candidates matched the formula output (max diff 7.8e-3 on gate_up act outputs of scale about 1e2, 2e-4 on down), and the tiny27b logits and greedy tests passed with `tune = "force"`.

Best candidate against the formula, per shape (us):

| shape | tokens | formula | best | gain |
|---|---|---|---|---|
| qkv 14336x5120 fp8 | 1 / 4 / 8 / 16 | 36.1 / 38.0 / 39.9 / 55.0 | 35.8 / 37.4 / 39.0 / 54.1 | 0.8 / 1.6 / 2.3 / 1.6% |
| lin_in 16384x5120 fp8 | 1 / 4 / 8 / 16 | 46.3 / 48.5 / 50.2 / 70.6 | none better | 0% |
| o 5120x6144 fp8 | 1 / 4 / 8 / 16 | 18.2 / 18.7 / 19.2 / 24.9 | none better | 0% |
| down 5120x17408 fp4 | 1 / 4 / 8 / 16 / 32 | 29.7 / 31.6 / 33.2 / 44.7 / 79.5 | 29.7 / 31.6 / 33.2 / 44.7 / 77.1 | 0 / 0 / 0 / 0 / 3.0% |
| gate_up 34816x5120 fp4 | 4 / 8 / 16 / 32 | 65.1 / 68.5 / 100.8 / 156.4 | 62.7 / 67.7 / 95.1 / 156.0 | 3.7 / 1.2 / 5.7 / 0.3% |

Decision: remove step 2 from the branch, keep the conclusion that `f4b_splits` and `f8b_splits` (`BatchShape::splits`) are within noise of the best split. The gaps reported in #413 for `lin_in` and `qkv` against vLLM come from the kernel body, not from the split count.
