# Engine defaults and profiles (#598)

Every engine option lives in the option structs (`ServeOptions`, `SpeculativeOptions`, `RuntimeOptions`,
`PrefixCacheOptions`, `LoaderOptions`). `crates/specrt/src/options.rs` flattens them into a schema (key, kind, default,
group, label, help) that the load dialog renders. A test fails when an option is declared without a label, help line
and group, so a new option shows up in the app without UI work.

## Profiles

| profile | what it sets |
|---|---|
| max performance (default) | MTP on, CUDA graphs on, fused kernels on, fast loader on with its #583 defaults, prefix cache on, 8 parallel requests |
| low memory | no MTP draft head copy, prefix cache off with 0 snapshots, loader host budget 512 MB |
| custom | shown as soon as one advanced field differs from the profile |

The choice is kept per model in `settings.json` (`model_engine`) with an app-wide default (`default_profile`, Settings).
The MTP draft head precision and draft count default to the model spec (nvfp4_g16, 2 tokens on the 27B, see
`docs/specrt-nvfp4.md`).

## Parallel requests: measured on the 27B

Qwen3.8-27B NVFP4, MTP k=2 with an nvfp4_g16 draft head, RTX PRO 6000 Max-Q, vLLM on the other card, 3 runs each,
512 token prompts, 256 token replies, `crates/specrt/scripts/seqs598.sh`.

| concurrent requests | decode tok/s aggregate (3 runs) | per sequence | p95 inter-token |
|---|---|---|---|
| 1 | 96.4, 98.2, 96.3 | 97 | 28 ms |
| 2 | 171.1, 175.8, 169.9 | 86 | 30 ms |
| 4 | 269.3, 261.5, 255.8 | 65 | 35 ms |
| 8 | 362.7, 367.5, 370.7 | 46 | 43 ms |

Single stream with the cap set to 1, 4 and 8 (`max_num_seqs`): 97.8/98.6/98.8, 98.1/99.2/99.1 and 96.4/98.2/96.3 tok/s.
The cap does not slow a lone request, and aggregate throughput still grows at 8 while each stream stays above 45
tok/s. The cost of a slot is one recurrent state (150 MB on the 27B), 1.2 GB for 8, taken out of the memory the
user chose. So the default is 8 (`DEFAULT_MAX_NUM_SEQS`, was 32 in `ServeOptions` and 4 in the app): the largest
count measured, with single-stream decode at full speed. The KV pool is shared, so it does not grow with the count.
