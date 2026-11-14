# LM Studio load panel against ours

Sources: lmstudio.ai docs (per-model defaults, parallel requests, speculative decoding, `lms load`) and the
screenshots saved in `docs/gui/lmstudio/` (`load-model.png`, `save-load-changes.png`,
`speculative-decoding-setting.png`, `model-settings-gear.webp`).

## What LM Studio does

- **Two steps.** Pick a model from a filterable list (sort by recency or size). A checkbox at the foot,
  "Manually choose model load parameters" (or hold Option), decides whether the parameters page opens at all.
  Without it the model loads with its saved defaults.
- **Up front** (the parameters page, `save-load-changes.png`): context length as a number box over a slider with
  "Model supports up to N tokens" under it, then GPU offload as `13 / 28` layers with a slider and a bin icon that
  resets it. A changed field shows a blue rail on the left and the bin key.
- **Below, in one flat list:** evaluation batch size, RoPE frequency base and scale (a checkbox, "Auto" until
  ticked), keep model in memory, try mmap, seed, flash attention (tagged Experimental). A "Show advanced settings"
  toggle reveals more, for example Max Concurrent Predictions (default 4, parallel requests).
- **Footer:** "Remember settings for <model>" checkbox, Cancel, Load Model (Cmd+Enter). The same fields live in the
  gear dialog under My Models, so per-model defaults are edited without loading.
- **Memory:** no live figure in the panel. The model size sits beside the title (3.42 GB). A resource estimate
  exists only in the CLI (`lms load --estimate-only`, which honours context, GPU offload, flash attention and
  vision) and as a guardrail check at load time. Offload is by layers, not by bytes.
- **Speculative decoding** is not a load parameter. It is a section in the chat sidebar (Power User mode or
  higher) where the user picks a second, smaller draft model. LM Studio's own advice is that it can make things
  slower. Per-request sampling lives in the same sidebar, with the same bin-key reset per field.
- **No profiles.** There are presets, but they cover chat settings (prompt, sampling), not load parameters.

## What we copy

- The two-level shape: the few fields everyone needs first, a toggle for the rest.
- "Model supports up to N tokens" under the context field, and the model size next to the title.
- A per-field reset key plus a marker on a field that differs from its default.
- "Remember settings for this model" in the footer, and the same dialog reachable without loading (the gear
  dialog; ours is the load dialog opened from the model detail page).
- Fields that are off by default say "Auto" instead of a made-up number.

## What we do differently

- **Memory is the main knob, in GB, with a live answer.** LM Studio makes the user guess layers and read a CLI
  estimate. We show weights, KV, state, total and free on the card as the user drags, and what the KV pool buys:
  concurrency at the full context and at 4k tokens. A context that does not fit the memory chosen turns red and
  disables Load.
- **No layer offload slider.** The engine keeps the model on the GPU; the memory control sizes the KV pool.
- **MTP is a plain switch, on by default.** The model carries its own draft head, so there is no second model to
  find. LM Studio's draft-model dance and its slowdown warning do not apply.
- **Defaults are the measured best, not "Auto".** A profile (Max performance, Low memory, Custom) sets the lot.
  LM Studio has no equivalent.
- **Every engine option is listed**, generated from the option structs in specrt, so a new option shows up
  without UI work. LM Studio hand-picks about ten.
- **Advanced is folded**, but unlike theirs it is complete, grouped (MTP, prefix cache, serving, runtime, loader)
  and each field shows its default.
