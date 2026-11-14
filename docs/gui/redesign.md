# App redesign: an inference server, LM Studio style

Issue #577. Goal g5: an LM Studio style desktop app with a model library and downloads, model
details, chat and server control. Benchmarking and comparison are not a user-facing feature (non-goal);
the `inference-bench bench`, `batch` and `matrix` commands and the result store in `crates/core` stay
as dev tooling for g6 and have no screen.

## The flow

```
pick a model (search)  ->  load it (per-load options)  ->  chat with it, or serve it
        models                 load dialog                     chat / server
```

The loaded model and the memory it takes are visible on every page: a model bar on chat and server,
and the status bar at the bottom everywhere.

## What the old app got wrong

| old | problem | new |
|---|---|---|
| bench tab, bench targets in settings | a benchmark product | removed from the app, CLI only |
| logs tab | a page of its own for a server concern | a pane on the server page |
| `specrt_ctx` and `gpu_bandwidth_gbs` in settings | context is a per-model load choice; bandwidth is a property of the card, not a preference | context slider in the load dialog; bandwidth input and the decode ceiling row are gone |
| model list unsearchable | a library of dozens of checkpoints cannot be scanned | search field filters by name, repo, architecture, quant and format |
| placement bands on the model detail | multi-host deployment is a server concern | stays in the server TOML, read-only on the server page |
| llama.cpp and vLLM support chips on every model | comparison with other engines | one chip: whether this app runs the model |
| icon of bars and a measured line | a benchmark chart | a server stack with status lights |

## Screens

Rail: **chat**, **models**, **server**, **settings**. Four items, no bench, no logs.

### Model bar (chat and server)

Purpose: the one place that says what is loaded and lets you change it, as in LM Studio's top bar.

```
+----------------------------------------------------------------------------------------+
| (o) ready  [ Qwen3-0.6B  4k ctx ]  [ select a model v ]  [ UNLOAD ]       VRAM 11 / 96 GB |
+----------------------------------------------------------------------------------------+
```

- Lamp and chip: loading, ready, failed (the egui_bench chat header states).
- Picker: a popup with a search field above the usable models. Choosing one opens the load dialog
  for it. It never loads silently, so the context choice is always made.
- `unload` frees the card. `load` appears when nothing is loaded and a model is picked.
- VRAM is the card's used and total, measured (cyan).

### Load dialog

A modal, 520 wide, opened from the model bar, the library cards and the model detail.

```
LOAD QWEN3-0.6B
+- context ---------------------------------------------------------+
| context length   [=========o-----------------------]  8192 tokens   |
|                  512                              40960 (model max)|
+- memory ----------------------------------------------------------+
| weights          1.2 GB          (from the spec and checkpoint)     |
| kv cache         0.9 GB          (8192 tokens)                      |
| state            0 B                                                |
| total            2.1 GB   of  38.4 GB free on the GPU               |
| (ok) fits in free memory                                            |
+--------------------------------------------------------------------+
| [x] remember for this model                                         |
|                                        [ LOAD ]  [ CANCEL ]         |
+--------------------------------------------------------------------+
```

Fields:

- **context length**: slider from 512 to the model's own maximum (`max_position_embeddings` from
  the config via the detail view, bounded by the engine's `MAX_CTX`), step 512, amber because the
  user sets it. The default is the last choice for this model, else the app default (8192),
  clamped to the maximum, and reduced to the largest value that fits free memory when it would not.
- **memory**: weights plus kv cache plus recurrent state from `ModelAnatomy`, the same numbers
  the engine uses, against `total - used` from the GPU poll. Cyan, as measured or derived from the
  device. The verdict line is `status(ok)`: green when it fits, red "does not fit" when it does not.
  With no GPU reading the line says the free memory is unknown and does not block loading.
- **remember for this model**: stores the choice in `settings.json` under the model name.
- **kv cache type**: not shown. The spec has no kv dtype yet (#324). When it does, the dialog lists
  the values the spec and `runtime` accept, never a list of its own (`ui-follows-engine`).
- Load stays enabled when the estimate is red: the estimate is a guess and the engine is the
  authority. The engine's out of memory error comes back as the load failure on the model bar.

### Models

Purpose: library and downloads in one place.

```
MODELS                                                          [ CHECK SPECS NOW ]
[ my models 12 | discover ]
[ search models by name, architecture or quant ............................... ]
 models 12   shown 3   usable 2                                   [x] show unusable
 +- Qwen3-0.6B ------------------------------------------ [safetensors] [loaded] -+
 | size 1.2 GB  arch Qwen3ForCausalLM  quant BF16  context 40960                  |
 | [supported] [vision: no]                                                       |
 | [ LOAD ] [ DETAILS ] [ REVEAL ] [ DELETE ]                                     |
 +--------------------------------------------------------------------------------+
 not here? [ SEARCH HUGGING FACE FOR "qwen" ]
```

- The search field filters the library live. Every word must match the name, repo, architecture,
  quant or format, case-insensitively. Nothing to submit.
- `discover` keeps the Hugging Face search (submit to query), file listing and resumable
  downloads. The library pane offers a shortcut into it with the typed text.
- `load` opens the load dialog. `details` opens the detail view.

### Model detail

What the model is and what loading costs it. Opened from a card, with `back`, `load` and
`compare with` (two checkpoints side by side on one scale is a model view, not a benchmark).

```
[BACK] Qwen3-0.6B  [config.json] [BF16]                       [LOAD]  compare with [ none v]
+- summary -------------------------------------------------------+
| architecture  Qwen3ForCausalLM   parameters 0.6 B   quant BF16   |
| max context   40960              vision no           mtp none    |
+- layers --- block diagram --- vision tower --- parameters and memory --- memory against context
+- numbers --------------------------------------------------------+
```

Removed: placement bands over the layer strip, the decode ceiling row (it needed a bandwidth the
user had to type). `memory against context` stays: it is the cost of loading, drawn against the
card's free memory.

### Server

Purpose: OpenAI server control, its request log and the logs.

```
SERVER
[model bar]
+- local server ----------------------------------------- (o) running -+
| address  http://127.0.0.1:1234/v1        serving  Qwen3-0.6B          |
| [ STOP SERVER ]                                                       |
+- settings (next start) --- port, api key, allow other machines -------+
+- placement ---------------- read-only, from the server TOML ----------+
| objective latency   context 8192   slots 1                            |
| alpha   24 GiB   layers 0..30        beta   24 GiB   layers 30..60    |
+- use it ----------------------------------------------------------+
[ requests | logs ]
 time      request                     status   ms
```

- `requests` is the access table; `logs` is the old logs page (level, search, copy, save, clear).
- `placement` shows only when `server.toml` in the data folder configures devices. It is the
  same `placement` table `inference-bench serve` reads, so the app and the daemon cannot disagree.
  There is no placement editor: multi-host setups are written in the TOML, validated by
  `inference-bench plan`, and nothing about them belongs on a model page. No other screen needs
  multi-host UI.

### Settings

App-wide only: **model folders**, **hugging face token**, **spec updates**, **app** (theme, data
folder). Gone: bench targets, native engine context, gpu bandwidth.

Migration: `specrt_ctx` becomes `default_ctx` (serde alias, so old files read), the starting
point of the load dialog's slider for a model with no remembered choice. `gpu_bandwidth_gbs`,
`endpoints` and `placement` are dropped; serde ignores them in an old file.

## Chat

Unchanged apart from the model bar replacing the ad hoc picker, and the context bar reading the
context the model was loaded with instead of a global setting.

## Icon

Dark tile, a stack of three server units with status lights: the top unit lit cyan (serving), an
amber light for the user's setting, green for ready. It reads as a local server at 16 px and in the
house palette; no bars, no measured line.

## Order of work

1. This document.
2. Remove the bench UI and the settings it needed; drop bandwidth and placement from the detail.
3. Library search.
4. Load dialog with slider and estimates, per-model memory in settings.
5. Model bar.
6. Rail of four, logs folded into server, placement on the server page.
7. Icon, screenshots, README.
