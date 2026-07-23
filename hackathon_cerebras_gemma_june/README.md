# Future Shield

Future Shield is a **multiverse pre-action simulator** for safety-critical AI.
Before an action commits to reality, the engine forks N candidate actions,
probes each one K ticks into the future with Cerebras-hosted Gemma
(`gemma-4-31b`), and only commits the surviving branch to the live mainline.
The result: real-time scenario-aware "what if I do this?" reasoning at
~1500+ tokens/sec on Cerebras hardware.

Built for the **Cerebras × Gemma June hackathon**.

## How it works

```
mainline t0 ──► [selector picks N candidate actions]
                ├── fork A: probe K steps ahead → safe   ✓ commit candidate
                ├── fork B: probe K steps ahead → failure ✗ killed
                └── fork C: probe K steps ahead → stall   ✗ killed
                       │
                       ▼
            survivor collapses to mainline t1
```

Every fork and every committed mainline tick carries its own scenario-specific
reasoning (a short recap + bullet points) produced by Gemma. When the run
ends, a final Gemma call writes a ~100-word verdict (`SUCCEEDED` /
`CRASHED` / `STALLED` / `INCONCLUSIVE`) that cites the actual entities from
the scenario.

There is **no mock and no fallback**. If the Cerebras call fails, the fork
fails surfaced in the UI and in the JSONL log.

## Two-file demo

By design the demo is just two files:

- `multiverse_1.py` engine + asyncio control loop + stdlib HTTP server.
- `multiverse_1.html` single-page UI (vanilla JS, dynamic SVG, polls `/api/state`).

## Setup

You need a Cerebras API key (free signup at https://cloud.cerebras.ai). Export
it as `CEREBRAS_API_KEY` before launching:

**macOS / Linux:**

```bash
export CEREBRAS_API_KEY="csk-..."
python multiverse_1.py --port 8762
```

**Windows (PowerShell):**

```powershell
$env:CEREBRAS_API_KEY = "csk-..."
python .\multiverse_1.py --port 8762
```

Then open <http://127.0.0.1:8762/>.

The server starts paused. Press **▶ Play** in the UI when you're ready to
spend model calls. Tune `forks/tick`, `look-ahead`, `max wait per Gemma call`,
and the scenario text from the sidebar before pressing Play.

## Telemetry

The top metric bar shows:

- `tokens` exact `(prompt + completion − cached)` across every Cerebras call
  in the run, read from the response's `usage` field. Not estimated.
- `tps` real generation throughput from Cerebras's `time_info.completion_time`
  (model-side only excludes queue, prompt eval, network).

## Logs

Every selector call, probe call, and run lifecycle event is appended to:

```
logs/future_shield_backend.jsonl
```

Each record includes the full prompt, raw model output, parsed packet,
exact token counts, and timing. The UI's `Run Evidence` panel shows the same
records click a row to inspect.

## API surface

```
GET  /              UI (multiverse_1.html)
GET  /api/health    provider/model/endpoint/log path
GET  /api/state     current multiverse snapshot (mainline, forks, run_summary, meter)
GET  /api/logs      recent JSONL records
POST /api/play      start or resume the loop
POST /api/pause     pause the loop
POST /api/reset     reset the current run
POST /api/config    apply sidebar controls
POST /api/generate-scenario   ask Gemma to write a 100-word scenario from a seed
```

## CLI flags

```
--host HOST                  bind address (default 127.0.0.1)
--port PORT                  port (default 8762)
--cerebras-endpoint URL      override endpoint
--model NAME                 override model (default gemma-4-31b)
--max-output-tokens N        per-call max output tokens (default 800)
--temperature F              sampling temperature (default 0.3)
--no-serve                   headless: run one simulation, print final snapshot, exit
```

## License

MIT; see [`../LICENSE`](../LICENSE).
