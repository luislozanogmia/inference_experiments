# Gemma 4 31B Tool-State Hallucination

This directory contains a cross-provider reproduction of a Gemma 4 31B failure
mode involving tool execution state.

When a prompt requires tool use but the request supplies no callable tool
schemas, the model can emit tool-call syntax as ordinary assistant text. In a
multi-turn conversation, it can later treat that unexecuted text as a completed
action and report invented results, including exact-looking command output.

The valid-tools controls show a necessary boundary: when real schemas are
supplied, the same model can return proper structured tool calls. The problem is
therefore not that Gemma 4 is generally incapable of tool calling. The failure
appears when generated tool-shaped text is allowed to become unsupported
execution state.

## Evidence overview

| Execution path | No-tools result | Multi-turn result | Valid-tools control |
|---|---|---|---|
| Cerebras API and Pi harness | Tool-shaped prose with no structured call | Unsupported completion claim and invented test output | Structured tool call returned |
| OpenRouter: OpenInference and Novita | No structured call; Novita emitted textual tool syntax | 10/10 sessions claimed unsupported success | Both providers returned structured calls |
| Direct AWS GPU, official checkpoint in NF4 | Raw tool-call and tool-response tokens without execution | 5/5 sessions claimed unsupported success | Official parser recognized a structured call |

Across the no-tools false-success runs, no schemas were supplied, no tool-result
messages existed, no external actions occurred, and the claimed files were not
created.

## Files

### Cerebras

- [`cerebras_gemma_tool_call_error.md`](cerebras_gemma_tool_call_error.md) —
  Pi and raw Cerebras evidence, interpretation, and suggested next steps.
- [`cerebras_reproduce.py`](cerebras_reproduce.py) — standard-library
  reproducer for raw Cerebras controls plus optional Pi scenarios.

### OpenRouter

- [`openrouter_gemma_tool_call_error.md`](openrouter_gemma_tool_call_error.md) —
  pinned OpenInference and Novita results.
- [`openrouter_reproduce.py`](openrouter_reproduce.py) — two-provider API
  reproducer with routing evidence.

### Direct AWS GPU

- [`aws_gemma_tool_call_error.md`](aws_gemma_tool_call_error.md) — direct
  official-checkpoint experiment, transcripts, limitations, and next steps.
- [`aws_reproduce.py`](aws_reproduce.py) — pinned Transformers/NF4 GPU
  reproducer.

## Quick reproduction

Python 3.10 or newer is required. Run commands from this directory.

### Cerebras API

```bash
export CEREBRAS_API_KEY="..."
python3 cerebras_reproduce.py --mode raw-no-tools
python3 cerebras_reproduce.py --mode raw-with-tools
```

### OpenRouter

```bash
export OPENROUTER_API_KEY="..."
python3 openrouter_reproduce.py --dry-run
python3 openrouter_reproduce.py --mode all
```

### Direct checkpoint

The validated low-cost configuration used an AWS `g5.xlarge` with a 24 GB A10G.
Follow the pinned environment setup in
[`aws_gemma_tool_call_error.md`](aws_gemma_tool_call_error.md#section-11), then
run:

```bash
python3 aws_reproduce.py \
  --output gemma-probe/results \
  --workspace gemma-probe/workspace \
  --repeats 5 \
  --max-new-tokens 768
```

Each reproducer writes timestamped machine-readable evidence and refuses to
overwrite an existing output directory.

## Interpretation boundary

The evidence rules out explanations confined to Hermes, Pi, Cerebras,
OpenRouter, or a single hosted provider. It supports a Gemma 4 execution-state
failure shared across inference paths. It does **not** yet isolate the official
weights from chat-template or token-handling behavior common to those paths.
The direct AWS run used NF4 quantization with BF16 compute, so a direct BF16
replication remains an important control.

Applications must execute only structured tool-call objects and must accept an
action as completed only after a matching tool-result message. Tool-looking
assistant prose is not execution evidence.
