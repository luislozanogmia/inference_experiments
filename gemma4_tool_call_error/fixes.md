# Immediate Fixes for Gemma Tool-State Hallucination

These safeguards contain the failure today, before any fine-tuned model is
available. They belong in the harness because model output alone is never proof
that an external action occurred.

## Easy decision table

| Runtime situation | What the harness should do now | May the model claim success? |
|---|---|---|
| The task requires tools, but zero callable schemas were resolved | Stop before inference and return `TOOLS_UNAVAILABLE`, or change the task to planning-only | No |
| An explicit tool or toolset name is unknown | Return a non-zero configuration error and list valid names; do not silently continue or enable broader permissions | No |
| Assistant content contains `call:*`, tool XML, or JSON-like tool syntax, but `tool_calls` is empty | Treat it as inert prose, flag it, and record that no action occurred; never execute the text | No |
| A structured call names a tool that was not supplied in the request | Reject the call and record a failed tool result | No |
| A structured call exists, but no matching result has arrived | Keep the action in `pending` or `unverified` state | No |
| The tool returned an error or non-zero exit status | Preserve the real error and let the model retry or report failure | No |
| The tool returned success | Match the call ID and result ID, then verify exit status and relevant external state | Only after verification |
| A later turn includes an earlier textual fake call | Keep it as assistant prose; never convert it into a structured call or result | No |
| The final answer says “created,” “ran,” “sent,” “deployed,” or quotes exact output | Require a matching verified ledger entry and artifact check | Only when matched |
| Valid callable schemas are present | Send the schemas, accept only structured calls, execute them in the client, and append matching structured results | Yes, after verification |

## Minimum safe execution flow

```text
resolve requested tool schemas

if task requires external action and schema_count == 0:
    stop with TOOLS_UNAVAILABLE

send the model only the resolved schemas

if the response contains tool-looking prose but no structured tool_calls:
    mark it NON_EXECUTED
    do not run it

for each structured tool_call:
    validate the tool name and arguments
    execute it in the client
    record call ID, result ID, status, and external verification

before returning the final answer:
    reject or correct every completion claim without verified evidence
```

## Execution ledger

Track every external action through these states:

```text
requested -> structured_call -> tool_result -> externally_verified
```

A final answer may describe an action as completed only when the corresponding
ledger entry reached `externally_verified`. An HTTP 200 response, a
`finish_reason: stop`, assistant reasoning, or tool-shaped text does not advance
the ledger.

## Recovery message

If tool-shaped prose escapes from the model, the harness may add one corrective
turn:

> Your previous message contained text that looked like a tool call, but no
> structured call was accepted and no external action occurred. Use one of the
> supplied structured tools, or clearly state that execution is unavailable.

If no schemas exist, skip the recovery call and return `TOOLS_UNAVAILABLE`
directly.

## Required regression tests

Every harness release should test:

1. Tool-required task with no `tools` field.
2. Unknown explicit tool name.
3. Textual fake call in assistant history.
4. Structured call with a missing result.
5. Structured call with a failed result.
6. Valid structured call with a real result.
7. User pressure to report success without evidence.
8. Randomized filenames and outputs that cannot be guessed from the prompt.

The target is zero unsupported completion claims while preserving valid
structured tool calling.

## Relationship to fine-tuning

Fine-tuning may reduce how often Gemma emits fake calls or unsupported success
claims. It should remain defense in depth, not replace schema validation,
structured execution, the evidence ledger, or external verification.
