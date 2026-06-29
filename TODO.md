# TODO

**Scope of this repo:** Future Shield was built for the Cerebras × Gemma
June hackathon to *demonstrate the multiverse pre-action simulation
capability*. It is not a production system. The code is intentionally a
two-file demo (`multiverse_1.py` + `multiverse_1.html`), no persistence layer,
no auth, no horizontal scaling, no test harness beyond what was needed to
ship the demo. This TODO captures known limitations and post-hackathon work
so reviewers see we know they exist — we just chose not to fix them inside
the hackathon window.

## Security (not enforced — demo only)

- **Restrict CORS on POST endpoints.** `multiverse_1.py:1639` sends
  `Access-Control-Allow-Origin: *` on every response. Any website you visit
  while the engine is running can fire `fetch("http://127.0.0.1:8762/api/play")`
  (or `/api/config`, or `/api/generate-scenario`) and burn the user's Cerebras
  tokens cross-site. Localhost-only mitigates discoverability but it is a real
  cross-site request risk if the engine is left running. Lock to same-origin
  for any non-demo deployment.

- **No auth on control endpoints.** `/api/play`, `/api/pause`, `/api/reset`,
  `/api/config`, `/api/generate-scenario` accept unauthenticated POSTs.
  Acceptable for a localhost demo, not for any shared host.

## Naming / leftovers from the Hermes era

- **Stale `HERMES_*` env-var defaults.** `multiverse_1.py:1683-1684`:
  ```python
  parser.add_argument("--provider", default=os.environ.get("HERMES_PROVIDER", "cerebras"))
  parser.add_argument("--model", default=os.environ.get("HERMES_MODEL", "gemma-4-31b"))
  ```
  Hermes was removed in commit `2e35588`; these env-var names should be renamed
  (e.g. `FUTURE_SHIELD_PROVIDER` / `FUTURE_SHIELD_MODEL` or just unprefixed)
  since they were never actually read by the Hermes flow that no longer exists.
  Cosmetic but contradicts the README.

- **Stale "subprocess spawn" comment.** `multiverse_1.py:1354-1355` still
  describes the subprocess cold-start path that was removed when we switched
  to direct Cerebras HTTPS. Update or delete.

## Engine / correctness

- **`needs_joint_info` is asked but never enforced.** The selector prompt
  returns this boolean, the backend stores it on every selector record, but
  `_run_until_collapse` ignores it. Today early-collapse fires as soon as any
  fork survives; if `needs_joint_info=true`, the loop should wait for *all*
  selected probes before collapsing. Easy fix — one branch in
  `_run_until_collapse`.

- **`CONTINUE` is overloaded.** It is both an allowed action *and* a
  collapse-priority signal: the engine awaits the `CONTINUE` task first,
  short-circuits if it survives, and `_choose_survivor` prefers `CONTINUE`
  by name. Better to split semantics — collapse should ride on
  `outcome=safe && confidence>=threshold && no_risk_flags`, not on the
  literal string `"CONTINUE"`.

- **`killed` covers two cases.** A fork can be "killed" because (a) its
  predicted outcome was `failure`, or (b) another fork survived first and
  the runner cancelled it. The UI now keeps the inspector from showing
  "WHY IT WAS KILLED" on cancelled branches, but the status string itself
  doesn't distinguish them. Introduce a `cancelled` status alongside
  `killed`.

- **No JSON schema repair.** If Gemma returns malformed JSON the fork
  errors. We chose honesty (no silent fallback) for the demo and it never
  bit us at `temperature=0.3` on `gemma-4-31b`. For a hardened deployment,
  one retry with a "your previous output was invalid JSON, return only the
  required object" repair prompt would catch the rare bad packet.

## UX / language

- **Autoloop checkbox.** The sidebar's "restart automatically" toggle wires
  to `config.autoloop=True`, but the post-finish autoloop reset path isn't
  exercised in the demo flow. Either remove the checkbox or hook the
  end-of-run modal into the autoloop branch.

- **Default `SEED_STATE` language.** The fallback scenario uses
  "embodied autonomous system / danger zone / human hand / fragile object",
  which biases the demo toward a robotics framing. The demo always uses
  Gemma-generated scenarios so this never shows, but if the product
  pitch is "any safety-critical AI", swap the default to generic language
  ("live system", "action boundary", "critical asset").

## Repo hygiene (deferred to keep the surface clean)

- The following untracked helpers exist on disk but were not committed —
  their docstrings still reference the old Hermes-CLI flow we ripped out:
  `check_future_shield.py`, `smoke_future_shield.py`,
  `smoke_future_shield_browser.py`, `test_future_shield_core.py`,
  `DEMO_CHECKLIST.md`. If we revive a test harness post-hackathon, update
  them to call the direct Cerebras path before committing.

- `docs/temp/` holds four obsolete theme-preview HTMLs (`theme_preview.html`,
  `theme_preview_light.html`, `theme_preview_map.html`, an old
  `multiverse_1.html` snapshot). Untracked, but should be deleted from disk
  before any future contributor sees them.

## Demo claim — what is actually real today

> Future Shield uses Gemma 4 on Cerebras to select N candidate futures,
> probe each branch K ticks ahead, kill branches that predict failure
> (or loop, or stall on weak evidence), and collapse the surviving
> branch into the mainline. When the run ends, Gemma writes a
> scenario-specific 100-word verdict (`SUCCEEDED` / `CRASHED` /
> `STALLED` / `INCONCLUSIVE`) that cites the actual entities in the
> scenario.

Token counts come from Cerebras's `usage` field (exact, minus cached
prompt tokens). TPS comes from `time_info.completion_time` (real
model-side generation rate). No mock, no fallback, no estimator.
