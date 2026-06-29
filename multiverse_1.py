"""Future Shield — multiverse engine + map UI server.

Mainline ticks left → right through time.
At every tick, N exploratory forks probe the next few ticks ahead via the model.
Each fork either dies on a predicted problem, stalls on weak evidence,
loops on a repeated state, or survives. The preferred survivor commits to the
mainline, unfinished alternatives are killed, and the present advances.

Real Cerebras inference through the local `hermes` CLI. No mock, no fallback.
If the model call fails, the fork fails — and that surfaces in the UI and the log.

Run:
    python multiverse_1.py                # serve at http://127.0.0.1:8762
    python multiverse_1.py --no-serve     # headless, prints final snapshot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


RUN_LOG_PATH = Path(__file__).with_name("logs") / "future_shield_backend.jsonl"
_LOG_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Model prompts
# ─────────────────────────────────────────────────────────────────────────────

SELECTOR_PROMPT = """You are the Future Shield Future Selector with {n} agents working under you.
Each agent will try one different future and come back with a recommendation.

You are standing at MAINLINE / REALITY.
Your task is to fork the next multiverse step.

Read the current world state and choose exactly {n} candidate futures to probe,
once for each agent. Each selected future will be placed in one agent prompt.

Do not collapse to an action yet.
Do not simulate the futures yet.
Only choose which futures are worth sending to Future Probes.

A good selection should include:
- the default continuation if it is allowed
- at least one safety-preserving intervention
- any action that could prevent failure fastest
- any action that resolves weak evidence, delay risk, or irreversible action risk

Goal:
{goal}

Previous state:
{previous_state}

Current state:
{current_state}

Action types:
{action_types}

Futures:
{futures}

Question:
Are the actions collapsable now, or does the system need all selected future
probes together in this round before a safe collapse can happen?

Return ONLY JSON, no markdown, no commentary:

{{
  "phase": "FUTURE_SELECTOR",
  "selected_actions": ["<ACTION>", "..."],
  "rejected_actions": [
    {{
      "action": "<ACTION>",
      "reason": "<why not worth probing now>"
    }}
  ],
  "selector_reason": "<one concise sentence>",
  "needs_joint_info": <true | false>
}}
"""

PROBE_PROMPT = """You are the Future Shield realtime probe.

A safety-critical agent is about to take ONE action and you must simulate
what happens if that action is taken for `probe_depth` consecutive ticks
from the current world state.

Important timing rule:
- probe_depth=1 means only the next immediate segment.
- Do not reject an action at depth 1 because of risk that appears only after
  a later segment; that later risk belongs to the next control-loop tick.

Goal: {goal}
Allowed actions: {allowed_actions}
Probed action: {action}
Probe depth (ticks ahead): {probe_depth}
Previous state: {previous_state}
Current state: {current_state}

Decide one of these outcomes:
  - "safe"    : action keeps the system inside its action boundary
  - "failure" : action leads to a failure state, unsafe zone entry, damage, or unstable state
  - "stall"   : evidence is too weak to predict, system loses momentum
  - "loop"    : the predicted state repeats an earlier mainline state

Return ONLY this JSON, no markdown, no commentary:

{{
  "predicted_state": "<one short sentence about the world after probe_depth ticks>",
  "outcome": "safe" | "failure" | "stall" | "loop",
  "reason": "<one short sentence on why>",
  "confidence": <float 0.0 to 1.0>
}}
"""


DEFAULT_ALLOWED_ACTIONS = [
    "CONTINUE",
    "SAFE_STOP",
    "DIVERT_PATH",
    "SLOW_DOWN",
    "OBSERVE_MORE",
    "ESCALATE_TO_HUMAN",
]


DEFAULT_ACTION_TYPES = [
    "continuation / default action",
    "safety-preserving intervention",
    "information-gathering or delay-risk resolution",
    "escalation or handoff",
]


SEED_STATE = (
    "embodied autonomous system still outside a danger zone with one short safe segment available; "
    "recent visual frames show a human hand and fragile object near the action boundary; "
    "state signals indicate rising proximity risk if the system keeps advancing beyond the next segment"
)


# ─────────────────────────────────────────────────────────────────────────────
# Config + meters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class MultiverseConfig:
    forks_per_tick: int = 3
    look_ahead: int = 2
    max_ticks: int = 6
    tick_seconds: float = 0.1
    hermes_exe: str = "hermes"
    hermes_skill: str = ""
    hermes_max_turns: int = 1
    provider: str = "cerebras"
    model: str = "gemma-4-31b"
    goal: str = "prevent the live system from entering a failure state"
    seed_state: str = SEED_STATE
    allowed_actions: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_ACTIONS))
    call_timeout: float = 30.0
    seed: int = 7
    autoloop: bool = True


@dataclass(slots=True)
class TokenMeter:
    started_at: float = field(default_factory=time.monotonic)
    tokens: int = 0
    calls: int = 0

    def add(self, text: str) -> None:
        self.calls += 1
        self.tokens += estimate_tokens(text)

    @property
    def tps(self) -> float:
        elapsed = max(time.monotonic() - self.started_at, 0.001)
        return self.tokens / elapsed

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


# ─────────────────────────────────────────────────────────────────────────────
# Domain model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class MainlineTick:
    index: int
    state: str
    action: str
    confidence: float
    timestamp: float
    forks_evaluated: int = 0
    forks_killed: int = 0


TRACKS = ("top", "mid", "bottom")


@dataclass(slots=True)
class Fork:
    fork_id: str
    from_tick: int
    action: str
    track: str = "mid"
    depth: int = 0
    status: str = "live"  # live | killed | stalled | looped | survived | errored
    reason: str = ""
    confidence: float = 0.0
    predicted_state: str = ""
    path: list[str] = field(default_factory=list)
    latest_probe: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.fork_id,
            "from_tick": self.from_tick,
            "action": self.action,
            "track": self.track,
            "depth": self.depth,
            "status": self.status,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "predicted_state": self.predicted_state,
            "path": list(self.path),
            "latest_probe": dict(self.latest_probe),
        }


@dataclass(slots=True)
class MultiverseEvent:
    ts: str
    tag: str  # fork | kill | stall | loop | commit | reset | error
    msg: str

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "tag": self.tag, "msg": self.msg}


# ─────────────────────────────────────────────────────────────────────────────
# Multiverse engine
# ─────────────────────────────────────────────────────────────────────────────

class Multiverse:
    def __init__(self, config: MultiverseConfig) -> None:
        self.config = config
        self.meter = TokenMeter()
        self.mainline: list[MainlineTick] = []
        self.forks: list[Fork] = []
        self.events: deque[MultiverseEvent] = deque(maxlen=64)
        self.latest_selector: dict[str, Any] = {}
        self.now_tick: int = 0
        self.running: bool = False
        self.finished: bool = False
        self.stop_reason: str = ""
        self.run_id: str = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._stop_flag = asyncio.Event()
        self._paused_at: float = 0.0
        self._total_paused: float = 0.0
        self._seed_mainline()

    def pause_clock(self) -> None:
        if self._paused_at == 0.0:
            self._paused_at = time.monotonic()

    def resume_clock(self) -> None:
        if self._paused_at != 0.0:
            self._total_paused += time.monotonic() - self._paused_at
            self._paused_at = 0.0

    def _effective_elapsed(self) -> float:
        now = time.monotonic()
        live_pause = (now - self._paused_at) if self._paused_at else 0.0
        return max(now - self.meter.started_at - self._total_paused - live_pause, 0.0)

    # ---- snapshot for the UI ---------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "now_tick": self.now_tick,
                "max_ticks": self.config.max_ticks,
                "look_ahead": self.config.look_ahead,
                "forks_per_tick": self.config.forks_per_tick,
                "running": self.running,
                "finished": self.finished,
                "stop_reason": self.stop_reason,
                "provider": self.config.provider,
                "model": self.config.model,
                "goal": self.config.goal,
                "run_id": self.run_id,
                "log_path": str(RUN_LOG_PATH),
                "latest_selector": dict(self.latest_selector),
                "mainline": [
                    {
                        "tick": t.index,
                        "state": t.state,
                        "action": t.action,
                        "confidence": round(t.confidence, 3),
                        "forks_evaluated": t.forks_evaluated,
                        "forks_killed": t.forks_killed,
                    }
                    for t in self.mainline
                ],
                "forks": [f.to_dict() for f in self.forks],
                "events": [e.to_dict() for e in list(self.events)[-32:]],
                "meter": {
                    "tokens": self.meter.tokens,
                    "calls": self.meter.calls,
                    "tps": round(self.meter.tokens / max(self._effective_elapsed(), 0.001), 1),
                    "elapsed": round(self._effective_elapsed(), 1),
                },
            }

    # ---- main loop --------------------------------------------------------

    async def run(self) -> None:
        self.running = True
        self._event("reset", f"multiverse started · goal: {self.config.goal}")
        append_run_log({
            "event": "run_started",
            "run_id": self.run_id,
            "config": asdict(self.config),
            "snapshot": self.snapshot(),
        })
        try:
            while self.now_tick < self.config.max_ticks and not self._stop_flag.is_set():
                await self._advance_one_tick()
                if self.stop_reason:
                    break
                if self.config.tick_seconds > 0:
                    await asyncio.sleep(self.config.tick_seconds)
            if not self.stop_reason:
                self.stop_reason = "max_ticks_reached"
        finally:
            self.running = False
            self.finished = True
            self._event("commit", f"run complete · reason: {self.stop_reason}")
            append_run_log({
                "event": "run_finished",
                "run_id": self.run_id,
                "stop_reason": self.stop_reason,
                "snapshot": self.snapshot(),
            })

    def request_stop(self) -> None:
        self._stop_flag.set()

    # ---- internals --------------------------------------------------------

    def _seed_mainline(self) -> None:
        self.mainline.append(
            MainlineTick(
                index=0,
                state=self.config.seed_state,
                action="START",
                confidence=1.0,
                timestamp=time.monotonic(),
            )
        )

    async def _advance_one_tick(self) -> None:
        current = self.mainline[-1]
        previous = self.mainline[-2].state if len(self.mainline) > 1 else "none"
        try:
            selected_actions = await self._select_actions(current, previous)
        except Exception as exc:  # noqa: BLE001
            self.stop_reason = f"selector_failed:{exc}"
            self._event("error", f"future selector failed at t{current.index}: {exc}")
            return
        new_forks: list[Fork] = []
        for index, action in enumerate(selected_actions):
            fork = Fork(
                fork_id=f"f-{current.index}-{index + 1}",
                from_tick=current.index,
                action=action,
                track=TRACKS[index % len(TRACKS)],
            )
            new_forks.append(fork)
        with self._lock:
            self.forks.extend(new_forks)
        self._event(
            "fork",
            f"t{current.index} → t{current.index + 1}   selector chose {len(new_forks)} forks: "
            + " · ".join(f.action for f in new_forks),
        )

        await self._run_until_collapse(new_forks, current)

        survivors = [f for f in new_forks if f.status == "survived"]
        killed = sum(1 for f in new_forks if f.status != "survived")
        if survivors:
            chosen = self._choose_survivor(survivors)
            commit = MainlineTick(
                index=current.index + 1,
                state=chosen.predicted_state or f"state after {chosen.action}",
                action=chosen.action,
                confidence=chosen.confidence,
                timestamp=time.monotonic(),
                forks_evaluated=len(new_forks),
                forks_killed=killed,
            )
            with self._lock:
                self.mainline.append(commit)
                self.now_tick = commit.index
            self._event(
                "commit",
                f"survivor {chosen.action} (conf {chosen.confidence:.2f}) "
                f"committed · mainline → t{commit.index}",
            )
            if chosen.action in {"SAFE_STOP", "INTERVENE_NOW", "PAUSE_SYSTEM", "HOLD_POSITION"}:
                self.stop_reason = f"terminal_action:{chosen.action}"
        else:
            self._event(
                "stall",
                f"no survivor at t{current.index} → mainline halts "
                f"({killed} forks killed/stalled)",
            )
            self.stop_reason = "no_survivor"

    async def _run_until_collapse(self, forks: list[Fork], current: MainlineTick) -> None:
        tasks = {asyncio.create_task(self._run_fork(f, current)): f for f in forks}
        continue_task = next((task for task, fork in tasks.items()
                              if fork.action == "CONTINUE"), None)

        if continue_task is not None:
            try:
                await continue_task
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            continue_fork = tasks[continue_task]
            if continue_fork.status == "survived":
                await self._kill_unfinished(tasks, chosen=continue_fork, current=current)
                return

        while tasks:
            pending = {task for task in tasks if not task.done()}
            if not pending:
                break
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                fork = tasks[task]
                try:
                    await task
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
                if fork.status == "survived":
                    await self._kill_unfinished(tasks, chosen=fork, current=current)
                    return

        await asyncio.gather(*tasks.keys(), return_exceptions=True)

    async def _kill_unfinished(
        self,
        tasks: dict[asyncio.Task[None], Fork],
        chosen: Fork,
        current: MainlineTick,
    ) -> None:
        pending = [(task, fork) for task, fork in tasks.items()
                   if fork is not chosen and not task.done()]
        for task, fork in pending:
            task.cancel()
            fork.status = "killed"
            fork.reason = "process killed: branch collapsed before completion"
            fork.finished_at = time.monotonic()
            self._record_cancelled_probe(fork, current.state, current.state, fork.reason)
            self._event("kill", f"{fork.fork_id}  {fork.action}  process killed after {chosen.action} committed")
        if pending:
            await asyncio.gather(*(task for task, _ in pending), return_exceptions=True)

    def _record_cancelled_probe(
        self,
        fork: Fork,
        current_state: str,
        previous_state: str,
        reason: str,
    ) -> None:
        probe_depth = max(1, int(fork.depth or 1))
        prompt = PROBE_PROMPT.format(
            allowed_actions=", ".join(self.config.allowed_actions),
            goal=self.config.goal,
            action=fork.action,
            probe_depth=probe_depth,
            current_state=current_state,
            previous_state=previous_state or "none",
        )
        prompt_tokens = estimate_tokens(prompt)
        probe_record = {
            **fork.latest_probe,
            "event": "model_probe",
            "run_id": self.run_id,
            "fork_id": fork.fork_id,
            "started_at": fork.latest_probe.get("started_at") or time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "ok": False,
            "cancelled": True,
            "provider": self.config.provider,
            "model": self.config.model,
            "hermes_exe": self.config.hermes_exe,
            "hermes_skill": self.config.hermes_skill,
            "hermes_max_turns": self.config.hermes_max_turns,
            "action": fork.action,
            "probe_depth": probe_depth,
            "previous_state": fork.latest_probe.get("previous_state") or previous_state,
            "current_state": fork.latest_probe.get("current_state") or current_state,
            "prompt": fork.latest_probe.get("prompt") or prompt,
            "prompt_tokens_estimate": fork.latest_probe.get("prompt_tokens_estimate") or prompt_tokens,
            "error": reason,
            "raw_output": fork.latest_probe.get("raw_output") or "",
            "total_tokens_estimate": fork.latest_probe.get("total_tokens_estimate") or prompt_tokens,
            "total_tokens_per_second": fork.latest_probe.get("total_tokens_per_second") or 0.0,
        }
        should_log = not bool(fork.latest_probe.get("prompt"))
        fork.latest_probe = dict(probe_record)
        if should_log:
            append_run_log(probe_record)

    def _choose_survivor(self, survivors: list[Fork]) -> Fork:
        continue_survivors = [f for f in survivors if f.action == "CONTINUE"]
        if continue_survivors:
            return max(continue_survivors, key=lambda f: f.confidence)
        return max(survivors, key=lambda f: f.confidence)

    async def _select_actions(self, current: MainlineTick, previous_state: str) -> list[str]:
        requested = max(1, min(int(self.config.forks_per_tick), len(self.config.allowed_actions)))
        prompt = SELECTOR_PROMPT.format(
            n=requested,
            goal=self.config.goal,
            previous_state=previous_state or "none",
            current_state=current.state,
            action_types="\n".join(f"- {item}" for item in DEFAULT_ACTION_TYPES),
            futures="\n".join(f"- {action}" for action in self.config.allowed_actions),
        )
        started_perf = time.perf_counter()
        started_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
        prompt_tokens = estimate_tokens(prompt)
        log_base = {
            "event": "future_selector",
            "run_id": self.run_id,
            "tick": current.index,
            "started_at": started_at,
            "provider": self.config.provider,
            "model": self.config.model,
            "hermes_exe": self.config.hermes_exe,
            "hermes_skill": self.config.hermes_skill,
            "hermes_max_turns": self.config.hermes_max_turns,
            "requested_futures": requested,
            "allowed_actions": list(self.config.allowed_actions),
            "previous_state": previous_state,
            "current_state": current.state,
            "prompt": prompt,
            "prompt_tokens_estimate": prompt_tokens,
        }
        self.latest_selector = {
            **log_base,
            "ok": None,
            "status": "running",
            "raw_output": "",
            "total_tokens_estimate": prompt_tokens,
            "total_tokens_per_second": 0.0,
        }
        raw = ""
        try:
            raw = await call_hermes(
                hermes_exe=self.config.hermes_exe,
                hermes_skill=self.config.hermes_skill,
                hermes_max_turns=self.config.hermes_max_turns,
                provider=self.config.provider,
                model=self.config.model,
                prompt=prompt,
                timeout=self.config.call_timeout,
            )
            duration = time.perf_counter() - started_perf
            output_tokens = estimate_tokens(raw)
            packet = parse_any_json(raw)
            selected = normalize_selector_actions(
                packet.get("selected_actions"),
                self.config.allowed_actions,
                requested,
            )
            selector_record = {
                **log_base,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "ok": True,
                "duration_seconds": duration,
                "raw_output": raw,
                "parsed_packet": packet,
                "selected_actions": selected,
                "rejected_actions": packet.get("rejected_actions", []),
                "selector_reason": str(packet.get("selector_reason", "")).strip(),
                "needs_joint_info": bool(packet.get("needs_joint_info", False)),
                "output_tokens_estimate": output_tokens,
                "total_tokens_estimate": prompt_tokens + output_tokens,
                "output_tokens_per_second": output_tokens / max(duration, 0.001),
                "total_tokens_per_second": (prompt_tokens + output_tokens) / max(duration, 0.001),
            }
            append_run_log(selector_record)
            self.latest_selector = dict(selector_record)
            self.meter.add(raw)
            self._event(
                "selector",
                f"t{current.index} selector → " + " · ".join(selected),
            )
            return selected
        except Exception as exc:
            duration = time.perf_counter() - started_perf
            selector_record = {
                **log_base,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "ok": False,
                "duration_seconds": duration,
                "error": str(exc),
                "raw_output": raw,
                "total_tokens_estimate": prompt_tokens + (estimate_tokens(raw) if raw else 0),
                "total_tokens_per_second": (prompt_tokens + (estimate_tokens(raw) if raw else 0)) / max(duration, 0.001),
            }
            append_run_log(selector_record)
            self.latest_selector = dict(selector_record)
            raise

    async def _run_fork(self, fork: Fork, anchor: MainlineTick) -> None:
        state = anchor.state
        previous = anchor.state
        for probe in range(1, self.config.look_ahead + 1):
            fork.depth = probe
            try:
                packet = await self._step_packet(state, previous, fork.action, probe, fork)
            except TimeoutError as exc:
                fork.status = "killed"
                fork.reason = f"process killed: {exc}"
                fork.finished_at = time.monotonic()
                self._event("kill", f"{fork.fork_id}  {fork.action}  {fork.reason}")
                return
            except Exception as exc:  # noqa: BLE001
                fork.status = "errored"
                fork.reason = f"engine call failed: {exc}"
                fork.finished_at = time.monotonic()
                self._event("error", f"{fork.fork_id}  {fork.action}  ⨯  {fork.reason}")
                return
            outcome, predicted, reason, confidence = normalize_probe_packet(packet)
            fork.path.append(f"{fork.action}→{outcome or 'invalid'}")
            if predicted:
                fork.predicted_state = predicted
            fork.confidence = max(fork.confidence, confidence)

            if outcome not in {"safe", "failure", "stall", "loop"}:
                fork.status = "errored"
                fork.reason = f"invalid outcome from model: {outcome!r}"
                fork.finished_at = time.monotonic()
                self._event("error", f"{fork.fork_id}  {fork.action}  ⨯  {fork.reason}")
                return

            if outcome == "failure":
                fork.status = "killed"
                fork.reason = reason or f"failure state predicted in {probe} tick(s)"
                fork.finished_at = time.monotonic()
                self._event("kill", f"{fork.fork_id}  {fork.action}  ✕  {fork.reason}")
                return
            if outcome == "loop":
                fork.status = "looped"
                fork.reason = reason or "state repeats earlier mainline state"
                fork.finished_at = time.monotonic()
                self._event("loop", f"{fork.fork_id}  {fork.action}  ↺  {fork.reason}")
                return
            if outcome == "stall":
                fork.status = "stalled"
                fork.reason = reason or "evidence below threshold"
                fork.finished_at = time.monotonic()
                self._event("stall", f"{fork.fork_id}  {fork.action}  ⚠  {fork.reason}")
                return

            next_state = predicted or state
            if fork.action == "CONTINUE":
                fork.status = "survived"
                fork.reason = "next segment safe; collapse and re-evaluate from the new present"
                fork.predicted_state = next_state
                fork.finished_at = time.monotonic()
                self._event(
                    "commit",
                    f"{fork.fork_id}  {fork.action}  ●  next segment survived "
                    f"(conf {fork.confidence:.2f})",
                )
                return

            previous = state
            state = next_state

        fork.status = "survived"
        fork.reason = "no problem predicted across probe window"
        fork.predicted_state = state
        fork.finished_at = time.monotonic()
        self._event(
            "commit",
            f"{fork.fork_id}  {fork.action}  ●  survived "
            f"(depth {fork.depth}, conf {fork.confidence:.2f})",
        )

    async def _step_packet(
        self,
        current_state: str,
        previous_state: str,
        action: str,
        probe_depth: int,
        fork: Fork,
    ) -> dict[str, Any]:
        prompt = PROBE_PROMPT.format(
            allowed_actions=", ".join(self.config.allowed_actions),
            goal=self.config.goal,
            action=action,
            probe_depth=probe_depth,
            current_state=current_state,
            previous_state=previous_state or "none",
        )
        started_perf = time.perf_counter()
        started_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
        prompt_tokens = estimate_tokens(prompt)
        log_base = {
            "event": "model_probe",
            "run_id": self.run_id,
            "fork_id": fork.fork_id,
            "started_at": started_at,
            "provider": self.config.provider,
            "model": self.config.model,
            "hermes_exe": self.config.hermes_exe,
            "hermes_skill": self.config.hermes_skill,
            "hermes_max_turns": self.config.hermes_max_turns,
            "action": action,
            "probe_depth": probe_depth,
            "previous_state": previous_state,
            "current_state": current_state,
            "prompt": prompt,
            "prompt_tokens_estimate": prompt_tokens,
        }
        raw = ""
        fork.latest_probe = {
            **log_base,
            "ok": None,
            "status": "running",
            "raw_output": "",
            "total_tokens_estimate": prompt_tokens,
            "total_tokens_per_second": 0.0,
        }
        try:
            raw = await call_hermes(
                hermes_exe=self.config.hermes_exe,
                hermes_skill=self.config.hermes_skill,
                hermes_max_turns=self.config.hermes_max_turns,
                provider=self.config.provider,
                model=self.config.model,
                prompt=prompt,
                timeout=self.config.call_timeout,
            )
            duration = time.perf_counter() - started_perf
            output_tokens = estimate_tokens(raw)
            packet = parse_any_json(raw)
            probe_record = {
                **log_base,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "ok": True,
                "duration_seconds": duration,
                "raw_output": raw,
                "parsed_packet": packet,
                "output_tokens_estimate": output_tokens,
                "total_tokens_estimate": prompt_tokens + output_tokens,
                "output_tokens_per_second": output_tokens / max(duration, 0.001),
                "total_tokens_per_second": (prompt_tokens + output_tokens) / max(duration, 0.001),
            }
            append_run_log(probe_record)
            fork.latest_probe = dict(probe_record)
            self.meter.add(raw)
            return packet
        except asyncio.CancelledError:
            duration = time.perf_counter() - started_perf
            probe_record = {
                **log_base,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "ok": False,
                "cancelled": True,
                "duration_seconds": duration,
                "error": "process killed: branch collapsed before completion",
                "raw_output": raw,
                "total_tokens_estimate": prompt_tokens,
                "total_tokens_per_second": prompt_tokens / max(duration, 0.001),
            }
            append_run_log(probe_record)
            fork.latest_probe = dict(probe_record)
            raise
        except Exception as exc:
            duration = time.perf_counter() - started_perf
            probe_record = {
                **log_base,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "ok": False,
                "duration_seconds": duration,
                "error": str(exc),
                "total_tokens_estimate": prompt_tokens,
                "total_tokens_per_second": prompt_tokens / max(duration, 0.001),
            }
            if raw:
                output_tokens = estimate_tokens(raw)
                probe_record.update({
                    "raw_output": raw,
                    "output_tokens_estimate": output_tokens,
                    "total_tokens_estimate": prompt_tokens + output_tokens,
                    "output_tokens_per_second": output_tokens / max(duration, 0.001),
                    "total_tokens_per_second": (prompt_tokens + output_tokens) / max(duration, 0.001),
                })
            append_run_log(probe_record)
            fork.latest_probe = dict(probe_record)
            raise

    def _event(self, tag: str, msg: str) -> None:
        ev = MultiverseEvent(ts=time.strftime("%H:%M:%S"), tag=tag, msg=msg)
        with self._lock:
            self.events.append(ev)


# ─────────────────────────────────────────────────────────────────────────────
# Hermes bridge + JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

async def call_hermes(
    hermes_exe: str,
    hermes_skill: str,
    hermes_max_turns: int,
    provider: str,
    model: str,
    prompt: str,
    timeout: float,
) -> str:
    command = [
        hermes_exe, "chat",
        "-q", prompt,
        "-Q",
        "--provider", provider,
        "-m", model,
        "--max-turns", str(hermes_max_turns),
    ]
    if hermes_skill.strip():
        command[2:2] = ["-s", hermes_skill.strip()]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        raise TimeoutError(f"{hermes_exe} exceeded {timeout:.1f}s") from None
    except asyncio.CancelledError:
        process.kill()
        await process.communicate()
        raise
    if process.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip()
                           or f"{hermes_exe} exited {process.returncode}")
    return strip_session_footer(stdout.decode("utf-8", errors="replace").strip())


def strip_session_footer(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines()
        if not line.strip().startswith("session_id:")
    ).strip()


def parse_any_json(text: str) -> dict[str, Any]:
    cleaned = strip_session_footer(text)
    candidates = [cleaned]
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= first < last:
        candidates.append(cleaned[first:last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("response did not contain a JSON object")


def normalize_selector_actions(
    selected_actions: Any,
    allowed_actions: list[str],
    requested: int,
) -> list[str]:
    if not isinstance(selected_actions, list):
        raise ValueError("selector response missing selected_actions list")
    allowed_lookup = {action.upper(): action for action in allowed_actions}
    selected: list[str] = []
    seen: set[str] = set()
    for raw_action in selected_actions:
        key = str(raw_action).strip().upper()
        if key not in allowed_lookup:
            raise ValueError(f"selector chose unsupported action: {raw_action!r}")
        if key in seen:
            raise ValueError(f"selector repeated action: {raw_action!r}")
        seen.add(key)
        selected.append(allowed_lookup[key])
    if len(selected) != requested:
        raise ValueError(f"selector chose {len(selected)} action(s), expected {requested}")
    return selected


def normalize_probe_packet(packet: dict[str, Any]) -> tuple[str, str, str, float]:
    outcome = str(packet.get("outcome", "")).lower()
    predicted = str(packet.get("predicted_state", "")).strip()
    reason = str(packet.get("reason", "")).strip()
    try:
        confidence = float(packet.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return outcome, predicted, reason, confidence


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def append_run_log(entry: dict[str, Any]) -> None:
    enriched = {
        "logged_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        **entry,
    }
    with _LOG_LOCK:
        RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def tail_run_log(limit: int = 40) -> list[dict[str, Any]]:
    if limit < 1:
        limit = 1
    limit = min(limit, 200)
    if not RUN_LOG_PATH.is_file():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    with _LOG_LOCK:
        with RUN_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    packet = json.loads(line)
                except json.JSONDecodeError as exc:
                    packet = {
                        "event": "log_decode_error",
                        "error": str(exc),
                        "raw_line": line[:500],
                    }
                rows.append(packet)
    return list(rows)


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def clamp_int(value: int, lower: int, upper: int | None = None) -> int:
    value = max(lower, value)
    if upper is not None:
        value = min(value, upper)
    return value


def clamp_float(value: float, lower: float, upper: float | None = None) -> float:
    value = max(lower, value)
    if upper is not None:
        value = min(value, upper)
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Background runner — keeps the multiverse loop alive in its own thread
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundRunner:
    def __init__(self, config: MultiverseConfig) -> None:
        self.config = config
        self.multiverse = Multiverse(config)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._reset_lock = threading.Lock()
        self._paused = threading.Event()
        self._paused.set()  # starts paused — nothing runs until play() is called
        self.multiverse.pause_clock()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="multiverse-loop")
        self._thread.start()

    def play(self) -> None:
        with self._reset_lock:
            if self.multiverse.finished:
                self.multiverse = Multiverse(self.config)
                if self._paused.is_set():
                    self.multiverse.pause_clock()
            if self._paused.is_set():
                self._paused.clear()
                self.multiverse.resume_clock()

    def pause(self) -> None:
        if not self._paused.is_set():
            self._paused.set()
            self.multiverse.pause_clock()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._cancel_current)

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def reset(self) -> None:
        with self._reset_lock:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._cancel_current)
            # reset returns to a clean paused state — user must press play to run
            self._paused.set()
            self.config = MultiverseConfig(**{**asdict(self.config),
                                              "seed": self.config.seed + 1})
            self.multiverse = Multiverse(self.config)
            self.multiverse.pause_clock()

    def apply_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        with self._reset_lock:
            current = asdict(self.config)
            for key, value in updates.items():
                if key not in current:
                    continue
                if key == "allowed_actions":
                    if isinstance(value, str):
                        value = [a.strip().upper() for a in value.split(",") if a.strip()]
                    elif isinstance(value, list):
                        value = [str(a).strip().upper() for a in value if str(a).strip()]
                    if not value:
                        continue
                if key in ("forks_per_tick", "look_ahead", "max_ticks",
                           "hermes_max_turns", "seed"):
                    value = int(value)
                    if key == "forks_per_tick":
                        action_count = len(current.get("allowed_actions") or [])
                        value = clamp_int(value, 1, max(action_count, 1))
                    elif key == "look_ahead":
                        value = clamp_int(value, 1, 20)
                    elif key == "max_ticks":
                        value = clamp_int(value, 1, 200)
                    elif key == "hermes_max_turns":
                        value = clamp_int(value, 1, 20)
                    elif key == "seed":
                        value = clamp_int(value, 0)
                if key in ("tick_seconds", "call_timeout"):
                    value = float(value)
                    if key == "tick_seconds":
                        value = clamp_float(value, 0.0, 30.0)
                    elif key == "call_timeout":
                        value = clamp_float(value, 2.0, 600.0)
                if key == "autoloop":
                    value = coerce_bool(value)
                current[key] = value
            new_config = MultiverseConfig(**current)
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._cancel_current)
            self.config = new_config
            self.multiverse = Multiverse(new_config)
            if self._paused.is_set():
                self.multiverse.pause_clock()
            return current

    def stop(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._cancel_current)

    def _cancel_current(self) -> None:
        self.multiverse.request_stop()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.3)
                    continue
                self._loop.run_until_complete(self.multiverse.run())
                if self._stop.is_set():
                    break
                if self.multiverse.finished:
                    self._paused.set()
                    self.multiverse.pause_clock()
                    append_run_log({
                        "event": "run_finished_paused",
                        "run_id": self.multiverse.run_id,
                        "stop_reason": self.multiverse.stop_reason,
                        "autoloop_requested": self.config.autoloop,
                    })
                    continue
                time.sleep(2.0)
                with self._reset_lock:
                    old_run_id = self.multiverse.run_id
                    self.config = MultiverseConfig(**{**asdict(self.config),
                                                      "seed": self.config.seed + 1})
                    self.multiverse = Multiverse(self.config)
                    append_run_log({
                        "event": "autoloop_reset",
                        "old_run_id": old_run_id,
                        "run_id": self.multiverse.run_id,
                        "config": asdict(self.config),
                    })
        finally:
            self._loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────────────────────────────

class FutureShieldHandler(BaseHTTPRequestHandler):
    server_version = "FutureShield/0.3"

    def do_GET(self) -> None:  # noqa: N802
        split = urlsplit(self.path)
        path = split.path
        if path in {"/", "/multiverse_1.html"}:
            self._send_file(Path(__file__).with_name("multiverse_1.html"))
            return
        if path == "/api/state":
            runner: BackgroundRunner = self.server.runner  # type: ignore[attr-defined]
            snap = runner.multiverse.snapshot()
            snap["paused"] = runner.is_paused()
            self._send_json(snap)
            return
        if path == "/api/health":
            config: MultiverseConfig = self.server.config  # type: ignore[attr-defined]
            self._send_json({
                "ok": True,
                "provider": config.provider,
                "model": config.model,
                "hermes_exe": config.hermes_exe,
                "hermes_skill": config.hermes_skill,
                "log_path": str(RUN_LOG_PATH),
            })
            return
        if path == "/api/logs":
            params = parse_qs(split.query)
            try:
                limit = int(params.get("limit", ["40"])[0])
            except (TypeError, ValueError):
                limit = 40
            self._send_json({
                "ok": True,
                "log_path": str(RUN_LOG_PATH),
                "entries": tail_run_log(limit),
            })
            return
        if path == "/api/config":
            runner: BackgroundRunner = self.server.runner  # type: ignore[attr-defined]
            self._send_json({"ok": True, "config": asdict(runner.config)})
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/play":
            runner: BackgroundRunner = self.server.runner  # type: ignore[attr-defined]
            append_run_log({
                "event": "play_clicked",
                "run_id": runner.multiverse.run_id,
                "paused_before": runner.is_paused(),
                "snapshot": runner.multiverse.snapshot(),
            })
            runner.play()
            self._send_json({"ok": True, "paused": False})
            return
        if path == "/api/pause":
            runner: BackgroundRunner = self.server.runner  # type: ignore[attr-defined]
            append_run_log({
                "event": "pause_clicked",
                "run_id": runner.multiverse.run_id,
                "paused_before": runner.is_paused(),
                "snapshot": runner.multiverse.snapshot(),
            })
            runner.pause()
            self._send_json({"ok": True, "paused": True})
            return
        if path == "/api/reset":
            runner: BackgroundRunner = self.server.runner  # type: ignore[attr-defined]
            old_run_id = runner.multiverse.run_id
            runner.reset()
            append_run_log({
                "event": "reset_clicked",
                "old_run_id": old_run_id,
                "run_id": runner.multiverse.run_id,
                "snapshot": runner.multiverse.snapshot(),
            })
            self._send_json({"ok": True})
            return
        if path == "/api/config":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                updates = json.loads(body or "{}")
                if not isinstance(updates, dict):
                    raise ValueError("body must be a JSON object")
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": f"invalid request: {exc}"}, status=400)
                return
            runner: BackgroundRunner = self.server.runner  # type: ignore[attr-defined]
            try:
                applied = runner.apply_config(updates)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": f"invalid config: {exc}"}, status=400)
                return
            append_run_log({
                "event": "config_applied",
                "run_id": runner.multiverse.run_id,
                "updates": updates,
                "config": applied,
                "snapshot": runner.multiverse.snapshot(),
            })
            self._send_json({"ok": True, "config": applied})
            return
        self.send_error(404, "Not found")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.path.startswith("/api/state"):
            return  # poll noise
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        sys.stderr.write(f"{timestamp} {self.address_string()} {format % args}\n")

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404, f"missing: {path.name}")
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve_html(config: MultiverseConfig, host: str, port: int) -> None:
    runner = BackgroundRunner(config)
    runner.start()
    server = ThreadingHTTPServer((host, port), FutureShieldHandler)
    server.config = config            # type: ignore[attr-defined]
    server.runner = runner            # type: ignore[attr-defined]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp}  Future Shield -> http://{host}:{port}/")
    print(f"{stamp}  provider={config.provider}  model={config.model}  skill={config.hermes_skill}")
    print(f"{stamp}  forks/tick={config.forks_per_tick}  look-ahead={config.look_ahead}  "
          f"max-ticks={config.max_ticks}")
    print(
        f"{stamp}  endpoints: GET /  GET /api/state  GET /api/health  "
        "GET /api/logs  POST /api/play  POST /api/pause  POST /api/reset"
    )
    print(f"{stamp}  logs: {RUN_LOG_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        runner.stop()
        server.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Future Shield — multiverse map server + engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--forks-per-tick", type=int, default=3)
    parser.add_argument("--look-ahead", type=int, default=2)
    parser.add_argument("--max-ticks", type=int, default=6)
    parser.add_argument("--tick-seconds", type=float, default=0.1)
    parser.add_argument("--hermes-exe", default=os.environ.get("HERMES_EXE", "hermes"))
    parser.add_argument("--hermes-skill", default=os.environ.get("HERMES_SKILL", ""))
    parser.add_argument("--hermes-max-turns", type=int, default=1)
    parser.add_argument("--provider", default=os.environ.get("HERMES_PROVIDER", "cerebras"))
    parser.add_argument("--model", default=os.environ.get("HERMES_MODEL", "gemma-4-31b"))
    parser.add_argument("--goal", default="prevent the live system from entering a failure state")
    parser.add_argument("--seed-state", default=SEED_STATE)
    parser.add_argument("--allowed-actions", default=",".join(DEFAULT_ALLOWED_ACTIONS))
    parser.add_argument("--call-timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-autoloop", action="store_true", help="run once instead of looping")
    parser.add_argument("--no-serve", action="store_true", help="headless, print final snapshot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8762)
    return parser


def parse_actions(raw: str) -> list[str]:
    actions = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return actions or list(DEFAULT_ALLOWED_ACTIONS)


def config_from_args(args: argparse.Namespace) -> MultiverseConfig:
    return MultiverseConfig(
        forks_per_tick=max(1, args.forks_per_tick),
        look_ahead=max(1, args.look_ahead),
        max_ticks=max(1, args.max_ticks),
        tick_seconds=max(0.0, args.tick_seconds),
        hermes_exe=args.hermes_exe,
        hermes_skill=args.hermes_skill,
        hermes_max_turns=max(1, args.hermes_max_turns),
        provider=args.provider,
        model=args.model,
        goal=args.goal,
        seed_state=args.seed_state,
        allowed_actions=parse_actions(args.allowed_actions),
        call_timeout=max(2.0, args.call_timeout),
        seed=args.seed,
        autoloop=not args.no_autoloop,
    )


async def headless_run(config: MultiverseConfig) -> None:
    multiverse = Multiverse(config)
    await multiverse.run()
    snapshot = multiverse.snapshot()
    print(json.dumps({
        "mainline": snapshot["mainline"],
        "forks": [f for f in snapshot["forks"] if f["status"] != "live"],
        "stop_reason": snapshot["stop_reason"],
        "meter": snapshot["meter"],
    }, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    if args.no_serve:
        asyncio.run(headless_run(config))
        return
    serve_html(config, args.host, args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nmultiverse interrupted")
        sys.exit(130)
