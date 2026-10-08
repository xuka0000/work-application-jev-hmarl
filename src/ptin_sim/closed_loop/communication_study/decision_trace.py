"""Append-only decision evidence. Audit truth is never an input to selection.

Records describe measured transitions, not counterfactual causal effects.
Missing scores and unmodelled device positions remain explicitly unavailable.
"""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import gzip
import json
import math
from pathlib import Path
import time
from typing import Mapping


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_value(v) for v in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"value": None, "status": "nonfinite", "raw": str(value)}
        return value
    raise TypeError(f"Unsupported evidence type: {type(value).__name__}")


class DecisionTrace:
    """One immutable attempt directory; flush every record, including failures."""

    def __init__(self, directory: Path, *, identity: dict, parameters: dict, compression="none",
                 initial_devices=None,gzip_level=1):
        if compression not in {"none", "gzip"}:
            raise ValueError("Unsupported trace compression")
        if not isinstance(gzip_level,int) or not 0<=gzip_level<=9:raise ValueError('Invalid gzip level')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / ("events.jsonl.gz" if compression == "gzip" else "events.jsonl")
        self._stream = gzip.open(self.path, "xt", encoding="utf-8",compresslevel=gzip_level) if compression == "gzip" else self.path.open("x", encoding="utf-8")
        self.sequence = 0
        self.previous_hash = "0" * 64
        self.pending = None
        self.offsets = {"messages": 0, "physical_states": 0, "service_states": 0, "resource_events": 0}
        self.emit("attempt_started", 0.0, {
            "identity": identity, "parameters": parameters,
            "storage":dict(schema_version=1,compression=compression,gzip_level=gzip_level if compression=='gzip' else None),
            "audit_truth_policy_visible": False,
            "device_positions": initial_devices,
            "device_position_status": "explicit_initial_device_state" if initial_devices is not None
            else "not_modelled_in_current_aggregate_plant",
        })

    def emit(self, kind, physical_time_s, payload):
        if not math.isfinite(physical_time_s) or physical_time_s < 0:
            raise ValueError("Invalid physical timestamp")
        record = json_value(dict(sequence=self.sequence, kind=kind,
            physical_time_s=physical_time_s, observed_utc=datetime.now(timezone.utc).isoformat(),
            previous_sha256=self.previous_hash, payload=payload))
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        record["sha256"] = digest
        if self._stream.closed:
            raise RuntimeError("Evidence stream already closed")
        # Reuse the exact canonical payload already serialized for hashing.
        # All fields remain present; avoid a second traversal of large states.
        self._stream.write(encoded[:-1]+',"sha256":"'+digest+'"}\n')
        self._stream.flush()
        self.previous_hash = digest
        self.sequence += 1
        return record["sequence"]

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.close()

    @staticmethod
    def truth_snapshot(env):
        # Deliberately excludes the growing trace list, avoids quadratic logs.
        state = {f.name: getattr(env._state, f.name) for f in fields(env._state)
                 if f.name != "trace_rows"}
        return json_value(dict(state=state, active_failed_edges=env._active_failed_edges(),
                               time_s=env.time_s, truth_boundary=env.truth_boundary))

    def drain(self, env):
        sources = {"messages": env.bus.events, "physical_states": env._physical_audits,
                   "service_states": env._service, "resource_events": getattr(env, "_resource_events", [])}
        for name, rows in sources.items():
            if len(rows) < self.offsets[name]:
                raise ValueError("Environment reset inside an attempt")
            for row in rows[self.offsets[name]:]:
                data = json_value(row)
                stamp = data.get("event_s", data.get("time_s"))
                if stamp is None:
                    raise ValueError(f"Missing physical time in {name}")
                self.emit(name, stamp, data)
                self.offsets[name] += 1

    def begin(self, env, observation):
        if self.pending is not None:
            raise RuntimeError("Unfinished decision")
        self.drain(env)
        candidates = json_value(env.available_actions())
        self.pending = self.emit("decision_observation", env.time_s, {
            "policy_observation": observation, "candidate_actions": candidates,
            "audit_truth_before": self.truth_snapshot(env),
        })
        self.candidate_ids = {a["action_id"] for a in candidates}
        return self.pending

    def selected(self, env, action, *, diagnostics=None, elapsed_s=None):
        if self.pending is None or action.action_id not in self.candidate_ids:
            raise ValueError("Selection is not from the recorded candidate set")
        self.emit("decision_selected", env.time_s, {
            "decision_sequence": self.pending, "action": action,
            "selection_wall_s": elapsed_s, "diagnostics": diagnostics,
            "diagnostics_status": "recorded" if diagnostics is not None else "not_exposed_by_policy",
        })

    def completed(self, env, result):
        observation, reward, terminated, truncated, info = result
        self.drain(env)
        self.emit("execution_feedback", env.time_s, {
            "decision_sequence": self.pending, "next_policy_observation": observation,
            "audit_truth_after": self.truth_snapshot(env), "reward": reward,
            "terminated": terminated, "truncated": truncated, "execution_info": info,
        })
        self.pending = None

    def step(self, env, policy, observation):
        """The recorder cannot change candidates, scores, environment or policy."""
        self.begin(env, observation)
        try:
            started = time.perf_counter()
            action = policy.select(env, observation)
            elapsed = time.perf_counter() - started
            self.selected(env, action, diagnostics=getattr(policy, "last_decision_trace", None), elapsed_s=elapsed)
            result = env.step(action)
            self.completed(env, result)
            return result
        except Exception as exc:
            self.drain(env)
            self.emit("decision_failed", env.time_s, {
                "decision_sequence": self.pending, "exception_type": type(exc).__name__,
                "message": str(exc), "audit_truth_after": self.truth_snapshot(env),
            })
            raise


def verify_trace(path: Path):
    previous = "0" * 64
    count = 0
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            digest = row.pop("sha256")
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if row["sequence"] != index or row["previous_sha256"] != previous or hashlib.sha256(encoded.encode("utf-8")).hexdigest() != digest:
                raise ValueError(f"Broken evidence chain at record {index}")
            previous = digest
            count += 1
    return {"records": count, "tail_sha256": previous}
