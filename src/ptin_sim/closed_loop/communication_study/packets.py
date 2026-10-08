"""Causal application-message events over a reproducible synthetic channel.

This is an application-layer experiment model, not an ns-3 protocol stack or
field RF validation. Channel randomness is keyed by link and physical time,
never by the policy name, number of rejected actions, or call order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import hashlib
import heapq
import math
import random
from typing import Any, Callable


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class ChannelConfig:
    profile: str = "good"
    seed: int = 1
    horizon_s: float = 14400.0
    slot_s: float = 30.0
    loss_rate: float = 0.01
    burst_slots: int = 10
    delay_ms: float = 35.0
    jitter_ms: float = 30.0
    blackout_start_s: float | None = None
    blackout_duration_s: float = 0.0
    blackout_scope: str = "all_links"

    def __post_init__(self) -> None:
        if self.profile not in {"good", "iid", "burst", "blackout"}:
            raise ValueError("unknown communication profile")
        if self.blackout_scope not in {"all_links", "dispatcher_isolation", "terminal_external_isolation"}:
            raise ValueError("unknown blackout scope")
        for key in ("horizon_s", "slot_s", "delay_ms", "jitter_ms", "blackout_duration_s"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {key}")
        if self.horizon_s <= 0 or self.slot_s <= 0 or self.burst_slots <= 0:
            raise ValueError("positive horizon, slot duration and burst length required")
        if not math.isfinite(self.loss_rate) or not 0 <= self.loss_rate <= 1:
            raise ValueError("loss rate must lie in [0,1]")
        if self.blackout_start_s is not None and (
            not math.isfinite(self.blackout_start_s) or self.blackout_start_s < 0
        ):
            raise ValueError("invalid blackout start")


class SyntheticChannel:
    def __init__(self, config: ChannelConfig) -> None:
        self.config = config
        self._masks: dict[str, tuple[bool, ...]] = {}

    def mask(self, link: str) -> tuple[bool, ...]:
        """Exact marginal count on the predeclared, fixed transmission grid."""
        if link in self._masks:
            return self._masks[link]
        c = self.config
        n = math.ceil(c.horizon_s / c.slot_s)
        count = round(c.loss_rate * n)
        rng = random.Random(_seed(c.seed, link, "loss"))
        if c.profile == "burst":
            # Pack losses into separated runs, preserving the same total count.
            groups = max(1, math.ceil(count / c.burst_slots))
            slots: list[int] = []
            for g in range(groups):
                length = min(c.burst_slots, count - len(slots))
                start = math.floor(g * n / groups)
                slots.extend(range(start, start + length))
            phase = rng.randrange(n)
            lost = {(s + phase) % n for s in slots}
        else:
            lost = set(rng.sample(range(n), count))
        mask = tuple(i in lost for i in range(n))
        if sum(mask) != count:
            raise ValueError("invalid burst construction for the declared loss rate")
        self._masks[link] = mask
        return mask

    def transmission(self, *, link: str, at_s: float) -> tuple[bool, float, str]:
        c = self.config
        if not math.isfinite(at_s) or not 0 <= at_s < c.horizon_s:
            return False, 0.0, "outside_channel_horizon"
        if c.blackout_start_s is not None and (
            c.blackout_start_s <= at_s < c.blackout_start_s + c.blackout_duration_s
        ):
            endpoints = link.split(">")
            isolated = c.blackout_scope == "all_links"
            if c.blackout_scope == "dispatcher_isolation":
                isolated = "dispatcher" in endpoints
            elif c.blackout_scope == "terminal_external_isolation":
                isolated = "terminal" in endpoints and not any(e.startswith("local:") for e in endpoints)
            if isolated:
                return False, 0.0, "central_blackout" if c.blackout_scope == "all_links" else c.blackout_scope
        slot = math.floor(at_s / c.slot_s)
        if self.mask(link)[slot]:
            return False, 0.0, "channel_loss"
        # Delay depends on physical send time; repeated queries do not resample.
        u = _seed(c.seed, link, f"{at_s:.9f}", "delay") / 2**64
        return True, (c.delay_ms + c.jitter_ms * u) / 1000.0, "delivered"


@dataclass(frozen=True)
class Message:
    message_id: str
    source: str
    destination: str
    kind: str
    target_id: str
    generated_s: float
    sent_s: float
    expires_s: float
    state_version: int
    payload: dict[str, Any]
    reply_to: str = ""
    payload_bytes: int = 512

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.generated_s, self.sent_s, self.expires_s)):
            raise ValueError("finite message timestamps required")
        if not 0 <= self.generated_s <= self.sent_s <= self.expires_s:
            raise ValueError("invalid generation/send/expiry ordering")
        if not self.message_id or not self.source or not self.destination:
            raise ValueError("message identity is required")
        if self.state_version < 0 or self.payload_bytes <= 0:
            raise ValueError("invalid version or payload size")


@dataclass(frozen=True)
class ReceivedMessage:
    message: Message
    received_s: float

    def age_s(self, now_s: float) -> float:
        if now_s < self.received_s:
            raise ValueError("cannot read a future arrival")
        return now_s - self.message.generated_s


class MessageBus:
    def __init__(self, channel: SyntheticChannel, *, router=None) -> None:
        self.channel = channel
        self.router = router
        self._routes: dict[str, dict] = {}
        self.now_s = 0.0
        self.events: list[dict[str, Any]] = []
        self._pending: list[tuple[float, int, Message]] = []
        self._counter = 0
        self._ids: set[str] = set()
        self._received: dict[tuple[str, str, str], ReceivedMessage] = {}
        self._telemetry_sources: dict[tuple[str, str, str], ReceivedMessage] = {}
        self._all_received: dict[str, ReceivedMessage] = {}
        self._acked: dict[str, float] = {}

    def send(self, message: Message, *, link_available: bool = True) -> None:
        if abs(message.sent_s - self.now_s) > 1e-8:
            raise ValueError("send only at the current physical clock")
        if message.message_id in self._ids:
            raise ValueError("duplicate message ID")
        message = replace(message, payload=deepcopy(message.payload))
        self._ids.add(message.message_id)
        ok, delay, reason = self.channel.transmission(
            link=f"{message.source}>{message.destination}", at_s=self.now_s
        )
        if not link_available:
            ok, reason = False, "physical_link_unavailable"
        route = None
        if self.router is not None:
            route = self.router.admit(message, self.now_s)
            self._routes[message.message_id] = route
            if not route["admitted"]:
                ok, reason = False, route["reason"]
        self._event("send", message, self.now_s, outcome="pending" if ok else reason, route=route)
        if ok:
            self._counter += 1
            heapq.heappush(self._pending, (self.now_s + delay, self._counter, message))
        else:
            self._event("drop", message, self.now_s, outcome=reason)

    def advance(self, until_s: float, *, on_receive: Callable[[ReceivedMessage], None] | None = None,
                include_endpoint: bool = True) -> None:
        if not math.isfinite(until_s) or until_s < self.now_s - 1e-9:
            raise ValueError("physical clock cannot move backwards")
        # At most a nanosecond of representation error, never a model step.
        until_s = max(until_s, self.now_s)
        while self._pending and (self._pending[0][0] <= until_s if include_endpoint
                                 else self._pending[0][0] < until_s):
            received_s, _, message = heapq.heappop(self._pending)
            self.now_s = received_s
            if received_s > message.expires_s:
                self._event("drop", message, received_s, outcome="expired_in_transit")
                continue
            if self.router is not None:
                route = self._routes[message.message_id]
                valid, reason = self.router.validate(message, route, received_s)
                if not valid:
                    self._event("drop", message, received_s, outcome=reason, route=route)
                    continue
            received = ReceivedMessage(message, received_s)
            key = (message.destination, message.kind, message.target_id)
            old = self._received.get(key)
            newer = old is None or (message.state_version, message.generated_s) >= (
                old.message.state_version, old.message.generated_s
            )
            self._all_received[message.message_id] = received
            if newer:
                self._received[key] = received
            source_newer = False
            if message.kind == 'telemetry':
                source_key = (message.destination, message.source, message.target_id)
                previous = self._telemetry_sources.get(source_key)
                source_newer = previous is None or (message.state_version, message.generated_s) >= (
                    previous.message.state_version, previous.message.generated_s)
                if source_newer:
                    self._telemetry_sources[source_key] = received
            self._event("receive", message, received_s, outcome="accepted" if newer else "superseded")
            if message.kind == "ack":
                original = self._all_received.get(message.reply_to)
                if original is None or (
                    original.message.source != message.destination
                    or original.message.destination != message.source
                    or original.message.target_id != message.target_id
                    or original.message.state_version != message.state_version
                ):
                    self._event("reject_ack", message, received_s, outcome="identity_or_version_mismatch")
                else:
                    self._acked[message.reply_to] = received_s
            # A different sensor's slightly older report may carry its own
            # device state. Global target freshness must not discard it.
            if (newer or source_newer or (message.kind == 'ack' and message.reply_to in self._acked)) and on_receive is not None:
                on_receive(received)
        self.now_s = until_s

    def next_arrival_s(self) -> float:
        return self._pending[0][0] if self._pending else math.inf

    def latest(self, destination: str, kind: str, target_id: str) -> ReceivedMessage | None:
        return self._received.get((destination, kind, target_id))

    def acknowledged(self, message_id: str) -> bool:
        return message_id in self._acked

    def acknowledge(self, received: ReceivedMessage, *, deadline_s: float = 0.250) -> None:
        m = received.message
        if received.received_s > self.now_s or m.message_id not in self._all_received:
            raise ValueError("cannot acknowledge an unreceived message")
        self.send(Message(
            message_id=f"ack:{m.message_id}", source=m.destination, destination=m.source,
            kind="ack", target_id=m.target_id, generated_s=self.now_s, sent_s=self.now_s,
            expires_s=self.now_s + deadline_s, state_version=m.state_version,
            payload={}, reply_to=m.message_id, payload_bytes=64,
        ))

    def _event(self, kind: str, message: Message, at_s: float, **extra: Any) -> None:
        self.events.append({
            "event": kind, "event_s": at_s,
            # Keep the actual transmitted observation/authorization, not only
            # its ID. asdict snapshots payloads before later state changes.
            **asdict(message), **extra,
        })


def execution_evidence(
    *, cached: ReceivedMessage | None, now_s: float, target_id: str,
    observed_state_version: int, require_freshness: bool = True,
    require_reconciliation: bool = True, reconciled: bool = True,
    max_age_s: float = 900.0,
) -> tuple[bool, tuple[str, ...]]:
    """Policy-side packet checks; the independent truth audit is separate."""
    reasons = []
    if cached is None:
        return False, ("no_received_authorization",)
    m = cached.message
    if cached.received_s > now_s:
        raise ValueError("future message cannot authorize execution")
    if m.target_id != target_id or not m.payload.get("authorized", False):
        reasons.append("authorization_mismatch")
    if now_s > m.expires_s:
        reasons.append("authorization_expired")
    if require_freshness and (cached.age_s(now_s) > max_age_s or m.state_version != observed_state_version):
        reasons.append("stale_state")
    if require_reconciliation and not reconciled:
        reasons.append("reconciliation_pending")
    return not reasons, tuple(reasons)
