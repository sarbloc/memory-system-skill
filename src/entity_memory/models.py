"""Data models: Fact, Entity, Event."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Fact:
    text: str
    added: str  # ISO date — transaction-time: when we recorded the fact
    source: str  # event ID that produced this
    expires: Optional[str] = None  # ISO date or None (permanent)
    last_seen: Optional[str] = None  # updated on duplicate hit
    hit_count: int = 1
    embedding: Optional[list[float]] = field(default=None, repr=False)
    # Bi-temporal fields (issue #21). All optional/None so legacy payloads
    # deserialize unchanged — a fact with none of these set behaves exactly as
    # before (current, with valid-time falling back to transaction-time).
    valid_from: Optional[str] = None  # ISO date the fact became true in the world
    superseded_at: Optional[str] = None  # ISO date it stopped being true (None = current)
    superseded_by: Optional[str] = None  # identifier (text) of the replacing fact

    @property
    def is_current(self) -> bool:
        """A fact is current unless it has been superseded."""
        return self.superseded_at is None

    def valid_at(self, as_of: str) -> bool:
        """True if this fact was in effect on the given ISO date.

        In effect from ``valid_from`` (falling back to ``added`` when valid-time
        is unknown) through ``superseded_at`` (exclusive — the fact is already
        gone on the day it is superseded, which is the day its replacement takes
        effect). ISO date strings compare correctly lexicographically, so no
        datetime parsing is needed.

        A fact with a TTL (``expires``) is also gone once that date has passed:
        an as-of query after expiry must not resurrect it. ``expires`` may be a
        date or a datetime, so compare on the date prefix only.
        """
        start = self.valid_from or self.added
        if start > as_of:
            return False
        if self.superseded_at is not None and self.superseded_at <= as_of:
            return False
        if self.expires is not None and self.expires[:10] <= as_of:
            return False
        return True


@dataclass
class Entity:
    id: str  # "person:alice", "project:dashboard"
    type: str  # person | project | tool | preference | decision
    facts: list[Fact] = field(default_factory=list)
    last_updated: str = ""


@dataclass
class Event:
    id: str  # UUID
    text: str
    timestamp: str  # ISO datetime
    source: str  # conversation | task | cron (legacy categorisation)
    agent: str = "main"
    extracted: bool = False
    expires: Optional[str] = None  # ISO datetime
    # Provenance fields added for Endurance integration. All optional;
    # legacy callers (OpenClaw via the existing CLI) leave them None.
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    profile: Optional[str] = None
    trigger_source: Optional[str] = None  # telegram_message | ha_motion | ...
