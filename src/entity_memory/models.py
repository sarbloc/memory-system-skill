"""Data models: Fact, Entity, Event."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Fact:
    text: str
    added: str  # ISO date
    source: str  # event ID that produced this
    expires: Optional[str] = None  # ISO date or None (permanent)
    last_seen: Optional[str] = None  # updated on duplicate hit
    hit_count: int = 1
    embedding: Optional[list[float]] = field(default=None, repr=False)


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
    source: str  # conversation | task | cron
    agent: str = "main"
    extracted: bool = False
    expires: Optional[str] = None  # ISO datetime
