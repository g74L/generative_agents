"""The material object-effect boundary for the existing Smallville event path.

Classification is Smallville compatibility policy, not universal OCE physics.
Neither classification, proposal construction nor grounding publishes events.
ReverieServer commits accepted decisions to the single live Maze.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, AbstractSet


class ObjectEffectKind(str, Enum):
  NO_OBJECT_EFFECT = "NO_OBJECT_EFFECT"
  MATERIAL_OBJECT_EFFECT_PROPOSAL = "MATERIAL_OBJECT_EFFECT_PROPOSAL"
  MALFORMED_OBJECT_EFFECT_PROPOSAL = "MALFORMED_OBJECT_EFFECT_PROPOSAL"


class ObjectEffectRejectionReason(str, Enum):
  MALFORMED_OBJECT_EFFECT = "MALFORMED_OBJECT_EFFECT"
  TARGET_ADDRESS_UNKNOWN = "TARGET_ADDRESS_UNKNOWN"
  TARGET_TILE_MISMATCH = "TARGET_TILE_MISMATCH"


class ObjectEffectStatus(str, Enum):
  ACCEPTED = "ACCEPTED"
  REJECTED = "REJECTED"


def classify_object_effect(event: tuple) -> ObjectEffectKind:
  """Classify Scratch's four-field object tuple without consulting the world.

  Presence means non-None; semantic plausibility and value types are outside
  this wave. Social/waiting payloads must be empty to use compatibility.
  <random> explicitly denotes a location fallback even with generated text.
  Sentinel recognition follows the substring checks in Smallville execute.
  """
  address, predicate, value, description = event
  if not address:
    return ObjectEffectKind.NO_OBJECT_EFFECT
  if "<random>" in address:
    return ObjectEffectKind.NO_OBJECT_EFFECT
  populated = tuple(field is not None for field in
                    (predicate, value, description))
  if not any(populated):
    return ObjectEffectKind.NO_OBJECT_EFFECT
  if "<persona>" in address or "<waiting>" in address:
    return ObjectEffectKind.MALFORMED_OBJECT_EFFECT_PROPOSAL
  if not all(populated):
    return ObjectEffectKind.MALFORMED_OBJECT_EFFECT_PROPOSAL
  return ObjectEffectKind.MATERIAL_OBJECT_EFFECT_PROPOSAL


@dataclass(frozen=True)
class SmallvilleObjectEffectProposal:
  source_actor: str
  target_address: str
  predicate: str
  value: str
  description: str
  publication_tile: tuple[int, int]

  @classmethod
  def from_event(cls, source_actor: str, event: tuple,
                 publication_tile: tuple[int, int]) -> SmallvilleObjectEffectProposal:
    """Copy the classified material tuple; acquiring a value confers no authority."""
    address, predicate, value, description = event
    return cls(source_actor, address, predicate, value, description,
               publication_tile)

  def as_event(self) -> tuple:
    return (self.target_address, self.predicate, self.value, self.description)


@dataclass(frozen=True)
class ObjectEffectDecision:
  proposal: SmallvilleObjectEffectProposal
  status: ObjectEffectStatus
  reason: ObjectEffectRejectionReason | None = None


def ground_object_effect(
    proposal: SmallvilleObjectEffectProposal,
    address_tiles: Mapping[str, AbstractSet[tuple[int, int]]],
) -> ObjectEffectDecision:
  """Read the live Maze address index; apply only the two spatial requirements."""
  if proposal.target_address not in address_tiles:
    return ObjectEffectDecision(
      proposal, ObjectEffectStatus.REJECTED,
      ObjectEffectRejectionReason.TARGET_ADDRESS_UNKNOWN)
  if proposal.publication_tile not in address_tiles[proposal.target_address]:
    return ObjectEffectDecision(
      proposal, ObjectEffectStatus.REJECTED,
      ObjectEffectRejectionReason.TARGET_TILE_MISMATCH)
  return ObjectEffectDecision(proposal, ObjectEffectStatus.ACCEPTED)


class ObjectEffectTransitionError(RuntimeError):
  """Stop the current tick without manufacturing an alternate consequence."""

  def __init__(self, reason: ObjectEffectRejectionReason,
               source_actor: str, target_address: str):
    self.reason = reason
    self.source_actor = source_actor
    self.target_address = target_address
    super().__init__(f"{reason.value}: actor={source_actor}, target={target_address}")
