"""Authoritative factual memory consequences for completed conversations.

This module deliberately owns no schedule, world, relationship, or narrative
interpretation state.  It turns one validated MODEL_END conversation into two
independently owned Chat nodes with one stable shared experience identity.
"""
from dataclasses import dataclass
import datetime
from enum import Enum
import hashlib
import json

from persona.cognitive_modules.conversation_contract import (
  ConversationResult,
  ConversationTermination,
)
from persona.prompt_template.gpt_structure import get_embedding
from persona.prompt_template.llm_provider import (
  MEMORY_WRITE,
  embedding_call_context,
)


SOCIAL_CONVERSATION_SCHEMA = "oce.social_conversation_experience.v1"
EXPERIENCE_KEYWORD_PREFIX = "__oce_social_experience__:"
# Chat nodes are not candidates in new_retrieve(), and current direct Chat
# consumers do not use poignancy.  Zero explicitly means factual but not yet
# cognitively calibrated; it is not an actor-specific salience judgment.
UNINTERPRETED_CHAT_POIGNANCY = 0


class SocialConsequenceCommitStatus(str, Enum):
  COMMITTED = "COMMITTED"
  ALREADY_COMMITTED = "ALREADY_COMMITTED"


class SocialConsequenceInvariantError(RuntimeError):
  """A persisted marker does not identify the expected factual Chat node."""


class PartialSocialConsequenceStateError(SocialConsequenceInvariantError):
  """Exactly one participant contains the shared experience consequence."""


class SocialConsequenceCommitError(RuntimeError):
  """The social consequence could not be committed before any bilateral state."""


class FatalPartialSocialConsequenceCommitError(SocialConsequenceCommitError):
  """Mutation began but both actor-local consequences were not established."""


def _nonempty_text(value, field):
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"{field} must be non-empty text")
  return value


def _canonical_transcript(value):
  if not isinstance(value, (tuple, list)):
    raise ValueError("transcript must be a sequence")
  transcript = tuple(tuple(row) for row in value)
  if not transcript or any(
      len(row) != 2 or not all(isinstance(item, str) and item.strip()
                               for item in row)
      for row in transcript):
    raise ValueError("transcript must contain validated utterances")
  return transcript


@dataclass(frozen=True)
class SocialConversationExperience:
  experience_id: str
  initiator: str
  target: str
  started_at: datetime.datetime
  termination: ConversationTermination
  transcript: tuple[tuple[str, str], ...]
  location: str | None = None
  location_provenance: str | None = None

  def __post_init__(self):
    _nonempty_text(self.experience_id, "experience_id")
    _nonempty_text(self.initiator, "initiator")
    _nonempty_text(self.target, "target")
    if self.initiator == self.target:
      raise ValueError("conversation participants must be distinct")
    if not isinstance(self.started_at, datetime.datetime):
      raise ValueError("started_at must be a datetime")
    if self.termination != ConversationTermination.MODEL_END:
      raise ValueError("only MODEL_END is a completed social experience")
    transcript = _canonical_transcript(self.transcript)
    if transcript != self.transcript:
      raise ValueError("transcript must be an immutable canonical tuple")
    if any(row[0] not in (self.initiator, self.target) for row in transcript):
      raise ValueError("transcript contains a non-participant speaker")
    if (self.location is None) != (self.location_provenance is None):
      raise ValueError("location and provenance must be supplied together")
    if self.location is not None:
      _nonempty_text(self.location, "location")
      _nonempty_text(self.location_provenance, "location_provenance")


@dataclass(frozen=True)
class SocialConsequenceCommitResult:
  status: SocialConsequenceCommitStatus
  experience: SocialConversationExperience
  initiator_chat: object
  target_chat: object


def _experience_payload(initiator, target, started_at, transcript, termination):
  return {
    "schema": SOCIAL_CONVERSATION_SCHEMA,
    "initiator": initiator,
    "target": target,
    "started_at": started_at.isoformat(timespec="seconds"),
    "termination": termination.value,
    "transcript": [list(row) for row in transcript],
  }


def build_social_conversation_experience(
    conversation, initiator, target, started_at, *, location=None,
    location_provenance=None):
  """Validate and identify one completed interaction without side effects."""
  if not isinstance(conversation, ConversationResult):
    raise TypeError("conversation must be a ConversationResult")
  conversation.require_complete()
  initiator = _nonempty_text(initiator, "initiator")
  target = _nonempty_text(target, "target")
  if initiator == target:
    raise ValueError("conversation participants must be distinct")
  if not isinstance(started_at, datetime.datetime):
    raise ValueError("started_at must be a datetime")
  # AssociativeMemory persists timezone-free second precision. Reject values
  # that would be lossy rather than letting two distinct times share an ID or
  # silently breaking consistency after save/reload.
  if started_at.microsecond or (started_at.tzinfo is not None
                                and started_at.utcoffset() is not None):
    raise ValueError("started_at must use timezone-free second precision")
  transcript = _canonical_transcript(conversation.transcript)
  if any(row[0] not in (initiator, target) for row in transcript):
    raise ValueError("transcript contains a non-participant speaker")
  payload = _experience_payload(
    initiator, target, started_at, transcript, conversation.termination)
  digest = hashlib.sha256(json.dumps(
    payload, ensure_ascii=False, sort_keys=True,
    separators=(",", ":")).encode("utf-8")).hexdigest()
  return SocialConversationExperience(
    experience_id=f"{SOCIAL_CONVERSATION_SCHEMA}:{digest}",
    initiator=initiator,
    target=target,
    started_at=started_at,
    termination=conversation.termination,
    transcript=transcript,
    location=location,
    location_provenance=location_provenance,
  )


def experience_keyword(experience_id):
  return EXPERIENCE_KEYWORD_PREFIX + _nonempty_text(
    experience_id, "experience_id")


def factual_chat_description(experience):
  """Exact, deterministic factual content; no summary or interpretation."""
  if not isinstance(experience, SocialConversationExperience):
    raise TypeError("experience must be a SocialConversationExperience")
  serialized = json.dumps(
    [list(row) for row in experience.transcript], ensure_ascii=False,
    separators=(",", ":"))
  return f"in a conversation; transcript: {serialized}"


def chat_nodes_for_experience(memory, experience_id):
  marker = experience_keyword(experience_id).lower()
  return tuple(node for node in memory.seq_chat
               if marker in {str(keyword).lower()
                             for keyword in node.keywords})


def _matches(node, memory, experience, actor, counterpart, description):
  keywords = {str(keyword).lower() for keyword in node.keywords}
  return all((
    getattr(node, "type", None) == "chat",
    node.created == experience.started_at,
    node.expiration is None,
    node.subject == actor,
    node.predicate == "chat with",
    node.object == counterpart,
    node.description == description,
    node.embedding_key == description,
    node.poignancy == UNINTERPRETED_CHAT_POIGNANCY,
    counterpart.lower() in keywords,
    experience_keyword(experience.experience_id).lower() in keywords,
    _canonical_transcript(node.filling) == experience.transcript,
    node.embedding_key in memory.embeddings,
  ))


def require_consistent_experience_chat(
    memory, experience, actor, counterpart):
  """Return the one matching Chat or fail on duplicate/data mismatch."""
  nodes = chat_nodes_for_experience(memory, experience.experience_id)
  if len(nodes) != 1:
    raise SocialConsequenceInvariantError(
      "experience marker must identify exactly one actor-local Chat")
  description = factual_chat_description(experience)
  if not _matches(nodes[0], memory, experience, actor, counterpart,
                  description):
    raise SocialConsequenceInvariantError(
      "experience marker exists but Chat data does not match")
  return nodes[0]


def require_scratch_experience_chat(memory, experience_id, actor,
                                    counterpart, transcript):
  """Validate the perception bridge without reconstructing world authority."""
  nodes = chat_nodes_for_experience(memory, experience_id)
  if len(nodes) != 1:
    raise SocialConsequenceInvariantError(
      "Scratch experience marker lacks exactly one committed Chat")
  node = nodes[0]
  if not all((
      node.subject == actor,
      node.predicate == "chat with",
      node.object == counterpart,
      _canonical_transcript(node.filling) == _canonical_transcript(transcript),
  )):
    raise SocialConsequenceInvariantError(
      "Scratch experience marker does not match committed Chat data")
  return node


def _persona_name(persona):
  name = _nonempty_text(getattr(persona, "name", None), "persona.name")
  scratch_name = _nonempty_text(
    getattr(getattr(persona, "scratch", None), "name", None),
    "persona.scratch.name")
  if name != scratch_name:
    raise SocialConsequenceInvariantError(
      "persona and Scratch identities do not match")
  return name


@embedding_call_context(MEMORY_WRITE)
def commit_social_conversation_experience(
    conversation, initiator_persona, target_persona, *, started_at=None,
    embedding_fn=None):
  """Commit exactly one factual actor-local Chat consequence per participant.

  All validation and external embedding work occurs before the first memory
  mutation. Existing bilateral state is checked without another embedding.
  Unexpected failure after the first write is surfaced as fatal partial state;
  this V1 never silently repairs or rolls back social memory.
  """
  initiator = _persona_name(initiator_persona)
  target = _persona_name(target_persona)
  if started_at is None:
    started_at = initiator_persona.scratch.curr_time
  experience = build_social_conversation_experience(
    conversation, initiator, target, started_at)
  description = factual_chat_description(experience)
  marker = experience_keyword(experience.experience_id)
  memories = (initiator_persona.a_mem, target_persona.a_mem)
  if memories[0] is memories[1]:
    raise SocialConsequenceInvariantError(
      "participants must not share one AssociativeMemory instance")
  existing = tuple(chat_nodes_for_experience(memory, experience.experience_id)
                   for memory in memories)

  if bool(existing[0]) != bool(existing[1]):
    raise PartialSocialConsequenceStateError(
      "social experience is committed for exactly one participant")
  if existing[0]:
    init_node = require_consistent_experience_chat(
      memories[0], experience, initiator, target)
    target_node = require_consistent_experience_chat(
      memories[1], experience, target, initiator)
    return SocialConsequenceCommitResult(
      SocialConsequenceCommitStatus.ALREADY_COMMITTED,
      experience, init_node, target_node)

  for memory in memories:
    if not callable(getattr(memory, "add_chat", None)):
      raise SocialConsequenceInvariantError(
        "participant memory cannot accept Chat consequences")

  embed = embedding_fn or get_embedding
  try:
    embedding = embed(description)
  except Exception as error:
    raise SocialConsequenceCommitError(
      "social experience embedding preparation failed") from error
  embedding_pair = (description, embedding)

  def add(memory, actor, counterpart):
    return memory.add_chat(
      experience.started_at, None, actor, "chat with", counterpart,
      description, {actor, counterpart, marker},
      UNINTERPRETED_CHAT_POIGNANCY, embedding_pair,
      experience.transcript)

  try:
    init_node = add(memories[0], initiator, target)
  except Exception as error:
    if chat_nodes_for_experience(memories[0], experience.experience_id):
      raise FatalPartialSocialConsequenceCommitError(
        "first writer failed after mutating social memory") from error
    raise SocialConsequenceCommitError(
      "social experience commit failed before bilateral mutation") from error
  try:
    init_node = require_consistent_experience_chat(
      memories[0], experience, initiator, target)
  except Exception as error:
    raise FatalPartialSocialConsequenceCommitError(
      "first actor-local consequence is inconsistent") from error
  try:
    target_node = add(memories[1], target, initiator)
  except Exception as error:
    raise FatalPartialSocialConsequenceCommitError(
      "fatal partial social consequence commit") from error

  try:
    target_node = require_consistent_experience_chat(
      memories[1], experience, target, initiator)
  except Exception as error:
    raise FatalPartialSocialConsequenceCommitError(
      "bilateral write completed with inconsistent social consequence data") \
      from error
  return SocialConsequenceCommitResult(
    SocialConsequenceCommitStatus.COMMITTED,
    experience, init_node, target_node)
