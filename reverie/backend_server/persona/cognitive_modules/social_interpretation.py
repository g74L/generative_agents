"""Typed actor-local interpretation of authoritative OCE social memory.

This module turns one validated factual Chat into at most one persistent
actor-owned Thought. It owns no world, schedule, relationship score,
conversation, or trigger policy. A Thought records what an interaction means
to one actor; it is not authoritative truth about the counterpart or world.
"""
from dataclasses import dataclass, replace
import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import re

from persona.cognitive_modules.social_consequence import (
  EXPERIENCE_KEYWORD_PREFIX,
  SOCIAL_CONVERSATION_SCHEMA,
  UNINTERPRETED_CHAT_POIGNANCY,
)
from persona.prompt_template.chat_runtime import (
  ModernChatRequest,
  ModernChatRuntimeError,
  run_modern_chat,
  validate_modern_chat_caller,
)
from persona.prompt_template.gpt_structure import get_embedding
from persona.prompt_template.llm_provider import (
  REFLECTION,
  embedding_call_context,
  get_llm_replay_context,
  logical_call,
  use_llm_replay_context,
)
from persona.prompt_template.modern_openai_provider import (
  LLMConnectionError,
  LLMEmptyOutputError,
  LLMIncompleteResponseError,
  LLMMalformedResponseError,
  LLMProviderError,
  LLMRateLimitError,
  LLMRefusalError,
  LLMServerError,
  LLMTimeoutError,
)


SOCIAL_INTERPRETATION_SCHEMA = "oce.social_interpretation.v1"
INTERPRETATION_KEYWORD_PREFIX = "__oce_social_interpretation__:"
SOCIAL_INTERPRETATION_CALLER = "social_interpretation"
MAX_FORMAT_ATTEMPTS = 3
# Zero is a compatibility value, not a judgment that the Thought is
# cognitively unimportant. This contract does not yet calibrate salience.
UNINTERPRETED_SOCIAL_THOUGHT_POIGNANCY = 0

_EXPERIENCE_ID = re.compile(
  rf"^{re.escape(SOCIAL_CONVERSATION_SCHEMA)}:[0-9a-f]{{64}}$")


class SocialInterpretationStatus(str, Enum):
  SUCCESS = "SUCCESS"
  INVALID_CONTEXT = "INVALID_CONTEXT"
  INVALID_OUTPUT = "INVALID_OUTPUT"
  REFUSAL = "REFUSAL"
  TIMEOUT = "TIMEOUT"
  UNAVAILABLE = "UNAVAILABLE"
  PROVIDER_ERROR = "PROVIDER_ERROR"


class SocialInterpretationCommitStatus(str, Enum):
  COMMITTED = "COMMITTED"
  ALREADY_COMMITTED = "ALREADY_COMMITTED"


class SocialInterpretationContextError(ValueError):
  """The source does not satisfy the OCE interpretation context contract."""


class SocialInterpretationInvariantError(RuntimeError):
  """Persisted interpretation state is duplicate or inconsistent."""


class DuplicateSocialInterpretationError(SocialInterpretationInvariantError):
  """More than one Thought claims the same actor/experience slot."""


class SocialInterpretationDataMismatchError(
    SocialInterpretationInvariantError):
  """A marker exists but its Thought data or evidence lineage disagrees."""


class SocialInterpretationCommitError(RuntimeError):
  """A validated interpretation could not be committed."""


class InvalidSocialInterpretationOutput(ValueError):
  """Provider text does not match the exact output schema."""


def _text(value):
  return isinstance(value, str) and bool(value.strip())


def _canonical_transcript(value, actor_name, counterpart_name):
  if not isinstance(value, (tuple, list)):
    raise SocialInterpretationContextError("INVALID_TRANSCRIPT")
  transcript = tuple(tuple(row) for row in value)
  if (not transcript or any(
      len(row) != 2 or not all(_text(item) for item in row)
      for row in transcript)
      or any(row[0] not in (actor_name, counterpart_name)
             for row in transcript)):
    raise SocialInterpretationContextError("INVALID_TRANSCRIPT")
  return transcript


def _factual_description(transcript):
  serialized = json.dumps(
    [list(row) for row in transcript], ensure_ascii=False,
    separators=(",", ":"))
  return f"in a conversation; transcript: {serialized}"


def _experience_id_candidates(actor_name, counterpart_name, created,
                              transcript):
  """Reconstruct both possible participant orientations from persisted data."""
  values = []
  for initiator, target in ((actor_name, counterpart_name),
                            (counterpart_name, actor_name)):
    payload = {
      "schema": SOCIAL_CONVERSATION_SCHEMA,
      "initiator": initiator,
      "target": target,
      "started_at": created.isoformat(timespec="seconds"),
      "termination": "MODEL_END",
      "transcript": [list(row) for row in transcript],
    }
    digest = hashlib.sha256(json.dumps(
      payload, ensure_ascii=False, sort_keys=True,
      separators=(",", ":")).encode("utf-8")).hexdigest()
    values.append(f"{SOCIAL_CONVERSATION_SCHEMA}:{digest}")
  return tuple(values)


def _experience_id_from_chat(chat_node):
  keywords = getattr(chat_node, "keywords", None)
  if not isinstance(keywords, (set, tuple, list)):
    raise SocialInterpretationContextError("INVALID_CHAT_KEYWORDS")
  markers = [str(value) for value in keywords
             if isinstance(value, str)
             and value.lower().startswith(EXPERIENCE_KEYWORD_PREFIX)]
  if len(markers) != 1:
    reason = ("MISSING_EXPERIENCE_MARKER" if not markers
              else "MULTIPLE_EXPERIENCE_MARKERS")
    raise SocialInterpretationContextError(reason)
  marker = markers[0]
  if not marker.startswith(EXPERIENCE_KEYWORD_PREFIX):
    raise SocialInterpretationContextError("MALFORMED_EXPERIENCE_MARKER")
  experience_id = marker[len(EXPERIENCE_KEYWORD_PREFIX):]
  if not _EXPERIENCE_ID.fullmatch(experience_id):
    raise SocialInterpretationContextError("MALFORMED_EXPERIENCE_MARKER")
  return experience_id


@dataclass(frozen=True)
class SocialInterpretationRequest:
  actor_name: str
  actor_identity: str
  counterpart_name: str
  experience_id: str
  experience_time: datetime.datetime
  factual_description: str
  transcript: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SocialInterpretationResult:
  status: SocialInterpretationStatus
  interpretation: str | None = None
  attempt_count: int = 0
  reason: str | None = None

  def __post_init__(self):
    if not isinstance(self.status, SocialInterpretationStatus):
      raise ValueError("invalid interpretation status")
    if type(self.attempt_count) is not int or self.attempt_count < 0:
      raise ValueError("invalid attempt count")
    if self.status == SocialInterpretationStatus.SUCCESS:
      if not _text(self.interpretation) or self.reason is not None:
        raise ValueError(
          "success requires one interpretation and no failure reason")
    elif self.interpretation is not None or not _text(self.reason):
      raise ValueError(
        "non-success requires a structural reason and no interpretation")


@dataclass(frozen=True)
class SocialInterpretationCommitResult:
  status: SocialInterpretationCommitStatus
  interpretation_id: str
  source_chat: object
  thought: object

  @property
  def interpretation(self):
    return self.thought.description


def validate_social_interpretation_context(request):
  if not isinstance(request, SocialInterpretationRequest):
    return "INVALID_REQUEST_TYPE"
  for field in ("actor_name", "actor_identity", "counterpart_name",
                "experience_id", "factual_description"):
    if not _text(getattr(request, field)):
      return "INVALID_" + field.upper()
  if request.actor_name == request.counterpart_name:
    return "INVALID_COUNTERPART_NAME"
  if not _EXPERIENCE_ID.fullmatch(request.experience_id):
    return "INVALID_EXPERIENCE_ID"
  if not isinstance(request.experience_time, datetime.datetime):
    return "INVALID_EXPERIENCE_TIME"
  if (request.experience_time.microsecond
      or (request.experience_time.tzinfo is not None
          and request.experience_time.utcoffset() is not None)):
    return "INVALID_EXPERIENCE_TIME"
  try:
    transcript = _canonical_transcript(
      request.transcript, request.actor_name, request.counterpart_name)
  except SocialInterpretationContextError:
    return "INVALID_TRANSCRIPT"
  if transcript != request.transcript:
    return "NONCANONICAL_TRANSCRIPT"
  if request.factual_description != _factual_description(transcript):
    return "INVALID_FACTUAL_DESCRIPTION"
  return None


def build_social_interpretation_request(actor, counterpart_name, chat_node):
  """Build actor-local context from one authoritative OCE factual Chat.

  Only the interpreting actor is accepted, so counterpart private persona
  fields cannot enter the request accidentally.
  """
  actor_name = getattr(actor, "name", None)
  scratch = getattr(actor, "scratch", None)
  scratch_name = getattr(scratch, "name", None)
  if not _text(actor_name) or actor_name != scratch_name:
    raise SocialInterpretationContextError("INVALID_ACTOR_NAME")
  if not _text(counterpart_name) or counterpart_name == actor_name:
    raise SocialInterpretationContextError("INVALID_COUNTERPART_NAME")
  if getattr(chat_node, "type", None) != "chat":
    raise SocialInterpretationContextError("INVALID_SOURCE_CHAT_TYPE")
  if getattr(chat_node, "subject", None) != actor_name:
    raise SocialInterpretationContextError("ACTOR_MISMATCH")
  if getattr(chat_node, "predicate", None) != "chat with":
    raise SocialInterpretationContextError("INVALID_SOURCE_CHAT_PREDICATE")
  if getattr(chat_node, "object", None) != counterpart_name:
    raise SocialInterpretationContextError("COUNTERPART_MISMATCH")
  memory = getattr(actor, "a_mem", None)
  node_id = getattr(chat_node, "node_id", None)
  if (not _text(node_id)
      or getattr(memory, "id_to_node", {}).get(node_id) is not chat_node
      or chat_node not in getattr(memory, "seq_chat", ())):
    raise SocialInterpretationContextError("UNPERSISTED_SOURCE_CHAT")
  experience_id = _experience_id_from_chat(chat_node)
  created = getattr(chat_node, "created", None)
  transcript = _canonical_transcript(
    getattr(chat_node, "filling", None), actor_name, counterpart_name)
  description = getattr(chat_node, "description", None)
  if description != _factual_description(transcript):
    raise SocialInterpretationContextError("INVALID_FACTUAL_DESCRIPTION")
  if (not isinstance(created, datetime.datetime) or created.microsecond
      or (created.tzinfo is not None and created.utcoffset() is not None)):
    raise SocialInterpretationContextError("INVALID_EXPERIENCE_TIME")
  keywords = {str(keyword).lower() for keyword in chat_node.keywords}
  if not all((
      chat_node.expiration is None,
      chat_node.embedding_key == description,
      chat_node.poignancy == UNINTERPRETED_CHAT_POIGNANCY,
      actor_name.lower() in keywords,
      counterpart_name.lower() in keywords,
      description in memory.embeddings,
      experience_id in _experience_id_candidates(
        actor_name, counterpart_name, created, transcript),
  )):
    raise SocialInterpretationContextError("INCONSISTENT_SOURCE_CHAT")
  try:
    identity = scratch.get_str_iss()
  except (AttributeError, TypeError, ValueError):
    raise SocialInterpretationContextError("INVALID_ACTOR_IDENTITY") from None
  request = SocialInterpretationRequest(
    actor_name=actor_name,
    actor_identity=identity,
    counterpart_name=counterpart_name,
    experience_id=experience_id,
    experience_time=created,
    factual_description=description,
    transcript=transcript,
  )
  reason = validate_social_interpretation_context(request)
  if reason:
    raise SocialInterpretationContextError(reason)
  return request


def render_social_interpretation_prompt(request):
  reason = validate_social_interpretation_context(request)
  if reason:
    raise SocialInterpretationContextError(reason)
  template = (Path(__file__).resolve().parents[1] / "prompt_template"
              / "oce" / "social_interpretation.txt").read_text(
                encoding="utf-8")
  return template.format(
    actor=request.actor_name,
    counterpart=request.counterpart_name,
    identity=request.actor_identity,
    experience_time=request.experience_time.isoformat(sep=" "),
    factual_description=request.factual_description,
    transcript=json.dumps(request.transcript, ensure_ascii=False),
  )


def parse_social_interpretation_output(text):
  def unique(pairs):
    value = {}
    for key, item in pairs:
      if key in value:
        raise InvalidSocialInterpretationOutput("DUPLICATE_FIELD")
      value[key] = item
    return value

  def invalid_constant(value):
    del value
    raise InvalidSocialInterpretationOutput("INVALID_JSON_CONSTANT")

  if not isinstance(text, str):
    raise InvalidSocialInterpretationOutput("NON_TEXT_OUTPUT")
  try:
    value = json.loads(
      text, object_pairs_hook=unique, parse_constant=invalid_constant)
  except json.JSONDecodeError as error:
    raise InvalidSocialInterpretationOutput("INVALID_JSON") from error
  if not isinstance(value, dict) or set(value) != {"interpretation"}:
    raise InvalidSocialInterpretationOutput("INVALID_KEYS")
  interpretation = value["interpretation"]
  if not _text(interpretation):
    raise InvalidSocialInterpretationOutput("INVALID_INTERPRETATION_TEXT")
  return interpretation


def _provider_failure(error):
  if isinstance(error, LLMRefusalError):
    return SocialInterpretationStatus.REFUSAL
  if isinstance(error, LLMTimeoutError):
    return SocialInterpretationStatus.TIMEOUT
  if isinstance(error, (LLMConnectionError, LLMRateLimitError,
                        LLMServerError)):
    return SocialInterpretationStatus.UNAVAILABLE
  if isinstance(error, (LLMEmptyOutputError, LLMIncompleteResponseError,
                        LLMMalformedResponseError)):
    return SocialInterpretationStatus.INVALID_OUTPUT
  if isinstance(error, (LLMProviderError, ModernChatRuntimeError)):
    return SocialInterpretationStatus.PROVIDER_ERROR
  raise error


def generate_social_interpretation(request):
  """Generate actor-local meaning without mutating any memory or world state."""
  reason = validate_social_interpretation_context(request)
  if reason:
    return SocialInterpretationResult(
      SocialInterpretationStatus.INVALID_CONTEXT, reason=reason)
  prompt = render_social_interpretation_prompt(request)
  context = replace(
    get_llm_replay_context(), caller_id=SOCIAL_INTERPRETATION_CALLER,
    actor_id=request.actor_name, cognitive_category=REFLECTION)
  with use_llm_replay_context(context), logical_call():
    for attempt in range(1, MAX_FORMAT_ATTEMPTS + 1):
      try:
        validate_modern_chat_caller(SOCIAL_INTERPRETATION_CALLER)
        output = run_modern_chat(ModernChatRequest(
          messages=({"role": "user", "content": prompt},),
          temperature=0, max_tokens=None)).content
        interpretation = parse_social_interpretation_output(output)
        return SocialInterpretationResult(
          SocialInterpretationStatus.SUCCESS, interpretation,
          attempt_count=attempt)
      except InvalidSocialInterpretationOutput as error:
        reason = str(error)
      except (LLMProviderError, ModernChatRuntimeError) as error:
        status = _provider_failure(error)
        reason = type(error).__name__
        if status != SocialInterpretationStatus.INVALID_OUTPUT:
          return SocialInterpretationResult(
            status, attempt_count=attempt, reason=reason)
  return SocialInterpretationResult(
    SocialInterpretationStatus.INVALID_OUTPUT,
    attempt_count=MAX_FORMAT_ATTEMPTS, reason=reason)


def social_interpretation_id(request):
  reason = validate_social_interpretation_context(request)
  if reason:
    raise SocialInterpretationContextError(reason)
  payload = {
    "schema": SOCIAL_INTERPRETATION_SCHEMA,
    "experience_id": request.experience_id,
    "actor_name": request.actor_name,
    "counterpart_name": request.counterpart_name,
  }
  digest = hashlib.sha256(json.dumps(
    payload, ensure_ascii=False, sort_keys=True,
    separators=(",", ":")).encode("utf-8")).hexdigest()
  return f"{SOCIAL_INTERPRETATION_SCHEMA}:{digest}"


def interpretation_keyword(interpretation_id):
  if not _text(interpretation_id):
    raise ValueError("interpretation_id must be non-empty text")
  return INTERPRETATION_KEYWORD_PREFIX + interpretation_id


def _marked_thoughts(memory, interpretation_id):
  marker = interpretation_keyword(interpretation_id).lower()
  return tuple(node for node in memory.seq_thought
               if marker in {str(keyword).lower()
                             for keyword in node.keywords})


def _require_existing(actor, request, source_chat):
  interpretation_id = social_interpretation_id(request)
  nodes = _marked_thoughts(actor.a_mem, interpretation_id)
  if len(nodes) > 1:
    raise DuplicateSocialInterpretationError(
      "interpretation marker identifies multiple Thoughts")
  if not nodes:
    return None
  node = nodes[0]
  keywords = {str(keyword).lower() for keyword in node.keywords}
  marker = interpretation_keyword(interpretation_id).lower()
  interpretation_markers = tuple(
    keyword for keyword in keywords
    if keyword.startswith(INTERPRETATION_KEYWORD_PREFIX))
  if not all((
      getattr(node, "type", None) == "thought",
      actor.a_mem.id_to_node.get(node.node_id) is node,
      isinstance(node.created, datetime.datetime),
      node.expiration is None,
      node.subject == request.actor_name,
      node.predicate == "interprets",
      node.object == request.counterpart_name,
      _text(node.description),
      node.embedding_key == node.description,
      node.poignancy == UNINTERPRETED_SOCIAL_THOUGHT_POIGNANCY,
      interpretation_markers == (marker,),
      request.actor_name.lower() in keywords,
      request.counterpart_name.lower() in keywords,
      node.filling == [source_chat.node_id],
      node.embedding_key in actor.a_mem.embeddings,
      actor.a_mem.id_to_node.get(source_chat.node_id) is source_chat,
  )):
    raise SocialInterpretationDataMismatchError(
      "interpretation marker exists but Thought data or lineage does not match")
  return SocialInterpretationCommitResult(
    SocialInterpretationCommitStatus.ALREADY_COMMITTED,
    interpretation_id, source_chat, node)


@embedding_call_context(REFLECTION)
def commit_social_interpretation(
    actor, request, source_chat, result, *, embedding_fn=None):
  """Commit one validated result as an actor-local Thought with Chat lineage."""
  if not isinstance(result, SocialInterpretationResult):
    raise TypeError("result must be a SocialInterpretationResult")
  if result.status != SocialInterpretationStatus.SUCCESS:
    raise SocialInterpretationCommitError(
      "only a successful interpretation can be committed")
  rebuilt = build_social_interpretation_request(
    actor, request.counterpart_name, source_chat)
  if rebuilt != request:
    raise SocialInterpretationInvariantError(
      "source Chat no longer matches the interpretation request")
  existing = _require_existing(actor, request, source_chat)
  if existing:
    return existing
  if not callable(getattr(actor.a_mem, "add_thought", None)):
    raise SocialInterpretationInvariantError(
      "actor memory cannot accept Thought consequences")
  created = getattr(actor.scratch, "curr_time", None)
  if (not isinstance(created, datetime.datetime) or created.microsecond
      or (created.tzinfo is not None and created.utcoffset() is not None)):
    raise SocialInterpretationContextError("INVALID_INTERPRETATION_TIME")
  interpretation_id = social_interpretation_id(request)
  marker = interpretation_keyword(interpretation_id)
  embed = embedding_fn or get_embedding
  try:
    embedding = embed(result.interpretation)
  except Exception as error:
    raise SocialInterpretationCommitError(
      "interpretation embedding preparation failed") from error
  try:
    thought = actor.a_mem.add_thought(
      created, None, request.actor_name, "interprets",
      request.counterpart_name, result.interpretation,
      {request.actor_name, request.counterpart_name, marker},
      UNINTERPRETED_SOCIAL_THOUGHT_POIGNANCY,
      (result.interpretation, embedding), [source_chat.node_id])
  except Exception as error:
    raise SocialInterpretationCommitError(
      "interpretation Thought commit failed") from error
  committed = _require_existing(actor, request, source_chat)
  if committed is None:
    raise SocialInterpretationInvariantError(
      "interpretation write did not establish its marker")
  return SocialInterpretationCommitResult(
    SocialInterpretationCommitStatus.COMMITTED,
    interpretation_id, source_chat, thought)


def interpret_social_experience(
    actor, counterpart_name, chat_node, *, embedding_fn=None):
  """Explicitly generate and commit, or return a typed cognitive failure.

  The pre-provider marker check makes repeated calls and calls after reload
  free of both provider and embedding work.
  """
  try:
    request = build_social_interpretation_request(
      actor, counterpart_name, chat_node)
  except SocialInterpretationContextError as error:
    return SocialInterpretationResult(
      SocialInterpretationStatus.INVALID_CONTEXT, reason=str(error))
  existing = _require_existing(actor, request, chat_node)
  if existing:
    return existing
  generated = generate_social_interpretation(request)
  if generated.status != SocialInterpretationStatus.SUCCESS:
    return generated
  return commit_social_interpretation(
    actor, request, chat_node, generated, embedding_fn=embedding_fn)


# Readable alias for callers that name the source surface rather than the
# shared experience represented by the authoritative Chat.
interpret_social_chat = interpret_social_experience
