"""Narrow typed cognition boundary for agent_chat_v2; no social side effects.

Historical wrappers are intentionally not used. Only validated successes carry
semantic content. Runtime transport retries/accounting retain their ownership.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
import datetime
from enum import Enum
import json
from pathlib import Path

from persona.cognitive_modules.social_talk_decision import (
  TalkDecisionStatus as ConversationStatus,
)
from persona.prompt_template.chat_runtime import (
  ModernChatRequest, ModernChatRuntimeError, run_modern_chat,
  validate_modern_chat_caller,
)
from persona.prompt_template.llm_provider import (
  CONVERSATION, get_llm_replay_context, logical_call, use_llm_replay_context,
)
from persona.prompt_template.modern_openai_provider import (
  LLMConnectionError, LLMEmptyOutputError, LLMIncompleteResponseError,
  LLMMalformedResponseError, LLMProviderError, LLMRateLimitError,
  LLMRefusalError, LLMServerError, LLMTimeoutError,
)

MAX_FORMAT_ATTEMPTS = 3
MAX_CONVERSATION_ROUNDS = 8


def _text(value):
  return isinstance(value, str) and bool(value.strip())


def _transcript(value):
  return (isinstance(value, tuple) and all(
    isinstance(row, tuple) and len(row) == 2 and all(_text(v) for v in row)
    for row in value))


@dataclass(frozen=True)
class ConversationRelationshipRequest:
  speaker: str
  target: str
  memories: tuple[str, ...]


@dataclass(frozen=True)
class ConversationUtteranceRequest:
  speaker: str
  target: str
  identity: str
  memories: tuple[str, ...]
  relationship: str
  transcript: tuple[tuple[str, str], ...]
  current_time: datetime.datetime
  location: str
  speaker_activity: str
  target_activity: str
  previous_chat: str | None = None


def _validate_result(status, attempt_count, reason, has_content):
  if not isinstance(status, ConversationStatus):
    raise ValueError('invalid conversation status')
  if type(attempt_count) is not int or attempt_count < 0:
    raise ValueError('invalid attempt count')
  if status == ConversationStatus.SUCCESS:
    if not has_content or reason is not None:
      raise ValueError('success requires validated content and no failure reason')
  elif has_content or not _text(reason):
    raise ValueError('non-success requires a structural reason and no cognition')


@dataclass(frozen=True)
class ConversationRelationshipResult:
  status: ConversationStatus
  relationship: str | None = None
  attempt_count: int = 0
  reason: str | None = None

  def __post_init__(self):
    _validate_result(self.status, self.attempt_count, self.reason,
                     self.relationship is not None)
    if self.status == ConversationStatus.SUCCESS and not _text(self.relationship):
      raise ValueError('relationship must be non-empty text')


@dataclass(frozen=True)
class ConversationUtteranceResult:
  status: ConversationStatus
  utterance: str | None = None
  end: bool | None = None
  attempt_count: int = 0
  reason: str | None = None

  def __post_init__(self):
    _validate_result(self.status, self.attempt_count, self.reason,
                     self.utterance is not None or self.end is not None)
    if self.status == ConversationStatus.SUCCESS:
      if not _text(self.utterance) or type(self.end) is not bool:
        raise ValueError('utterance requires non-empty text and boolean end')


class ConversationTermination(str, Enum):
  MODEL_END = 'MODEL_END'
  SAFETY_CEILING = 'SAFETY_CEILING'
  FAILURE = 'FAILURE'


@dataclass(frozen=True)
class ConversationFailure:
  surface: str
  speaker: str
  target: str
  status: ConversationStatus
  attempt_count: int
  reason: str

  def __post_init__(self):
    _validate_result(self.status, self.attempt_count, self.reason, False)
    if not all(_text(v) for v in (self.surface, self.speaker, self.target)):
      raise ValueError('failure requires structural actor/surface identifiers')


@dataclass(frozen=True)
class ConversationResult:
  transcript: tuple[tuple[str, str], ...]
  termination: ConversationTermination
  failure: ConversationFailure | None = None

  def __post_init__(self):
    if not _transcript(self.transcript):
      raise ValueError('invalid conversation transcript')
    if not isinstance(self.termination, ConversationTermination):
      raise ValueError('invalid termination')
    if self.termination == ConversationTermination.FAILURE:
      if not isinstance(self.failure, ConversationFailure):
        raise ValueError('failed conversation requires structural cause')
    elif self.failure is not None or not self.transcript:
      raise ValueError('non-failure conversation requires speech and no failure')

  @property
  def turn_count(self):
    return len(self.transcript)

  def require_complete(self):
    if self.termination == ConversationTermination.FAILURE:
      raise ConversationCognitionUnavailableError(self)
    if self.termination == ConversationTermination.SAFETY_CEILING:
      raise ConversationIncompleteError(self)
    return self

  def __iter__(self):
    """Read-only success view for the existing generate_convo/summary bridge.

    The bridge retains this typed object, including termination. Only MODEL_END
    becomes iterable for legacy consequence consumers. Explicit .transcript
    remains available for inspecting incomplete and failed diagnostics.
    """
    self.require_complete()
    return iter(self.transcript)


class ConversationCognitionUnavailableError(RuntimeError):
  def __init__(self, result):
    self.result = result
    self.status = result.failure.status
    self.surface = result.failure.surface
    super().__init__(f'conversation: {self.surface}: {self.status.value}')


class ConversationIncompleteError(RuntimeError):
  """Valid bounded cognition that lacks explicit semantic completion."""
  def __init__(self, result):
    self.result = result
    self.termination = result.termination
    self.turn_count = result.turn_count
    super().__init__(f'conversation: {self.termination.value}: incomplete')


def _memories(retrieved, attribute):
  if not isinstance(retrieved, Mapping):
    return None
  values = []
  for nodes in retrieved.values():
    if not isinstance(nodes, (tuple, list)):
      return None
    values.extend(getattr(node, attribute, None) for node in nodes)
  return tuple(values) if all(_text(v) for v in values) else None


def build_relationship_request(speaker, target, retrieved):
  # Preserve relationship semantics: actor-local embedding keys and names only.
  return ConversationRelationshipRequest(speaker.scratch.name, target.scratch.name,
                                          _memories(retrieved, 'embedding_key'))


def build_utterance_request(maze, speaker, target, retrieved, transcript, relationship):
  scratch = speaker.scratch
  now = scratch.curr_time
  identity = scratch.get_str_iss() if isinstance(now, datetime.datetime) else None
  previous = None
  if isinstance(now, datetime.datetime) and speaker.a_mem.seq_chat:
    for node in speaker.a_mem.seq_chat:
      if node.object == target.scratch.name:
        if not isinstance(node.created, datetime.datetime) or not _text(node.description):
          previous = ''  # Invalid supplied memory, not absence of memory.
        else:
          minutes = int((now - node.created).total_seconds() / 60)
          previous = f'{minutes} minutes ago, {scratch.name} and {target.scratch.name} were {node.description}. This is after that conversation.'
        break
    # Preserve the existing eight-hour context window, without fixing retrieval.
    oldest = speaker.a_mem.seq_chat[-1].created
    if not isinstance(oldest, datetime.datetime):
      previous = ''
    elif previous != '' and int((now - oldest).total_seconds() / 60) > 480:
      previous = None
  tile = maze.access_tile(scratch.curr_tile)
  return ConversationUtteranceRequest(
    speaker=scratch.name, target=target.scratch.name, identity=identity,
    memories=_memories(retrieved, 'description'),
    relationship=(relationship.relationship
      if isinstance(relationship, ConversationRelationshipResult)
      and relationship.status == ConversationStatus.SUCCESS else None),
    transcript=(tuple(tuple(row) for row in transcript)
                if isinstance(transcript, (list, tuple)) else None),
    current_time=now, location=f"{tile['arena']} in {tile['sector']}",
    speaker_activity=scratch.act_description,
    target_activity=target.scratch.act_description, previous_chat=previous)


def validate_relationship_context(request):
  for name in ('speaker', 'target'):
    if not _text(getattr(request, name)):
      return 'INVALID_' + name.upper()
  if not isinstance(request.memories, tuple) or not all(_text(v) for v in request.memories):
    return 'INVALID_LOCAL_MEMORIES'
  return None


def validate_utterance_context(request):
  reason = validate_relationship_context(request)
  if reason:
    return reason
  for name in ('identity', 'relationship', 'location', 'speaker_activity', 'target_activity'):
    if not _text(getattr(request, name)):
      return 'INVALID_' + name.upper()
  if not isinstance(request.current_time, datetime.datetime):
    return 'INVALID_CURRENT_TIME'
  if not _transcript(request.transcript) or any(
      row[0] not in (request.speaker, request.target) for row in request.transcript):
    return 'INVALID_TRANSCRIPT'
  if request.previous_chat is not None and not _text(request.previous_chat):
    return 'INVALID_PREVIOUS_CHAT'
  return None


def _render(template, **values):
  path = Path(__file__).resolve().parents[1] / 'prompt_template/oce' / template
  return path.read_text(encoding='utf-8').format(**values)


def render_relationship_prompt(request):
  return _render('conversation_relationship.txt', speaker=request.speaker,
    target=request.target, memories=json.dumps(request.memories, ensure_ascii=False))


def render_utterance_prompt(request):
  return _render('conversation_utterance.txt', identity=request.identity,
    speaker=request.speaker, target=request.target, relationship=request.relationship,
    memories=json.dumps(request.memories, ensure_ascii=False),
    transcript=json.dumps(request.transcript, ensure_ascii=False),
    time=request.current_time.isoformat(sep=' '), location=request.location,
    speaker_activity=request.speaker_activity, target_activity=request.target_activity,
    previous_chat=request.previous_chat or 'No previous conversation locally recalled.',
    turn_context=('The conversation has not started; the speaker initiates.'
                  if not request.transcript else 'Continue the conversation as the speaker.'))


class InvalidConversationOutput(ValueError):
  pass


def _json_object(text, keys):
  def unique(pairs):
    result = {}
    for key, value in pairs:
      if key in result:
        raise InvalidConversationOutput('DUPLICATE_FIELD')
      result[key] = value
    return result
  def invalid_constant(value):
    raise InvalidConversationOutput('INVALID_JSON_CONSTANT')
  if not isinstance(text, str):
    raise InvalidConversationOutput('NON_TEXT_OUTPUT')
  try:
    result = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
  except json.JSONDecodeError as error:
    raise InvalidConversationOutput('INVALID_JSON') from error
  if not isinstance(result, dict) or set(result) != set(keys):
    raise InvalidConversationOutput('INVALID_KEYS')
  return result


def parse_relationship_output(text):
  value = _json_object(text, ('relationship',))['relationship']
  if not _text(value):
    raise InvalidConversationOutput('INVALID_RELATIONSHIP_TEXT')
  return value


def parse_utterance_output(text):
  value = _json_object(text, ('utterance', 'end'))
  if not _text(value['utterance']):
    raise InvalidConversationOutput('INVALID_UTTERANCE_TEXT')
  if type(value['end']) is not bool:
    raise InvalidConversationOutput('INVALID_END_BOOLEAN')
  return value['utterance'], value['end']


def provider_failure(error):
  """Map known infrastructure errors only; callers never catch arbitrary bugs."""
  if isinstance(error, LLMRefusalError):
    return ConversationStatus.REFUSAL
  if isinstance(error, LLMTimeoutError):
    return ConversationStatus.TIMEOUT
  if isinstance(error, (LLMConnectionError, LLMRateLimitError, LLMServerError)):
    return ConversationStatus.UNAVAILABLE
  if isinstance(error, (LLMEmptyOutputError, LLMIncompleteResponseError, LLMMalformedResponseError)):
    return ConversationStatus.INVALID_OUTPUT
  if isinstance(error, (LLMProviderError, ModernChatRuntimeError)):
    return ConversationStatus.PROVIDER_ERROR
  raise error


def _generate(request, relationship):
  # This helper is private to these two concrete conversation surfaces.
  result_type = ConversationRelationshipResult if relationship else ConversationUtteranceResult
  validate = validate_relationship_context if relationship else validate_utterance_context
  reason = validate(request)
  if reason:
    return result_type(ConversationStatus.INVALID_CONTEXT, reason=reason)
  prompt = render_relationship_prompt(request) if relationship else render_utterance_prompt(request)
  caller = 'agent_chat_summarize_relationship' if relationship else 'iterative_chat_utterance'
  context = replace(get_llm_replay_context(), caller_id=caller,
                    actor_id=request.speaker, cognitive_category=CONVERSATION)
  with use_llm_replay_context(context), logical_call():
    for attempt in range(1, MAX_FORMAT_ATTEMPTS + 1):
      try:
        validate_modern_chat_caller(caller)
        response = run_modern_chat(ModernChatRequest(
          messages=({'role': 'user', 'content': prompt},), temperature=0,
          max_tokens=None)).content
        if relationship:
          return result_type(ConversationStatus.SUCCESS, parse_relationship_output(response),
                             attempt_count=attempt)
        utterance, end = parse_utterance_output(response)
        return result_type(ConversationStatus.SUCCESS, utterance, end, attempt_count=attempt)
      except InvalidConversationOutput as error:
        reason = str(error)
      except (LLMProviderError, ModernChatRuntimeError) as error:
        status = provider_failure(error)
        reason = type(error).__name__
        if status != ConversationStatus.INVALID_OUTPUT:
          return result_type(status, attempt_count=attempt, reason=reason)
  return result_type(ConversationStatus.INVALID_OUTPUT,
                     attempt_count=MAX_FORMAT_ATTEMPTS, reason=reason)


def interpret_relationship(request):
  return _generate(request, relationship=True)


def generate_utterance(request):
  return _generate(request, relationship=False)
