"""Typed talk decisions over Smallville's existing Completion Compatibility path.

This contract only decides whether to initiate a conversation. It never applies
chat, schedule or world effects. The historical prompt/wrapper remain available
as legacy evidence; the OCE planner consumes only successful results here.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import datetime
from enum import Enum
import json
from pathlib import Path

from persona.prompt_template.completion_runtime import (
  ModernCompletionCompatError,
  ModernCompletionCompatRequest,
  run_modern_completion_compat,
)
from persona.prompt_template.llm_provider import logical_call
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


class TalkDecision(str, Enum):
  TALK = "yes"
  DO_NOT_TALK = "no"


class TalkDecisionStatus(str, Enum):
  SUCCESS = "SUCCESS"
  INVALID_CONTEXT = "INVALID_CONTEXT"
  INVALID_OUTPUT = "INVALID_OUTPUT"
  REFUSAL = "REFUSAL"
  TIMEOUT = "TIMEOUT"
  UNAVAILABLE = "UNAVAILABLE"
  PROVIDER_ERROR = "PROVIDER_ERROR"


@dataclass(frozen=True)
class SocialTalkDecisionRequest:
  initiator_name: str
  initiator_identity: str
  target_name: str
  current_time: datetime.datetime
  initiator_activity: str
  target_activity: str
  retrieved_events: tuple[str, ...]
  retrieved_thoughts: tuple[str, ...]
  last_chat_time: datetime.datetime | None = None
  last_chat_description: str | None = None


@dataclass(frozen=True)
class SocialTalkDecisionResult:
  status: TalkDecisionStatus
  decision: TalkDecision | None = None
  # Number of contract attempts. Each may include existing runtime transport
  # retries; physical attempts and their costs remain in provider telemetry.
  attempt_count: int = 0
  reason: str | None = None

  def __post_init__(self):
    if not isinstance(self.status, TalkDecisionStatus):
      raise ValueError("invalid talk result status")
    if self.status == TalkDecisionStatus.SUCCESS:
      if not isinstance(self.decision, TalkDecision) or self.reason is not None:
        raise ValueError("successful talk result requires exactly one decision")
    elif self.decision is not None or not self.reason:
      raise ValueError("non-success talk result requires a reason and no decision")


def _present_text(value):
  return isinstance(value, str) and bool(value.strip())


def _memory_descriptions(retrieved, key):
  """Translate only actor-local retrieval; None marks an invalid structure."""
  if not isinstance(retrieved, Mapping):
    return None
  nodes = retrieved.get(key)
  if not isinstance(nodes, (tuple, list)):
    return None
  descriptions = tuple(getattr(node, "description", None) for node in nodes)
  return descriptions if all(_present_text(item) for item in descriptions) else None


def build_social_talk_request(initiator, target, retrieved):
  """Read self identity, observable activities and initiator-local memories.

  In particular, target.get_str_iss/innate/learned are never consulted.
  Missing time is preserved for validation without calling the date-dependent
  legacy identity formatter. Unexpected failures of that formatter propagate.
  """
  current_time = initiator.scratch.curr_time
  identity = (initiator.scratch.get_str_iss()
              if isinstance(current_time, datetime.datetime) else None)
  last_chat = initiator.a_mem.get_last_chat(target.name)
  return SocialTalkDecisionRequest(
    initiator_name=initiator.name, initiator_identity=identity,
    target_name=target.name, current_time=current_time,
    initiator_activity=initiator.scratch.act_description,
    target_activity=target.scratch.act_description,
    retrieved_events=_memory_descriptions(retrieved, "events"),
    retrieved_thoughts=_memory_descriptions(retrieved, "thoughts"),
    last_chat_time=getattr(last_chat, "created", None) if last_chat else None,
    last_chat_description=(getattr(last_chat, "description", None) or ""
                           if last_chat else None),
  )


def validate_social_talk_context(request):
  """Return a stable missing/invalid-context reason, or None when sufficient."""
  for field in ("initiator_name", "target_name", "initiator_identity",
                "initiator_activity", "target_activity"):
    if not _present_text(getattr(request, field)):
      return "MISSING_" + field.upper()
  if not isinstance(request.current_time, datetime.datetime):
    return "INVALID_CURRENT_TIME"
  for field in ("retrieved_events", "retrieved_thoughts"):
    value = getattr(request, field)
    if not isinstance(value, tuple) or not all(_present_text(item) for item in value):
      return "INVALID_" + field.upper()
  if request.last_chat_time is not None or request.last_chat_description is not None:
    if (not isinstance(request.last_chat_time, datetime.datetime)
        or not _present_text(request.last_chat_description)):
      return "INVALID_LAST_CHAT_CONTEXT"
  return None


def render_social_talk_prompt(request):
  """Render the dedicated contract template without rewriting supplied context."""
  template = (Path(__file__).resolve().parents[1] / "prompt_template"
              / "oce" / "social_talk_decision.txt").read_text(encoding="utf-8")
  last_chat = "No previous chat known."
  if request.last_chat_time is not None:
    last_chat = (f"{request.last_chat_time.isoformat(sep=' ')}: "
                 f"{request.last_chat_description}")
  return template.format(
    identity=request.initiator_identity,
    events=json.dumps(request.retrieved_events, ensure_ascii=False),
    thoughts=json.dumps(request.retrieved_thoughts, ensure_ascii=False),
    time=request.current_time.isoformat(sep=" "),
    initiator=request.initiator_name, target=request.target_name,
    initiator_activity=request.initiator_activity,
    target_activity=request.target_activity, last_chat=last_chat)


class _InvalidTalkOutput(ValueError):
  pass


def parse_social_talk_output(text):
  """Require exactly one JSON decision key; reject duplicates and extra prose."""
  def unique_object(pairs):
    result = {}
    for key, value in pairs:
      if key in result:
        raise _InvalidTalkOutput("DUPLICATE_DECISION_FIELD")
      result[key] = value
    return result

  if not isinstance(text, str):
    raise _InvalidTalkOutput("NON_TEXT_OUTPUT")
  try:
    value = json.loads(text, object_pairs_hook=unique_object)
  except json.JSONDecodeError as error:
    raise _InvalidTalkOutput("INVALID_DECISION_JSON") from error
  if (not isinstance(value, dict) or set(value) != {"decision"}
      or value["decision"] not in ("yes", "no")):
    raise _InvalidTalkOutput("INVALID_DECISION_SCHEMA")
  return TalkDecision(value["decision"])


def decide_social_talk(request):
  """Validate context, then try the same output contract at most five times.

  Transport retries remain owned by Completion Compatibility. One outer logical
  call retains accounting across output retries. Cost-guard failures and unknown
  programming exceptions propagate; no exception is converted to yes or no.
  """
  reason = validate_social_talk_context(request)
  if reason is not None:
    return SocialTalkDecisionResult(TalkDecisionStatus.INVALID_CONTEXT, reason=reason)
  prompt = render_social_talk_prompt(request)
  with logical_call():
    for attempt in range(1, 6):
      try:
        # source_model identifies legacy intent only. The actual provider model
        # is selected exclusively by the active Completion Compatibility config.
        provider_request = ModernCompletionCompatRequest(
          prompt=prompt, source_model="text-davinci-003", caller_id="decide_to_talk",
          temperature=0, max_tokens=512, top_p=1, frequency_penalty=0,
          presence_penalty=0, stream=False, stop=None)
        text = run_modern_completion_compat(provider_request)
        decision = parse_social_talk_output(text)
      except (_InvalidTalkOutput, LLMIncompleteResponseError,
              LLMEmptyOutputError, LLMMalformedResponseError) as error:
        reason = str(error) if isinstance(error, _InvalidTalkOutput) else type(error).__name__
        continue
      except LLMRefusalError:
        return SocialTalkDecisionResult(
          TalkDecisionStatus.REFUSAL, attempt_count=attempt, reason="PROVIDER_REFUSAL")
      except LLMTimeoutError:
        return SocialTalkDecisionResult(
          TalkDecisionStatus.TIMEOUT, attempt_count=attempt, reason="PROVIDER_TIMEOUT")
      except (LLMConnectionError, LLMRateLimitError, LLMServerError) as error:
        return SocialTalkDecisionResult(
          TalkDecisionStatus.UNAVAILABLE, attempt_count=attempt, reason=type(error).__name__)
      except (LLMProviderError, ModernCompletionCompatError) as error:
        return SocialTalkDecisionResult(
          TalkDecisionStatus.PROVIDER_ERROR, attempt_count=attempt, reason=type(error).__name__)
      return SocialTalkDecisionResult(
        TalkDecisionStatus.SUCCESS, decision, attempt_count=attempt)
  return SocialTalkDecisionResult(
    TalkDecisionStatus.INVALID_OUTPUT, attempt_count=5, reason=reason)


class CognitiveDecisionUnavailableError(RuntimeError):
  """Structural diagnostic for a failed talk decision, without provider text."""

  def __init__(self, request, result):
    self.surface = "decide_to_talk"
    self.status = result.status
    self.actor = request.initiator_name
    self.target = request.target_name
    self.attempt_count = result.attempt_count
    self.reason = result.reason
    super().__init__(
      f"{self.surface}: status={self.status.value}, actor={self.actor}, "
      f"target={self.target}, attempts={self.attempt_count}, reason={self.reason}")


def consume_social_talk_result(request, result):
  if result.status != TalkDecisionStatus.SUCCESS:
    raise CognitiveDecisionUnavailableError(request, result)
  return result.decision == TalkDecision.TALK
