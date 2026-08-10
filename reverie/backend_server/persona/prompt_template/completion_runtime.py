"""Explicit legacy-Completion-to-modern-Chat compatibility runtime."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Mapping, Optional

from persona.prompt_template.chat_runtime import (
  M5_CHAT_MODEL,
  validate_modern_chat_provider_result,
)
from persona.prompt_template.llm_provider import (
  COMPLETION_COMPAT,
  completion_compat,
  create_llm_provider,
  get_completion_compat_provider,
  get_llm_replay_context,
  LLMReplayContext,
  logical_call,
  use_completion_compat_provider,
  use_llm_replay_context,
)
from persona.prompt_template.llm_provider_config import (
  MODERN_OPENAI,
  MODERN_SDK_MODE,
  MODERN_TRANSPORT,
  LLMProviderConfig,
)
from persona.prompt_template.modern_openai_provider import (
  LLMConnectionError,
  LLMRateLimitError,
  LLMServerError,
  LLMTimeoutError,
  use_legacy_completion_length_output,
)


COMPLETION_COMPAT_MODEL = M5_CHAT_MODEL
LEGACY_COMPLETION_INTENT_MODELS = (
  "text-davinci-002",
  "text-davinci-003",
)
FORBIDDEN_MODERN_RUNTIME_MODELS = (
  "text-davinci-002",
  "text-davinci-003",
  "gpt-3.5-turbo",
  "text-embedding-ada-002",
)
M6_REPLAY_CALLER_ALLOWLIST = (
  "wake_up_hour",
  "daily_plan",
  "generate_hourly_schedule",
  "task_decomp",
  "action_sector",
  "action_arena",
  "action_game_object",
  "event_triple",
  "act_obj_event_triple",
  "new_decomp_schedule",
  "decide_to_talk",
  "decide_to_react",
  "focal_pt",
  "insight_and_guidance",
  "generate_next_convo_line",
  "generate_whisper_inner_thought",
  "planning_thought_on_convo",
  "memo_on_convo",
)
M6_DEFERRED_CALLERS = (
  "create_conversation",
  "extract_keywords",
  "keyword_to_thoughts",
  "convo_to_thoughts",
)
_RETRYABLE_ERRORS = (
  LLMConnectionError, LLMRateLimitError, LLMServerError, LLMTimeoutError)


class ModernCompletionCompatError(RuntimeError):
  pass


class ModernCompletionCompatInactiveError(ModernCompletionCompatError):
  pass


class ModernCompletionCompatRequestError(ModernCompletionCompatError,
                                         ValueError):
  pass


class LegacyModelInvocationDetectedError(ModernCompletionCompatRequestError):
  pass


class UnsupportedLegacyCompletionIntentError(
    LegacyModelInvocationDetectedError):
  pass


class CompletionCompatCallerNotAllowedError(ModernCompletionCompatRequestError):
  pass


def validate_completion_compat_caller(caller_id):
  if (not isinstance(caller_id, str) or not caller_id.strip()
      or caller_id not in M6_REPLAY_CALLER_ALLOWLIST):
    safe_caller = caller_id if isinstance(caller_id, str) else type(caller_id).__name__
    if isinstance(safe_caller, str):
      safe_caller = safe_caller.strip()[:128] or "missing"
    raise CompletionCompatCallerNotAllowedError(
      f"CompletionCompat caller is not allowed: {safe_caller}")
  return caller_id


def assert_no_legacy_model_invocation(model):
  if model in FORBIDDEN_MODERN_RUNTIME_MODELS:
    raise LegacyModelInvocationDetectedError(
      "Legacy model invocation is forbidden in the modern runtime")
  return model


@dataclass(frozen=True)
class ModernCompletionRuntimeConfig:
  provider_kind: str = MODERN_OPENAI
  transport_kind: str = MODERN_TRANSPORT
  sdk_mode: str = MODERN_SDK_MODE
  model: str = COMPLETION_COMPAT_MODEL
  operation: str = COMPLETION_COMPAT
  application_retry_count: int = 0
  request_timeout_seconds: float = 600.0

  def __post_init__(self):
    if (self.provider_kind, self.transport_kind, self.sdk_mode) != (
        MODERN_OPENAI, MODERN_TRANSPORT, MODERN_SDK_MODE):
      raise ValueError(
        "Modern CompletionCompat provider, transport, and SDK mode must match")
    assert_no_legacy_model_invocation(self.model)
    if self.model != COMPLETION_COMPAT_MODEL:
      raise ValueError("M6 supports only its canonical pinned model")
    if self.operation != COMPLETION_COMPAT:
      raise ValueError("M6 supports only COMPLETION_COMPAT")
    if (type(self.application_retry_count) is not int
        or self.application_retry_count < 0):
      raise ValueError("application_retry_count must be non-negative")
    if (not isinstance(self.request_timeout_seconds, Real)
        or isinstance(self.request_timeout_seconds, bool)
        or self.request_timeout_seconds <= 0):
      raise ValueError("request_timeout_seconds must be positive")

  def provider_config(self):
    return LLMProviderConfig(
      provider_kind=self.provider_kind,
      transport_kind=self.transport_kind,
      sdk_mode=self.sdk_mode,
      sdk_retry_count=0,
      request_timeout_seconds=float(self.request_timeout_seconds),
      store_responses=False,
    )


def build_modern_completion_runtime_config(**changes):
  return ModernCompletionRuntimeConfig(**changes)


def _validate_real(value, field_name, minimum=None, maximum=None):
  if not isinstance(value, Real) or isinstance(value, bool):
    raise ModernCompletionCompatRequestError(
      f"{field_name} must be numeric")
  if minimum is not None and value < minimum:
    raise ModernCompletionCompatRequestError(
      f"{field_name} is below its supported range")
  if maximum is not None and value > maximum:
    raise ModernCompletionCompatRequestError(
      f"{field_name} is above its supported range")


def _freeze_stop(stop):
  def freeze_text(value):
    if not value:
      raise ModernCompletionCompatRequestError(
        "stop text must be non-empty")
    if value.isspace():
      # Historical wrappers use LF/CRLF as stop sequences.  Horizontal-only
      # whitespace is not meaningful, and a bare CR is deliberately rejected;
      # every CR in a whitespace-only stop must be part of an exact CRLF pair.
      if "\n" not in value or any(
          char == "\r" and (
            index + 1 == len(value) or value[index + 1] != "\n")
          for index, char in enumerate(value)):
        raise ModernCompletionCompatRequestError(
          "stop text must not contain only horizontal whitespace")
    return value

  if stop is None:
    return None
  if isinstance(stop, str):
    return freeze_text(stop)
  if isinstance(stop, (list, tuple)):
    if not stop or not all(isinstance(item, str) for item in stop):
      raise ModernCompletionCompatRequestError(
        "stop sequences must contain non-empty text")
    return tuple(freeze_text(item) for item in stop)
  raise ModernCompletionCompatRequestError(
    "stop must be text or a sequence of text")


@dataclass(frozen=True)
class ModernCompletionCompatRequest:
  prompt: str
  source_model: str
  caller_id: str
  temperature: float
  max_tokens: int
  top_p: float
  frequency_penalty: float
  presence_penalty: float
  stream: bool
  stop: Any = None

  def __post_init__(self):
    validate_completion_compat_caller(self.caller_id)
    if not isinstance(self.prompt, str):
      raise ModernCompletionCompatRequestError("prompt must be text")
    if self.source_model not in LEGACY_COMPLETION_INTENT_MODELS:
      raise UnsupportedLegacyCompletionIntentError(
        "Unsupported legacy Completion intent in the modern runtime")
    _validate_real(self.temperature, "temperature")
    if type(self.max_tokens) is not int or self.max_tokens <= 0:
      raise ModernCompletionCompatRequestError(
        "max_tokens must be a positive integer")
    _validate_real(self.top_p, "top_p", 0, 1)
    _validate_real(self.frequency_penalty, "frequency_penalty", -2, 2)
    _validate_real(self.presence_penalty, "presence_penalty", -2, 2)
    if self.stream is not False:
      raise ModernCompletionCompatRequestError(
        "streaming is not supported by CompletionCompat V0")
    object.__setattr__(self, "stop", _freeze_stop(self.stop))


_runtime_config = ContextVar(
  "modern_completion_compat_runtime_config", default=None)


def get_modern_completion_runtime_config():
  return _runtime_config.get()


def is_modern_completion_runtime_active():
  return get_modern_completion_runtime_config() is not None


@contextmanager
def use_modern_completion_runtime(config, modern_client_adapter):
  if not isinstance(config, ModernCompletionRuntimeConfig):
    raise TypeError("config must be ModernCompletionRuntimeConfig")
  provider = create_llm_provider(config.provider_config(), modern_client_adapter)
  token = _runtime_config.set(config)
  try:
    with use_completion_compat_provider(provider):
      yield provider
  finally:
    _runtime_config.reset(token)


def request_from_legacy_parameters(prompt, parameters, caller_id=None):
  if not isinstance(parameters, Mapping):
    raise ModernCompletionCompatRequestError(
      "legacy Completion parameters must be a mapping")
  required = (
    "engine", "temperature", "max_tokens", "top_p",
    "frequency_penalty", "presence_penalty", "stream", "stop")
  missing = [name for name in required if name not in parameters]
  if missing:
    raise ModernCompletionCompatRequestError(
      "legacy Completion parameters are incomplete")
  return ModernCompletionCompatRequest(
    prompt=prompt,
    source_model=parameters["engine"],
    caller_id=caller_id,
    temperature=parameters["temperature"],
    max_tokens=parameters["max_tokens"],
    top_p=parameters["top_p"],
    frequency_penalty=parameters["frequency_penalty"],
    presence_penalty=parameters["presence_penalty"],
    stream=parameters["stream"],
    stop=parameters["stop"],
  )


def run_modern_completion_compat(request):
  if not isinstance(request, ModernCompletionCompatRequest):
    raise TypeError("request must be ModernCompletionCompatRequest")
  config = get_modern_completion_runtime_config()
  if config is None or get_completion_compat_provider() is None:
    raise ModernCompletionCompatInactiveError(
      "CompletionCompat requires an explicit runtime context")
  assert_no_legacy_model_invocation(config.model)

  def discard_response_metadata():
    consume = getattr(
      get_completion_compat_provider(), "consume_response_metadata", None)
    if callable(consume):
      consume()

  effective_context = replace(
    get_llm_replay_context(), caller_id=request.caller_id)
  with use_llm_replay_context(effective_context):
    with logical_call():
      for attempt in range(config.application_retry_count + 1):
        try:
          with use_legacy_completion_length_output():
            response = completion_compat(
              model=config.model,
              prompt=request.prompt,
              temperature=request.temperature,
              max_tokens=request.max_tokens,
              top_p=request.top_p,
              frequency_penalty=request.frequency_penalty,
              presence_penalty=request.presence_penalty,
              stop=request.stop,
              result_validator=validate_modern_chat_provider_result,
              on_result_validation_error=discard_response_metadata,
            )
          return response["choices"][0]["message"]["content"]
        except _RETRYABLE_ERRORS:
          if attempt >= config.application_retry_count:
            raise
  raise AssertionError("unreachable CompletionCompat retry state")


def run_legacy_completion_compat(prompt, parameters, caller_id=None):
  return run_modern_completion_compat(
    request_from_legacy_parameters(prompt, parameters, caller_id=caller_id))
