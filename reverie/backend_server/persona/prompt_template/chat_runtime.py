"""Explicit, Chat-only modern runtime; the repository default stays legacy."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

from persona.prompt_template.llm_provider import (
  CHAT,
  chat_completion,
  create_llm_provider,
  get_chat_provider,
  logical_call,
  use_chat_provider,
)
from persona.prompt_template.llm_provider_config import (
  MODERN_OPENAI,
  MODERN_SDK_MODE,
  MODERN_TRANSPORT,
  LLMProviderConfig,
)
from persona.prompt_template.modern_openai_provider import (
  LLMConnectionError,
  LLMMalformedResponseError,
  LLMRateLimitError,
  LLMServerError,
  LLMTimeoutError,
  ModernChatResponseValidationError,
  NormalizedUsage,
  ProviderResponseMetadata,
  validate_normalized_usage,
)


M5_CHAT_MODEL = "gpt-4o-mini"
SUPPORTED_CHAT_ROLES = ("system", "user", "assistant")
_RETRYABLE_ERRORS = (
  LLMConnectionError, LLMRateLimitError, LLMServerError, LLMTimeoutError)


class ModernChatRuntimeError(RuntimeError):
  pass


class ModernChatRuntimeInactiveError(ModernChatRuntimeError):
  pass


class ModernChatRequestError(ModernChatRuntimeError, ValueError):
  pass


class ModernChatModelMismatchError(ModernChatRequestError):
  pass


class ModernChatCallerNotAllowedError(ModernChatRequestError):
  pass


class ModernChatValidationError(ModernChatRuntimeError):
  pass


M5_REPLAY_CALLER_ALLOWLIST = (
  "pronunciatio", "act_obj_desc", "event_poignancy")


def validate_modern_chat_caller(caller_id):
  if (not isinstance(caller_id, str) or not caller_id.strip()
      or caller_id not in M5_REPLAY_CALLER_ALLOWLIST):
    raise ModernChatCallerNotAllowedError(
      "Modern Chat caller is missing or not authorized")
  return caller_id


@dataclass(frozen=True)
class ModernChatRuntimeConfig:
  provider_kind: str = MODERN_OPENAI
  transport_kind: str = MODERN_TRANSPORT
  sdk_mode: str = MODERN_SDK_MODE
  chat_model: str = M5_CHAT_MODEL
  operation: str = CHAT
  application_retry_count: int = 0
  request_timeout_seconds: float = 600.0

  def __post_init__(self):
    if (self.provider_kind, self.transport_kind, self.sdk_mode) != (
        MODERN_OPENAI, MODERN_TRANSPORT, MODERN_SDK_MODE):
      raise ValueError("Modern Chat provider, transport, and SDK mode must match")
    if not isinstance(self.chat_model, str) or not self.chat_model.strip():
      raise ValueError("chat_model is required")
    if self.chat_model != M5_CHAT_MODEL:
      raise ValueError("M5 supports only its canonical Chat model")
    if self.operation != CHAT:
      raise ValueError("M5 supports only the CHAT operation")
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


def build_modern_chat_runtime_config(**changes):
  return ModernChatRuntimeConfig(**changes)


@dataclass(frozen=True)
class ModernChatRequest:
  messages: Tuple[Mapping[str, str], ...]
  model: Optional[str] = None
  temperature: Optional[float] = None
  max_tokens: Optional[int] = None
  stop: Any = None
  response_format: Optional[Mapping[str, Any]] = None

  def __post_init__(self):
    if not isinstance(self.messages, (list, tuple)) or not self.messages:
      raise ModernChatRequestError("messages must be a non-empty sequence")
    frozen_messages = []
    for message in self.messages:
      if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
        raise ModernChatRequestError("each message requires role and content")
      if message["role"] not in SUPPORTED_CHAT_ROLES:
        raise ModernChatRequestError("unsupported Chat message role")
      if not isinstance(message["content"], str):
        raise ModernChatRequestError("Chat message content must be text")
      frozen_messages.append(MappingProxyType({
        "role": message["role"], "content": message["content"]}))
    object.__setattr__(self, "messages", tuple(frozen_messages))
    if self.model is not None and (
        not isinstance(self.model, str) or not self.model.strip()):
      raise ModernChatRequestError("model must be non-empty when provided")
    if self.temperature is not None and (
        not isinstance(self.temperature, Real)
        or isinstance(self.temperature, bool)):
      raise ModernChatRequestError("temperature must be numeric")
    if self.max_tokens is not None and (
        type(self.max_tokens) is not int or self.max_tokens <= 0):
      raise ModernChatRequestError("max_tokens must be positive")
    if self.stop is not None:
      if isinstance(self.stop, str):
        if not self.stop.strip():
          raise ModernChatRequestError("stop text must be non-blank")
      elif isinstance(self.stop, (list, tuple)):
        if not self.stop or not all(
            isinstance(item, str) and item.strip() for item in self.stop):
          raise ModernChatRequestError(
            "stop sequences must contain non-blank text")
        object.__setattr__(self, "stop", tuple(self.stop))
      else:
        raise ModernChatRequestError(
          "stop must be text or a sequence of text")
    if (self.response_format is not None
        and not isinstance(self.response_format, Mapping)):
      raise ModernChatRequestError("response_format must be a mapping")
    if self.response_format is not None:
      object.__setattr__(self, "response_format",
                         _freeze_value(self.response_format))


@dataclass(frozen=True)
class ModernChatResult:
  content: str
  requested_model: str
  response_model: Optional[str]
  finish_reason: Optional[str]
  request_id: Optional[str]
  response_status: Optional[str]
  usage: NormalizedUsage

  @property
  def model(self):
    """Compatibility alias: model means the provider-returned model."""
    return self.response_model


_runtime_config = ContextVar("modern_chat_runtime_config", default=None)


def get_modern_chat_runtime_config():
  return _runtime_config.get()


@contextmanager
def use_modern_chat_runtime(config, modern_client_adapter):
  if not isinstance(config, ModernChatRuntimeConfig):
    raise TypeError("config must be ModernChatRuntimeConfig")
  provider = create_llm_provider(config.provider_config(), modern_client_adapter)
  token = _runtime_config.set(config)
  try:
    with use_chat_provider(provider):
      yield provider
  finally:
    _runtime_config.reset(token)


def _freeze_value(value):
  if isinstance(value, Mapping):
    frozen = {}
    for key, item in value.items():
      if not isinstance(key, str):
        raise ModernChatRequestError("mapping keys must be text")
      frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)
  if isinstance(value, (list, tuple)):
    return tuple(_freeze_value(item) for item in value)
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  raise ModernChatRequestError("request mappings contain an unsupported value")


def _thaw_value(value):
  if isinstance(value, Mapping):
    return {key: _thaw_value(item) for key, item in value.items()}
  if isinstance(value, tuple):
    return [_thaw_value(item) for item in value]
  return value


def _validated_optional_result_text(value, field_name, max_length=512):
  if value is None:
    return None
  if (not isinstance(value, str) or not value.strip()
      or len(value) > max_length):
    raise ModernChatResponseValidationError(
      f"Modern Chat provider returned invalid {field_name}")
  return value


def validate_modern_chat_provider_result(response):
  try:
    choice = response["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice.get("finish_reason")
    response_model = response.get("model")
    metadata = response.get("_normalized_metadata")
  except (AttributeError, KeyError, IndexError, TypeError) as error:
    raise ModernChatResponseValidationError(
      "Modern Chat provider returned an invalid normalized result") from error
  if not isinstance(content, str) or not content.strip():
    raise ModernChatResponseValidationError(
      "Modern Chat provider returned empty or invalid output text")
  if not isinstance(metadata, ProviderResponseMetadata):
    raise ModernChatResponseValidationError(
      "Modern Chat provider returned invalid response metadata")
  _validated_optional_result_text(response_model, "response model", 256)
  _validated_optional_result_text(finish_reason, "finish reason", 128)
  _validated_optional_result_text(metadata.request_id, "request ID")
  _validated_optional_result_text(
    metadata.response_model, "response model", 256)
  _validated_optional_result_text(
    metadata.finish_reason, "finish reason", 128)
  _validated_optional_result_text(
    metadata.response_status, "response status", 128)
  if response_model != metadata.response_model:
    raise ModernChatResponseValidationError(
      "Modern Chat provider returned inconsistent model metadata")
  if finish_reason != metadata.finish_reason:
    raise ModernChatResponseValidationError(
      "Modern Chat provider returned inconsistent finish metadata")
  validate_normalized_usage(NormalizedUsage(
    input_tokens=metadata.input_tokens,
    output_tokens=metadata.output_tokens,
    cached_input_tokens=metadata.cached_input_tokens,
    reasoning_tokens=metadata.reasoning_tokens,
  ))


def _result_from_provider(response, requested_model):
  validate_modern_chat_provider_result(response)
  content = response["choices"][0]["message"]["content"]
  finish_reason = response["choices"][0].get("finish_reason")
  metadata = response["_normalized_metadata"]
  return ModernChatResult(
    content=content,
    requested_model=requested_model,
    response_model=response.get("model"),
    finish_reason=finish_reason,
    request_id=metadata.request_id,
    response_status=metadata.response_status,
    usage=NormalizedUsage(
      input_tokens=metadata.input_tokens,
      output_tokens=metadata.output_tokens,
      cached_input_tokens=metadata.cached_input_tokens,
      reasoning_tokens=metadata.reasoning_tokens,
    ),
  )


def run_modern_chat(request, validator: Optional[Callable[[str], bool]] = None):
  if not isinstance(request, ModernChatRequest):
    raise TypeError("request must be ModernChatRequest")
  config = get_modern_chat_runtime_config()
  if config is None:
    raise ModernChatRuntimeInactiveError(
      "Modern Chat requires an explicit runtime context")
  model = request.model or config.chat_model
  if model != config.chat_model:
    raise ModernChatModelMismatchError(
      "request model does not match the active modern Chat runtime")
  messages = [dict(message) for message in request.messages]
  with logical_call():
    for attempt in range(config.application_retry_count + 1):
      try:
        def validate_result(response):
          validate_modern_chat_provider_result(response)
          if validator is not None:
            content = response["choices"][0]["message"]["content"]
            if not validator(content):
              raise ModernChatValidationError(
                "Modern Chat output validation failed")

        def discard_response_metadata():
          consume_metadata = getattr(
            get_chat_provider(), "consume_response_metadata", None)
          if callable(consume_metadata):
            consume_metadata()

        response = chat_completion(
          model=model,
          messages=messages,
          temperature=request.temperature,
          max_tokens=request.max_tokens,
          stop=request.stop,
          response_format=(_thaw_value(request.response_format)
                           if request.response_format is not None else None),
          result_validator=validate_result,
          on_result_validation_error=discard_response_metadata,
        )
        return _result_from_provider(response, model)
      except _RETRYABLE_ERRORS:
        if attempt >= config.application_retry_count:
          raise
  raise AssertionError("unreachable modern Chat retry state")
