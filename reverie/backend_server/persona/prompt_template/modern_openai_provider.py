"""Optional modern OpenAI SDK adapter used only when explicitly selected.

This module never imports ``OpenAI`` at import time.  M2 keeps application
retries in the existing wrappers and configures modern SDK retries to zero.
"""
from contextvars import ContextVar
from dataclasses import dataclass
import importlib
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, Optional, Tuple

from persona.prompt_template.llm_provider_config import (
  LLMProviderConfig,
  MODERN_OPENAI,
  modern_openai_config,
)


class LLMProviderError(RuntimeError):
  pass


class ModernOpenAISdkUnavailableError(LLMProviderError):
  pass


class LLMAuthenticationError(LLMProviderError):
  pass


class LLMAuthorizationError(LLMProviderError):
  pass


class LLMModelNotFoundError(LLMProviderError):
  pass


class LLMInvalidRequestError(LLMProviderError):
  pass


class LLMUnsupportedParameterError(LLMInvalidRequestError):
  pass


class LLMTimeoutError(LLMProviderError):
  pass


class LLMConnectionError(LLMProviderError):
  pass


class LLMRateLimitError(LLMProviderError):
  pass


class LLMServerError(LLMProviderError):
  pass


class LLMIncompleteResponseError(LLMProviderError):
  pass


class LLMRefusalError(LLMProviderError):
  pass


class LLMEmptyOutputError(LLMProviderError):
  pass


class LLMMalformedResponseError(LLMProviderError):
  pass


class LLMUnsupportedOperationError(LLMProviderError):
  pass


class ModernCompletionNotEnabledError(LLMUnsupportedOperationError):
  pass


@dataclass(frozen=True)
class NormalizedUsage:
  input_tokens: Optional[int] = None
  output_tokens: Optional[int] = None
  cached_input_tokens: Optional[int] = None
  reasoning_tokens: Optional[int] = None


@dataclass(frozen=True)
class NormalizedTextResponse:
  text: str
  model: Optional[str]
  request_id: Optional[str]
  finish_reason: Optional[str]
  status: Optional[str]
  usage: NormalizedUsage


@dataclass(frozen=True)
class NormalizedEmbeddingResponse:
  vector: Tuple[float, ...]
  model: Optional[str]
  request_id: Optional[str]
  usage: NormalizedUsage


@dataclass(frozen=True)
class ProviderResponseMetadata:
  request_id: Optional[str] = None
  response_model: Optional[str] = None
  finish_reason: Optional[str] = None
  response_status: Optional[str] = None
  input_tokens: Optional[int] = None
  output_tokens: Optional[int] = None
  cached_input_tokens: Optional[int] = None
  reasoning_tokens: Optional[int] = None


def _field(value, name, default=None):
  if isinstance(value, dict):
    return value.get(name, default)
  return getattr(value, name, default)


def _error_code(error):
  code = getattr(error, "code", None)
  if code:
    return str(code).lower()
  body = getattr(error, "body", None)
  body_error = _field(body, "error", {})
  return str(_field(body_error, "code", "")).lower()


def map_modern_sdk_error(error):
  """Map SDK errors without copying exception text or response content."""
  name = type(error).__name__
  status = getattr(error, "status_code", None)
  code = _error_code(error)
  if name == "AuthenticationError" or status == 401:
    return LLMAuthenticationError("Modern OpenAI authentication failed")
  if name == "PermissionDeniedError" or status == 403:
    return LLMAuthorizationError("Modern OpenAI authorization failed")
  if name == "NotFoundError" or status == 404:
    return LLMModelNotFoundError("Modern OpenAI model or resource not found")
  if code in ("unsupported_parameter", "unsupported_value"):
    return LLMUnsupportedParameterError(
      "Modern OpenAI request contains an unsupported parameter")
  if name in ("BadRequestError", "UnprocessableEntityError") or status in (400, 422):
    return LLMInvalidRequestError("Modern OpenAI request is invalid")
  if name in ("APITimeoutError", "TimeoutError"):
    return LLMTimeoutError("Modern OpenAI request timed out")
  if name in ("APIConnectionError", "ConnectionError"):
    return LLMConnectionError("Modern OpenAI connection failed")
  if name == "RateLimitError" or status == 429:
    return LLMRateLimitError("Modern OpenAI rate limit reached")
  if name == "InternalServerError" or (isinstance(status, int) and status >= 500):
    return LLMServerError("Modern OpenAI server error")
  return LLMProviderError("Modern OpenAI request failed")


def _usage_from_response(response):
  usage = _field(response, "usage")
  if usage is None:
    return NormalizedUsage()

  malformed_message = "Malformed usage metadata in modern OpenAI response"
  missing = object()
  usage_fields = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
  )

  def member(container, name):
    if isinstance(container, Mapping):
      return container.get(name, missing)
    return getattr(container, name, missing)

  def validate_container(container, supported_fields):
    if isinstance(container, Mapping):
      return
    if isinstance(container, (str, bytes, list, tuple, set, bool)):
      raise LLMMalformedResponseError(malformed_message)
    if not any(hasattr(container, field) for field in supported_fields):
      raise LLMMalformedResponseError(malformed_message)

  def token_count(value):
    if value is missing or value is None:
      return None
    if type(value) is not int or value < 0:
      raise LLMMalformedResponseError(malformed_message)
    return value

  def token_detail(details, field):
    if details is missing or details is None:
      return None
    validate_container(details, (field,))
    return token_count(member(details, field))

  validate_container(usage, usage_fields)
  prompt_tokens = token_count(member(usage, "prompt_tokens"))
  completion_tokens = token_count(member(usage, "completion_tokens"))
  input_tokens = token_count(member(usage, "input_tokens"))
  output_tokens = token_count(member(usage, "output_tokens"))
  token_count(member(usage, "total_tokens"))
  prompt_details = member(usage, "prompt_tokens_details")
  completion_details = member(usage, "completion_tokens_details")
  return NormalizedUsage(
    input_tokens=(prompt_tokens if prompt_tokens is not None else input_tokens),
    output_tokens=(completion_tokens
                   if completion_tokens is not None else output_tokens),
    cached_input_tokens=token_detail(prompt_details, "cached_tokens"),
    reasoning_tokens=token_detail(completion_details, "reasoning_tokens"),
  )


def normalize_chat_response(response):
  choices = _field(response, "choices")
  if not isinstance(choices, (list, tuple)) or not choices:
    raise LLMMalformedResponseError("Modern chat response has no choices")
  choice = choices[0]
  message = _field(choice, "message")
  if message is None:
    raise LLMMalformedResponseError("Modern chat response has no message")
  refusal = _field(message, "refusal")
  if refusal:
    raise LLMRefusalError("Modern chat response was refused")
  status = _field(response, "status")
  finish_reason = _field(choice, "finish_reason")
  if status == "incomplete" or finish_reason in ("length", "content_filter"):
    raise LLMIncompleteResponseError("Modern chat response is incomplete")
  text = _field(message, "content")
  if text is None:
    raise LLMEmptyOutputError("Modern chat response contains no output text")
  if not isinstance(text, str):
    raise LLMMalformedResponseError("Modern chat output text is malformed")
  if not text.strip():
    raise LLMEmptyOutputError("Modern chat response contains empty output")
  return NormalizedTextResponse(
    text=text,
    model=_field(response, "model"),
    request_id=(_field(response, "_request_id")
                or _field(response, "request_id")),
    finish_reason=finish_reason,
    status=status or "completed",
    usage=_usage_from_response(response),
  )


def normalize_embedding_response(response):
  data = _field(response, "data")
  if not isinstance(data, (list, tuple)) or not data:
    raise LLMMalformedResponseError("Modern embedding response has no data")
  vector = _field(data[0], "embedding")
  if (not isinstance(vector, (list, tuple)) or not vector
      or not all(isinstance(item, Real) and not isinstance(item, bool)
                 and math.isfinite(item) for item in vector)):
    raise LLMMalformedResponseError("Modern embedding vector is malformed")
  return NormalizedEmbeddingResponse(
    vector=tuple(float(item) for item in vector),
    model=_field(response, "model"),
    request_id=(_field(response, "_request_id")
                or _field(response, "request_id")),
    usage=_usage_from_response(response),
  )


def _create_modern_sdk_client(config):
  try:
    openai_module = importlib.import_module("openai")
    client_class = getattr(openai_module, "OpenAI")
  except (ImportError, AttributeError) as error:
    raise ModernOpenAISdkUnavailableError(
      "Modern OpenAI SDK is not installed in this environment") from error
  try:
    return client_class(
      max_retries=config.sdk_retry_count,
      timeout=config.request_timeout_seconds,
    )
  except Exception as error:
    raise map_modern_sdk_error(error) from error


class ModernOpenAIClientAdapter:
  """Normalize concrete modern SDK objects behind a small testable surface."""

  def __init__(self, config=None, client=None, client_factory=None):
    self.config = config or modern_openai_config()
    if self.config.provider_kind != MODERN_OPENAI:
      raise ValueError("Modern adapter requires MODERN_OPENAI configuration")
    factory = client_factory or _create_modern_sdk_client
    self.client = client if client is not None else factory(self.config)

  def create_chat(self, *, model, messages):
    try:
      response = self.client.chat.completions.create(
        model=model,
        messages=messages,
        store=self.config.store_responses,
      )
    except LLMProviderError:
      raise
    except Exception as error:
      raise map_modern_sdk_error(error) from error
    return normalize_chat_response(response)

  def create_embedding(self, *, model, input):
    try:
      response = self.client.embeddings.create(model=model, input=input)
    except LLMProviderError:
      raise
    except Exception as error:
      raise map_modern_sdk_error(error) from error
    return normalize_embedding_response(response)


class ModernOpenAIProvider:
  """M2 Chat/Embedding seam; Completion intentionally remains unavailable."""

  provider_identity = "openai-modern"
  embedding_space_provider = "openai"

  def __init__(self, config=None, client_adapter=None):
    self.config = config or modern_openai_config()
    self.provider_kind = self.config.provider_kind
    self.transport_kind = self.config.transport_kind
    self.client_adapter = (client_adapter
                           or ModernOpenAIClientAdapter(self.config))
    self._metadata = ContextVar(
      f"modern_provider_metadata_{id(self)}", default=None)

  def _remember(self, response):
    usage = response.usage
    self._metadata.set(ProviderResponseMetadata(
      request_id=response.request_id,
      response_model=response.model,
      finish_reason=getattr(response, "finish_reason", None),
      response_status=getattr(response, "status", None),
      input_tokens=usage.input_tokens,
      output_tokens=usage.output_tokens,
      cached_input_tokens=usage.cached_input_tokens,
      reasoning_tokens=usage.reasoning_tokens,
    ))

  def consume_response_metadata(self):
    metadata = self._metadata.get()
    self._metadata.set(None)
    return metadata

  def chat_completion(self, *, model, messages):
    response = self.client_adapter.create_chat(model=model, messages=messages)
    self._remember(response)
    return {
      "choices": [{
        "message": {"content": response.text},
        "finish_reason": response.finish_reason,
      }],
      "model": response.model,
    }

  def embedding(self, *, input, model):
    response = self.client_adapter.create_embedding(model=model, input=input)
    self._remember(response)
    return {
      "data": [{"embedding": list(response.vector)}],
      "model": response.model,
    }

  def text_completion(self, **kwargs):
    raise ModernCompletionNotEnabledError(
      "Modern text Completion is not enabled in M2")
