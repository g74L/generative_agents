"""Minimal provider seam for the legacy OpenAI transport.

Retry, validation, parsing, and fail-safe behavior intentionally remain in
``gpt_structure``.  This module forwards physical API attempts, keeps
privacy-safe in-memory metadata, and owns the bounded exact embedding cache.
"""
from collections import OrderedDict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from numbers import Real
from threading import RLock
import time
from types import SimpleNamespace
from typing import Any, Deque, Dict, Iterator, List, Optional, Protocol, Tuple

import openai

from persona.memory_structures.embedding_space import (
  EmbeddingVectorValidationError,
  LEGACY_ADA_002_MANIFEST,
  assert_runtime_embedding_request,
  get_runtime_embedding_manifest,
)
from persona.prompt_template.llm_provider_config import (
  FAKE,
  FAKE_TRANSPORT,
  LEGACY_OPENAI,
  LEGACY_TRANSPORT,
  MODERN_OPENAI,
  LLMProviderConfig,
  get_llm_provider_config,
  use_llm_provider_config,
)


CHAT = "CHAT"
COMPLETION = "COMPLETION"
EMBEDDING = "EMBEDDING"
SUCCESS = "SUCCESS"
ERROR = "ERROR"
HIT = "HIT"
MISS = "MISS"
DISABLED = "DISABLED"
BYPASS = "BYPASS"

EMBEDDING_VERSION = "legacy-embedding-v0"
EMBEDDING_NORMALIZATION_VERSION = (
  LEGACY_ADA_002_MANIFEST.normalization_version)
DEFAULT_EMBEDDING_CACHE_CAPACITY = 1024
DEFAULT_EMBEDDING_CACHE_ENABLED = True

UNSPECIFIED = "UNSPECIFIED"
PERCEPTION = "PERCEPTION"
RETRIEVAL = "RETRIEVAL"
PLANNING = "PLANNING"
REFLECTION = "REFLECTION"
CONVERSATION = "CONVERSATION"
MEMORY_WRITE = "MEMORY_WRITE"
IDENTITY = "IDENTITY"
WORLD_GROUNDING = "WORLD_GROUNDING"
EMBEDDING_CALL_CATEGORIES = (
  UNSPECIFIED,
  PERCEPTION,
  RETRIEVAL,
  PLANNING,
  REFLECTION,
  CONVERSATION,
  MEMORY_WRITE,
  IDENTITY,
  WORLD_GROUNDING,
)


class LLMProvider(Protocol):
  """Transport operations used by the existing legacy wrappers."""

  provider_identity: str
  embedding_space_provider: Optional[str]
  provider_kind: str
  transport_kind: str

  def chat_completion(self, *, model: str, messages: List[Dict[str, str]]) -> Any:
    ...

  def text_completion(self, *, model: str, prompt: str, temperature: float,
                      max_tokens: int, top_p: float, frequency_penalty: float,
                      presence_penalty: float, stream: bool, stop: Any) -> Any:
    ...

  def embedding(self, *, input: List[str], model: str) -> Any:
    ...


class OpenAILegacyProvider:
  """Exact adapter for the OpenAI 0.x APIs used by the baseline."""

  provider_identity = "openai-legacy"
  embedding_space_provider = "openai"
  provider_kind = LEGACY_OPENAI
  transport_kind = LEGACY_TRANSPORT

  def chat_completion(self, *, model, messages):
    return openai.ChatCompletion.create(model=model, messages=messages)

  def text_completion(self, *, model, prompt, temperature, max_tokens, top_p,
                      frequency_penalty, presence_penalty, stream, stop):
    return openai.Completion.create(
      model=model,
      prompt=prompt,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty,
      stream=stream,
      stop=stop,
    )

  def embedding(self, *, input, model):
    return openai.Embedding.create(input=input, model=model)


@dataclass(frozen=True)
class TelemetryEvent:
  operation: str
  logical_call_id: str
  physical_attempt: int
  model_or_engine: str
  outcome: str
  elapsed_seconds: float
  input_fingerprint: str
  error_type: Optional[str] = None
  provider_kind: Optional[str] = None
  transport_kind: Optional[str] = None
  request_id: Optional[str] = None
  response_model: Optional[str] = None
  finish_reason: Optional[str] = None
  response_status: Optional[str] = None
  input_tokens: Optional[int] = None
  output_tokens: Optional[int] = None
  cached_input_tokens: Optional[int] = None
  reasoning_tokens: Optional[int] = None


@dataclass(frozen=True)
class EmbeddingLogicalEvent:
  logical_call_id: str
  model: str
  provider_identity: str
  category: str
  cache_outcome: str
  cache_key_fingerprint: str


@dataclass(frozen=True)
class EmbeddingCacheStats:
  logical_embedding_requests: int
  physical_embedding_attempts: int
  cache_hits: int
  cache_misses: int
  cache_entries: int
  evictions: int
  enabled: bool
  capacity: int


@dataclass
class _LogicalCallState:
  call_id: str
  physical_attempts: int = 0


_logical_call_state: ContextVar[Optional[_LogicalCallState]] = ContextVar(
  "legacy_llm_logical_call", default=None)
_embedding_call_category: ContextVar[str] = ContextVar(
  "embedding_call_category", default=UNSPECIFIED)
_logical_call_ids = itertools.count(1)
_telemetry: List[TelemetryEvent] = []
_embedding_logical_events: List[EmbeddingLogicalEvent] = []
_default_provider: LLMProvider = OpenAILegacyProvider()
_provider: LLMProvider = _default_provider
_embedding_cache = OrderedDict()
_embedding_cache_enabled = DEFAULT_EMBEDDING_CACHE_ENABLED
_embedding_cache_capacity = DEFAULT_EMBEDDING_CACHE_CAPACITY
_embedding_cache_stats = {
  "logical_embedding_requests": 0,
  "physical_embedding_attempts": 0,
  "cache_hits": 0,
  "cache_misses": 0,
  "evictions": 0,
}
_embedding_category_stats = {
  category: {
    "logical_requests": 0,
    "physical_attempts": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "evictions": 0,
  }
  for category in EMBEDDING_CALL_CATEGORIES
}
_embedding_cache_lock = RLock()


def get_provider() -> LLMProvider:
  return _provider


def set_provider(provider: LLMProvider) -> LLMProvider:
  """Install a provider and return the previously active provider."""
  global _provider
  previous = _provider
  _provider = provider
  return previous


def reset_provider() -> None:
  global _provider
  _provider = _default_provider


@contextmanager
def use_provider(provider: LLMProvider) -> Iterator[LLMProvider]:
  previous = set_provider(provider)
  try:
    yield provider
  finally:
    set_provider(previous)


@contextmanager
def logical_call() -> Iterator[str]:
  """Group wrapper-level retries under one logical call identifier."""
  current = _logical_call_state.get()
  if current is not None:
    yield current.call_id
    return

  state = _LogicalCallState(f"llm-{next(_logical_call_ids)}")
  token = _logical_call_state.set(state)
  try:
    yield state.call_id
  finally:
    _logical_call_state.reset(token)


@contextmanager
def embedding_call_context(category: str, preserve_existing: bool = False):
  """Attribute embedding measurement while preserving nested context safely."""
  if category not in EMBEDDING_CALL_CATEGORIES:
    raise ValueError(f"Unknown embedding call category: {category}")
  current = _embedding_call_category.get()
  if preserve_existing and current != UNSPECIFIED:
    yield current
    return

  token = _embedding_call_category.set(category)
  try:
    yield category
  finally:
    _embedding_call_category.reset(token)


def get_embedding_call_category() -> str:
  return _embedding_call_category.get()


def reset_embedding_call_context() -> None:
  _embedding_call_category.set(UNSPECIFIED)


def clear_telemetry() -> None:
  _telemetry.clear()
  _embedding_logical_events.clear()


def get_telemetry() -> Tuple[TelemetryEvent, ...]:
  return tuple(_telemetry)


def get_embedding_logical_events() -> Tuple[EmbeddingLogicalEvent, ...]:
  return tuple(_embedding_logical_events)


def set_embedding_cache_enabled(enabled: bool) -> None:
  global _embedding_cache_enabled
  with _embedding_cache_lock:
    _embedding_cache_enabled = bool(enabled)


def set_embedding_cache_capacity(capacity: int) -> None:
  if capacity < 1:
    raise ValueError("Embedding cache capacity must be at least 1")
  global _embedding_cache_capacity
  with _embedding_cache_lock:
    _embedding_cache_capacity = capacity
    while len(_embedding_cache) > capacity:
      _, (_, owner_category) = _embedding_cache.popitem(last=False)
      _embedding_cache_stats["evictions"] += 1
      _embedding_category_stats[owner_category]["evictions"] += 1


def clear_embedding_cache() -> None:
  with _embedding_cache_lock:
    _embedding_cache.clear()


def reset_embedding_cache() -> None:
  """Restore the enabled, bounded V0 cache and reset all cache statistics."""
  global _embedding_cache_enabled, _embedding_cache_capacity
  with _embedding_cache_lock:
    _embedding_cache.clear()
    _embedding_cache_enabled = DEFAULT_EMBEDDING_CACHE_ENABLED
    _embedding_cache_capacity = DEFAULT_EMBEDDING_CACHE_CAPACITY
    for key in _embedding_cache_stats:
      _embedding_cache_stats[key] = 0
    for category_stats in _embedding_category_stats.values():
      for key in category_stats:
        category_stats[key] = 0
  reset_embedding_call_context()


def reset_embedding_cache_statistics() -> None:
  """Reset global/category counters without removing reusable cache entries."""
  with _embedding_cache_lock:
    for key in _embedding_cache_stats:
      _embedding_cache_stats[key] = 0
    for category_stats in _embedding_category_stats.values():
      for key in category_stats:
        category_stats[key] = 0


def reset_embedding_measurement_all() -> None:
  """Reset cache, embedding telemetry, statistics, configuration, and context."""
  reset_embedding_cache()
  _embedding_logical_events.clear()
  _telemetry[:] = [event for event in _telemetry
                   if event.operation != EMBEDDING]


def get_embedding_measurement_snapshot() -> Dict[str, Any]:
  """Return a content-free snapshot of global and per-category measurements."""
  global_stats = get_embedding_cache_stats()
  with _embedding_cache_lock:
    by_category = {}
    for category in EMBEDDING_CALL_CATEGORIES:
      stats = _embedding_category_stats[category]
      logical_requests = stats["logical_requests"]
      by_category[category] = {
        "logical_requests": logical_requests,
        "physical_attempts": stats["physical_attempts"],
        "cache_hits": stats["cache_hits"],
        "cache_misses": stats["cache_misses"],
        "cache_hit_rate": (
          stats["cache_hits"] / logical_requests
          if logical_requests else 0.0),
        "evictions": stats["evictions"],
      }
    return {
      "global": {
        "logical_embedding_requests": global_stats.logical_embedding_requests,
        "physical_embedding_attempts": global_stats.physical_embedding_attempts,
        "cache_hits": global_stats.cache_hits,
        "cache_misses": global_stats.cache_misses,
        "cache_entries": global_stats.cache_entries,
        "evictions": global_stats.evictions,
        "enabled": global_stats.enabled,
        "capacity": global_stats.capacity,
      },
      "by_category": by_category,
    }


def get_embedding_cache_stats() -> EmbeddingCacheStats:
  with _embedding_cache_lock:
    return EmbeddingCacheStats(
      logical_embedding_requests=(
        _embedding_cache_stats["logical_embedding_requests"]),
      physical_embedding_attempts=(
        _embedding_cache_stats["physical_embedding_attempts"]),
      cache_hits=_embedding_cache_stats["cache_hits"],
      cache_misses=_embedding_cache_stats["cache_misses"],
      cache_entries=len(_embedding_cache),
      evictions=_embedding_cache_stats["evictions"],
      enabled=_embedding_cache_enabled,
      capacity=_embedding_cache_capacity,
    )


def _fingerprint(kwargs: Dict[str, Any]) -> str:
  serialized = json.dumps(kwargs, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=repr)
  return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _invoke(operation: str, method_name: str, kwargs: Dict[str, Any],
            result_validator=None, on_result_validation_error=None) -> Any:
  state = _logical_call_state.get()
  if state is None:
    with logical_call():
      return _invoke(
        operation, method_name, kwargs,
        result_validator=result_validator,
        on_result_validation_error=on_result_validation_error)

  state.physical_attempts += 1
  if operation == EMBEDDING:
    with _embedding_cache_lock:
      _embedding_cache_stats["physical_embedding_attempts"] += 1
      category = _embedding_call_category.get()
      _embedding_category_stats[category]["physical_attempts"] += 1
  attempt = state.physical_attempts
  started = time.perf_counter()
  fingerprint = _fingerprint(kwargs)
  model_or_engine = str(kwargs.get("model", ""))
  provider_kind = getattr(_provider, "provider_kind", None)
  transport_kind = getattr(_provider, "transport_kind", None)
  try:
    result = getattr(_provider, method_name)(**kwargs)
    if result_validator is not None:
      try:
        result_validator(result)
      except Exception:
        if on_result_validation_error is not None:
          on_result_validation_error()
        raise
  except Exception as error:
    _telemetry.append(TelemetryEvent(
      operation=operation,
      logical_call_id=state.call_id,
      physical_attempt=attempt,
      model_or_engine=model_or_engine,
      outcome=ERROR,
      elapsed_seconds=time.perf_counter() - started,
      input_fingerprint=fingerprint,
      error_type=type(error).__name__,
      provider_kind=provider_kind,
      transport_kind=transport_kind,
    ))
    raise

  metadata = None
  consume_metadata = getattr(_provider, "consume_response_metadata", None)
  if callable(consume_metadata):
    metadata = consume_metadata()
  _telemetry.append(TelemetryEvent(
    operation=operation,
    logical_call_id=state.call_id,
    physical_attempt=attempt,
    model_or_engine=model_or_engine,
    outcome=SUCCESS,
    elapsed_seconds=time.perf_counter() - started,
    input_fingerprint=fingerprint,
    provider_kind=provider_kind,
    transport_kind=transport_kind,
    request_id=getattr(metadata, "request_id", None),
    response_model=getattr(metadata, "response_model", None),
    finish_reason=getattr(metadata, "finish_reason", None),
    response_status=getattr(metadata, "response_status", None),
    input_tokens=getattr(metadata, "input_tokens", None),
    output_tokens=getattr(metadata, "output_tokens", None),
    cached_input_tokens=getattr(metadata, "cached_input_tokens", None),
    reasoning_tokens=getattr(metadata, "reasoning_tokens", None),
  ))
  return result


def chat_completion(*, model: str, messages: List[Dict[str, str]]) -> Any:
  return _invoke(CHAT, "chat_completion", {
    "model": model,
    "messages": messages,
  })


def text_completion(*, model: str, prompt: str, temperature: float,
                    max_tokens: int, top_p: float, frequency_penalty: float,
                    presence_penalty: float, stream: bool, stop: Any) -> Any:
  return _invoke(COMPLETION, "text_completion", {
    "model": model,
    "prompt": prompt,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "top_p": top_p,
    "frequency_penalty": frequency_penalty,
    "presence_penalty": presence_penalty,
    "stream": stream,
    "stop": stop,
  })


def _provider_identity(provider: LLMProvider) -> str:
  identity = getattr(provider, "provider_identity", None)
  if identity:
    return str(identity)
  provider_type = type(provider)
  return (f"{provider_type.__module__}.{provider_type.__qualname__}:"
          f"instance-{id(provider)}")


def _normalize_embedding_text(text: str) -> str:
  normalized = text.replace("\n", " ")
  if not normalized:
    normalized = "this is blank"
  return normalized


def _valid_embedding_vector(vector: Any) -> bool:
  return (isinstance(vector, (list, tuple))
          and bool(vector)
          and all(isinstance(value, Real) and not isinstance(value, bool)
                  for value in vector))


def _validate_runtime_embedding_vector(vector, runtime_manifest):
  """Validate non-legacy vectors against their complete local manifest."""
  if not isinstance(vector, list):
    raise EmbeddingVectorValidationError(
      "runtime embedding response expected a vector list")
  if len(vector) != runtime_manifest.dimensions:
    raise EmbeddingVectorValidationError(
      "runtime embedding response has incompatible dimensions")
  squared_norm = 0.0
  for value in vector:
    if (not isinstance(value, Real) or isinstance(value, bool)
        or not math.isfinite(value)):
      raise EmbeddingVectorValidationError(
        "runtime embedding response contains a non-finite numeric value")
    squared_norm += float(value) * float(value)
  if squared_norm == 0.0:
    raise EmbeddingVectorValidationError(
      "runtime embedding response has zero norm")


def _cache_key(provider_identity: str, model: str,
               embedding_space_version: str, normalization_version: str,
               normalized_text: str):
  return (
    provider_identity,
    model,
    EMBEDDING_VERSION,
    embedding_space_version,
    normalization_version,
    normalized_text,
  )


def embedding(*, input: List[str], model: str) -> Any:
  runtime_manifest = get_runtime_embedding_manifest()
  logical_provider = getattr(_provider, "embedding_space_provider", None)
  if logical_provider is not None:
    assert_runtime_embedding_request(
      logical_provider, model, runtime_manifest.normalization_version,
      runtime_manifest=runtime_manifest)
  with logical_call() as logical_call_id:
    category = _embedding_call_category.get()
    with _embedding_cache_lock:
      _embedding_cache_stats["logical_embedding_requests"] += 1
      _embedding_category_stats[category]["logical_requests"] += 1

    if len(input) != 1:
      with _embedding_cache_lock:
        _embedding_cache_stats["cache_misses"] += 1
        _embedding_category_stats[category]["cache_misses"] += 1
      _embedding_logical_events.append(EmbeddingLogicalEvent(
        logical_call_id, model, _provider_identity(_provider), category, BYPASS,
        _fingerprint({"input_count": len(input), "model": model})))
      return _invoke(EMBEDDING, "embedding", {
        "input": input,
        "model": model,
      })

    normalized_text = _normalize_embedding_text(input[0])
    provider_identity = _provider_identity(_provider)
    key = _cache_key(
      provider_identity, model, runtime_manifest.embedding_space_version,
      runtime_manifest.normalization_version, normalized_text)
    key_fingerprint = _fingerprint({
      "provider_identity": provider_identity,
      "model": model,
      "embedding_version": EMBEDDING_VERSION,
      "embedding_space_version": runtime_manifest.embedding_space_version,
      "normalization_version": runtime_manifest.normalization_version,
      "normalized_text": normalized_text,
    })

    with _embedding_cache_lock:
      if not _embedding_cache_enabled:
        cache_outcome = DISABLED
        _embedding_cache_stats["cache_misses"] += 1
        _embedding_category_stats[category]["cache_misses"] += 1
      elif key in _embedding_cache:
        vector, owner_category = _embedding_cache.pop(key)
        vector = list(vector)
        _embedding_cache[key] = (tuple(vector), owner_category)
        _embedding_cache_stats["cache_hits"] += 1
        _embedding_category_stats[category]["cache_hits"] += 1
        _embedding_logical_events.append(EmbeddingLogicalEvent(
          logical_call_id, model, provider_identity, category, HIT,
          key_fingerprint))
        return {"data": [{"embedding": vector}]}
      else:
        cache_outcome = MISS
        _embedding_cache_stats["cache_misses"] += 1
        _embedding_category_stats[category]["cache_misses"] += 1

    _embedding_logical_events.append(EmbeddingLogicalEvent(
      logical_call_id, model, provider_identity, category, cache_outcome,
      key_fingerprint))
    strict_runtime = runtime_manifest != LEGACY_ADA_002_MANIFEST

    def validate_response(response):
      try:
        vector = response["data"][0]["embedding"]
      except (KeyError, IndexError, TypeError) as error:
        raise EmbeddingVectorValidationError(
          "runtime embedding response has no vector") from error
      _validate_runtime_embedding_vector(vector, runtime_manifest)

    def discard_invalid_response_metadata():
      consume_metadata = getattr(_provider, "consume_response_metadata", None)
      if callable(consume_metadata):
        consume_metadata()

    response = _invoke(EMBEDDING, "embedding", {
      "input": [normalized_text],
      "model": model,
    }, result_validator=validate_response if strict_runtime else None,
      on_result_validation_error=(
        discard_invalid_response_metadata if strict_runtime else None))

    if cache_outcome == MISS:
      try:
        vector = response["data"][0]["embedding"]
      except (KeyError, IndexError, TypeError):
        return response
      if _valid_embedding_vector(vector):
        with _embedding_cache_lock:
          _embedding_cache[key] = (tuple(vector), category)
          _embedding_cache.move_to_end(key)
          while len(_embedding_cache) > _embedding_cache_capacity:
            _, (_, owner_category) = _embedding_cache.popitem(last=False)
            _embedding_cache_stats["evictions"] += 1
            _embedding_category_stats[owner_category]["evictions"] += 1
    return response


@dataclass(frozen=True)
class FakeCall:
  operation: str
  arguments: Dict[str, Any]


class FakeProvider:
  """Queue-based in-memory provider for network-free wrapper tests."""

  _identity_counter = itertools.count(1)

  def __init__(self, provider_identity=None, embedding_space_provider=None):
    self.provider_identity = (provider_identity
                              or f"fake-{next(self._identity_counter)}")
    self.embedding_space_provider = embedding_space_provider
    self.provider_kind = FAKE
    self.transport_kind = FAKE_TRANSPORT
    self.calls: List[FakeCall] = []
    self._chat_results: Deque[Any] = deque()
    self._completion_results: Deque[Any] = deque()
    self._embedding_results: Deque[Any] = deque()

  def queue_chat_response(self, content: str) -> None:
    self._chat_results.append({"choices": [{"message": {"content": content}}]})

  def queue_completion_response(self, text: str) -> None:
    self._completion_results.append(
      SimpleNamespace(choices=[SimpleNamespace(text=text)]))

  def queue_embedding_response(self, vector: List[float]) -> None:
    self._embedding_results.append({"data": [{"embedding": vector}]})

  def queue_error(self, operation: str, error: Exception) -> None:
    queues = {
      CHAT: self._chat_results,
      COMPLETION: self._completion_results,
      EMBEDDING: self._embedding_results,
    }
    queues[operation].append(error)

  def _next(self, operation: str, queue: Deque[Any], arguments: Dict[str, Any]) -> Any:
    self.calls.append(FakeCall(operation, dict(arguments)))
    if not queue:
      raise AssertionError(f"No fake {operation} response configured")
    result = queue.popleft()
    if isinstance(result, Exception):
      raise result
    return result

  def chat_completion(self, *, model, messages):
    return self._next(CHAT, self._chat_results, {
      "model": model, "messages": messages})

  def text_completion(self, *, model, prompt, temperature, max_tokens, top_p,
                      frequency_penalty, presence_penalty, stream, stop):
    return self._next(COMPLETION, self._completion_results, {
      "model": model,
      "prompt": prompt,
      "temperature": temperature,
      "max_tokens": max_tokens,
      "top_p": top_p,
      "frequency_penalty": frequency_penalty,
      "presence_penalty": presence_penalty,
      "stream": stream,
      "stop": stop,
    })

  def embedding(self, *, input, model):
    return self._next(EMBEDDING, self._embedding_results, {
      "input": input, "model": model})


def create_llm_provider(config=None, modern_client_adapter=None):
  """Create the explicitly selected provider; never fall back silently."""
  selected = config or get_llm_provider_config()
  if not isinstance(selected, LLMProviderConfig):
    raise TypeError("LLM provider config must be LLMProviderConfig")
  if selected.provider_kind == LEGACY_OPENAI:
    return OpenAILegacyProvider()
  if selected.provider_kind == FAKE:
    return FakeProvider(provider_identity="configured-fake")
  if selected.provider_kind == MODERN_OPENAI:
    from persona.prompt_template.modern_openai_provider import (
      ModernOpenAIProvider,
    )
    return ModernOpenAIProvider(
      config=selected, client_adapter=modern_client_adapter)
  raise ValueError(f"Unsupported provider_kind: {selected.provider_kind}")


@contextmanager
def use_configured_provider(config, modern_client_adapter=None):
  """Scope transport configuration and its provider to the current context."""
  provider = create_llm_provider(config, modern_client_adapter)
  with use_llm_provider_config(config):
    with use_provider(provider):
      yield provider
