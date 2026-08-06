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
COMPLETION_COMPAT = "COMPLETION_COMPAT"
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

  def chat_completion(self, *, model: str, messages: List[Dict[str, str]],
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None, stop: Any = None,
                      response_format: Any = None,
                      top_p: Optional[float] = None,
                      frequency_penalty: Optional[float] = None,
                      presence_penalty: Optional[float] = None) -> Any:
    ...

  def text_completion(self, *, model: str, prompt: str, temperature: float,
                      max_tokens: int, top_p: float, frequency_penalty: float,
                      presence_penalty: float, stream: bool, stop: Any) -> Any:
    ...

  def embedding(self, *, input: List[str], model: str) -> Any:
    ...


class LLMAttemptObserver(Protocol):
  """Optional orchestration hook around validated physical LLM attempts."""

  def before_attempt(self, *, operation: str, model: str,
                     logical_call_id: str, physical_attempt: int,
                     replay_context: "LLMReplayContext") -> None:
    ...

  def after_attempt(self, event: "TelemetryEvent") -> None:
    ...


class OpenAILegacyProvider:
  """Exact adapter for the OpenAI 0.x APIs used by the baseline."""

  provider_identity = "openai-legacy"
  embedding_space_provider = "openai"
  provider_kind = LEGACY_OPENAI
  transport_kind = LEGACY_TRANSPORT

  def chat_completion(self, *, model, messages, temperature=None,
                      max_tokens=None, stop=None, response_format=None,
                      top_p=None, frequency_penalty=None,
                      presence_penalty=None):
    arguments = {"model": model, "messages": messages}
    for name, value in (
        ("temperature", temperature), ("max_tokens", max_tokens),
        ("stop", stop), ("response_format", response_format),
        ("top_p", top_p), ("frequency_penalty", frequency_penalty),
        ("presence_penalty", presence_penalty)):
      if value is not None:
        arguments[name] = value
    return openai.ChatCompletion.create(**arguments)

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
  caller_id: Optional[str] = None
  cognitive_category: Optional[str] = None
  actor_id: Optional[str] = None
  simulation_id: Optional[str] = None
  simulation_step: Optional[int] = None


@dataclass(frozen=True)
class LLMReplayContext:
  """Immutable replay attribution captured with each physical LLM attempt."""
  caller_id: Optional[str] = None
  cognitive_category: Optional[str] = None
  actor_id: Optional[str] = None
  simulation_id: Optional[str] = None
  simulation_step: Optional[int] = None

  def __post_init__(self):
    for field_name in (
        "caller_id", "cognitive_category", "actor_id", "simulation_id"):
      value = getattr(self, field_name)
      if value is not None and (
          not isinstance(value, str) or not value.strip() or len(value) > 512):
        raise ValueError(f"{field_name} must be non-blank text or None")
    if (self.simulation_step is not None
        and (type(self.simulation_step) is not int
             or self.simulation_step < 0)):
      raise ValueError(
        "simulation_step must be a non-negative integer or None")


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
_llm_replay_context: ContextVar[LLMReplayContext] = ContextVar(
  "llm_replay_context", default=LLMReplayContext())
_chat_provider_override: ContextVar[Optional[LLMProvider]] = ContextVar(
  "chat_provider_override", default=None)
_completion_provider_override: ContextVar[Optional[LLMProvider]] = ContextVar(
  "completion_provider_override", default=None)
_completion_compat_provider_override: ContextVar[Optional[LLMProvider]] = (
  ContextVar("completion_compat_provider_override", default=None))
_embedding_provider_override: ContextVar[Optional[LLMProvider]] = ContextVar(
  "embedding_provider_override", default=None)
_llm_attempt_observer: ContextVar[Optional[LLMAttemptObserver]] = ContextVar(
  "llm_attempt_observer", default=None)
_provider_operation_scope = ContextVar(
  "provider_operation_scope", default=None)
_provider_operation_fallback = ContextVar(
  "provider_operation_fallback", default=None)
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


def get_chat_provider() -> LLMProvider:
  """Return the Chat-only override, or the unchanged general provider."""
  provider = _chat_provider_override.get()
  return provider if provider is not None else _general_provider_for(CHAT)


def get_completion_provider() -> LLMProvider:
  provider = _completion_provider_override.get()
  return provider if provider is not None else _general_provider_for(COMPLETION)


def get_completion_compat_provider() -> Optional[LLMProvider]:
  """Return only an explicitly scoped CompletionCompat provider."""
  return _completion_compat_provider_override.get()


def get_embedding_provider() -> LLMProvider:
  provider = _embedding_provider_override.get()
  return provider if provider is not None else _general_provider_for(EMBEDDING)


def _general_provider_for(operation):
  operations = _provider_operation_scope.get()
  if operations is None or operation in operations:
    return _provider
  return _provider_operation_fallback.get() or _default_provider


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
def use_provider(provider: LLMProvider, operations=None) -> Iterator[LLMProvider]:
  previous = set_provider(provider)
  selected_operations = ((CHAT, COMPLETION, EMBEDDING)
                         if operations is None else tuple(operations))
  if any(item not in (CHAT, COMPLETION, EMBEDDING)
         for item in selected_operations):
    set_provider(previous)
    raise ValueError("provider operation scope contains an unknown operation")
  operation_token = _provider_operation_scope.set(
    frozenset(selected_operations))
  fallback_token = _provider_operation_fallback.set(previous)
  try:
    yield provider
  finally:
    _provider_operation_fallback.reset(fallback_token)
    _provider_operation_scope.reset(operation_token)
    set_provider(previous)


@contextmanager
def use_chat_provider(provider: LLMProvider) -> Iterator[LLMProvider]:
  """Override Chat only; Completion and Embedding keep the general provider."""
  token = _chat_provider_override.set(provider)
  try:
    yield provider
  finally:
    _chat_provider_override.reset(token)


@contextmanager
def use_completion_provider(provider: LLMProvider) -> Iterator[LLMProvider]:
  token = _completion_provider_override.set(provider)
  try:
    yield provider
  finally:
    _completion_provider_override.reset(token)


@contextmanager
def use_completion_compat_provider(
    provider: LLMProvider) -> Iterator[LLMProvider]:
  token = _completion_compat_provider_override.set(provider)
  try:
    yield provider
  finally:
    _completion_compat_provider_override.reset(token)


@contextmanager
def use_embedding_provider(provider: LLMProvider) -> Iterator[LLMProvider]:
  token = _embedding_provider_override.set(provider)
  try:
    yield provider
  finally:
    _embedding_provider_override.reset(token)


def get_llm_attempt_observer() -> Optional[LLMAttemptObserver]:
  return _llm_attempt_observer.get()


def install_llm_attempt_observer(observer: LLMAttemptObserver):
  """Install an orchestration observer and return its ContextVar token."""
  if observer is None:
    raise TypeError("observer is required")
  return _llm_attempt_observer.set(observer)


def reset_llm_attempt_observer(token) -> None:
  _llm_attempt_observer.reset(token)


@contextmanager
def use_llm_attempt_observer(
    observer: LLMAttemptObserver) -> Iterator[LLMAttemptObserver]:
  token = install_llm_attempt_observer(observer)
  try:
    yield observer
  finally:
    reset_llm_attempt_observer(token)


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


def get_llm_replay_context() -> LLMReplayContext:
  return _llm_replay_context.get()


@contextmanager
def use_llm_replay_context(
    context: LLMReplayContext) -> Iterator[LLMReplayContext]:
  if not isinstance(context, LLMReplayContext):
    raise TypeError("context must be LLMReplayContext")
  token = _llm_replay_context.set(context)
  try:
    yield context
  finally:
    _llm_replay_context.reset(token)


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

  attempt = state.physical_attempts + 1
  fingerprint = _fingerprint(kwargs)
  model_or_engine = str(kwargs.get("model", ""))
  active_provider = {
    CHAT: get_chat_provider,
    COMPLETION: get_completion_provider,
    COMPLETION_COMPAT: get_completion_compat_provider,
    EMBEDDING: get_embedding_provider,
  }[operation]()
  if active_provider is None:
    raise RuntimeError("CompletionCompat provider is not configured")
  provider_kind = getattr(active_provider, "provider_kind", None)
  transport_kind = getattr(active_provider, "transport_kind", None)
  replay_context = get_llm_replay_context()
  attempt_observer = get_llm_attempt_observer()
  if attempt_observer is not None:
    attempt_observer.before_attempt(
      operation=operation, model=model_or_engine,
      logical_call_id=state.call_id, physical_attempt=attempt,
      replay_context=replay_context)
  state.physical_attempts = attempt
  if operation == EMBEDDING:
    with _embedding_cache_lock:
      _embedding_cache_stats["physical_embedding_attempts"] += 1
      category = _embedding_call_category.get()
      _embedding_category_stats[category]["physical_attempts"] += 1
  started = time.perf_counter()
  try:
    result = getattr(active_provider, method_name)(**kwargs)
    if result_validator is not None:
      try:
        result_validator(result)
      except Exception:
        if on_result_validation_error is not None:
          on_result_validation_error()
        raise
  except Exception as error:
    error_request_id = getattr(error, "request_id", None)
    if (not isinstance(error_request_id, str)
        or not error_request_id.strip() or len(error_request_id) > 512):
      error_request_id = None

    def safe_error_text(name, maximum):
      value = getattr(error, name, None)
      if (isinstance(value, str) and value.strip()
          and len(value) <= maximum):
        return value
      return None

    def safe_error_token(name):
      value = getattr(error, name, None)
      return value if type(value) is int and value >= 0 else None

    event = TelemetryEvent(
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
      request_id=error_request_id,
      response_model=safe_error_text("response_model", 256),
      finish_reason=safe_error_text("finish_reason", 128),
      response_status=safe_error_text("response_status", 128),
      input_tokens=safe_error_token("input_tokens"),
      output_tokens=safe_error_token("output_tokens"),
      cached_input_tokens=safe_error_token("cached_input_tokens"),
      reasoning_tokens=safe_error_token("reasoning_tokens"),
      caller_id=replay_context.caller_id,
      cognitive_category=replay_context.cognitive_category,
      actor_id=replay_context.actor_id,
      simulation_id=replay_context.simulation_id,
      simulation_step=replay_context.simulation_step,
    )
    _telemetry.append(event)
    if attempt_observer is not None:
      attempt_observer.after_attempt(event)
    raise

  metadata = None
  consume_metadata = getattr(active_provider, "consume_response_metadata", None)
  if callable(consume_metadata):
    metadata = consume_metadata()
  event = TelemetryEvent(
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
    caller_id=replay_context.caller_id,
    cognitive_category=replay_context.cognitive_category,
    actor_id=replay_context.actor_id,
    simulation_id=replay_context.simulation_id,
    simulation_step=replay_context.simulation_step,
  )
  _telemetry.append(event)
  if attempt_observer is not None:
    attempt_observer.after_attempt(event)
  return result


def chat_completion(*, model: str, messages: List[Dict[str, str]],
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None, stop: Any = None,
                    response_format: Any = None, result_validator=None,
                    on_result_validation_error=None,
                    top_p: Optional[float] = None,
                    frequency_penalty: Optional[float] = None,
                    presence_penalty: Optional[float] = None) -> Any:
  arguments = {"model": model, "messages": messages}
  for name, value in (
      ("temperature", temperature), ("max_tokens", max_tokens),
      ("stop", stop), ("response_format", response_format),
      ("top_p", top_p), ("frequency_penalty", frequency_penalty),
      ("presence_penalty", presence_penalty)):
    if value is not None:
      arguments[name] = value
  return _invoke(
    CHAT, "chat_completion", arguments,
    result_validator=result_validator,
    on_result_validation_error=on_result_validation_error)


def completion_compat(*, model: str, prompt: str, temperature: float,
                      max_tokens: int, top_p: float,
                      frequency_penalty: float, presence_penalty: float,
                      stop: Any, result_validator=None,
                      on_result_validation_error=None) -> Any:
  """Execute a legacy prompt through an explicitly scoped modern Chat seam."""
  arguments = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": temperature,
    "max_tokens": max_tokens,
    "top_p": top_p,
    "frequency_penalty": frequency_penalty,
    "presence_penalty": presence_penalty,
  }
  if stop is not None:
    arguments["stop"] = stop
  return _invoke(
    COMPLETION_COMPAT, "chat_completion", arguments,
    result_validator=result_validator,
    on_result_validation_error=on_result_validation_error)


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
  active_provider = get_embedding_provider()
  logical_provider = getattr(active_provider, "embedding_space_provider", None)
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
        logical_call_id, model, _provider_identity(active_provider), category, BYPASS,
        _fingerprint({"input_count": len(input), "model": model})))
      return _invoke(EMBEDDING, "embedding", {
        "input": input,
        "model": model,
      })

    normalized_text = _normalize_embedding_text(input[0])
    provider_identity = _provider_identity(active_provider)
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
      consume_metadata = getattr(active_provider, "consume_response_metadata", None)
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

  def chat_completion(self, *, model, messages, temperature=None,
                      max_tokens=None, stop=None, response_format=None,
                      top_p=None, frequency_penalty=None,
                      presence_penalty=None):
    arguments = {"model": model, "messages": messages}
    for name, value in (
        ("temperature", temperature), ("max_tokens", max_tokens),
        ("stop", stop), ("response_format", response_format),
        ("top_p", top_p), ("frequency_penalty", frequency_penalty),
        ("presence_penalty", presence_penalty)):
      if value is not None:
        arguments[name] = value
    return self._next(CHAT, self._chat_results, arguments)

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
    if config.provider_kind == MODERN_OPENAI:
      with use_provider(provider, operations=()):
        with use_embedding_provider(provider):
          yield provider
    else:
      with use_provider(provider):
        yield provider
