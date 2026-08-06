"""Minimal, offline-only orchestration for one controlled Smallville step.

R0H deliberately supports one actor, one step, and the historical wake-up-hour
planning wrapper.  It composes existing modern runtimes and persistence seams;
it does not alter cognition or provide a general replay framework.
"""
from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass, field
import datetime
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from persona.memory_structures.embedding_space import EmbeddingSpaceManifest
from persona.memory_structures.scratch import Scratch
from persona.prompt_template import run_gpt_prompt
from persona.prompt_template.chat_runtime import (
  M5_CHAT_MODEL,
  ModernChatRuntimeConfig,
  build_modern_chat_runtime_config,
  use_modern_chat_runtime,
)
from persona.prompt_template.completion_runtime import (
  COMPLETION_COMPAT_MODEL,
  FORBIDDEN_MODERN_RUNTIME_MODELS,
  M6_DEFERRED_CALLERS,
  ModernCompletionRuntimeConfig,
  build_modern_completion_runtime_config,
  use_modern_completion_runtime,
)
from persona.prompt_template.cost_ledger import (
  CostLedgerContext,
  ModelPricing,
  PricingSnapshot,
  use_cost_ledger_context,
)
from persona.prompt_template.embedding_runtime import (
  TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
  TEXT_EMBEDDING_3_SMALL_MODEL,
  EmbeddingRuntimeConfig,
  build_modern_embedding_runtime_config,
  use_embedding_runtime,
)
from persona.prompt_template.embedding_store_bootstrap import (
  ModernEmbeddingStoreBootstrapRequest,
  bootstrap_modern_embedding_store,
)
from persona.prompt_template.llm_provider import (
  CHAT,
  COMPLETION_COMPAT,
  EMBEDDING,
  LLMReplayContext,
  TelemetryEvent,
  get_embedding_cache_stats,
  get_llm_replay_context,
  get_telemetry,
  install_llm_attempt_observer,
  reset_llm_attempt_observer,
  use_llm_replay_context,
)
from persona.prompt_template.llm_provider_config import (
  MODERN_OPENAI,
  MODERN_SDK_MODE,
  MODERN_TRANSPORT,
)
from persona.prompt_template.modern_openai_provider import (
  LLMProviderError,
  ModernOpenAIClientAdapter,
  NormalizedEmbeddingResponse,
  NormalizedTextResponse,
  NormalizedUsage,
)
from persona.prompt_template.replay_cost_guard import (
  ReplayCostAccountingUnavailableError,
  ReplayCostCeiling,
  ReplayCostGuardConfig,
  ReplayCostGuardError,
  ReplayCostGuardState,
  get_replay_cost_guard,
  use_replay_cost_guard,
)


SUCCESS = "SUCCESS"
PLANNING = "PLANNING"
WAKE_UP_CALLER = "wake_up_hour"
WAKE_UP_PATH = "planning.wake_up_hour"
FORBIDDEN_COGNITIVE_CALLERS = frozenset(M6_DEFERRED_CALLERS)
R0H_PRICING_SNAPSHOT = PricingSnapshot(
  snapshot_id="r0h-offline-pricing-v0",
  schema_version=1,
  currency="USD",
  created_at="2026-08-06",
  models=(
    ModelPricing(
      model=M5_CHAT_MODEL,
      input_per_million=Decimal("0.15"),
      cached_input_per_million=Decimal("0.075"),
      output_per_million=Decimal("0.60"),
      effective_from="2026-08-06",
      source_label="R0H deterministic offline fixture"),
    ModelPricing(
      model=TEXT_EMBEDDING_3_SMALL_MODEL,
      embedding_input_per_million=Decimal("0.02"),
      effective_from="2026-08-06",
      source_label="R0H deterministic offline fixture"),
  ),
  source_note="Pinned deterministic prices for the offline R0H harness",
)


class ControlledReplayError(RuntimeError):
  """Base error for safe harness validation and execution failures."""


class ControlledReplayConfigurationError(ControlledReplayError, ValueError):
  pass


class ControlledReplayLegacyConfigurationError(
    ControlledReplayConfigurationError):
  pass


class ControlledReplayPathForbiddenError(ControlledReplayError):
  pass


class ReplayLogicalCallLimitExceededError(ReplayCostGuardError):
  def __init__(self, maximum):
    self.maximum = maximum
    super().__init__("Controlled replay logical-call limit reached")


class ReplayPhysicalAttemptLimitExceededError(ReplayCostGuardError):
  def __init__(self, maximum):
    self.maximum = maximum
    super().__init__("Controlled replay physical-attempt limit reached")


@dataclass(frozen=True)
class ControlledReplayProfile:
  replay_id: str
  simulation_id: str
  actor_id: str
  step: int
  maximum_cost: Decimal
  maximum_logical_calls: int
  maximum_physical_attempts: int
  conversation_enabled: bool = False
  reflection_enabled: bool = False

  def __post_init__(self):
    for name in ("replay_id", "simulation_id", "actor_id"):
      value = getattr(self, name)
      if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ControlledReplayConfigurationError(
          f"{name} must be non-blank text")
    if type(self.step) is not int or self.step < 0:
      raise ControlledReplayConfigurationError(
        "step must be a non-negative integer")
    if (isinstance(self.maximum_cost, bool)
        or not isinstance(self.maximum_cost, Decimal)):
      raise ControlledReplayConfigurationError(
        "maximum_cost must be a Decimal")
    if not self.maximum_cost.is_finite() or self.maximum_cost <= 0:
      raise ControlledReplayConfigurationError(
        "maximum_cost must be finite and positive")
    for name in ("maximum_logical_calls", "maximum_physical_attempts"):
      value = getattr(self, name)
      if type(value) is not int or value <= 0:
        raise ControlledReplayConfigurationError(
          f"{name} must be a positive integer")
    if type(self.conversation_enabled) is not bool:
      raise ControlledReplayConfigurationError(
        "conversation_enabled must be boolean")
    if type(self.reflection_enabled) is not bool:
      raise ControlledReplayConfigurationError(
        "reflection_enabled must be boolean")
    if self.conversation_enabled:
      raise ControlledReplayConfigurationError(
        "conversation is unsupported by controlled replay V0")
    if self.reflection_enabled:
      raise ControlledReplayConfigurationError(
        "reflection is unsupported by controlled replay V0")


@dataclass(frozen=True)
class ControlledReplayActorFixture:
  actor_id: str
  persona: Any
  prompt_input: Tuple[str, ...]

  def __post_init__(self):
    if not isinstance(self.actor_id, str) or not self.actor_id.strip():
      raise ControlledReplayConfigurationError("fixture actor_id is required")
    if self.persona is None:
      raise ControlledReplayConfigurationError("fixture persona is required")
    if (not isinstance(self.prompt_input, tuple) or not self.prompt_input
        or not all(isinstance(item, str) for item in self.prompt_input)):
      raise ControlledReplayConfigurationError(
        "prompt_input must be a non-empty immutable text tuple")


@dataclass(frozen=True)
class ControlledReplayPersonaFixture:
  scratch: Scratch


def build_controlled_replay_actor_fixture(
    actor_id, scratch_path, simulation_meta_path):
  """Build the historical prompt fixture from canonical simulation time."""
  simulation_meta_path = Path(simulation_meta_path)
  try:
    simulation_meta = json.loads(
      simulation_meta_path.read_text(encoding="utf-8"))
  except OSError as error:
    raise ControlledReplayConfigurationError(
      "simulation metadata is unavailable") from error
  except (json.JSONDecodeError, TypeError) as error:
    raise ControlledReplayConfigurationError(
      "simulation metadata is malformed") from error

  if not isinstance(simulation_meta, dict):
    raise ControlledReplayConfigurationError(
      "simulation metadata is malformed")
  serialized_curr_time = simulation_meta.get("curr_time")
  if not isinstance(serialized_curr_time, str) or not serialized_curr_time.strip():
    raise ControlledReplayConfigurationError(
      "simulation metadata curr_time is unavailable")
  try:
    curr_time = datetime.datetime.strptime(
      serialized_curr_time, "%B %d, %Y, %H:%M:%S")
  except ValueError as error:
    raise ControlledReplayConfigurationError(
      "simulation metadata curr_time is malformed") from error

  scratch = Scratch(str(Path(scratch_path)))
  scratch.curr_time = curr_time
  persona = ControlledReplayPersonaFixture(scratch=scratch)
  return ControlledReplayActorFixture(
    actor_id=actor_id,
    persona=persona,
    prompt_input=(
      scratch.get_str_iss(),
      scratch.get_str_lifestyle(),
      scratch.get_str_firstname(),
    ),
  )


@dataclass(frozen=True)
class ControlledReplayEnvironmentFixture:
  simulation_id: str
  step: int

  def __post_init__(self):
    if not isinstance(self.simulation_id, str) or not self.simulation_id.strip():
      raise ControlledReplayConfigurationError(
        "fixture simulation_id is required")
    if type(self.step) is not int or self.step < 0:
      raise ControlledReplayConfigurationError(
        "fixture step must be a non-negative integer")


class DeterministicReplayFakeAdapter:
  """In-memory modern-client adapter with deterministic normalized results."""

  offline_fake = True

  def __init__(self, text_responses=("7 am",), *, input_tokens=20,
               output_tokens=3, cached_input_tokens=0, reasoning_tokens=0,
               embedding_input_tokens=1):
    self._text_responses = list(text_responses)
    self.input_tokens = input_tokens
    self.output_tokens = output_tokens
    self.cached_input_tokens = cached_input_tokens
    self.reasoning_tokens = reasoning_tokens
    self.embedding_input_tokens = embedding_input_tokens
    self.calls = []

  def create_chat(self, **kwargs):
    self.calls.append(("create_chat", dict(kwargs)))
    if not self._text_responses:
      raise AssertionError("No deterministic text response configured")
    response = self._text_responses.pop(0)
    if isinstance(response, BaseException):
      raise response
    return NormalizedTextResponse(
      text=response,
      model=kwargs["model"],
      request_id=f"r0h-text-{len(self.calls)}",
      finish_reason="stop",
      status="completed",
      usage=NormalizedUsage(
        self.input_tokens, self.output_tokens,
        self.cached_input_tokens, self.reasoning_tokens),
    )

  def create_embedding(self, **kwargs):
    self.calls.append(("create_embedding", dict(kwargs)))
    return NormalizedEmbeddingResponse(
      vector=(1.0,) + (0.0,) * 1535,
      model=kwargs["model"],
      request_id=f"r0h-embedding-{len(self.calls)}",
      usage=NormalizedUsage(self.embedding_input_tokens, None),
    )


@dataclass(frozen=True)
class ControlledReplayProviders:
  chat_adapter: Any
  completion_adapter: Any
  embedding_adapter: Any
  chat_config: ModernChatRuntimeConfig = field(
    default_factory=build_modern_chat_runtime_config)
  completion_config: ModernCompletionRuntimeConfig = field(
    default_factory=build_modern_completion_runtime_config)
  embedding_config: EmbeddingRuntimeConfig = field(
    default_factory=build_modern_embedding_runtime_config)
  pricing_snapshot: PricingSnapshot = R0H_PRICING_SNAPSHOT
  live_api_enabled: bool = False

  def __post_init__(self):
    if type(self.live_api_enabled) is not bool:
      raise ControlledReplayConfigurationError(
        "live_api_enabled must be boolean")
    adapters = (
      ("chat_adapter", self.chat_adapter, "create_chat"),
      ("completion_adapter", self.completion_adapter, "create_chat"),
      ("embedding_adapter", self.embedding_adapter, "create_embedding"),
    )
    for name, adapter, method in adapters:
      offline_fake = getattr(adapter, "offline_fake", False) is True
      authorized_live = (
        self.live_api_enabled
        and isinstance(adapter, ModernOpenAIClientAdapter))
      if ((not offline_fake and not authorized_live)
          or not callable(getattr(adapter, method, None))):
        raise ControlledReplayConfigurationError(
          f"{name} must be an offline fake or explicitly enabled modern adapter")
    _validate_modern_runtime_configuration(self)


@dataclass(frozen=True)
class ControlledReplayTelemetryEntry:
  replay_id: str
  simulation_id: str
  actor_id: str
  step: int
  cognitive_category: str
  caller_id: str
  operation: str
  logical_call_id: str
  physical_attempt: int
  requested_model: str
  returned_model: Optional[str]
  outcome: str
  error_type: Optional[str]


@dataclass(frozen=True)
class ControlledReplayLedgerEntry:
  replay_id: str
  simulation_id: str
  actor_id: str
  step: int
  cognitive_category: str
  caller_id: str
  operation: str
  logical_call_id: str
  physical_attempt: int
  estimated_cost: Optional[Decimal]


@dataclass(frozen=True)
class ControlledReplayReport:
  replay_id: str
  simulation_id: str
  actor_id: str
  step: int
  status: str
  selected_cognitive_path: str
  logical_calls: int
  physical_attempts: int
  operation_counts: Tuple[Tuple[str, int], ...]
  models_requested: Tuple[str, ...]
  models_returned: Tuple[str, ...]
  retry_count: int
  input_tokens: int
  output_tokens: int
  cached_tokens: int
  reasoning_tokens: int
  embedding_cache_hits: int
  embedding_cache_misses: int
  accumulated_cost: Decimal
  cost_ceiling: Decimal
  remaining_cost: Decimal
  legacy_detections: int
  error_type: Optional[str]
  primary_error_type: Optional[str]
  underlying_provider_error_type: Optional[str]
  store_manifest_identity: str
  parser_result_type: Optional[str]
  wake_up_hour: Optional[int]
  cognitive_output_digest: Optional[str]
  telemetry: Tuple[ControlledReplayTelemetryEntry, ...]
  ledger: Tuple[ControlledReplayLedgerEntry, ...]


class ReplayCallCountGuardState:
  def __init__(self, profile):
    self.maximum_logical_calls = profile.maximum_logical_calls
    self.maximum_physical_attempts = profile.maximum_physical_attempts
    self._logical_call_ids = []
    self._physical_attempts = 0

  @property
  def logical_calls(self):
    return len(self._logical_call_ids)

  @property
  def physical_attempts(self):
    return self._physical_attempts

  def check_before_attempt(self, logical_call_id):
    if (logical_call_id not in self._logical_call_ids
        and self.logical_calls >= self.maximum_logical_calls):
      raise ReplayLogicalCallLimitExceededError(
        self.maximum_logical_calls)
    if self.physical_attempts >= self.maximum_physical_attempts:
      raise ReplayPhysicalAttemptLimitExceededError(
        self.maximum_physical_attempts)

  def record_attempt(self, logical_call_id):
    if logical_call_id not in self._logical_call_ids:
      self._logical_call_ids.append(logical_call_id)
    self._physical_attempts += 1


_active_call_count_guard = ContextVar(
  "active_replay_call_count_guard", default=None)


def get_replay_call_count_guard():
  return _active_call_count_guard.get()


@contextmanager
def use_replay_call_count_guard(profile):
  state = ReplayCallCountGuardState(profile)
  token = _active_call_count_guard.set(state)
  try:
    yield state
  finally:
    _active_call_count_guard.reset(token)


class _CombinedReplayObserver:
  def __init__(self, call_guard, cost_guard):
    self.call_guard = call_guard
    self.cost_guard = cost_guard

  def before_attempt(self, **kwargs):
    logical_call_id = kwargs["logical_call_id"]
    self.call_guard.check_before_attempt(logical_call_id)
    self.cost_guard.before_attempt(**kwargs)
    self.call_guard.record_attempt(logical_call_id)

  def after_attempt(self, event):
    self.cost_guard.after_attempt(event)


@contextmanager
def _use_combined_replay_observer(call_guard, cost_guard):
  token = install_llm_attempt_observer(
    _CombinedReplayObserver(call_guard, cost_guard))
  try:
    yield
  finally:
    reset_llm_attempt_observer(token)


@contextmanager
def _historical_wrapper_environment():
  """Resolve historical relative templates and suppress debug prompt output."""
  previous_directory = Path.cwd()
  previous_debug = getattr(run_gpt_prompt, "debug", False)
  backend_server = Path(__file__).resolve().parent
  run_gpt_prompt.debug = False
  try:
    os.chdir(backend_server)
    with redirect_stdout(io.StringIO()):
      yield
  finally:
    os.chdir(previous_directory)
    run_gpt_prompt.debug = previous_debug


def assert_controlled_replay_path_allowed(caller_id):
  if caller_id in FORBIDDEN_COGNITIVE_CALLERS:
    raise ControlledReplayPathForbiddenError(
      "Conversation and reflection callers are forbidden in R0H")
  if caller_id != WAKE_UP_CALLER:
    raise ControlledReplayPathForbiddenError(
      "Controlled replay V0 supports only its selected planning caller")
  return caller_id


def _validate_modern_runtime_configuration(providers):
  configs = (
    providers.chat_config, providers.completion_config,
    getattr(providers.embedding_config, "provider_config", None))
  for config in configs:
    if config is None or not all(hasattr(config, name) for name in (
        "provider_kind", "transport_kind", "sdk_mode")):
      raise ControlledReplayConfigurationError(
        "Every modern operation requires its explicit runtime configuration")
    if (config.provider_kind, config.transport_kind, config.sdk_mode) != (
        MODERN_OPENAI, MODERN_TRANSPORT, MODERN_SDK_MODE):
      raise ControlledReplayLegacyConfigurationError(
        "Controlled replay requires the modern provider and transport")
  if (not isinstance(providers.chat_config, ModernChatRuntimeConfig)
      or not isinstance(
        providers.completion_config, ModernCompletionRuntimeConfig)
      or not isinstance(providers.embedding_config, EmbeddingRuntimeConfig)):
    raise ControlledReplayConfigurationError(
      "Every modern operation requires its canonical runtime config type")
  requested_models = (
    providers.chat_config.chat_model,
    providers.completion_config.model,
    providers.embedding_config.embedding_model,
  )
  if any(model in FORBIDDEN_MODERN_RUNTIME_MODELS
         for model in requested_models):
    raise ControlledReplayLegacyConfigurationError(
      "A legacy model is forbidden in controlled replay")
  if requested_models != (
      M5_CHAT_MODEL, COMPLETION_COMPAT_MODEL,
      TEXT_EMBEDDING_3_SMALL_MODEL):
    raise ControlledReplayConfigurationError(
      "Controlled replay requires all canonical pinned models")
  if not isinstance(providers.pricing_snapshot, PricingSnapshot):
    raise ControlledReplayConfigurationError(
      "pricing_snapshot must be a PricingSnapshot")


def _manifest_identity(manifest: EmbeddingSpaceManifest):
  return ":".join((manifest.provider, manifest.model,
                   str(manifest.dimensions),
                   manifest.embedding_space_version,
                   manifest.normalization_version))


def _trusted_provider_error_type_names():
  pending = list(LLMProviderError.__subclasses__())
  trusted = {LLMProviderError.__name__}
  while pending:
    error_type = pending.pop()
    trusted.add(error_type.__name__)
    pending.extend(error_type.__subclasses__())
  return frozenset(trusted)


TRUSTED_PROVIDER_ERROR_TYPE_NAMES = _trusted_provider_error_type_names()


def _telemetry_entry(profile, event):
  return ControlledReplayTelemetryEntry(
    replay_id=profile.replay_id,
    simulation_id=event.simulation_id,
    actor_id=event.actor_id,
    step=event.simulation_step,
    cognitive_category=event.cognitive_category,
    caller_id=event.caller_id,
    operation=event.operation,
    logical_call_id=event.logical_call_id,
    physical_attempt=event.physical_attempt,
    requested_model=event.model_or_engine,
    returned_model=event.response_model,
    outcome=event.outcome,
    error_type=(
      event.error_type
      if event.error_type in TRUSTED_PROVIDER_ERROR_TYPE_NAMES
      else "UNKNOWN_PROVIDER_ERROR" if event.error_type else None),
  )


def _underlying_provider_error_type(error, events):
  if isinstance(error, LLMProviderError):
    return type(error).__name__
  if (not isinstance(error, ReplayCostAccountingUnavailableError)
      or not any(event.outcome == "ERROR" for event in events)):
    return None
  cause = error.__cause__ or error.__context__
  seen = set()
  while cause is not None and id(cause) not in seen:
    seen.add(id(cause))
    if isinstance(cause, LLMProviderError):
      return type(cause).__name__
    next_cause = cause.__cause__ or cause.__context__
    if next_cause is None:
      return "UNKNOWN_PROVIDER_ERROR"
    cause = next_cause
  return "UNKNOWN_PROVIDER_ERROR"


def _build_report(profile, events, call_guard, cost_guard, manifest,
                  cache_before, cache_after, output=None, error=None):
  cost_snapshot = cost_guard.snapshot()
  records = cost_guard.records()
  counts = {}
  for event in events:
    counts[event.operation] = counts.get(event.operation, 0) + 1
  telemetry = tuple(_telemetry_entry(profile, event) for event in events)
  ledger = tuple(ControlledReplayLedgerEntry(
    replay_id=profile.replay_id,
    simulation_id=record.simulation_id,
    actor_id=record.actor_id,
    step=record.simulation_step,
    cognitive_category=record.cognitive_category,
    caller_id=record.caller_id,
    operation=record.operation,
    logical_call_id=record.logical_call_id,
    physical_attempt=record.attempt,
    estimated_cost=record.estimated_total_cost_usd,
  ) for record in records)
  requested = tuple(dict.fromkeys(event.model_or_engine for event in events))
  returned = tuple(dict.fromkeys(
    event.response_model for event in events if event.response_model))
  legacy_detections = sum(
    event.model_or_engine in FORBIDDEN_MODERN_RUNTIME_MODELS
    or event.response_model in FORBIDDEN_MODERN_RUNTIME_MODELS
    for event in events)
  digest = None if output is None else hashlib.sha256(
    repr(output).encode("utf-8")).hexdigest()
  primary_error_type = None if error is None else type(error).__name__
  return ControlledReplayReport(
    replay_id=profile.replay_id,
    simulation_id=profile.simulation_id,
    actor_id=profile.actor_id,
    step=profile.step,
    status=SUCCESS if error is None else "FAILED",
    selected_cognitive_path=WAKE_UP_PATH,
    logical_calls=call_guard.logical_calls,
    physical_attempts=call_guard.physical_attempts,
    operation_counts=tuple(sorted(counts.items())),
    models_requested=requested,
    models_returned=returned,
    retry_count=max(0, call_guard.physical_attempts - call_guard.logical_calls),
    input_tokens=sum(event.input_tokens or 0 for event in events),
    output_tokens=sum(event.output_tokens or 0 for event in events),
    cached_tokens=sum(event.cached_input_tokens or 0 for event in events),
    reasoning_tokens=sum(event.reasoning_tokens or 0 for event in events),
    embedding_cache_hits=(cache_after.cache_hits - cache_before.cache_hits),
    embedding_cache_misses=(cache_after.cache_misses - cache_before.cache_misses),
    accumulated_cost=cost_snapshot.accumulated_cost,
    cost_ceiling=cost_snapshot.ceiling,
    remaining_cost=cost_snapshot.remaining_cost,
    legacy_detections=legacy_detections,
    error_type=primary_error_type,
    primary_error_type=primary_error_type,
    underlying_provider_error_type=(
      _underlying_provider_error_type(error, events)),
    store_manifest_identity=_manifest_identity(manifest),
    parser_result_type=None if output is None else type(output).__name__,
    wake_up_hour=(output if type(output) is int else None),
    cognitive_output_digest=digest,
    telemetry=telemetry,
    ledger=ledger,
  )


def _validate_fixture_identity(profile, actor_fixture, environment_fixture):
  if not isinstance(actor_fixture, ControlledReplayActorFixture):
    raise TypeError("actor_fixture must be ControlledReplayActorFixture")
  if not isinstance(environment_fixture, ControlledReplayEnvironmentFixture):
    raise TypeError(
      "environment_fixture must be ControlledReplayEnvironmentFixture")
  if actor_fixture.actor_id != profile.actor_id:
    raise ControlledReplayConfigurationError(
      "actor fixture identity does not match the replay profile")
  if (environment_fixture.simulation_id != profile.simulation_id
      or environment_fixture.step != profile.step):
    raise ControlledReplayConfigurationError(
      "environment fixture identity does not match the replay profile")


def run_controlled_replay_step(profile, actor_fixture, environment_fixture,
                               providers, store_path):
  """Run one offline wake-up planning decision and return a safe report."""
  if not isinstance(profile, ControlledReplayProfile):
    raise TypeError("profile must be ControlledReplayProfile")
  if not isinstance(providers, ControlledReplayProviders):
    raise TypeError("providers must be ControlledReplayProviders")
  _validate_fixture_identity(profile, actor_fixture, environment_fixture)
  _validate_modern_runtime_configuration(providers)
  assert_controlled_replay_path_allowed(WAKE_UP_CALLER)

  store_path = Path(store_path)
  bootstrap_result = bootstrap_modern_embedding_store(
    ModernEmbeddingStoreBootstrapRequest(
      target_path=store_path, allowed_parent=store_path.parent))
  manifest = bootstrap_result.manifest
  telemetry_start = len(get_telemetry())
  cache_before = get_embedding_cache_stats()
  replay_context = LLMReplayContext(
    cognitive_category=PLANNING,
    actor_id=profile.actor_id,
    simulation_id=profile.simulation_id,
    simulation_step=profile.step,
  )
  ledger_context = CostLedgerContext(
    simulation_id=profile.simulation_id,
    simulation_step=profile.step,
    actor_id=profile.actor_id,
    cognitive_category=PLANNING,
  )
  cost_config = ReplayCostGuardConfig(
    replay_id=profile.replay_id,
    simulation_id=profile.simulation_id,
    ceiling=ReplayCostCeiling(profile.maximum_cost),
    pricing_snapshot=providers.pricing_snapshot,
  )

  with use_embedding_runtime(
      providers.embedding_config, providers.embedding_adapter,
      store_path=store_path, legacy_assumption_allowed=False):
    with use_modern_chat_runtime(
        providers.chat_config, providers.chat_adapter):
      with use_modern_completion_runtime(
          providers.completion_config, providers.completion_adapter):
        with use_llm_replay_context(replay_context):
          with use_cost_ledger_context(ledger_context):
            with use_replay_call_count_guard(profile) as call_guard:
              with use_replay_cost_guard(cost_config) as cost_guard:
                with _use_combined_replay_observer(call_guard, cost_guard):
                  try:
                    with _historical_wrapper_environment():
                      output = run_gpt_prompt.run_gpt_prompt_wake_up_hour(
                        actor_fixture.persona,
                        test_input=list(actor_fixture.prompt_input))[0]
                  except Exception as error:
                    events = get_telemetry()[telemetry_start:]
                    report = _build_report(
                      profile, events, call_guard, cost_guard, manifest,
                      cache_before, get_embedding_cache_stats(), error=error)
                    try:
                      error.controlled_replay_report = report
                    except Exception:
                      pass
                    raise
                  return _build_report(
                    profile, get_telemetry()[telemetry_start:], call_guard,
                    cost_guard, manifest, cache_before,
                    get_embedding_cache_stats(), output=output)


def controlled_replay_contexts_are_reset():
  """Content-free diagnostic used by the R0H teardown tests."""
  from persona.prompt_template.chat_runtime import get_modern_chat_runtime_config
  from persona.prompt_template.completion_runtime import (
    get_modern_completion_runtime_config,
  )
  from persona.memory_structures.embedding_space import (
    LEGACY_ADA_002_MANIFEST, get_runtime_embedding_manifest,
  )
  return (
    get_modern_chat_runtime_config() is None
    and get_modern_completion_runtime_config() is None
    and get_llm_replay_context() == LLMReplayContext()
    and get_replay_cost_guard() is None
    and get_replay_call_count_guard() is None
    and get_runtime_embedding_manifest() == LEGACY_ADA_002_MANIFEST
  )
