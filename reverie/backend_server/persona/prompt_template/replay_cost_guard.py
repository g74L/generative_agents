"""Replay-scoped, privacy-safe enforcement over the factual cost ledger.

The guard is opt-in and belongs to replay orchestration.  It observes validated
physical-attempt telemetry, delegates all price calculation to ``cost_ledger``,
and prevents further calls once accounting is unavailable or the ceiling is
reached.  R0 supports sequential replay only; concurrent attempts sharing one
guard are rejected before their provider call.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from threading import RLock
from typing import Optional, Tuple

from persona.prompt_template.cost_ledger import (
  CostLedgerContext,
  CostLedgerRecord,
  PricingSnapshot,
  build_cost_ledger_records,
  summarize_cost_ledger,
)
from persona.prompt_template.llm_provider import (
  CHAT,
  COMPLETION_COMPAT,
  EMBEDDING,
  LLMReplayContext,
  TelemetryEvent,
  install_llm_attempt_observer,
  reset_llm_attempt_observer,
)


COVERED_OPERATIONS = frozenset((CHAT, COMPLETION_COMPAT, EMBEDDING))


def _event_bound_cost_context(event: TelemetryEvent) -> CostLedgerContext:
  """Preserve each call-time attribution field, including explicit None."""
  return CostLedgerContext(
    caller_id=event.caller_id,
    cognitive_category=event.cognitive_category,
    actor_id=event.actor_id,
    simulation_id=event.simulation_id,
    simulation_step=event.simulation_step,
  )


class ReplayCostGuardError(RuntimeError):
  """Base class for controlled replay cost enforcement failures."""


class ReplayCostCeilingExceededError(ReplayCostGuardError):
  def __init__(self, accumulated_cost, maximum_cost, operation, model):
    self.accumulated_cost = accumulated_cost
    self.maximum_cost = maximum_cost
    self.operation = operation
    self.model = model
    super().__init__(
      f"Replay cost ceiling exceeded: {accumulated_cost} > {maximum_cost} "
      f"after {operation} using {model}")


class ReplayCostAccountingUnavailableError(ReplayCostGuardError):
  def __init__(self, operation, model):
    self.operation = operation
    self.model = model
    super().__init__(
      f"Replay cost accounting unavailable for {operation} using {model}")


class ReplayCostGuardAlreadyTrippedError(ReplayCostGuardError):
  def __init__(self, accumulated_cost, maximum_cost):
    self.accumulated_cost = accumulated_cost
    self.maximum_cost = maximum_cost
    super().__init__(
      f"Replay cost guard is tripped at {accumulated_cost} "
      f"with ceiling {maximum_cost}")


class ReplayCostConcurrentAttemptError(ReplayCostGuardError):
  def __init__(self):
    super().__init__("R0 replay cost guard supports sequential attempts only")


class ReplayCostContextMismatchError(ReplayCostGuardError):
  def __init__(self):
    super().__init__("LLM replay simulation does not match cost guard context")


@dataclass(frozen=True)
class ReplayCostCeiling:
  maximum_cost: Decimal

  def __post_init__(self):
    value = self.maximum_cost
    if isinstance(value, bool) or not isinstance(value, Decimal):
      raise TypeError("maximum_cost must be a canonical Decimal")
    if not value.is_finite() or value <= 0:
      raise ValueError("maximum_cost must be finite and greater than zero")


@dataclass(frozen=True)
class ReplayCostGuardConfig:
  replay_id: str
  simulation_id: str
  ceiling: ReplayCostCeiling
  pricing_snapshot: PricingSnapshot

  def __post_init__(self):
    for field_name in ("replay_id", "simulation_id"):
      value = getattr(self, field_name)
      if (not isinstance(value, str) or not value.strip()
          or len(value) > 512):
        raise ValueError(f"{field_name} must be non-blank text")
    if not isinstance(self.ceiling, ReplayCostCeiling):
      raise TypeError("ceiling must be ReplayCostCeiling")
    if not isinstance(self.pricing_snapshot, PricingSnapshot):
      raise TypeError("pricing_snapshot must be PricingSnapshot")


@dataclass(frozen=True)
class ReplayCostGuardSnapshot:
  replay_id: str
  simulation_id: str
  ceiling: Decimal
  accumulated_cost: Decimal
  remaining_cost: Decimal
  tripped: bool
  logical_calls: int
  physical_attempts: int
  cost_by_operation: Tuple[Tuple[str, Decimal], ...]


class ReplayCostGuardState:
  """Controlled mutable state shared by all operations in one replay."""

  def __init__(self, config: ReplayCostGuardConfig):
    if not isinstance(config, ReplayCostGuardConfig):
      raise TypeError("config must be ReplayCostGuardConfig")
    self.config = config
    self._records = []
    self._tripped = False
    self._in_flight = None
    self._lock = RLock()

  def _summary(self):
    return summarize_cost_ledger(tuple(self._records))

  def _accumulated_cost(self):
    value = self._summary().totals.estimated_total_cost_usd
    return value if value is not None else Decimal("0")

  def before_attempt(self, *, operation: str, model: str,
                     logical_call_id: str, physical_attempt: int,
                     replay_context: LLMReplayContext) -> None:
    with self._lock:
      accumulated = self._accumulated_cost()
      if self._tripped or accumulated >= self.config.ceiling.maximum_cost:
        self._tripped = True
        raise ReplayCostGuardAlreadyTrippedError(
          accumulated, self.config.ceiling.maximum_cost)
      if operation not in COVERED_OPERATIONS:
        self._tripped = True
        raise ReplayCostAccountingUnavailableError(operation, model)
      if replay_context.simulation_id != self.config.simulation_id:
        self._tripped = True
        raise ReplayCostContextMismatchError()
      if self._in_flight is not None:
        raise ReplayCostConcurrentAttemptError()
      self._in_flight = (logical_call_id, physical_attempt)

  def after_attempt(self, event: TelemetryEvent) -> None:
    with self._lock:
      try:
        try:
          record = build_cost_ledger_records(
            (event,), self.config.pricing_snapshot,
            context_resolver=_event_bound_cost_context)[0]
        except Exception as error:
          self._tripped = True
          raise ReplayCostAccountingUnavailableError(
            event.operation, event.model_or_engine) from error
        self._records.append(record)
        if record.estimated_total_cost_usd is None:
          self._tripped = True
          raise ReplayCostAccountingUnavailableError(
            event.operation, event.model_or_engine)
        accumulated = self._accumulated_cost()
        maximum = self.config.ceiling.maximum_cost
        if accumulated >= maximum:
          self._tripped = True
        if accumulated > maximum:
          raise ReplayCostCeilingExceededError(
            accumulated, maximum, event.operation, event.model_or_engine)
      finally:
        self._in_flight = None

  def snapshot(self) -> ReplayCostGuardSnapshot:
    with self._lock:
      summary = self._summary()
      accumulated = (summary.totals.estimated_total_cost_usd
                     if summary.totals.estimated_total_cost_usd is not None
                     else Decimal("0"))
      remaining = self.config.ceiling.maximum_cost - accumulated
      if remaining < 0:
        remaining = Decimal("0")
      breakdown = tuple(
        (operation, aggregate.estimated_total_cost_usd or Decimal("0"))
        for operation, aggregate in summary.by_operation)
      return ReplayCostGuardSnapshot(
        replay_id=self.config.replay_id,
        simulation_id=self.config.simulation_id,
        ceiling=self.config.ceiling.maximum_cost,
        accumulated_cost=accumulated,
        remaining_cost=remaining,
        tripped=self._tripped,
        logical_calls=summary.totals.logical_calls,
        physical_attempts=summary.totals.physical_attempts,
        cost_by_operation=breakdown,
      )

  def records(self) -> Tuple[CostLedgerRecord, ...]:
    with self._lock:
      return tuple(self._records)


@dataclass(frozen=True)
class _ReplayCostGuardInstallation:
  state: ReplayCostGuardState
  state_token: object
  observer_token: object


_active_replay_cost_guard: ContextVar[Optional[ReplayCostGuardState]] = (
  ContextVar("active_replay_cost_guard", default=None))


def get_replay_cost_guard() -> Optional[ReplayCostGuardState]:
  return _active_replay_cost_guard.get()


def install_replay_cost_guard(
    config: ReplayCostGuardConfig) -> _ReplayCostGuardInstallation:
  state = ReplayCostGuardState(config)
  state_token = _active_replay_cost_guard.set(state)
  try:
    observer_token = install_llm_attempt_observer(state)
  except Exception:
    _active_replay_cost_guard.reset(state_token)
    raise
  return _ReplayCostGuardInstallation(state, state_token, observer_token)


def reset_replay_cost_guard(
    installation: _ReplayCostGuardInstallation) -> None:
  if (not isinstance(installation, _ReplayCostGuardInstallation)
      or get_replay_cost_guard() is not installation.state):
    raise ValueError("installation is not the active replay cost guard")
  reset_llm_attempt_observer(installation.observer_token)
  _active_replay_cost_guard.reset(installation.state_token)


@contextmanager
def use_replay_cost_guard(config: ReplayCostGuardConfig):
  installation = install_replay_cost_guard(config)
  try:
    yield installation.state
  finally:
    reset_replay_cost_guard(installation)
