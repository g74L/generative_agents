"""Pure, privacy-safe replay cost accounting over existing LLM telemetry.

Telemetry remains the immutable source of call and usage facts.  Pricing is an
explicit, versioned input so historical records can be recalculated without
changing providers or the original events.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple


LEDGER_SCHEMA_VERSION = 1
PRICING_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1

COMPLETE = "COMPLETE"
PARTIAL = "PARTIAL"
UNAVAILABLE = "UNAVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"

PRICING_COMPLETE = "COMPLETE"
PRICING_PARTIAL = "PARTIAL"
PRICING_UNAVAILABLE = "UNAVAILABLE"

UNSPECIFIED = "UNSPECIFIED"
USD = "USD"
PER_MILLION_TOKENS = "PER_MILLION_TOKENS"
MONEY_QUANTUM = Decimal("0.000000000001")
MILLION = Decimal("1000000")


def _validate_optional_token(value, field_name):
  if value is not None and (type(value) is not int or value < 0):
    raise ValueError(f"{field_name} must be a non-negative integer or None")


def _validate_count(value, field_name, positive=False):
  minimum = 1 if positive else 0
  if type(value) is not int or value < minimum:
    qualifier = "positive" if positive else "non-negative"
    raise ValueError(f"{field_name} must be a {qualifier} integer")


def _validate_optional_decimal(value, field_name):
  if value is not None:
    if not isinstance(value, Decimal):
      raise TypeError(f"{field_name} must be Decimal or None")
    if not value.is_finite() or value < 0:
      raise ValueError(f"{field_name} must be finite and non-negative")


def _money(value):
  with localcontext() as context:
    context.prec = 40
    return value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True)
class ModelPricing:
  model: str
  currency: str = USD
  unit: str = PER_MILLION_TOKENS
  input_per_million: Optional[Decimal] = None
  cached_input_per_million: Optional[Decimal] = None
  output_per_million: Optional[Decimal] = None
  embedding_input_per_million: Optional[Decimal] = None
  effective_from: Optional[str] = None
  source_label: Optional[str] = None

  def __post_init__(self):
    if not self.model:
      raise ValueError("model is required")
    if self.currency != USD:
      raise ValueError("Replay Cost Ledger V0 supports USD pricing only")
    if self.unit != PER_MILLION_TOKENS:
      raise ValueError("Pricing unit must be PER_MILLION_TOKENS")
    for field_name in (
        "input_per_million", "cached_input_per_million",
        "output_per_million", "embedding_input_per_million"):
      _validate_optional_decimal(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class PricingSnapshot:
  snapshot_id: str
  schema_version: int
  currency: str
  created_at: str
  models: Tuple[ModelPricing, ...]
  source_note: str

  def __post_init__(self):
    if not self.snapshot_id:
      raise ValueError("snapshot_id is required")
    if self.schema_version != PRICING_SCHEMA_VERSION:
      raise ValueError("Unsupported pricing snapshot schema version")
    if self.currency != USD:
      raise ValueError("Replay Cost Ledger V0 supports USD snapshots only")
    if not isinstance(self.models, tuple):
      raise TypeError("models must be an immutable tuple")
    names = [item.model for item in self.models]
    if len(names) != len(set(names)):
      raise ValueError("Pricing snapshot contains duplicate models")
    if any(item.currency != self.currency for item in self.models):
      raise ValueError("Model and snapshot currencies must match")

  def pricing_for(self, model):
    return next((item for item in self.models if item.model == model), None)


@dataclass(frozen=True)
class CostLedgerContext:
  simulation_id: Optional[str] = None
  simulation_step: Optional[int] = None
  actor_id: Optional[str] = None
  cognitive_category: Optional[str] = None

  def __post_init__(self):
    if (self.simulation_step is not None
        and (type(self.simulation_step) is not int
             or self.simulation_step < 0)):
      raise ValueError("simulation_step must be a non-negative integer or None")


_cost_ledger_context = ContextVar(
  "cost_ledger_context", default=CostLedgerContext())


def get_cost_ledger_context():
  return _cost_ledger_context.get()


def set_cost_ledger_context(context):
  if not isinstance(context, CostLedgerContext):
    raise TypeError("context must be CostLedgerContext")
  return _cost_ledger_context.set(context)


def reset_cost_ledger_context(token=None):
  if token is None:
    _cost_ledger_context.set(CostLedgerContext())
  else:
    _cost_ledger_context.reset(token)


@contextmanager
def use_cost_ledger_context(context):
  token = set_cost_ledger_context(context)
  try:
    yield context
  finally:
    reset_cost_ledger_context(token)


@dataclass(frozen=True)
class CostLedgerRecord:
  schema_version: int
  record_id: str
  logical_call_id: Optional[str]
  operation: str
  provider_kind: Optional[str]
  transport_kind: Optional[str]
  model: Optional[str]
  attempt: int
  outcome: str
  elapsed_ms: Decimal
  error_type: Optional[str]
  request_id: Optional[str]
  input_tokens: Optional[int]
  output_tokens: Optional[int]
  cached_input_tokens: Optional[int]
  reasoning_tokens: Optional[int]
  total_tokens: Optional[int]
  token_usage_status: str
  pricing_snapshot_id: Optional[str]
  pricing_status: str
  estimated_input_cost_usd: Optional[Decimal]
  estimated_cached_input_cost_usd: Optional[Decimal]
  estimated_output_cost_usd: Optional[Decimal]
  estimated_total_cost_usd: Optional[Decimal]
  cognitive_category: Optional[str]
  actor_id: Optional[str]
  simulation_id: Optional[str]
  simulation_step: Optional[int]
  created_at: Optional[str]

  def __post_init__(self):
    if self.schema_version != LEDGER_SCHEMA_VERSION:
      raise ValueError("Unsupported cost ledger record schema version")
    if not self.record_id or not self.operation:
      raise ValueError("record_id and operation are required")
    _validate_count(self.attempt, "attempt", positive=True)
    if (not isinstance(self.elapsed_ms, Decimal)
        or not self.elapsed_ms.is_finite() or self.elapsed_ms < 0):
      raise ValueError("elapsed_ms must be a finite non-negative Decimal")
    if self.outcome not in ("SUCCESS", "ERROR"):
      raise ValueError("Unsupported ledger outcome")
    if self.token_usage_status not in (
        COMPLETE, PARTIAL, UNAVAILABLE, NOT_APPLICABLE):
      raise ValueError("Unsupported token usage status")
    if self.pricing_status not in (
        PRICING_COMPLETE, PRICING_PARTIAL, PRICING_UNAVAILABLE):
      raise ValueError("Unsupported pricing status")
    for field_name in (
        "input_tokens", "output_tokens", "cached_input_tokens",
        "reasoning_tokens", "total_tokens"):
      _validate_optional_token(getattr(self, field_name), field_name)
    if (self.token_usage_status in (UNAVAILABLE, NOT_APPLICABLE)
        and any(getattr(self, name) is not None for name in (
          "input_tokens", "output_tokens", "cached_input_tokens",
          "reasoning_tokens", "total_tokens"))):
      raise ValueError("Unavailable token usage cannot contain token values")
    if self.token_usage_status == COMPLETE:
      if self.operation == "EMBEDDING":
        if self.input_tokens is None:
          raise ValueError("Complete embedding usage requires input tokens")
      elif self.input_tokens is None or self.output_tokens is None:
        raise ValueError("Complete text usage requires input and output tokens")
      if (self.cached_input_tokens is not None
          and self.cached_input_tokens > self.input_tokens):
        raise ValueError("Cached input tokens cannot exceed input tokens")
    if self.total_tokens is not None:
      expected_total = (
        self.input_tokens if self.operation == "EMBEDDING"
        else (self.input_tokens + self.output_tokens
              if self.input_tokens is not None
              and self.output_tokens is not None else None))
      if expected_total is None or self.total_tokens != expected_total:
        raise ValueError("total_tokens must match the normalized usage fields")
    for field_name in (
        "estimated_input_cost_usd", "estimated_cached_input_cost_usd",
        "estimated_output_cost_usd", "estimated_total_cost_usd"):
      _validate_optional_decimal(getattr(self, field_name), field_name)
    if (self.pricing_status == PRICING_COMPLETE
        and self.estimated_total_cost_usd is None):
      raise ValueError("Complete pricing requires a total cost")
    if (self.pricing_status != PRICING_COMPLETE
        and self.estimated_total_cost_usd is not None):
      raise ValueError("Incomplete pricing cannot contain a total cost")
    if (self.simulation_step is not None
        and (type(self.simulation_step) is not int
             or self.simulation_step < 0)):
      raise ValueError("simulation_step must be non-negative or None")


@dataclass(frozen=True)
class TokenAggregate:
  known_value: int
  unknown_record_count: int

  def __post_init__(self):
    _validate_count(self.known_value, "known_value")
    _validate_count(self.unknown_record_count, "unknown_record_count")


@dataclass(frozen=True)
class LedgerAggregate:
  logical_calls: int
  physical_attempts: int
  successful_attempts: int
  failed_attempts: int
  retry_count: int
  input_tokens: TokenAggregate
  output_tokens: TokenAggregate
  cached_input_tokens: TokenAggregate
  reasoning_tokens: TokenAggregate
  total_tokens: TokenAggregate
  known_cost_record_count: int
  unknown_cost_record_count: int
  estimated_total_cost_usd: Optional[Decimal]

  def __post_init__(self):
    for field_name in (
        "logical_calls", "physical_attempts", "successful_attempts",
        "failed_attempts", "retry_count", "known_cost_record_count",
        "unknown_cost_record_count"):
      _validate_count(getattr(self, field_name), field_name)
    for field_name in (
        "input_tokens", "output_tokens", "cached_input_tokens",
        "reasoning_tokens", "total_tokens"):
      token_aggregate = getattr(self, field_name)
      if not isinstance(token_aggregate, TokenAggregate):
        raise TypeError(f"{field_name} must be TokenAggregate")
      if token_aggregate.unknown_record_count > self.physical_attempts:
        raise ValueError(
          f"{field_name} unknown count cannot exceed physical attempts")
    if self.successful_attempts + self.failed_attempts != self.physical_attempts:
      raise ValueError("Attempt outcome counts must equal physical attempts")
    if (self.known_cost_record_count + self.unknown_cost_record_count
        != self.physical_attempts):
      raise ValueError("Cost record counts must equal physical attempts")
    if self.logical_calls > self.physical_attempts:
      raise ValueError("Logical calls cannot exceed physical attempts")
    if self.retry_count > self.physical_attempts - self.logical_calls:
      raise ValueError("Retry count is inconsistent with call counts")
    _validate_optional_decimal(
      self.estimated_total_cost_usd, "estimated_total_cost_usd")
    if ((self.known_cost_record_count == 0)
        != (self.estimated_total_cost_usd is None)):
      raise ValueError("Known costs and aggregate total are inconsistent")


@dataclass(frozen=True)
class EmbeddingCacheLedgerSummary:
  logical_embedding_requests: int = 0
  physical_embedding_attempts: int = 0
  cache_hits: int = 0
  cache_misses: int = 0
  cache_hit_rate: Decimal = Decimal("0")
  evictions: int = 0
  avoided_embedding_calls: int = 0
  estimated_embedding_cost_avoided_usd: Optional[Decimal] = None

  def __post_init__(self):
    for field_name in (
        "logical_embedding_requests", "physical_embedding_attempts",
        "cache_hits", "cache_misses", "evictions",
        "avoided_embedding_calls"):
      _validate_count(getattr(self, field_name), field_name)
    if self.cache_hits > self.logical_embedding_requests:
      raise ValueError("Cache hits cannot exceed logical requests")
    if self.cache_hits + self.cache_misses != self.logical_embedding_requests:
      raise ValueError("Every logical embedding request must be hit or miss")
    if self.avoided_embedding_calls != self.cache_hits:
      raise ValueError("Avoided embedding calls must equal cache hits")
    if (not isinstance(self.cache_hit_rate, Decimal)
        or not self.cache_hit_rate.is_finite()
        or not Decimal("0") <= self.cache_hit_rate <= Decimal("1")):
      raise ValueError("cache_hit_rate must be a Decimal between zero and one")
    expected_hit_rate = (_money(
      Decimal(self.cache_hits) / Decimal(self.logical_embedding_requests))
      if self.logical_embedding_requests else _money(Decimal("0")))
    if self.cache_hit_rate != expected_hit_rate:
      raise ValueError("cache_hit_rate must match cache hit counters")
    _validate_optional_decimal(
      self.estimated_embedding_cost_avoided_usd,
      "estimated_embedding_cost_avoided_usd")


@dataclass(frozen=True)
class CostLedgerSummary:
  schema_version: int
  pricing_snapshot_id: Optional[str]
  totals: LedgerAggregate
  by_operation: Tuple[Tuple[str, LedgerAggregate], ...]
  by_model: Tuple[Tuple[str, LedgerAggregate], ...]
  by_provider: Tuple[Tuple[str, LedgerAggregate], ...]
  by_outcome: Tuple[Tuple[str, LedgerAggregate], ...]
  by_cognitive_category: Tuple[Tuple[str, LedgerAggregate], ...]
  by_actor: Tuple[Tuple[str, LedgerAggregate], ...]
  by_simulation_step: Tuple[Tuple[str, LedgerAggregate], ...]
  by_pricing_snapshot: Tuple[Tuple[str, LedgerAggregate], ...]
  embedding_cache: EmbeddingCacheLedgerSummary

  def __post_init__(self):
    if self.schema_version != SUMMARY_SCHEMA_VERSION:
      raise ValueError("Unsupported cost ledger summary schema version")
    if not isinstance(self.totals, LedgerAggregate):
      raise TypeError("totals must be LedgerAggregate")
    for field_name in (
        "by_operation", "by_model", "by_provider", "by_outcome",
        "by_cognitive_category", "by_actor", "by_simulation_step",
        "by_pricing_snapshot"):
      items = getattr(self, field_name)
      if not isinstance(items, tuple):
        raise TypeError(f"{field_name} must be an immutable tuple")
      keys = [key for key, aggregate in items
              if isinstance(key, str) and isinstance(aggregate, LedgerAggregate)]
      if len(keys) != len(items) or keys != sorted(set(keys)):
        raise ValueError(f"{field_name} must have unique sorted string keys")
    if not isinstance(self.embedding_cache, EmbeddingCacheLedgerSummary):
      raise TypeError("embedding_cache must be EmbeddingCacheLedgerSummary")
    snapshot_keys = [key for key, aggregate in self.by_pricing_snapshot
                     if key != UNSPECIFIED]
    expected_snapshot_id = (
      snapshot_keys[0] if len(snapshot_keys) == 1
      else "MULTIPLE" if snapshot_keys else None)
    if self.pricing_snapshot_id != expected_snapshot_id:
      raise ValueError(
        "pricing_snapshot_id must match by_pricing_snapshot")


def _event_field(event, name, default=None):
  if isinstance(event, Mapping):
    return event.get(name, default)
  return getattr(event, name, default)


def _normalize_optional_token_count(value):
  if value is None:
    return None, False
  if type(value) is int and value >= 0:
    return value, False
  return None, True


def _token_usage(event, operation, outcome):
  raw_values = {
    name: _event_field(event, name) for name in (
      "input_tokens", "output_tokens", "cached_input_tokens",
      "reasoning_tokens")}
  normalized = {
    name: _normalize_optional_token_count(value)
    for name, value in raw_values.items()}
  values = {name: value for name, (value, invalid) in normalized.items()}
  malformed = any(invalid for value, invalid in normalized.values())
  if all(value is None for value in raw_values.values()):
    status = NOT_APPLICABLE if outcome == "ERROR" else UNAVAILABLE
    return values, None, status
  incoherent_cached = (
    values["cached_input_tokens"] is not None
    and values["input_tokens"] is not None
    and values["cached_input_tokens"] > values["input_tokens"])
  if operation == "EMBEDDING":
    complete = values["input_tokens"] is not None
    total = values["input_tokens"] if complete else None
  else:
    complete = (values["input_tokens"] is not None
                and values["output_tokens"] is not None)
    total = (values["input_tokens"] + values["output_tokens"]
             if complete else None)
  status = COMPLETE if complete and not malformed and not incoherent_cached else PARTIAL
  return values, total, status


def _component_cost(tokens, rate):
  if tokens is None or rate is None:
    return None
  with localcontext() as context:
    context.prec = 40
    return _money(Decimal(tokens) * rate / MILLION)


def _costs(operation, usage, token_status, pricing):
  empty = (None, None, None, None)
  if pricing is None:
    return PRICING_UNAVAILABLE, empty
  if token_status != COMPLETE:
    return PRICING_PARTIAL, empty

  input_tokens = usage["input_tokens"]
  output_tokens = usage["output_tokens"]
  cached_tokens = usage["cached_input_tokens"]
  if operation == "EMBEDDING":
    embedding_cost = _component_cost(
      input_tokens, pricing.embedding_input_per_million)
    if embedding_cost is None:
      return PRICING_PARTIAL, empty
    return PRICING_COMPLETE, (
      embedding_cost, None, None, embedding_cost)

  if pricing.input_per_million is None or pricing.output_per_million is None:
    return PRICING_PARTIAL, empty
  if cached_tokens is None:
    input_cost = _component_cost(input_tokens, pricing.input_per_million)
    cached_cost = None
  elif cached_tokens == 0:
    input_cost = _component_cost(input_tokens, pricing.input_per_million)
    cached_cost = _money(Decimal("0"))
  else:
    input_cost = _component_cost(
      input_tokens - cached_tokens, pricing.input_per_million)
    cached_cost = _component_cost(
      cached_tokens, pricing.cached_input_per_million)
    if cached_cost is None:
      output_cost = _component_cost(
        output_tokens, pricing.output_per_million)
      return PRICING_PARTIAL, (input_cost, None, output_cost, None)
  output_cost = _component_cost(output_tokens, pricing.output_per_million)
  total_cost = _money(input_cost + output_cost + (cached_cost or Decimal(0)))
  return PRICING_COMPLETE, (
    input_cost, cached_cost, output_cost, total_cost)


def _coerce_context(value):
  if value is None:
    return CostLedgerContext()
  if isinstance(value, CostLedgerContext):
    return value
  if isinstance(value, Mapping):
    allowed = {item.name for item in fields(CostLedgerContext)}
    unexpected = set(value) - allowed
    if unexpected:
      raise ValueError(f"Unexpected cost context fields: {sorted(unexpected)}")
    return CostLedgerContext(**value)
  raise TypeError("context_resolver must return CostLedgerContext, mapping, or None")


def build_cost_ledger_records(
    telemetry_records: Iterable[Any],
    pricing_snapshot: Optional[PricingSnapshot] = None,
    context_resolver: Optional[Callable[[Any], Any]] = None,
    embedding_logical_events: Iterable[Any] = (),
):
  """Build immutable economic records without mutating source telemetry."""
  category_by_call = {
    _event_field(event, "logical_call_id"): _event_field(event, "category")
    for event in embedding_logical_events
    if _event_field(event, "logical_call_id") is not None
  }
  records = []
  for index, event in enumerate(tuple(telemetry_records), 1):
    operation = str(_event_field(event, "operation", ""))
    outcome = str(_event_field(event, "outcome", ""))
    logical_call_id = _event_field(event, "logical_call_id")
    attempt = _event_field(event, "physical_attempt", 0)
    if type(attempt) is not int or attempt < 1:
      raise ValueError("physical_attempt must be a positive integer")
    elapsed = _event_field(event, "elapsed_seconds", 0)
    if (not isinstance(elapsed, (int, float, Decimal))
        or isinstance(elapsed, bool) or elapsed < 0):
      raise ValueError("elapsed_seconds must be non-negative")
    context = _coerce_context(
      context_resolver(event) if context_resolver else get_cost_ledger_context())
    category = context.cognitive_category
    if category is None:
      category = category_by_call.get(logical_call_id)
    usage, total_tokens, token_status = _token_usage(
      event, operation, outcome)
    model = (_event_field(event, "response_model")
             or _event_field(event, "model_or_engine") or None)
    pricing = (pricing_snapshot.pricing_for(model)
               if pricing_snapshot is not None else None)
    pricing_status, costs = _costs(
      operation, usage, token_status, pricing)
    record_call_id = str(logical_call_id) if logical_call_id else None
    records.append(CostLedgerRecord(
      schema_version=LEDGER_SCHEMA_VERSION,
      record_id=(f"ledger-{record_call_id or 'orphan'}-{attempt}-{index}"),
      logical_call_id=record_call_id,
      operation=operation,
      provider_kind=_event_field(event, "provider_kind"),
      transport_kind=_event_field(event, "transport_kind"),
      model=model,
      attempt=attempt,
      outcome=outcome,
      elapsed_ms=(Decimal(str(elapsed)) * Decimal("1000")).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_EVEN),
      error_type=_event_field(event, "error_type"),
      request_id=_event_field(event, "request_id"),
      input_tokens=usage["input_tokens"],
      output_tokens=usage["output_tokens"],
      cached_input_tokens=usage["cached_input_tokens"],
      reasoning_tokens=usage["reasoning_tokens"],
      total_tokens=total_tokens,
      token_usage_status=token_status,
      pricing_snapshot_id=(pricing_snapshot.snapshot_id
                           if pricing_snapshot is not None else None),
      pricing_status=pricing_status,
      estimated_input_cost_usd=costs[0],
      estimated_cached_input_cost_usd=costs[1],
      estimated_output_cost_usd=costs[2],
      estimated_total_cost_usd=costs[3],
      cognitive_category=category,
      actor_id=context.actor_id,
      simulation_id=context.simulation_id,
      simulation_step=context.simulation_step,
      created_at=None,
    ))
  return tuple(records)


def _aggregate(records):
  records = tuple(records)
  logical_ids = {
    record.logical_call_id for record in records if record.logical_call_id}
  grouped_attempts = {}
  for record in records:
    if record.logical_call_id:
      grouped_attempts[record.logical_call_id] = (
        grouped_attempts.get(record.logical_call_id, 0) + 1)
  retry_count = sum(max(count - 1, 0)
                    for count in grouped_attempts.values())
  known_costs = [
    record.estimated_total_cost_usd for record in records
    if record.estimated_total_cost_usd is not None]
  estimated_total = (
    _money(sum(known_costs, Decimal("0"))) if known_costs else None)

  def token_aggregate(field_name):
    values = [getattr(record, field_name) for record in records]
    return TokenAggregate(
      known_value=sum(value for value in values if value is not None),
      unknown_record_count=sum(value is None for value in values),
    )

  return LedgerAggregate(
    logical_calls=len(logical_ids),
    physical_attempts=len(records),
    successful_attempts=sum(record.outcome == "SUCCESS" for record in records),
    failed_attempts=sum(record.outcome != "SUCCESS" for record in records),
    retry_count=retry_count,
    input_tokens=token_aggregate("input_tokens"),
    output_tokens=token_aggregate("output_tokens"),
    cached_input_tokens=token_aggregate("cached_input_tokens"),
    reasoning_tokens=token_aggregate("reasoning_tokens"),
    total_tokens=token_aggregate("total_tokens"),
    known_cost_record_count=len(known_costs),
    unknown_cost_record_count=len(records) - len(known_costs),
    estimated_total_cost_usd=estimated_total,
  )


def _breakdown(records, attribute):
  grouped = {}
  for record in records:
    value = getattr(record, attribute)
    key = UNSPECIFIED if value is None or value == "" else str(value)
    grouped.setdefault(key, []).append(record)
  return tuple((key, _aggregate(grouped[key])) for key in sorted(grouped))


def _embedding_cache_summary(
    measurement, embedding_logical_events, pricing_snapshot,
    avoided_embedding_token_counts):
  global_stats = (measurement or {}).get("global", measurement or {})

  def counter(name):
    value = global_stats.get(name, 0)
    _validate_count(value, name)
    return value

  logical = counter("logical_embedding_requests")
  physical = counter("physical_embedding_attempts")
  hits = counter("cache_hits")
  misses = counter("cache_misses")
  evictions = counter("evictions")
  hit_rate = (_money(Decimal(hits) / Decimal(logical))
              if logical else _money(Decimal("0")))

  avoided_cost = None
  hit_events = [event for event in embedding_logical_events
                if _event_field(event, "cache_outcome") == "HIT"]
  token_counts = avoided_embedding_token_counts or {}
  if hits == 0:
    avoided_cost = _money(Decimal("0"))
  elif (pricing_snapshot is not None and len(hit_events) == hits
        and all(_event_field(event, "logical_call_id") in token_counts
                for event in hit_events)):
    costs = []
    for event in hit_events:
      pricing = pricing_snapshot.pricing_for(_event_field(event, "model"))
      tokens = token_counts[_event_field(event, "logical_call_id")]
      _validate_optional_token(tokens, "avoided embedding token count")
      cost = _component_cost(
        tokens, pricing.embedding_input_per_million if pricing else None)
      if cost is None:
        costs = []
        break
      costs.append(cost)
    if costs:
      avoided_cost = _money(sum(costs, Decimal("0")))
  return EmbeddingCacheLedgerSummary(
    logical_embedding_requests=logical,
    physical_embedding_attempts=physical,
    cache_hits=hits,
    cache_misses=misses,
    cache_hit_rate=hit_rate,
    evictions=evictions,
    avoided_embedding_calls=hits,
    estimated_embedding_cost_avoided_usd=avoided_cost,
  )


def summarize_cost_ledger(
    records: Iterable[CostLedgerRecord],
    embedding_measurement: Optional[Mapping[str, Any]] = None,
    embedding_logical_events: Iterable[Any] = (),
    pricing_snapshot: Optional[PricingSnapshot] = None,
    avoided_embedding_token_counts: Optional[Mapping[str, int]] = None,
):
  records = tuple(records)
  embedding_events = tuple(embedding_logical_events)
  snapshot_ids = sorted({record.pricing_snapshot_id for record in records
                         if record.pricing_snapshot_id})
  snapshot_id = (snapshot_ids[0] if len(snapshot_ids) == 1
                 else "MULTIPLE" if snapshot_ids else None)
  return CostLedgerSummary(
    schema_version=SUMMARY_SCHEMA_VERSION,
    pricing_snapshot_id=snapshot_id,
    totals=_aggregate(records),
    by_operation=_breakdown(records, "operation"),
    by_model=_breakdown(records, "model"),
    by_provider=_breakdown(records, "provider_kind"),
    by_outcome=_breakdown(records, "outcome"),
    by_cognitive_category=_breakdown(records, "cognitive_category"),
    by_actor=_breakdown(records, "actor_id"),
    by_simulation_step=_breakdown(records, "simulation_step"),
    by_pricing_snapshot=_breakdown(records, "pricing_snapshot_id"),
    embedding_cache=_embedding_cache_summary(
      embedding_measurement, embedding_events, pricing_snapshot,
      avoided_embedding_token_counts),
  )


def _decimal_string(value):
  return format(value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_EVEN), "f")


def _aggregate_to_dict(aggregate):
  result = {}
  for item in fields(aggregate):
    value = getattr(aggregate, item.name)
    if isinstance(value, Decimal):
      result[item.name] = _decimal_string(value)
    elif isinstance(value, TokenAggregate):
      result[item.name] = {
        "known_value": value.known_value,
        "unknown_record_count": value.unknown_record_count,
      }
    else:
      result[item.name] = value
  return result


def _breakdown_to_dict(items):
  return {key: _aggregate_to_dict(value) for key, value in items}


def cost_ledger_summary_to_dict(summary):
  """Return deterministic JSON-safe summary data containing no content."""
  cache = {}
  for item in fields(summary.embedding_cache):
    value = getattr(summary.embedding_cache, item.name)
    cache[item.name] = (
      _decimal_string(value) if isinstance(value, Decimal) else value)
  return {
    "schema_version": summary.schema_version,
    "pricing_snapshot_id": summary.pricing_snapshot_id,
    "totals": _aggregate_to_dict(summary.totals),
    "by_operation": _breakdown_to_dict(summary.by_operation),
    "by_model": _breakdown_to_dict(summary.by_model),
    "by_provider": _breakdown_to_dict(summary.by_provider),
    "by_outcome": _breakdown_to_dict(summary.by_outcome),
    "by_cognitive_category": _breakdown_to_dict(
      summary.by_cognitive_category),
    "by_actor": _breakdown_to_dict(summary.by_actor),
    "by_simulation_step": _breakdown_to_dict(summary.by_simulation_step),
    "by_pricing_snapshot": _breakdown_to_dict(summary.by_pricing_snapshot),
    "embedding_cache": cache,
  }
