import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from pathlib import Path
import sys
import unittest


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.prompt_template.cost_ledger import (
  COMPLETE,
  NOT_APPLICABLE,
  PARTIAL,
  PRICING_COMPLETE,
  PRICING_PARTIAL,
  PRICING_UNAVAILABLE,
  UNAVAILABLE,
  CostLedgerContext,
  EmbeddingCacheLedgerSummary,
  ModelPricing,
  PricingSnapshot,
  TokenAggregate,
  build_cost_ledger_records,
  cost_ledger_summary_to_dict,
  get_cost_ledger_context,
  reset_cost_ledger_context,
  summarize_cost_ledger,
  use_cost_ledger_context,
)
from persona.prompt_template.llm_provider import (
  CHAT,
  EMBEDDING,
  ERROR,
  HIT,
  SUCCESS,
  EmbeddingLogicalEvent,
  TelemetryEvent,
)


class ReplayCostLedgerTests(unittest.TestCase):
  def setUp(self):
    reset_cost_ledger_context()
    self.snapshot = PricingSnapshot(
      snapshot_id="synthetic-test-v0",
      schema_version=1,
      currency="USD",
      created_at="2026-08-05T00:00:00Z",
      models=(
        ModelPricing(
          model="test-chat",
          input_per_million=Decimal("2"),
          cached_input_per_million=Decimal("1"),
          output_per_million=Decimal("4"),
          effective_from="2026-08-05",
          source_label="synthetic test pricing"),
        ModelPricing(
          model="other-chat",
          input_per_million=Decimal("3"),
          cached_input_per_million=Decimal("1.5"),
          output_per_million=Decimal("6"),
          effective_from="2026-08-05",
          source_label="synthetic test pricing"),
        ModelPricing(
          model="test-embedding",
          embedding_input_per_million=Decimal("0.5"),
          effective_from="2026-08-05",
          source_label="synthetic test pricing"),
      ),
      source_note="Synthetic prices for deterministic offline tests only",
    )

  def tearDown(self):
    reset_cost_ledger_context()

  def event(self, call_id="call-1", operation=CHAT, attempt=1,
            model="test-chat", outcome=SUCCESS, input_tokens=1000,
            output_tokens=100, cached_input_tokens=200,
            reasoning_tokens=20, **overrides):
    values = dict(
      operation=operation,
      logical_call_id=call_id,
      physical_attempt=attempt,
      model_or_engine=model,
      outcome=outcome,
      elapsed_seconds=0.0125,
      input_fingerprint="privacy-safe-fingerprint",
      error_type=None if outcome == SUCCESS else "SyntheticError",
      provider_kind="MODERN_OPENAI",
      transport_kind="OPENAI_SDK_1_PLUS",
      request_id="req-test" if outcome == SUCCESS else None,
      response_model=model if outcome == SUCCESS else None,
      finish_reason="stop" if outcome == SUCCESS else None,
      response_status="completed" if outcome == SUCCESS else None,
      input_tokens=input_tokens,
      output_tokens=output_tokens,
      cached_input_tokens=cached_input_tokens,
      reasoning_tokens=reasoning_tokens,
    )
    values.update(overrides)
    return TelemetryEvent(**values)

  def records(self, *events, snapshot_marker=True, **kwargs):
    snapshot = self.snapshot if snapshot_marker else None
    return build_cost_ledger_records(events, snapshot, **kwargs)

  def breakdown(self, summary, name):
    return dict(getattr(summary, name))

  def test_01_complete_chat_record_has_exact_decimal_cost(self):
    record = self.records(self.event())[0]
    self.assertEqual(COMPLETE, record.token_usage_status)
    self.assertEqual(PRICING_COMPLETE, record.pricing_status)
    self.assertEqual(Decimal("0.001600000000"),
                     record.estimated_input_cost_usd)
    self.assertEqual(Decimal("0.000200000000"),
                     record.estimated_cached_input_cost_usd)
    self.assertEqual(Decimal("0.000400000000"),
                     record.estimated_output_cost_usd)
    self.assertEqual(Decimal("0.002200000000"),
                     record.estimated_total_cost_usd)

  def test_02_complete_embedding_record_has_exact_cost(self):
    record = self.records(self.event(
      operation=EMBEDDING, model="test-embedding", input_tokens=500,
      output_tokens=None, cached_input_tokens=None,
      reasoning_tokens=None))[0]
    self.assertEqual(COMPLETE, record.token_usage_status)
    self.assertEqual(500, record.total_tokens)
    self.assertEqual(Decimal("0.000250000000"),
                     record.estimated_total_cost_usd)
    self.assertIsNone(record.estimated_output_cost_usd)

  def test_03_successful_usage_absence_is_unavailable_not_zero(self):
    record = self.records(self.event(
      input_tokens=None, output_tokens=None, cached_input_tokens=None,
      reasoning_tokens=None))[0]
    self.assertEqual(UNAVAILABLE, record.token_usage_status)
    self.assertIsNone(record.total_tokens)
    self.assertIsNone(record.estimated_total_cost_usd)

  def test_04_partial_usage_remains_partial(self):
    record = self.records(self.event(
      input_tokens=10, output_tokens=None, cached_input_tokens=None,
      reasoning_tokens=None))[0]
    self.assertEqual(PARTIAL, record.token_usage_status)
    self.assertEqual(PRICING_PARTIAL, record.pricing_status)
    self.assertIsNone(record.estimated_total_cost_usd)

  def test_05_absent_pricing_is_unavailable_not_zero(self):
    record = self.records(self.event(), snapshot_marker=False)[0]
    self.assertEqual(PRICING_UNAVAILABLE, record.pricing_status)
    self.assertIsNone(record.pricing_snapshot_id)
    self.assertIsNone(record.estimated_total_cost_usd)

  def test_06_unknown_model_does_not_invent_price(self):
    record = self.records(self.event(model="unknown-model"))[0]
    self.assertEqual(PRICING_UNAVAILABLE, record.pricing_status)
    self.assertEqual("synthetic-test-v0", record.pricing_snapshot_id)
    self.assertIsNone(record.estimated_total_cost_usd)

  def test_07_cached_input_uses_separate_price(self):
    record = self.records(self.event(
      input_tokens=100, cached_input_tokens=75, output_tokens=0))[0]
    self.assertEqual(Decimal("0.000050000000"),
                     record.estimated_input_cost_usd)
    self.assertEqual(Decimal("0.000075000000"),
                     record.estimated_cached_input_cost_usd)

  def test_08_cached_input_greater_than_input_is_partial(self):
    record = self.records(self.event(
      input_tokens=10, cached_input_tokens=11))[0]
    self.assertEqual(PARTIAL, record.token_usage_status)
    self.assertEqual(PRICING_PARTIAL, record.pricing_status)
    self.assertIsNone(record.estimated_total_cost_usd)

  def test_09_reasoning_tokens_are_not_double_billed(self):
    baseline = self.records(self.event(reasoning_tokens=0))[0]
    reasoning = self.records(self.event(reasoning_tokens=90))[0]
    self.assertEqual(baseline.estimated_total_cost_usd,
                     reasoning.estimated_total_cost_usd)
    self.assertEqual(90, reasoning.reasoning_tokens)

  def test_10_physical_retries_are_derived_per_logical_call(self):
    events = (
      self.event(attempt=1, outcome=ERROR, input_tokens=None,
                 output_tokens=None, cached_input_tokens=None,
                 reasoning_tokens=None),
      self.event(attempt=2),
      self.event(call_id="call-2", attempt=1),
    )
    totals = summarize_cost_ledger(self.records(*events)).totals
    self.assertEqual((2, 3, 1),
                     (totals.logical_calls, totals.physical_attempts,
                      totals.retry_count))

  def test_11_failed_attempt_without_usage_is_not_applicable(self):
    record = self.records(self.event(
      outcome=ERROR, input_tokens=None, output_tokens=None,
      cached_input_tokens=None, reasoning_tokens=None))[0]
    self.assertEqual(NOT_APPLICABLE, record.token_usage_status)
    self.assertEqual("SyntheticError", record.error_type)
    self.assertIsNone(record.estimated_total_cost_usd)

  def test_12_failed_logical_call_is_counted(self):
    record = self.records(self.event(
      outcome=ERROR, input_tokens=None, output_tokens=None,
      cached_input_tokens=None, reasoning_tokens=None))[0]
    totals = summarize_cost_ledger((record,)).totals
    self.assertEqual((1, 1, 0, 1), (
      totals.logical_calls, totals.physical_attempts,
      totals.successful_attempts, totals.failed_attempts))

  def test_13_multiple_models_have_independent_breakdowns(self):
    summary = summarize_cost_ledger(self.records(
      self.event(), self.event(call_id="call-2", model="other-chat")))
    self.assertEqual(["other-chat", "test-chat"],
                     list(self.breakdown(summary, "by_model")))

  def test_14_multiple_operations_have_independent_breakdowns(self):
    summary = summarize_cost_ledger(self.records(
      self.event(), self.event(
        call_id="call-2", operation=EMBEDDING,
        model="test-embedding", input_tokens=10, output_tokens=None,
        cached_input_tokens=None, reasoning_tokens=None)))
    self.assertEqual([CHAT, EMBEDDING],
                     list(self.breakdown(summary, "by_operation")))

  def test_15_embedding_category_reuses_existing_logical_events(self):
    logical = EmbeddingLogicalEvent(
      "call-1", "test-embedding", "fake", "RETRIEVAL", "MISS", "fp")
    record = self.records(self.event(
      operation=EMBEDDING, model="test-embedding", output_tokens=None,
      cached_input_tokens=None, reasoning_tokens=None),
      embedding_logical_events=(logical,))[0]
    self.assertEqual("RETRIEVAL", record.cognitive_category)

  def test_16_absent_context_is_preserved_as_unspecified_breakdown(self):
    summary = summarize_cost_ledger(self.records(self.event()))
    self.assertEqual(["UNSPECIFIED"],
                     list(self.breakdown(summary, "by_actor")))
    self.assertIsNone(self.records(self.event())[0].actor_id)

  def test_17_present_context_is_attached_without_provider_changes(self):
    context = CostLedgerContext(
      simulation_id="sim-1", simulation_step=7, actor_id="actor-1",
      cognitive_category="PLANNING")
    with use_cost_ledger_context(context):
      record = self.records(self.event())[0]
    self.assertEqual(("sim-1", 7, "actor-1", "PLANNING"), (
      record.simulation_id, record.simulation_step, record.actor_id,
      record.cognitive_category))

  def test_18_context_nesting_and_exception_reset(self):
    outer = CostLedgerContext(actor_id="outer")
    inner = CostLedgerContext(actor_id="inner")
    with use_cost_ledger_context(outer):
      self.assertEqual("outer", get_cost_ledger_context().actor_id)
      with self.assertRaises(RuntimeError):
        with use_cost_ledger_context(inner):
          self.assertEqual("inner", get_cost_ledger_context().actor_id)
          raise RuntimeError("synthetic")
      self.assertEqual("outer", get_cost_ledger_context().actor_id)
    self.assertEqual(CostLedgerContext(), get_cost_ledger_context())

  def test_19_decimal_precision_covers_small_and_million_token_costs(self):
    tiny_snapshot = PricingSnapshot(
      "tiny", 1, "USD", "2026-08-05",
      (ModelPricing(
        "test-chat", input_per_million=Decimal("0.000001"),
        output_per_million=Decimal("0")),), "synthetic")
    tiny = build_cost_ledger_records((self.event(
      input_tokens=1, output_tokens=0, cached_input_tokens=None),),
      tiny_snapshot)[0]
    million = self.records(self.event(
      input_tokens=1000000, output_tokens=0,
      cached_input_tokens=None))[0]
    self.assertEqual(Decimal("0.000000000001"),
                     tiny.estimated_total_cost_usd)
    self.assertEqual(Decimal("2.000000000000"),
                     million.estimated_total_cost_usd)
    with self.assertRaises(TypeError):
      ModelPricing("bad", input_per_million=1.5)

  def test_20_many_record_sum_is_exact_decimal(self):
    records = self.records(*(self.event(
      call_id=f"call-{index}", input_tokens=1, output_tokens=0,
      cached_input_tokens=None) for index in range(1000)))
    total = summarize_cost_ledger(records).totals.estimated_total_cost_usd
    self.assertEqual(Decimal("0.002000000000"), total)

  def test_21_json_export_is_deterministic_and_decimal_safe(self):
    summary = summarize_cost_ledger(self.records(
      self.event(model="other-chat"),
      self.event(call_id="call-2", model="test-chat")))
    first = json.dumps(cost_ledger_summary_to_dict(summary), sort_keys=True)
    second = json.dumps(cost_ledger_summary_to_dict(summary), sort_keys=True)
    self.assertEqual(first, second)
    self.assertIn('"estimated_total_cost_usd": "', first)
    self.assertLess(first.index("other-chat"), first.index("test-chat"))

  def test_22_summary_excludes_sensitive_content(self):
    secret_values = (
      "TOP-SECRET-PROMPT", "TOP-SECRET-OUTPUT", "sk-secret-key",
      "private transcript", "embedding-vector-secret")
    event = self.event(input_fingerprint="safe-fingerprint")
    serialized = json.dumps(cost_ledger_summary_to_dict(
      summarize_cost_ledger(self.records(event))), sort_keys=True)
    for secret in secret_values:
      self.assertNotIn(secret, serialized)
    for forbidden_key in (
        "prompt", "messages", "response_text", "embedding", "authorization"):
      self.assertNotIn(f'"{forbidden_key}"', serialized.lower())

  def test_23_embedding_cache_metrics_reuse_existing_snapshot(self):
    measurement = {"global": {
      "logical_embedding_requests": 5,
      "physical_embedding_attempts": 3,
      "cache_hits": 2,
      "cache_misses": 3,
      "evictions": 1,
    }}
    cache = summarize_cost_ledger(
      (), embedding_measurement=measurement).embedding_cache
    self.assertEqual((5, 3, 2, 3, 1, 2), (
      cache.logical_embedding_requests, cache.physical_embedding_attempts,
      cache.cache_hits, cache.cache_misses, cache.evictions,
      cache.avoided_embedding_calls))
    self.assertEqual(Decimal("0.400000000000"), cache.cache_hit_rate)

  def test_24_avoided_embedding_cost_is_none_without_token_facts(self):
    event = EmbeddingLogicalEvent(
      "hit-1", "test-embedding", "fake", "RETRIEVAL", HIT, "fp")
    cache = summarize_cost_ledger(
      (), embedding_measurement={"global": {
        "logical_embedding_requests": 1, "physical_embedding_attempts": 0,
        "cache_hits": 1, "cache_misses": 0, "evictions": 0}},
      embedding_logical_events=(event,), pricing_snapshot=self.snapshot,
    ).embedding_cache
    self.assertIsNone(cache.estimated_embedding_cost_avoided_usd)

  def test_25_avoided_embedding_cost_requires_tokens_and_pricing(self):
    event = EmbeddingLogicalEvent(
      "hit-1", "test-embedding", "fake", "RETRIEVAL", HIT, "fp")
    cache = summarize_cost_ledger(
      (), embedding_measurement={"global": {
        "logical_embedding_requests": 1, "physical_embedding_attempts": 0,
        "cache_hits": 1, "cache_misses": 0, "evictions": 0}},
      embedding_logical_events=(event,), pricing_snapshot=self.snapshot,
      avoided_embedding_token_counts={"hit-1": 500},
    ).embedding_cache
    self.assertEqual(Decimal("0.000250000000"),
                     cache.estimated_embedding_cost_avoided_usd)

  def test_26_source_telemetry_is_not_mutated(self):
    event = self.event()
    before = repr(event)
    records = self.records(event)
    self.assertEqual(before, repr(event))
    self.assertEqual(1000, event.input_tokens)
    with self.assertRaises(FrozenInstanceError):
      records[0].input_tokens = 0

  def test_27_zero_tokens_produce_known_zero_cost(self):
    record = self.records(self.event(
      input_tokens=0, output_tokens=0, cached_input_tokens=0,
      reasoning_tokens=0))[0]
    self.assertEqual(COMPLETE, record.token_usage_status)
    self.assertEqual(Decimal("0E-12"), record.estimated_total_cost_usd)

  def test_28_missing_cached_rate_makes_positive_cached_usage_partial(self):
    snapshot = PricingSnapshot(
      "partial", 1, "USD", "2026-08-05",
      (ModelPricing(
        "test-chat", input_per_million=Decimal("2"),
        output_per_million=Decimal("4")),), "synthetic")
    record = build_cost_ledger_records((self.event(),), snapshot)[0]
    self.assertEqual(PRICING_PARTIAL, record.pricing_status)
    self.assertIsNone(record.estimated_total_cost_usd)
    self.assertIsNotNone(record.estimated_input_cost_usd)

  def test_29_orphan_attempt_is_not_misclassified_as_retry(self):
    orphan = {
      "operation": CHAT, "logical_call_id": None, "physical_attempt": 1,
      "model_or_engine": "test-chat", "outcome": ERROR,
      "elapsed_seconds": 0, "input_tokens": None, "output_tokens": None,
      "cached_input_tokens": None, "reasoning_tokens": None,
    }
    totals = summarize_cost_ledger(
      build_cost_ledger_records((orphan,), self.snapshot)).totals
    self.assertEqual((0, 1, 0),
                     (totals.logical_calls, totals.physical_attempts,
                      totals.retry_count))

  def test_30_metadata_is_preserved_without_content(self):
    record = self.records(self.event())[0]
    self.assertEqual(("req-test", "MODERN_OPENAI", "OPENAI_SDK_1_PLUS"), (
      record.request_id, record.provider_kind, record.transport_kind))
    self.assertEqual(Decimal("12.500"), record.elapsed_ms)
    self.assertFalse(hasattr(record, "input_fingerprint"))

  def test_31_required_synthetic_replay_fixture_totals(self):
    events = (
      self.event(attempt=1, outcome=ERROR, input_tokens=None,
                 output_tokens=None, cached_input_tokens=None,
                 reasoning_tokens=None),
      self.event(attempt=2),
      self.event(call_id="call-2", operation=EMBEDDING,
                 model="test-embedding", input_tokens=500,
                 output_tokens=None, cached_input_tokens=None,
                 reasoning_tokens=None),
      self.event(call_id="call-3", input_tokens=None, output_tokens=None,
                 cached_input_tokens=None, reasoning_tokens=None),
    )
    summary = summarize_cost_ledger(self.records(*events))
    self.assertEqual((3, 4, 1, 2, 2), (
      summary.totals.logical_calls, summary.totals.physical_attempts,
      summary.totals.retry_count, summary.totals.known_cost_record_count,
      summary.totals.unknown_cost_record_count))
    self.assertEqual(Decimal("0.002450000000"),
                     summary.totals.estimated_total_cost_usd)

  def test_32_pricing_snapshot_is_immutable_and_rejects_duplicates(self):
    with self.assertRaises(FrozenInstanceError):
      self.snapshot.snapshot_id = "changed"
    duplicate = ModelPricing(
      "same", input_per_million=Decimal("1"),
      output_per_million=Decimal("1"))
    with self.assertRaises(ValueError):
      PricingSnapshot(
        "duplicate", 1, "USD", "2026-08-05",
        (duplicate, duplicate), "synthetic")

  def test_33_context_resolver_drives_all_context_breakdowns(self):
    events = (
      self.event(call_id="call-1", outcome=SUCCESS),
      self.event(call_id="call-2", outcome=ERROR, input_tokens=None,
                 output_tokens=None, cached_input_tokens=None,
                 reasoning_tokens=None),
    )
    contexts = {
      "call-1": CostLedgerContext(
        "sim-a", 1, "actor-a", "PLANNING"),
      "call-2": CostLedgerContext(
        "sim-a", 2, "actor-b", "REFLECTION"),
    }
    records = build_cost_ledger_records(
      events, self.snapshot,
      context_resolver=lambda event: contexts[event.logical_call_id])
    summary = summarize_cost_ledger(records)
    self.assertEqual(["PLANNING", "REFLECTION"], list(
      self.breakdown(summary, "by_cognitive_category")))
    self.assertEqual(["actor-a", "actor-b"],
                     list(self.breakdown(summary, "by_actor")))
    self.assertEqual(["1", "2"],
                     list(self.breakdown(summary, "by_simulation_step")))
    self.assertEqual([ERROR, SUCCESS],
                     list(self.breakdown(summary, "by_outcome")))
    self.assertEqual(["MODERN_OPENAI"],
                     list(self.breakdown(summary, "by_provider")))

  def test_34_unavailable_usage_is_explicit_in_summary_and_json(self):
    record = self.records(self.event(
      input_tokens=None, output_tokens=None, cached_input_tokens=None,
      reasoning_tokens=None))[0]
    summary = summarize_cost_ledger((record,))
    self.assertEqual(TokenAggregate(0, 1), summary.totals.input_tokens)
    self.assertEqual(TokenAggregate(0, 1), summary.totals.output_tokens)
    self.assertEqual(TokenAggregate(0, 1), summary.totals.total_tokens)
    report = cost_ledger_summary_to_dict(summary)
    self.assertEqual(
      {"known_value": 0, "unknown_record_count": 1},
      report["totals"]["input_tokens"])

  def test_35_real_zero_tokens_remain_known_zero(self):
    record = self.records(self.event(
      input_tokens=0, output_tokens=0, cached_input_tokens=0,
      reasoning_tokens=0))[0]
    totals = summarize_cost_ledger((record,)).totals
    self.assertEqual(TokenAggregate(0, 0), totals.input_tokens)
    self.assertEqual(TokenAggregate(0, 0), totals.output_tokens)
    self.assertEqual(TokenAggregate(0, 0), totals.total_tokens)

  def test_36_real_zero_and_unknown_remain_distinguishable(self):
    records = self.records(
      self.event(call_id="known-zero", input_tokens=0, output_tokens=0,
                 cached_input_tokens=0, reasoning_tokens=0),
      self.event(call_id="unknown", input_tokens=None, output_tokens=None,
                 cached_input_tokens=None, reasoning_tokens=None))
    totals = summarize_cost_ledger(records).totals
    self.assertEqual(TokenAggregate(0, 1), totals.input_tokens)
    self.assertEqual(TokenAggregate(0, 1), totals.total_tokens)

  def test_37_every_breakdown_preserves_unknown_token_counts(self):
    record = self.records(self.event(
      input_tokens=None, output_tokens=None, cached_input_tokens=None,
      reasoning_tokens=None))[0]
    summary = summarize_cost_ledger((record,))
    for name in (
        "by_operation", "by_model", "by_provider", "by_outcome",
        "by_cognitive_category", "by_actor", "by_simulation_step",
        "by_pricing_snapshot"):
      with self.subTest(breakdown=name):
        aggregate = getattr(summary, name)[0][1]
        self.assertEqual(TokenAggregate(0, 1), aggregate.input_tokens)

  def test_38_string_input_token_is_removed_without_conversion(self):
    record = self.records(self.event(
      input_tokens="10", output_tokens=5, cached_input_tokens=None,
      reasoning_tokens=None))[0]
    self.assertEqual(PARTIAL, record.token_usage_status)
    self.assertIsNone(record.input_tokens)
    self.assertEqual(5, record.output_tokens)
    totals = summarize_cost_ledger((record,)).totals
    self.assertEqual(TokenAggregate(0, 1), totals.input_tokens)
    self.assertEqual(TokenAggregate(5, 0), totals.output_tokens)

  def test_39_float_output_token_is_removed(self):
    record = self.records(self.event(output_tokens=5.0))[0]
    self.assertEqual(PARTIAL, record.token_usage_status)
    self.assertIsNone(record.output_tokens)
    self.assertEqual(TokenAggregate(0, 1),
                     summarize_cost_ledger((record,)).totals.output_tokens)

  def test_40_boolean_cached_token_is_removed(self):
    record = self.records(self.event(cached_input_tokens=True))[0]
    self.assertEqual(PARTIAL, record.token_usage_status)
    self.assertIsNone(record.cached_input_tokens)
    self.assertEqual(TokenAggregate(0, 1), summarize_cost_ledger(
      (record,)).totals.cached_input_tokens)

  def test_41_negative_reasoning_token_is_removed(self):
    record = self.records(self.event(reasoning_tokens=-1))[0]
    self.assertEqual(PARTIAL, record.token_usage_status)
    self.assertIsNone(record.reasoning_tokens)
    self.assertEqual(TokenAggregate(0, 1), summarize_cost_ledger(
      (record,)).totals.reasoning_tokens)

  def test_42_mixed_valid_and_malformed_usage_preserves_valid_fields(self):
    record = self.records(self.event(
      input_tokens=10, output_tokens=[], cached_input_tokens=2,
      reasoning_tokens=3))[0]
    self.assertEqual(PARTIAL, record.token_usage_status)
    self.assertEqual((10, None, 2, 3), (
      record.input_tokens, record.output_tokens,
      record.cached_input_tokens, record.reasoning_tokens))
    totals = summarize_cost_ledger((record,)).totals
    self.assertEqual(TokenAggregate(10, 0), totals.input_tokens)
    self.assertEqual(TokenAggregate(0, 1), totals.output_tokens)

  def test_43_partial_malformed_records_never_break_aggregation(self):
    malformed = (
      self.event(call_id="one", input_tokens={}, output_tokens=5),
      self.event(call_id="two", input_tokens=4, output_tokens="bad"),
      self.event(call_id="three", cached_input_tokens=False),
    )
    records = self.records(*malformed)
    summary = summarize_cost_ledger(records)
    self.assertEqual(3, summary.totals.physical_attempts)
    self.assertTrue(all(record.token_usage_status == PARTIAL
                        for record in records))

  def test_44_cached_greater_than_input_keeps_valid_audit_values(self):
    record = self.records(self.event(
      input_tokens=10, output_tokens=5, cached_input_tokens=11))[0]
    self.assertEqual(PARTIAL, record.token_usage_status)
    self.assertEqual((10, 5, 11, 15), (
      record.input_tokens, record.output_tokens,
      record.cached_input_tokens, record.total_tokens))
    self.assertIsNone(record.estimated_total_cost_usd)
    totals = summarize_cost_ledger((record,)).totals
    self.assertEqual(TokenAggregate(11, 0), totals.cached_input_tokens)

  def test_45_record_validation_rejects_invalid_invariants(self):
    record = self.records(self.event())[0]
    invalid = (
      {"attempt": 0},
      {"elapsed_ms": Decimal("-0.001")},
      {"outcome": "UNKNOWN"},
      {"token_usage_status": "UNKNOWN"},
      {"input_tokens": True},
      {"estimated_total_cost_usd": Decimal("-0.01")},
    )
    for changes in invalid:
      with self.subTest(changes=changes), self.assertRaises(ValueError):
        replace(record, **changes)

  def test_46_token_and_ledger_aggregate_validation_rejects_bad_counts(self):
    with self.assertRaises(ValueError):
      TokenAggregate(-1, 0)
    with self.assertRaises(ValueError):
      TokenAggregate(0, True)
    totals = summarize_cost_ledger(self.records(self.event())).totals
    with self.assertRaises(ValueError):
      replace(totals, known_cost_record_count=0,
              unknown_cost_record_count=0)
    with self.assertRaises(ValueError):
      replace(totals, input_tokens=TokenAggregate(0, 2))
    with self.assertRaises(ValueError):
      replace(totals, retry_count=1)

  def test_47_negative_cache_counters_are_rejected(self):
    with self.assertRaises(ValueError):
      EmbeddingCacheLedgerSummary(
        logical_embedding_requests=-1,
        cache_hit_rate=Decimal("0"))

  def test_48_incoherent_cache_counters_are_rejected(self):
    invalid = (
      {"logical_embedding_requests": 1, "cache_hits": 2,
       "cache_misses": 0, "avoided_embedding_calls": 2},
      {"logical_embedding_requests": 2, "cache_hits": 1,
       "cache_misses": 0, "avoided_embedding_calls": 1},
      {"logical_embedding_requests": 1, "cache_hits": 1,
       "cache_misses": 0, "avoided_embedding_calls": 0},
      {"logical_embedding_requests": 2, "cache_hits": 1,
       "cache_misses": 1, "avoided_embedding_calls": 1},
    )
    for values in invalid:
      with self.subTest(values=values), self.assertRaises(ValueError):
        EmbeddingCacheLedgerSummary(
          cache_hit_rate=Decimal("0"), **values)
    retry_summary = EmbeddingCacheLedgerSummary(
      logical_embedding_requests=1, physical_embedding_attempts=2,
      cache_hits=0, cache_misses=1, cache_hit_rate=Decimal("0"))
    self.assertEqual(2, retry_summary.physical_embedding_attempts)

  def test_49_multiple_snapshots_have_explicit_breakdown(self):
    second = PricingSnapshot(
      "synthetic-test-v1", 1, "USD", "2026-08-06",
      (ModelPricing(
        "test-chat", input_per_million=Decimal("4"),
        cached_input_per_million=Decimal("2"),
        output_per_million=Decimal("8")),), "synthetic")
    first_records = self.records(self.event(call_id="first"))
    second_records = build_cost_ledger_records(
      (self.event(call_id="second"),), second)
    summary = summarize_cost_ledger(first_records + second_records)
    self.assertEqual("MULTIPLE", summary.pricing_snapshot_id)
    self.assertEqual(
      ["synthetic-test-v0", "synthetic-test-v1"],
      list(dict(summary.by_pricing_snapshot)))
    self.assertEqual(
      summary.totals.estimated_total_cost_usd,
      sum((aggregate.estimated_total_cost_usd
           for key, aggregate in summary.by_pricing_snapshot), Decimal("0")))

  def test_50_non_usd_pricing_is_rejected_before_aggregation(self):
    with self.assertRaises(ValueError):
      ModelPricing(
        "eur-model", currency="EUR", input_per_million=Decimal("1"))
    with self.assertRaises(ValueError):
      PricingSnapshot("eur", 1, "EUR", "2026-08-05", (), "synthetic")

  def test_51_privacy_sentinels_in_source_do_not_reach_report(self):
    source = {
      "operation": CHAT,
      "logical_call_id": "privacy",
      "physical_attempt": 1,
      "model_or_engine": "test-chat",
      "outcome": ERROR,
      "elapsed_seconds": 0,
      "input_tokens": None,
      "output_tokens": None,
      "cached_input_tokens": None,
      "reasoning_tokens": None,
      "input_fingerprint": "SECRET_FINGERPRINT_SENTINEL",
      "request_id": "SECRET_REQUEST_ID_SENTINEL",
      "error_type": "SECRET_ERROR_SENTINEL",
      "prompt": "SECRET_PROMPT_SENTINEL",
      "response_text": "SECRET_OUTPUT_SENTINEL",
      "embedding_vector": "SECRET_EMBEDDING_SENTINEL",
      "authorization": "SECRET_AUTHORIZATION_SENTINEL",
      "api_key": "SECRET_API_KEY_SENTINEL",
      "memory": "SECRET_MEMORY_SENTINEL",
      "transcript": "SECRET_TRANSCRIPT_SENTINEL",
    }
    report = json.dumps(cost_ledger_summary_to_dict(summarize_cost_ledger(
      build_cost_ledger_records((source,), self.snapshot))), sort_keys=True)
    for value in source.values():
      if isinstance(value, str) and value.startswith("SECRET_"):
        self.assertNotIn(value, report)

  def test_52_token_aggregate_json_is_deterministic(self):
    records = self.records(
      self.event(call_id="zero", input_tokens=0, output_tokens=0,
                 cached_input_tokens=0, reasoning_tokens=0),
      self.event(call_id="unknown", input_tokens=None, output_tokens=None,
                 cached_input_tokens=None, reasoning_tokens=None))
    summary = summarize_cost_ledger(records)
    first = cost_ledger_summary_to_dict(summary)
    second = cost_ledger_summary_to_dict(summary)
    self.assertEqual(first, second)
    json.dumps(first)
    self.assertEqual(
      {"known_value": 0, "unknown_record_count": 1},
      first["totals"]["total_tokens"])

  def test_53_build_and_summary_do_not_mutate_any_source_container(self):
    telemetry = [self.event()]
    measurement = {"global": {
      "logical_embedding_requests": 1,
      "physical_embedding_attempts": 1,
      "cache_hits": 0,
      "cache_misses": 1,
      "evictions": 0,
    }}
    telemetry_before = repr(telemetry)
    measurement_before = repr(measurement)
    records = build_cost_ledger_records(telemetry, self.snapshot)
    summarize_cost_ledger(records, embedding_measurement=measurement)
    self.assertEqual(telemetry_before, repr(telemetry))
    self.assertEqual(measurement_before, repr(measurement))

  def test_54_original_synthetic_replay_keeps_cost_and_call_totals(self):
    events = (
      self.event(attempt=1, outcome=ERROR, input_tokens=None,
                 output_tokens=None, cached_input_tokens=None,
                 reasoning_tokens=None),
      self.event(attempt=2),
      self.event(call_id="call-2", operation=EMBEDDING,
                 model="test-embedding", input_tokens=500,
                 output_tokens=None, cached_input_tokens=None,
                 reasoning_tokens=None),
      self.event(call_id="call-3", input_tokens=None, output_tokens=None,
                 cached_input_tokens=None, reasoning_tokens=None),
    )
    summary = summarize_cost_ledger(self.records(*events))
    self.assertEqual((3, 4, 1, 2, 2), (
      summary.totals.logical_calls, summary.totals.physical_attempts,
      summary.totals.retry_count, summary.totals.known_cost_record_count,
      summary.totals.unknown_cost_record_count))
    self.assertEqual(Decimal("0.002450000000"),
                     summary.totals.estimated_total_cost_usd)

  def test_55_summary_validation_rejects_unsorted_breakdown(self):
    records = self.records(
      self.event(call_id="chat"),
      self.event(call_id="embedding", operation=EMBEDDING,
                 model="test-embedding", input_tokens=1,
                 output_tokens=None, cached_input_tokens=None,
                 reasoning_tokens=None))
    summary = summarize_cost_ledger(records)
    with self.assertRaises(ValueError):
      replace(summary, by_operation=tuple(reversed(summary.by_operation)))
    with self.assertRaises(ValueError):
      replace(summary, pricing_snapshot_id="wrong-snapshot")


if __name__ == "__main__":
  unittest.main()
