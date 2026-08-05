from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import socket
import sys
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.prompt_template.cost_ledger import (
  CostLedgerContext,
  ModelPricing,
  PricingSnapshot,
  summarize_cost_ledger,
  use_cost_ledger_context,
)
from persona.prompt_template.llm_provider import (
  CHAT,
  COMPLETION_COMPAT,
  EMBEDDING,
  LLMReplayContext,
  chat_completion,
  clear_telemetry,
  completion_compat,
  embedding,
  get_telemetry,
  logical_call,
  reset_embedding_cache,
  text_completion,
  use_chat_provider,
  use_completion_compat_provider,
  use_completion_provider,
  use_embedding_provider,
  use_llm_replay_context,
)
from persona.prompt_template.replay_cost_guard import (
  ReplayCostAccountingUnavailableError,
  ReplayCostCeiling,
  ReplayCostCeilingExceededError,
  ReplayCostConcurrentAttemptError,
  ReplayCostContextMismatchError,
  ReplayCostGuardAlreadyTrippedError,
  ReplayCostGuardConfig,
  ReplayCostGuardSnapshot,
  ReplayCostGuardState,
  get_replay_cost_guard,
  install_replay_cost_guard,
  reset_replay_cost_guard,
  use_replay_cost_guard,
)


CHAT_MODEL = "r0-chat"
COMPAT_MODEL = "r0-completion-compat"
EMBEDDING_MODEL = "text-embedding-ada-002"
SIMULATION_ID = "ego-vivens-lab-01"


class RecordingUsageProvider:
  provider_identity = "r0-offline-provider"
  embedding_space_provider = "openai"
  provider_kind = "MODERN_OPENAI"
  transport_kind = "OPENAI_SDK_1_PLUS"

  def __init__(self, usage=None, fail=False):
    self.usage = list(usage or [])
    self.fail = fail
    self.calls = []
    self._metadata = None

  def _record(self, operation, model):
    self.calls.append((operation, model))
    if self.fail:
      raise RuntimeError("synthetic provider failure")
    values = (self.usage.pop(0) if self.usage else
              ((1, None) if operation == EMBEDDING else (1, 1)))
    input_tokens, output_tokens = values
    self._metadata = SimpleNamespace(
      request_id=f"req-{len(self.calls)}", response_model=model,
      finish_reason="stop", response_status="completed",
      input_tokens=input_tokens, output_tokens=output_tokens,
      cached_input_tokens=None, reasoning_tokens=None)

  def chat_completion(self, *, model, messages, **kwargs):
    operation = (COMPLETION_COMPAT if model == COMPAT_MODEL else CHAT)
    self._record(operation, model)
    return {"choices": [{"message": {"content": "synthetic"}}]}

  def embedding(self, *, model, input):
    self._record(EMBEDDING, model)
    return {"data": [{"embedding": [1.0] + [0.0] * 1535}]}

  def text_completion(self, **kwargs):
    self._record("COMPLETION", kwargs.get("model", ""))
    return SimpleNamespace(choices=[SimpleNamespace(text="synthetic")])

  def consume_response_metadata(self):
    metadata = self._metadata
    self._metadata = None
    return metadata


class ReplayCostGuardTests(unittest.TestCase):
  def setUp(self):
    clear_telemetry()
    reset_embedding_cache()
    self.pricing = PricingSnapshot(
      snapshot_id="r0g-synthetic-v0", schema_version=1, currency="USD",
      created_at="stable", source_note="offline synthetic pricing",
      models=(
        ModelPricing(
          CHAT_MODEL, input_per_million=Decimal("1000000"),
          output_per_million=Decimal("1000000")),
        ModelPricing(
          COMPAT_MODEL, input_per_million=Decimal("1000000"),
          output_per_million=Decimal("1000000")),
        ModelPricing(
          EMBEDDING_MODEL,
          embedding_input_per_million=Decimal("1000000")),
      ))

  def tearDown(self):
    clear_telemetry()
    reset_embedding_cache()
    self.assertIsNone(get_replay_cost_guard())

  def config(self, maximum="10", replay_id="controlled-replay-v0-test",
             simulation_id=SIMULATION_ID):
    return ReplayCostGuardConfig(
      replay_id, simulation_id, ReplayCostCeiling(Decimal(maximum)),
      self.pricing)

  def context(self, actor="actor-A", step=0, simulation_id=SIMULATION_ID,
              caller="r0-orchestrator", category="PLANNING"):
    return LLMReplayContext(
      caller_id=caller, cognitive_category=category,
      actor_id=actor, simulation_id=simulation_id, simulation_step=step)

  def providers(self, provider):
    stack = ExitStack()
    stack.enter_context(use_chat_provider(provider))
    stack.enter_context(use_completion_compat_provider(provider))
    stack.enter_context(use_completion_provider(provider))
    stack.enter_context(use_embedding_provider(provider))
    return stack

  def chat(self):
    return chat_completion(
      model=CHAT_MODEL, messages=[{"role": "user", "content": "private"}])

  def compat(self):
    return completion_compat(
      model=COMPAT_MODEL, prompt="private", temperature=0,
      max_tokens=5, top_p=1, frequency_penalty=0,
      presence_penalty=0, stop=None)

  def embed(self, text="private embedding"):
    return embedding(input=[text], model=EMBEDDING_MODEL)

  def test_01_ceiling_validation_is_decimal_only_and_positive(self):
    for value in (0, -1, True, False, 1.0, "1", None):
      with self.subTest(value=repr(value)):
        with self.assertRaises((TypeError, ValueError)):
          ReplayCostCeiling(value)
    for value in (Decimal("0"), Decimal("-1"), Decimal("NaN"),
                  Decimal("Infinity"), Decimal("-Infinity")):
      with self.subTest(value=str(value)):
        with self.assertRaises(ValueError):
          ReplayCostCeiling(value)
    ceiling = ReplayCostCeiling(Decimal("0.0000000000001"))
    self.assertEqual(Decimal("0.0000000000001"), ceiling.maximum_cost)
    with self.assertRaises(FrozenInstanceError):
      ceiling.maximum_cost = Decimal("2")

  def test_02_config_requires_explicit_replay_identity_and_pricing(self):
    for replay_id, simulation_id in (("", SIMULATION_ID), ("   ", SIMULATION_ID),
                                     ("r0", ""), ("r0", "   ")):
      with self.subTest(replay_id=replay_id, simulation_id=simulation_id):
        with self.assertRaises(ValueError):
          self.config(replay_id=replay_id, simulation_id=simulation_id)
    with self.assertRaises(TypeError):
      ReplayCostGuardConfig("r0", SIMULATION_ID, object(), self.pricing)
    with self.assertRaises(TypeError):
      ReplayCostGuardConfig(
        "r0", SIMULATION_ID, ReplayCostCeiling(Decimal("1")), object())

  def test_03_default_off_preserves_all_three_operations(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()):
      self.chat()
      self.compat()
      self.embed()
    self.assertIsNone(get_replay_cost_guard())
    self.assertEqual(3, len(provider.calls))

  def test_04_under_ceiling_aggregates_all_operations(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config("6"))) as state:
      self.chat()
      self.compat()
      self.embed()
      snapshot = state.snapshot()
    self.assertEqual(Decimal("5.000000000000"), snapshot.accumulated_cost)
    self.assertEqual(Decimal("1.000000000000"), snapshot.remaining_cost)
    self.assertEqual(3, snapshot.logical_calls)
    self.assertEqual(3, snapshot.physical_attempts)
    self.assertEqual({
      CHAT: Decimal("2.000000000000"),
      COMPLETION_COMPAT: Decimal("2.000000000000"),
      EMBEDDING: Decimal("1.000000000000"),
    }, dict(snapshot.cost_by_operation))
    self.assertFalse(snapshot.tripped)

  def test_05_exact_ceiling_completes_then_blocks_pre_provider(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config("2"))) as state:
      self.chat()
      self.assertTrue(state.snapshot().tripped)
      with self.assertRaises(ReplayCostGuardAlreadyTrippedError):
        self.embed()
    self.assertEqual(1, len(provider.calls))

  def test_06_exceeded_ceiling_raises_post_call_and_stays_tripped(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config("0.5"))) as state:
      with self.assertRaises(ReplayCostCeilingExceededError):
        self.embed()
      self.assertEqual(Decimal("1.000000000000"),
                       state.snapshot().accumulated_cost)
      with self.assertRaises(ReplayCostGuardAlreadyTrippedError):
        self.chat()
    self.assertEqual(1, len(provider.calls))

  def test_07_completion_trip_blocks_embedding_cross_operation(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config("1"))):
      with self.assertRaises(ReplayCostCeilingExceededError):
        self.compat()
      with self.assertRaises(ReplayCostGuardAlreadyTrippedError):
        self.embed()
    self.assertEqual([(COMPLETION_COMPAT, COMPAT_MODEL)], provider.calls)

  def test_08_embedding_trip_blocks_chat_cross_operation(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config("0.5"))):
      with self.assertRaises(ReplayCostCeilingExceededError):
        self.embed()
      with self.assertRaises(ReplayCostGuardAlreadyTrippedError):
        self.chat()
    self.assertEqual([(EMBEDDING, EMBEDDING_MODEL)], provider.calls)

  def test_09_each_semantic_retry_attempt_consumes_budget(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config("5"))) as state, logical_call():
      self.chat()
      self.chat()
    snapshot = state.snapshot()
    self.assertEqual(1, snapshot.logical_calls)
    self.assertEqual(2, snapshot.physical_attempts)
    self.assertEqual(Decimal("4.000000000000"), snapshot.accumulated_cost)

  def test_10_unknown_usage_trips_fail_closed(self):
    provider = RecordingUsageProvider(usage=[(None, None)])
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config())) as state:
      with self.assertRaises(ReplayCostAccountingUnavailableError):
        self.chat()
      self.assertTrue(state.snapshot().tripped)
      with self.assertRaises(ReplayCostGuardAlreadyTrippedError):
        self.embed()
    self.assertEqual(1, len(provider.calls))

  def test_11_malformed_usage_trips_fail_closed(self):
    provider = RecordingUsageProvider(usage=[(-1, 1)])
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config())) as state:
      with self.assertRaises(ReplayCostAccountingUnavailableError):
        self.chat()
      self.assertTrue(state.snapshot().tripped)

  def test_12_provider_error_without_usage_trips_fail_closed(self):
    provider = RecordingUsageProvider(fail=True)
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config())) as state:
      with self.assertRaises(ReplayCostAccountingUnavailableError):
        self.chat()
      self.assertTrue(state.snapshot().tripped)

  def test_13_context_nesting_and_exception_restore_outer_state(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config("10", replay_id="outer"))) as outer:
      self.chat()
      with self.assertRaises(RuntimeError):
        with use_replay_cost_guard(
            self.config("10", replay_id="inner")) as inner:
          self.embed("inner-unique")
          self.assertIs(inner, get_replay_cost_guard())
          raise RuntimeError("synthetic")
      self.assertIs(outer, get_replay_cost_guard())
      self.assertEqual(Decimal("2.000000000000"),
                       outer.snapshot().accumulated_cost)
    self.assertIsNone(get_replay_cost_guard())

  def test_14_direct_install_get_and_reset(self):
    installation = install_replay_cost_guard(self.config())
    self.assertIs(installation.state, get_replay_cost_guard())
    reset_replay_cost_guard(installation)
    self.assertIsNone(get_replay_cost_guard())

  def test_15_distinct_replays_do_not_share_cost(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()):
      with use_replay_cost_guard(self.config(replay_id="R0-A")) as first:
        self.chat()
        first_snapshot = first.snapshot()
      with use_replay_cost_guard(self.config(replay_id="R0-B")) as second:
        self.embed("replay-b-unique")
        second_snapshot = second.snapshot()
    self.assertEqual(Decimal("2.000000000000"),
                     first_snapshot.accumulated_cost)
    self.assertEqual(Decimal("1.000000000000"),
                     second_snapshot.accumulated_cost)

  def test_16_actor_and_step_changes_share_one_budget(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_replay_cost_guard(self.config()) as state:
      with use_llm_replay_context(self.context("actor-A", 0)):
        self.chat()
      with use_llm_replay_context(self.context("actor-B", 1)):
        self.embed()
    self.assertEqual(Decimal("3.000000000000"),
                     state.snapshot().accumulated_cost)
    records = state.records()
    self.assertEqual(["actor-A", "actor-B"], [r.actor_id for r in records])
    self.assertEqual([0, 1], [r.simulation_step for r in records])

  def test_17_context_mismatch_hard_fails_before_provider(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(
        self.context(simulation_id="wrong")), use_replay_cost_guard(
          self.config()) as state:
      with self.assertRaises(ReplayCostContextMismatchError):
        self.chat()
      self.assertTrue(state.snapshot().tripped)
    self.assertEqual([], provider.calls)

  def test_18_snapshot_errors_and_records_are_content_private(self):
    provider = RecordingUsageProvider()
    secret = "PRIVATE-PROMPT-TRANSCRIPT-MEMORY"
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config("1"))) as state:
      with self.assertRaises(ReplayCostCeilingExceededError) as raised:
        chat_completion(
          model=CHAT_MODEL, messages=[{"role": "user", "content": secret}])
      exported = repr((state.snapshot(), state.records(), raised.exception))
    self.assertNotIn(secret, exported)
    self.assertNotIn("messages", exported.lower())
    snapshot = state.snapshot()
    self.assertIsInstance(snapshot, ReplayCostGuardSnapshot)
    with self.assertRaises(FrozenInstanceError):
      snapshot.tripped = False

  def test_19_synthetic_two_actor_replay_trips_on_third_operation(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_replay_cost_guard(
        self.config("4", replay_id="controlled-replay-v0-test")) as state:
      with use_llm_replay_context(self.context("actor-A", 0)):
        self.compat()
        self.embed("integration-embedding")
      with use_llm_replay_context(self.context("actor-B", 1)):
        with self.assertRaises(ReplayCostCeilingExceededError):
          self.chat()
        with self.assertRaises(ReplayCostGuardAlreadyTrippedError):
          self.embed("forbidden-after-trip")
    snapshot = state.snapshot()
    self.assertEqual(Decimal("5.000000000000"), snapshot.accumulated_cost)
    self.assertEqual(3, snapshot.physical_attempts)
    self.assertEqual(3, len(provider.calls))
    self.assertEqual(SIMULATION_ID, snapshot.simulation_id)

  def test_20_guard_total_is_exactly_ledger_total(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config())) as state:
      self.chat()
      self.embed()
    ledger_total = summarize_cost_ledger(
      state.records()).totals.estimated_total_cost_usd
    self.assertEqual(ledger_total, state.snapshot().accumulated_cost)

  def test_21_no_network_is_used(self):
    provider = RecordingUsageProvider()
    with patch.object(socket, "getaddrinfo",
                      side_effect=AssertionError("DNS reached")), patch.object(
        socket, "create_connection",
        side_effect=AssertionError("socket reached")), self.providers(
          provider), use_llm_replay_context(self.context()), (
          use_replay_cost_guard(self.config())):
      self.chat()

  def test_22_parallel_attempt_is_rejected_before_provider(self):
    state = ReplayCostGuardState(self.config())
    context = self.context()
    state.before_attempt(
      operation=CHAT, model=CHAT_MODEL, logical_call_id="call-1",
      physical_attempt=1, replay_context=context)
    with self.assertRaises(ReplayCostConcurrentAttemptError):
      state.before_attempt(
        operation=EMBEDDING, model=EMBEDDING_MODEL,
        logical_call_id="call-2", physical_attempt=1,
        replay_context=context)

  def test_23_missing_pricing_trips_fail_closed(self):
    pricing = PricingSnapshot(
      snapshot_id="missing-chat", schema_version=1, currency="USD",
      created_at="stable", source_note="offline", models=())
    config = ReplayCostGuardConfig(
      "r0-missing-pricing", SIMULATION_ID,
      ReplayCostCeiling(Decimal("10")), pricing)
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(config)) as state:
      with self.assertRaises(ReplayCostAccountingUnavailableError):
        self.chat()
      self.assertTrue(state.snapshot().tripped)

  def test_24_legacy_completion_is_blocked_pre_provider(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config())) as state:
      with self.assertRaises(ReplayCostAccountingUnavailableError):
        text_completion(
          model="text-davinci-003", prompt="private", temperature=0,
          max_tokens=5, top_p=1, frequency_penalty=0,
          presence_penalty=0, stream=False, stop=None)
      self.assertTrue(state.snapshot().tripped)
    self.assertEqual([], provider.calls)

  def test_25_caller_none_preserves_event_bound_attribution(self):
    provider = RecordingUsageProvider()
    context = self.context(
      actor="actor-no-caller", step=0, caller=None, category="PLANNING")
    with self.providers(provider), use_llm_replay_context(context), (
        use_replay_cost_guard(self.config())) as state:
      self.chat()
    record = state.records()[0]
    self.assertIsNone(record.caller_id)
    self.assertEqual("PLANNING", record.cognitive_category)
    self.assertEqual("actor-no-caller", record.actor_id)
    self.assertEqual(SIMULATION_ID, record.simulation_id)
    self.assertEqual(0, record.simulation_step)

  def test_26_late_attribution_cannot_override_event_bound_none(self):
    provider = RecordingUsageProvider()
    call_context = self.context(
      actor="actor-A", step=0, caller=None, category="PLANNING")
    late_context = CostLedgerContext(
      caller_id="late-caller", cognitive_category="REFLECTION",
      actor_id="actor-B", simulation_id="other-simulation",
      simulation_step=99)
    with self.providers(provider), use_llm_replay_context(call_context), (
        use_cost_ledger_context(late_context)), use_replay_cost_guard(
          self.config()) as state:
      self.chat()
    record = state.records()[0]
    self.assertEqual((None, "PLANNING", "actor-A", SIMULATION_ID, 0), (
      record.caller_id, record.cognitive_category, record.actor_id,
      record.simulation_id, record.simulation_step))

  def test_27_mixed_event_bound_attribution_is_field_exact(self):
    fixtures = (
      (None, "PLANNING", "actor-A", SIMULATION_ID, 0),
      ("authorized-caller", None, "actor-A", SIMULATION_ID, 1),
      (None, None, None, SIMULATION_ID, None),
      ("authorized-caller", "REFLECTION", None, SIMULATION_ID, 2),
    )
    for caller, category, actor, simulation, step in fixtures:
      with self.subTest(fixture=repr(
          (caller, category, actor, simulation, step))):
        provider = RecordingUsageProvider()
        context = self.context(
          actor=actor, step=step, simulation_id=simulation,
          caller=caller, category=category)
        with self.providers(provider), use_llm_replay_context(context), (
            use_replay_cost_guard(self.config())) as state:
          self.chat()
        record = state.records()[0]
        self.assertEqual((caller, category, actor, simulation, step), (
          record.caller_id, record.cognitive_category, record.actor_id,
          record.simulation_id, record.simulation_step))

  def test_28_partial_usage_trips_without_inventing_zero_cost(self):
    provider = RecordingUsageProvider(usage=[(1, None)])
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config())) as state:
      with self.assertRaises(ReplayCostAccountingUnavailableError):
        self.chat()
      snapshot = state.snapshot()
      self.assertTrue(snapshot.tripped)
      self.assertEqual(1, snapshot.physical_attempts)
      with self.assertRaises(ReplayCostGuardAlreadyTrippedError):
        self.compat()
    self.assertEqual(1, len(provider.calls))

  def test_29_embedding_cache_hit_has_no_physical_cost(self):
    provider = RecordingUsageProvider()
    with self.providers(provider), use_llm_replay_context(self.context()), (
        use_replay_cost_guard(self.config())) as state:
      self.embed("cache-hit-regression")
      before = state.snapshot()
      self.embed("cache-hit-regression")
      after = state.snapshot()
    self.assertEqual(1, len(provider.calls))
    self.assertEqual(before.physical_attempts, after.physical_attempts)
    self.assertEqual(before.accumulated_cost, after.accumulated_cost)


if __name__ == "__main__":
  unittest.main()
