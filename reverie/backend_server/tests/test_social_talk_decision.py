"""Offline cognitive-contract checks using real self identity and provider seams."""
import copy
from contextlib import ExitStack
from dataclasses import asdict, FrozenInstanceError, replace
import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
  sys.path.insert(0, str(BACKEND))

from controlled_replay import DeterministicReplayFakeAdapter
from maze import Maze
from persona.persona import Persona
from persona.memory_structures.scratch import Scratch
from persona.cognitive_modules import plan
from persona.cognitive_modules import social_talk_decision as talk
from persona.prompt_template import gpt_structure, modern_openai_provider as errors
from persona.prompt_template.completion_runtime import (
  build_modern_completion_runtime_config,
  COMPLETION_COMPAT_MODEL,
  use_modern_completion_runtime,
)
from persona.prompt_template.cost_ledger import (
  ModelPricing, PricingSnapshot, build_cost_ledger_records,
)
from persona.prompt_template.llm_provider import (
  clear_telemetry, get_telemetry, get_llm_replay_context,
  LLMReplayContext, use_llm_replay_context,
)
from persona.prompt_template.replay_cost_guard import (
  ReplayCostCeiling, ReplayCostCeilingExceededError,
  ReplayCostGuardConfig, use_replay_cost_guard,
)

STORAGE = BACKEND.parents[1] / "environment" / "frontend_server" / "storage"
CONTROL = STORAGE / "base_the_ville_n25"
TREATMENT = STORAGE / "base_the_ville_n25_cpi0_tom"
TOM = "Tom Moreno"
SAM = "Sam Moore"
NOW = datetime.datetime(2023, 2, 13, 13)
YES = '{"decision":"yes"}'
NO = '{"decision":"no"}'
Status = talk.TalkDecisionStatus


class SocialTalkDecisionTests(unittest.TestCase):
  def setUp(self):
    clear_telemetry()
    self.addCleanup(clear_telemetry)
    self.stack = ExitStack()
    self.addCleanup(self.stack.close)
    self.network = []

    def reject_network(*args, **kwargs):
      self.network.append("attempt")
      raise AssertionError("no network in cognitive-contract verification")

    for owner, name in ((socket, "create_connection"),
                        (socket.socket, "connect"), (socket.socket, "connect_ex")):
      self.stack.enter_context(patch.object(owner, name, reject_network))
    self.live_guards = [self.stack.enter_context(patch.object(
      errors.ModernOpenAIClientAdapter, method,
      side_effect=AssertionError("live provider forbidden")))
      for method in ("create_chat", "create_embedding")]
    self.tom = Persona(TOM, str(CONTROL / "personas" / TOM))
    self.sam = Persona(SAM, str(CONTROL / "personas" / SAM))
    for actor in (self.tom, self.sam):
      actor.scratch.curr_time = NOW
      actor.scratch.act_address = "the Ville:market:shop:counter"
      actor.scratch.act_description = "serving customers" if actor.name == TOM else "buying groceries"
      actor.scratch.act_event = (actor.name, "is", actor.scratch.act_description)
      actor.scratch.planned_path = []
    self.focus = {"curr_event": SimpleNamespace(subject=SAM, description="Sam is buying groceries"),
                  "events": [], "thoughts": []}
    self.personas = {TOM: self.tom, SAM: self.sam}
    self.context = LLMReplayContext(
      actor_id=TOM, simulation_id="talk-contract-offline", simulation_step=7,
      cognitive_category="CONVERSATION")
    self.stack.enter_context(use_llm_replay_context(self.context))
    previous_cwd = Path.cwd()
    try:
      os.chdir(BACKEND)
      self.maze = Maze("the_ville")
    finally:
      os.chdir(previous_cwd)

  def tearDown(self):
    self.assertEqual([], self.network)
    for guard in self.live_guards:
      guard.assert_not_called()

  def request(self):
    return talk.build_social_talk_request(self.tom, self.sam, self.focus)

  def execute(self, responses, request=None, retries=0):
    adapter = DeterministicReplayFakeAdapter(responses)
    config = build_modern_completion_runtime_config(application_retry_count=retries)
    with use_modern_completion_runtime(config, adapter):
      result = talk.decide_social_talk(request or self.request())
    return result, adapter

  def snapshot(self):
    return copy.deepcopy((vars(self.maze), vars(self.tom.scratch),
                          vars(self.sam.scratch), vars(self.tom.a_mem), vars(self.sam.a_mem)))

  def assert_snapshot(self, before):
    self.assertEqual(before, (vars(self.maze), vars(self.tom.scratch),
                             vars(self.sam.scratch), vars(self.tom.a_mem), vars(self.sam.a_mem)))

  def test_tom_control_treatment_exposes_real_self_identity_delta(self):
    paths = [source / "personas" / TOM / "bootstrap_memory" / "scratch.json"
             for source in (CONTROL, TREATMENT)]
    before_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    treatment_scratch = Scratch(str(paths[1]))
    treatment_scratch.curr_time = NOW
    treatment_scratch.act_description = self.tom.scratch.act_description
    treatment = SimpleNamespace(name=TOM, scratch=treatment_scratch, a_mem=self.tom.a_mem)
    control_request = self.request()
    treatment_request = talk.build_social_talk_request(treatment, self.sam, self.focus)
    self.assertEqual("rude, aggressive, energetic", self.tom.scratch.innate)
    self.assertEqual("rude, argumentative, easily provoked, energetic", treatment_scratch.innate)
    self.assertEqual({"initiator_identity"}, {
      key for key, value in asdict(control_request).items()
      if value != asdict(treatment_request)[key]})
    control_prompt = talk.render_social_talk_prompt(control_request)
    treatment_prompt = talk.render_social_talk_prompt(treatment_request)
    self.assertNotEqual(control_prompt, treatment_prompt)
    self.assertIn(treatment_scratch.innate, treatment_prompt)
    self.assertIn(treatment_scratch.learned, treatment_prompt)
    self.assertIn("long-standing problematic relationship with alcohol", treatment_prompt)
    self.assertIn("physical fight", treatment_prompt)
    self.assertIn("You don't like Sam Moore.", treatment_prompt)
    self.assertEqual(before_hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
    self.assertEqual((), get_telemetry())

  def test_target_private_identity_is_excluded_unless_known_locally(self):
    marker = "TARGET_PRIVATE_IDENTITY_3f91"
    self.sam.scratch.innate = marker
    self.sam.scratch.learned = marker
    self.sam.scratch.currently = marker
    with patch.object(self.sam.scratch, "get_str_iss",
                      side_effect=AssertionError("target private identity read")) as target_iss:
      request = self.request()
      self.assertNotIn(marker, repr(request))
      self.assertNotIn(marker, talk.render_social_talk_prompt(request))
      self.focus["thoughts"] = [SimpleNamespace(description=f"Tom remembers {marker}")]
      known_request = self.request()
      self.assertIn(marker, known_request.retrieved_thoughts[0])
      self.assertIn(marker, talk.render_social_talk_prompt(known_request))
      target_iss.assert_not_called()

  def test_required_context_is_checked_before_any_provider_attempt(self):
    request = self.request()
    invalid = {"initiator_identity": None, "initiator_name": "", "target_name": " ",
               "current_time": None, "initiator_activity": "", "target_activity": None,
               "retrieved_events": None, "retrieved_thoughts": (42,)}
    for field, value in invalid.items():
      with self.subTest(field=field):
        bad_request = replace(request, **{field: value})
        result, adapter = self.execute([YES], bad_request)
        self.assertEqual(Status.INVALID_CONTEXT, result.status)
        self.assertIsNone(result.decision)
        self.assertEqual(0, result.attempt_count)
        self.assertEqual([], adapter.calls)
        with self.assertRaises(talk.CognitiveDecisionUnavailableError):
          talk.consume_social_talk_result(bad_request, result)
    self.assertEqual((), get_telemetry())

  def test_missing_time_and_identity_builder_fail_as_context(self):
    self.tom.scratch.curr_time = None
    with patch.object(self.tom.scratch, "get_str_iss") as identity:
      result, adapter = self.execute([YES])
      identity.assert_not_called()
    self.assertEqual(Status.INVALID_CONTEXT, result.status)
    self.assertEqual([], adapter.calls)
    self.tom.scratch.curr_time = NOW
    with patch.object(self.tom.scratch, "get_str_iss", return_value=None):
      result, adapter = self.execute([YES])
    self.assertEqual(Status.INVALID_CONTEXT, result.status)
    self.assertEqual([], adapter.calls)

  def test_malformed_retrieved_context_does_not_become_empty_memory(self):
    for retrieved in (None, {}, {"events": "text", "thoughts": []},
                      {"events": [SimpleNamespace()], "thoughts": []},
                      {"events": [], "thoughts": [SimpleNamespace(description=None)]}):
      with self.subTest(retrieved=retrieved):
        request = talk.build_social_talk_request(self.tom, self.sam, retrieved)
        result, adapter = self.execute([YES], request)
        self.assertEqual(Status.INVALID_CONTEXT, result.status)
        self.assertEqual([], adapter.calls)

  def test_optional_empty_memory_and_absent_last_chat_are_valid(self):
    request = self.request()
    self.assertEqual((), request.retrieved_events)
    self.assertEqual((), request.retrieved_thoughts)
    self.assertIsNone(request.last_chat_description)
    self.assertIsNone(talk.validate_social_talk_context(request))
    with patch.object(self.tom.a_mem, "get_last_chat", return_value=SimpleNamespace(
        created=NOW - datetime.timedelta(days=1), description="Tom recalls a greeting")):
      request = self.request()
    self.assertIn("Tom recalls a greeting", talk.render_social_talk_prompt(request))
    self.assertIsNone(talk.validate_social_talk_context(request))
    with patch.object(self.tom.a_mem, "get_last_chat", return_value=SimpleNamespace()):
      self.assertEqual("INVALID_LAST_CHAT_CONTEXT", talk.validate_social_talk_context(self.request()))

  def test_strict_output_schema_rejects_legacy_prose_and_ambiguity(self):
    for text in ("yes", "no", 'Answer in yes or no: yes',
                 'Answer in "yes" or "no": yes', '```json\n' + YES + '\n```',
                 '{}', '[]', 'null', '"yes"', '{"decision":true}',
                 '{"decision":null}', '{"decision":"YES"}',
                 '{"decision":"yes","reason":"because"}',
                 '{"decision":"no","decision":"yes"}',
                 '{"decision":"yes"} trailing', '{"decision":',
                 '{"decision":["yes"]}', '{"decision":NaN}'):
      with self.subTest(text=text):
        with self.assertRaises(ValueError):
          talk.parse_social_talk_output(text)
    self.assertEqual(talk.TalkDecision.TALK, talk.parse_social_talk_output(" \n" + YES))
    self.assertEqual(talk.TalkDecision.DO_NOT_TALK, talk.parse_social_talk_output(NO))

  def test_valid_yes_and_no_are_typed_success_without_side_effects(self):
    for output, expected in ((YES, talk.TalkDecision.TALK), (NO, talk.TalkDecision.DO_NOT_TALK)):
      with self.subTest(output=output):
        before = self.snapshot()
        rng = random.getstate()
        numpy_rng = numpy.random.get_state()
        result, adapter = self.execute([output])
        self.assertEqual(Status.SUCCESS, result.status)
        self.assertEqual(expected, result.decision)
        self.assertEqual(1, result.attempt_count)
        self.assertEqual(1, len(adapter.calls))
        self.assert_snapshot(before)
        self.assertEqual(rng, random.getstate())
        numpy.testing.assert_array_equal(numpy_rng[1], numpy.random.get_state()[1])
        self.assertEqual(numpy_rng[2:], numpy.random.get_state()[2:])
        self.assertEqual(self.context, get_llm_replay_context())
        with self.assertRaises(FrozenInstanceError):
          result.decision = None

  def test_successful_decision_is_consumed_by_real_social_selection(self):
    for output, expected in ((YES, f"chat with {SAM}"), (NO, False)):
      with self.subTest(output=output):
        adapter = DeterministicReplayFakeAdapter([output])
        with use_modern_completion_runtime(build_modern_completion_runtime_config(), adapter):
          self.assertEqual(expected, plan._should_react(self.tom, self.focus, self.personas))
        self.assertEqual(1, len(adapter.calls))

  def test_output_repair_preserves_one_logical_call_and_same_prompt(self):
    result, adapter = self.execute(["invalid", YES])
    self.assertEqual(Status.SUCCESS, result.status)
    self.assertEqual(2, result.attempt_count)
    events = get_telemetry()
    self.assertEqual([1, 2], [event.physical_attempt for event in events])
    self.assertEqual(1, len({event.logical_call_id for event in events}))
    self.assertEqual(adapter.calls[0][1], adapter.calls[1][1])

  def _assert_plan_failure(self, responses, status, count):
    adapter = DeterministicReplayFakeAdapter(responses)
    before = self.snapshot()
    # Isolate earlier daily/action planning and focus selection; execute the
    # real plan -> _should_react -> generate_decide_to_talk -> provider path.
    with ExitStack() as stack:
      stack.enter_context(patch.object(self.tom.scratch, "act_check_finished", return_value=False))
      stack.enter_context(patch.object(plan, "_choose_retrieved", return_value=self.focus))
      guards = [stack.enter_context(patch.object(plan, method)) for method in (
        "_long_term_planning", "_determine_action", "_chat_react", "_wait_react",
        "_create_react", "generate_new_decomp_schedule", "run_gpt_prompt_decide_to_talk")]
      guards.extend(stack.enter_context(patch.object(gpt_structure, method))
                    for method in ("safe_generate_response", "GPT_request"))
      stack.enter_context(use_modern_completion_runtime(build_modern_completion_runtime_config(), adapter))
      with self.assertRaises(talk.CognitiveDecisionUnavailableError) as raised:
        plan.plan(self.tom, self.maze, self.personas, False, {"opportunity": self.focus})
      error = raised.exception
      self.assertEqual(status, error.status)
      self.assertEqual("decide_to_talk", error.surface)
      self.assertEqual((TOM, SAM, count), (error.actor, error.target, error.attempt_count))
      self.assertNotIn("private provider text", str(error))
      for guard in guards:
        guard.assert_not_called()
    self.assertEqual(count, len(adapter.calls))
    self.assert_snapshot(before)

  def test_exhausted_legacy_output_stops_plan_without_fallback_or_chat(self):
    legacy = 'Answer in "yes" or "no": yes'
    result, adapter = self.execute([legacy] * 5)
    self.assertEqual(Status.INVALID_OUTPUT, result.status)
    self.assertIsNone(result.decision)
    self.assertEqual(5, result.attempt_count)
    self.assertEqual(5, len(adapter.calls))
    self._assert_plan_failure([legacy] * 5, Status.INVALID_OUTPUT, 5)

  def test_provider_errors_map_to_non_actor_results_and_stop_plan(self):
    for error_type, expected in (
        (errors.LLMRefusalError, Status.REFUSAL),
        (errors.LLMTimeoutError, Status.TIMEOUT),
        (errors.LLMConnectionError, Status.UNAVAILABLE),
        (errors.LLMRateLimitError, Status.UNAVAILABLE),
        (errors.LLMServerError, Status.UNAVAILABLE),
        (errors.LLMAuthenticationError, Status.PROVIDER_ERROR),
        (errors.LLMAuthorizationError, Status.PROVIDER_ERROR),
        (errors.LLMModelNotFoundError, Status.PROVIDER_ERROR),
        (errors.LLMInvalidRequestError, Status.PROVIDER_ERROR),
        (errors.LLMUnsupportedOperationError, Status.PROVIDER_ERROR),
        (errors.ModernOpenAISdkUnavailableError, Status.PROVIDER_ERROR)):
      with self.subTest(error=error_type.__name__):
        error = error_type("private provider text")
        result, adapter = self.execute([error])
        self.assertEqual(expected, result.status)
        self.assertIsNone(result.decision)
        self.assertEqual(1, len(adapter.calls))
        self.assertNotIn("private provider text", repr(result))
        self._assert_plan_failure([error], expected, 1)

  def test_incomplete_empty_malformed_provider_results_exhaust_as_invalid_output(self):
    for error_type in (errors.LLMIncompleteResponseError, errors.LLMEmptyOutputError,
                       errors.LLMMalformedResponseError):
      with self.subTest(error=error_type.__name__):
        result, adapter = self.execute([error_type("private provider text")] * 5)
        self.assertEqual(Status.INVALID_OUTPUT, result.status)
        self.assertEqual(error_type.__name__, result.reason)
        self.assertIsNone(result.decision)
        self.assertEqual(5, len(adapter.calls))
    # Empty returned text also traverses provider normalization and validation.
    result, adapter = self.execute([""] * 5)
    self.assertEqual(Status.INVALID_OUTPUT, result.status)
    self.assertEqual(5, len(adapter.calls))

  def test_invalid_context_stops_plan_before_provider_or_consequences(self):
    with patch.object(self.tom.scratch, "get_str_iss", return_value=None):
      self._assert_plan_failure([YES], Status.INVALID_CONTEXT, 0)

  def test_inactive_runtime_is_provider_error_without_legacy_invocation(self):
    with patch.object(gpt_structure, "GPT_request") as legacy:
      result = talk.decide_social_talk(self.request())
    legacy.assert_not_called()
    self.assertEqual(Status.PROVIDER_ERROR, result.status)
    self.assertEqual("ModernCompletionCompatInactiveError", result.reason)
    self.assertIsNone(result.decision)
    self.assertEqual((), get_telemetry())

  def test_unexpected_programming_exceptions_escape(self):
    for error in (TypeError("defect"), AssertionError("defect"), RuntimeError("defect")):
      with self.subTest(error=type(error).__name__):
        with self.assertRaises(type(error)):
          self.execute([error])
    with patch.object(self.tom.scratch, "get_str_iss", side_effect=ValueError("defect")):
      with self.assertRaises(ValueError):
        self.request()

  def test_deterministic_eligibility_gates_still_avoid_talk_cognition(self):
    cases = (
      (self.tom, "act_address", ""), (self.sam, "act_address", None),
      (self.tom, "act_description", None), (self.sam, "act_description", ""),
      (self.tom, "act_description", "sleeping"), (self.sam, "act_description", "sleeping"),
      (self.tom, "curr_time", NOW.replace(hour=23)),
      (self.sam, "act_address", "<waiting> 0 0"),
      (self.tom, "act_address", "<waiting> 0 0"),
      (self.tom, "chatting_with", SAM), (self.sam, "chatting_with", TOM),
      (self.tom, "chatting_with_buffer", {SAM: 1}),
    )
    for actor, field, value in cases:
      with self.subTest(actor=actor.name, field=field, value=value):
        with patch.object(actor.scratch, field, value), \
            patch.object(plan, "generate_decide_to_talk") as cognitive:
          self.assertFalse(plan._should_react(self.tom, self.focus, self.personas))
        cognitive.assert_not_called()
    self.assertEqual((), get_telemetry())

  def pricing(self):
    return PricingSnapshot(
      "talk-synthetic", 1, "USD", "synthetic", (
        ModelPricing(COMPLETION_COMPAT_MODEL,
          input_per_million=Decimal("1000000"),
          output_per_million=Decimal("1000000")),), "offline only")

  def test_accounting_and_replay_attribution_cover_every_invalid_output_attempt(self):
    result, adapter = self.execute(["invalid"] * 5)
    self.assertEqual(Status.INVALID_OUTPUT, result.status)
    events = get_telemetry()
    self.assertEqual(5, len(events))
    self.assertEqual(1, len({event.logical_call_id for event in events}))
    self.assertEqual(list(range(1, 6)), [event.physical_attempt for event in events])
    for event in events:
      self.assertEqual(("COMPLETION_COMPAT", "decide_to_talk", TOM,
                        "talk-contract-offline", 7, "CONVERSATION", COMPLETION_COMPAT_MODEL),
        (event.operation, event.caller_id, event.actor_id, event.simulation_id,
         event.simulation_step, event.cognitive_category, event.model_or_engine))
    records = build_cost_ledger_records(events, self.pricing())
    self.assertEqual(5, len(records))
    self.assertTrue(all(record.estimated_total_cost_usd == Decimal("23") for record in records))
    self.assertEqual(1, len({repr(call[1]) for call in adapter.calls}))

  def test_cost_guard_still_stops_retries_and_escapes_as_infrastructure_failure(self):
    config = ReplayCostGuardConfig(
      "talk-offline", "talk-contract-offline", ReplayCostCeiling(Decimal("1")), self.pricing())
    adapter = DeterministicReplayFakeAdapter(["invalid"] * 5)
    with use_replay_cost_guard(config) as state, \
        use_modern_completion_runtime(build_modern_completion_runtime_config(), adapter):
      with self.assertRaises(ReplayCostCeilingExceededError):
        talk.decide_social_talk(self.request())
    self.assertEqual(1, len(adapter.calls))
    self.assertEqual(1, len(state.records()))

  def test_existing_transport_retry_accounting_is_preserved(self):
    result, adapter = self.execute([errors.LLMTimeoutError("timeout"), YES], retries=1)
    self.assertEqual(Status.SUCCESS, result.status)
    self.assertEqual(1, result.attempt_count)
    self.assertEqual(2, len(adapter.calls))
    events = get_telemetry()
    self.assertEqual([1, 2], [event.physical_attempt for event in events])
    self.assertEqual(1, len({event.logical_call_id for event in events}))

  def test_result_type_disallows_a_decision_on_non_success(self):
    for status in Status:
      if status != Status.SUCCESS:
        with self.assertRaises(ValueError):
          talk.SocialTalkDecisionResult(status, talk.TalkDecision.TALK, reason="failure")
    with self.assertRaises(ValueError):
      talk.SocialTalkDecisionResult(Status.SUCCESS)


if __name__ == "__main__":
  unittest.main()
