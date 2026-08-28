import datetime
from decimal import Decimal
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parents[1]
import sys
if str(BACKEND) not in sys.path:
  sys.path.insert(0, str(BACKEND))

import controlled_replay
import modern_smallville as subject
from persona.cognitive_modules import reflect as reflect_module
from persona.prompt_template import gpt_structure, run_gpt_prompt
from persona.memory_structures.associative_memory import AssociativeMemory
from persona.memory_structures.embedding_space import (
  LEGACY_ADA_002_MANIFEST,
  LegacyEmbeddingSpaceWarning,
  write_embedding_manifest)


class ModernTickFakeAdapter(controlled_replay.R1TDeterministicFakeAdapter):
  @staticmethod
  def response_for_caller(caller_id):
    if caller_id == "event_poignancy":
      return '{"output": "7"}'
    return controlled_replay.R1TDeterministicFakeAdapter.response_for_caller(
      caller_id)


class MalformedPoignancyFakeAdapter(ModernTickFakeAdapter):
  @staticmethod
  def response_for_caller(caller_id):
    if caller_id == "event_poignancy":
      return "not-json"
    return ModernTickFakeAdapter.response_for_caller(caller_id)


class NaturalConversationFakeAdapter(ModernTickFakeAdapter):
  def __init__(self):
    super().__init__()
    self.turn = 0

  def response_for_caller(self, caller_id):
    if caller_id == "decide_to_talk":
      return "yes"
    if caller_id == "agent_chat_summarize_relationship":
      return '{"output": "context"}'
    if caller_id == "iterative_chat_utterance":
      self.turn += 1
      end = "true" if self.turn == 2 else "false"
      return '{"utterance": "hello", "end": ' + end + "}"
    if caller_id == "summarize_conversation":
      return '{"output": "a brief exchange"}'
    if caller_id == "chat_poignancy":
      return '{"output": "6"}'
    return super().response_for_caller(caller_id)


class CeilingConversationFakeAdapter(NaturalConversationFakeAdapter):
  def response_for_caller(self, caller_id):
    if caller_id == "iterative_chat_utterance":
      return '{"utterance":"hello","end":false}'
    return super().response_for_caller(caller_id)


class ModernRunConfigTests(unittest.TestCase):
  def test_default_policy_is_valid(self):
    config = subject.ModernRunConfig(run_name="offline-valid")
    self.assertEqual(1, config.ticks)
    self.assertEqual((subject.COGNITIVE_ACTOR,), config.cognitive_actors)
    self.assertEqual(subject.PASSIVE_ACTORS, config.passive_actors)
    self.assertEqual(subject.VISIBLE_ACTORS, config.visible_actors)

  def test_all_cognitive_policy_is_valid(self):
    config = subject.ModernRunConfig(
      run_name="all-cognitive", cognitive_actors=subject.VISIBLE_ACTORS,
      passive_actors=())
    self.assertEqual(subject.VISIBLE_ACTORS, config.cognitive_actors)
    self.assertEqual((), config.passive_actors)

  def test_non_positive_ticks_are_rejected(self):
    for value in (0, -1):
      with self.subTest(value=value), self.assertRaises(
          subject.ModernRunConfigurationError):
        subject.ModernRunConfig(run_name="bad-ticks", ticks=value)

  def test_non_positive_cost_ceiling_is_rejected(self):
    for value in (Decimal("0"), Decimal("-0.01")):
      with self.subTest(value=value), self.assertRaises(
          subject.ModernRunConfigurationError):
        subject.ModernRunConfig(
          run_name="bad-cost", cost_ceiling_usd=value)

  def test_empty_name_is_rejected(self):
    with self.assertRaises(subject.ModernRunConfigurationError):
      subject.ModernRunConfig(run_name="")

  def test_invalid_actor_policy_is_rejected(self):
    with self.assertRaises(subject.ModernRunConfigurationError):
      subject.ModernRunConfig(
        run_name="bad-policy", cognitive_actors=("Maria Lopez",))

  def test_unknown_duplicate_overlap_and_order_are_rejected(self):
    policies = (
      (("Unknown",), subject.PASSIVE_ACTORS),
      ((subject.COGNITIVE_ACTOR, subject.COGNITIVE_ACTOR),
       subject.PASSIVE_ACTORS),
      ((subject.COGNITIVE_ACTOR,),
       (subject.COGNITIVE_ACTOR,) + subject.PASSIVE_ACTORS),
      (("Maria Lopez", subject.COGNITIVE_ACTOR), ("Klaus Mueller",)),
    )
    for cognitive, passive in policies:
      with self.subTest(cognitive=cognitive, passive=passive), \
          self.assertRaises(subject.ModernRunConfigurationError):
        subject.ModernRunConfig(
          run_name="bad-policy", cognitive_actors=cognitive,
          passive_actors=passive)

  def test_controlled_proximity_requires_r1m3c_policy(self):
    config = subject.ModernRunConfig(
      run_name="controlled", ticks=10,
      cognitive_actors=subject.VISIBLE_ACTORS, passive_actors=(),
      controlled_proximity=True)
    self.assertTrue(config.controlled_proximity)
    invalid = (
      {"ticks": 11, "cognitive_actors": subject.VISIBLE_ACTORS,
       "passive_actors": ()},
      {"ticks": 2},
    )
    for values in invalid:
      with self.subTest(values=values), self.assertRaises(
          subject.ModernRunConfigurationError):
        subject.ModernRunConfig(
          run_name="bad-controlled", controlled_proximity=True, **values)


class ModernResumeConfigTests(unittest.TestCase):
  def test_resume_config_is_explicit_and_valid(self):
    config = subject.ModernResumeConfig(
      source_run=Path("persisted-run"), run_name="continued-run", ticks=5,
      cost_ceiling_usd=Decimal("0.05"))
    self.assertEqual(Path("persisted-run"), config.source_run)
    self.assertEqual(5, config.ticks)

  def test_causal_social_observation_is_explicit(self):
    config = subject.ModernResumeConfig(
      source_run=Path("persisted-run"), run_name="continued-run",
      observe_causal_social_memory=True)
    self.assertTrue(config.observe_causal_social_memory)

  def test_resume_config_rejects_invalid_ticks(self):
    for value in (0, -1):
      with self.subTest(value=value), self.assertRaises(
          subject.ModernRunConfigurationError):
        subject.ModernResumeConfig(
          source_run=Path("persisted-run"), run_name="continued-run",
          ticks=value)

  def test_resume_config_rejects_non_path_source(self):
    with self.assertRaises(subject.ModernRunConfigurationError):
      subject.ModernResumeConfig(
        source_run="persisted-run", run_name="continued-run")


class FailureCallerAttributionTests(unittest.TestCase):
  """R1OBS-P1: a crash must never inherit an unrelated prior caller."""

  @staticmethod
  def _execution_state():
    return {"stage": "tick", "actor": "Maria Lopez", "tick": 3}

  @staticmethod
  def _result():
    return subject.ModernRunResult(
      verdict="MODERN_SMALLVILLE_HEADLESS_RUN_FAILED",
      run_directory=Path("unused"), cognitive_actors=(), passive_actors=(),
      completed_ticks=0, movement_count=0, initial_step=0, final_step=0,
      initial_time=None, final_time=None, logical_calls=0,
      physical_attempts=0, input_tokens=0, output_tokens=0,
      total_cost_usd=Decimal("0"), cost_ceiling_usd=Decimal("1"),
      save_passed=False, reload_passed=False, actor_move_counts=(),
      passive_provider_calls=0, passive_memory_mutations=0,
      legacy_fallback_count=0, retry_count=0,
      exception_type="RuntimeError", exception_message="boom")

  def test_exception_caller_beats_previous_telemetry(self):
    # Case A: the exception itself declares the true crash site (focal_pt);
    # an unrelated, previously successful caller must not override it.
    error = subject.ModernDeferredCallerError(
      "focal_pt", "create_chat", "Maria Lopez", 3)
    report = subject._build_failure_report(
      self._execution_state(), self._result(), error)
    self.assertEqual("focal_pt", report["caller"])
    self.assertEqual("create_chat", report["operation"])

  def test_missing_caller_does_not_inherit_previous_telemetry(self):
    # Case B (the regression this wave fixes): the crash carries no
    # caller of its own, so the reporter must not borrow the caller of
    # the last successful, unrelated telemetry event (event_poignancy).
    error = RuntimeError("unrelated failure with no caller attribute")
    report = subject._build_failure_report(
      self._execution_state(), self._result(), error)
    self.assertIsNone(report["caller"])
    self.assertIsNone(report["operation"])

  def test_no_telemetry_at_all_still_reports_unknown_caller(self):
    # Case C: no telemetry has ever been recorded; the reporter must
    # still degrade to an explicit unknown rather than raise.
    error = RuntimeError("first call in the run failed outright")
    report = subject._build_failure_report(
      self._execution_state(), self._result(), error)
    self.assertIsNone(report["caller"])
    self.assertIsNone(report["operation"])

  def test_execution_state_fields_are_untouched_by_the_fix(self):
    # Case D: stage/actor/tick come from the authoritative current
    # execution context and must survive the caller-attribution fix
    # unchanged.
    error = RuntimeError("boom")
    report = subject._build_failure_report(
      self._execution_state(), self._result(), error)
    self.assertEqual("tick", report["stage"])
    self.assertEqual("Maria Lopez", report["actor"])
    self.assertEqual(3, report["tick"])
    self.assertEqual("RuntimeError", report["exception_type"])
    self.assertEqual("boom", report["exception_message"])

  def test_no_error_yields_no_failure_section(self):
    report = subject._build_failure_report(
      self._execution_state(), self._result(), None)
    self.assertIsNone(report)

  def test_null_caller_round_trips_through_report_json(self):
    # Case E: a None caller must serialize as JSON null and read back
    # as None, matching report.json's on-disk contract.
    error = RuntimeError("unrelated failure with no caller attribute")
    failure = subject._build_failure_report(
      self._execution_state(), self._result(), error)
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "report.json"
      subject._write_json(path, {"failure": failure})
      self.assertIn('"caller": null', path.read_text(encoding="utf-8"))
      reloaded = subject._read_json(path)
      self.assertIsNone(reloaded["failure"]["caller"])
      self.assertIsNone(reloaded["failure"]["operation"])


class ReflectionLifecycleObserverTests(unittest.TestCase):
  """R1REF-A: the reflection lifecycle observer must be strictly read-only.

  These tests drive the observer directly against small, deterministic
  stand-ins for reflect.py's own functions (patched via unittest.mock, not
  modified) so the observer's own recording/wiring logic is proven without
  requiring a natural reflection trigger, an LLM provider, or a full
  ReverieServer -- consistent with "observe, don't force": this class does
  not simulate a natural reflection, it only proves the observer records
  faithfully whatever reflect.py's real functions would have handed it,
  and that reflect.py's own module namespace is untouched when disabled."""

  @staticmethod
  def _persona(name, importance_trigger_curr=150,
               importance_trigger_max=150):
    persona = SimpleNamespace(
      name=name,
      scratch=SimpleNamespace(
        importance_trigger_curr=importance_trigger_curr,
        importance_trigger_max=importance_trigger_max))
    counter = {"count": 0}

    def add_thought(created_ts, expiration, s, p, o, description, keywords,
                    poignancy, embedding_pair, filling):
      counter["count"] += 1
      return SimpleNamespace(
        node_id=f"node_{counter['count']}", type="thought",
        description=description, poignancy=poignancy, filling=filling,
        embedding_key=embedding_pair[0])

    persona.a_mem = SimpleNamespace(add_thought=add_thought)
    return persona

  @staticmethod
  def _server(*personas):
    return SimpleNamespace(
      personas={persona.name: persona for persona in personas})

  def test_disabled_observer_installs_nothing(self):
    persona = self._persona("Maria Lopez")
    server = self._server(persona)
    real_trigger = reflect_module.reflection_trigger
    real_run_reflect = reflect_module.run_reflect
    with subject._maybe_observe_reflection_lifecycle(
        False, server, {"tick": 0}, "sim") as observer:
      self.assertIsNone(observer)
      self.assertIs(real_trigger, reflect_module.reflection_trigger)
      self.assertIs(real_run_reflect, reflect_module.run_reflect)
    self.assertIs(real_trigger, reflect_module.reflection_trigger)
    self.assertIs(real_run_reflect, reflect_module.run_reflect)

  def test_observer_sees_threshold_transition(self):
    persona = self._persona(
      "Maria Lopez", importance_trigger_curr=-1, importance_trigger_max=150)
    server = self._server(persona)
    with patch.object(
        reflect_module, "reflection_trigger",
        lambda p: p.scratch.importance_trigger_curr <= 0):
      with subject._observe_reflection_lifecycle(
          server, {"tick": 3}, "sim") as observer:
        triggered = reflect_module.reflection_trigger(persona)
    self.assertTrue(triggered)
    self.assertEqual(1, len(observer.trigger_checks))
    check = observer.trigger_checks[0]
    self.assertEqual("Maria Lopez", check["actor"])
    self.assertEqual(3, check["simulation_step"])
    self.assertEqual(-1, check["importance_trigger_curr_before"])
    self.assertEqual(150, check["importance_trigger_max"])
    self.assertTrue(check["triggered"])

  def test_observer_handles_no_reflection_naturally(self):
    persona = self._persona(
      "Maria Lopez", importance_trigger_curr=40, importance_trigger_max=150)
    server = self._server(persona)
    with patch.object(
        reflect_module, "reflection_trigger",
        lambda p: p.scratch.importance_trigger_curr <= 0):
      with subject._observe_reflection_lifecycle(
          server, {"tick": 1}, "sim") as observer:
        triggered = reflect_module.reflection_trigger(persona)
    self.assertFalse(triggered)
    self.assertEqual([], observer.reflection_thoughts)
    self.assertEqual([], observer.focal_points)
    self.assertFalse(any(c["triggered"] for c in observer.trigger_checks))

  def test_observer_records_focal_points_content_free(self):
    persona = self._persona("Maria Lopez")
    server = self._server(persona)
    focal_points = ["Why did Klaus seem upset?", "What is the market like?"]
    with patch.object(
        reflect_module, "generate_focal_points",
        lambda p, n=3: focal_points):
      with subject._observe_reflection_lifecycle(
          server, {"tick": 5}, "sim") as observer:
        result = reflect_module.generate_focal_points(persona, 2)
    self.assertEqual(focal_points, result)
    self.assertEqual(1, len(observer.focal_points))
    record = observer.focal_points[0]
    self.assertEqual("Maria Lopez", record["actor"])
    self.assertEqual(2, record["requested_count"])
    self.assertEqual(
      [subject._content_hash(item) for item in focal_points],
      record["focal_point_hashes"])
    self.assertNotIn("Klaus", str(record))

  def test_observer_records_reflection_retrieval_refs(self):
    persona = self._persona("Maria Lopez")
    server = self._server(persona)
    node_a = SimpleNamespace(node_id="node_3", type="event")
    node_b = SimpleNamespace(node_id="node_7", type="thought")
    fake_result = {"Why did Klaus seem upset?": [node_a, node_b]}
    with patch.object(
        reflect_module, "new_retrieve",
        lambda p, focal_points, n_count=30: fake_result):
      with subject._observe_reflection_lifecycle(
          server, {"tick": 5}, "sim") as observer:
        result = reflect_module.new_retrieve(
          persona, ["Why did Klaus seem upset?"])
    self.assertIs(fake_result, result)
    self.assertEqual(1, len(observer.retrieval_events))
    event = observer.retrieval_events[0]
    self.assertEqual("Maria Lopez", event["actor"])
    self.assertEqual("reflection_new_retrieve", event["retrieval_context"])
    self.assertEqual(["node_3", "node_7"], event["retrieved_node_ids"])
    self.assertEqual(
      ["Maria Lopez::node_3", "Maria Lopez::node_7"],
      event["retrieved_node_refs"])
    self.assertEqual(["event", "thought"], event["retrieved_node_types"])

  def test_observer_records_insights_and_new_thought_lineage_refs(self):
    persona = self._persona("Klaus Mueller")
    server = self._server(persona)

    def fake_focal_points(p, n=3):
      return ["What did I learn about the market?"]

    def fake_retrieve(p, focal_points, n_count=30):
      node = SimpleNamespace(node_id="node_2", type="event")
      return {focal_points[0]: [node]}

    def fake_insights(p, nodes, n=5):
      # generate_insights_and_evidence's real contract already translates
      # evidence indices to node_id strings before returning (reflect.py
      # lines 51-53) -- this stand-in returns the same post-translation
      # shape run_reflect actually receives.
      return {"The market is competitive": [nodes[0].node_id]}

    def fake_run_reflect(p):
      # Stand-in for reflect.run_reflect's own call graph, used only to
      # exercise the observer's wrapping deterministically. reflect.py
      # itself is unmodified and untouched by this wave.
      focal_points = reflect_module.generate_focal_points(p, 3)
      retrieved = reflect_module.new_retrieve(p, focal_points)
      for focal_pt, nodes in retrieved.items():
        thoughts = reflect_module.generate_insights_and_evidence(p, nodes, 5)
        for thought, evidence in thoughts.items():
          p.a_mem.add_thought(
            "2026-01-01 08:00:00", None, p.name, "reflected on", thought,
            thought, {"market"}, 6, (thought, [0.1, 0.2]), evidence)

    with patch.object(
          reflect_module, "generate_focal_points", fake_focal_points), \
        patch.object(reflect_module, "new_retrieve", fake_retrieve), \
        patch.object(
          reflect_module, "generate_insights_and_evidence", fake_insights), \
        patch.object(reflect_module, "run_reflect", fake_run_reflect):
      with subject._observe_reflection_lifecycle(
          server, {"tick": 8}, "sim") as observer:
        reflect_module.run_reflect(persona)

    self.assertEqual(1, len(observer.insights))
    insight_record = observer.insights[0]
    self.assertEqual(["Klaus Mueller::node_2"],
                     insight_record["evidence_node_refs"])
    self.assertEqual(1, len(observer.reflection_thoughts))
    thought_record = observer.reflection_thoughts[0]
    self.assertEqual("Klaus Mueller", thought_record["actor"])
    self.assertEqual("node_1", thought_record["node_id"])
    self.assertEqual("Klaus Mueller::node_1", thought_record["node_ref"])
    self.assertEqual(["node_2"], thought_record["evidence_node_ids"])
    self.assertEqual(["Klaus Mueller::node_2"],
                     thought_record["evidence_node_refs"])
    self.assertEqual([], observer.conversation_followup_thoughts)

  def test_observer_distinguishes_conversation_followup_thoughts(self):
    persona = self._persona("Maria Lopez")
    server = self._server(persona)
    with subject._observe_reflection_lifecycle(
        server, {"tick": 8}, "sim") as observer:
      # Not inside run_reflect -- mirrors reflect()'s chatting_end_time
      # planning/memo thought branch, which is not the scientific
      # "reflection" this wave measures.
      persona.a_mem.add_thought(
        "2026-01-01 08:00:00", None, persona.name, "planned", "to relax",
        "planned to relax", {"relax"}, 4, ("planned to relax", [0.0]), [])
    self.assertEqual([], observer.reflection_thoughts)
    self.assertEqual(1, len(observer.conversation_followup_thoughts))

  def test_actor_qualified_refs_are_unambiguous_across_actors(self):
    maria = self._persona("Maria Lopez")
    klaus = self._persona("Klaus Mueller")
    server = self._server(maria, klaus)
    with subject._observe_reflection_lifecycle(
        server, {"tick": 1}, "sim") as observer:
      maria.a_mem.add_thought(
        "2026-01-01 08:00:00", None, maria.name, "reflected on", "x",
        "x", {"x"}, 5, ("x", [0.0]), [])
      klaus.a_mem.add_thought(
        "2026-01-01 08:00:00", None, klaus.name, "reflected on", "x",
        "x", {"x"}, 5, ("x", [0.0]), [])
    refs = {record["node_ref"]
            for record in observer.conversation_followup_thoughts}
    self.assertEqual({"Maria Lopez::node_1", "Klaus Mueller::node_1"}, refs)
    self.assertEqual(2, len(refs))

  def test_observer_records_are_json_serializable(self):
    persona = self._persona("Maria Lopez")
    server = self._server(persona)
    with patch.object(
        reflect_module, "reflection_trigger",
        lambda p: p.scratch.importance_trigger_curr <= 0), \
        patch.object(
          reflect_module, "generate_focal_points", lambda p, n=3: ["q"]):
      with subject._observe_reflection_lifecycle(
          server, {"tick": 2}, "sim") as observer:
        reflect_module.reflection_trigger(persona)
        reflect_module.generate_focal_points(persona, 1)
        persona.a_mem.add_thought(
          "2026-01-01 08:00:00", None, persona.name, "reflected on", "x",
          "x", {"x"}, 5, ("x", [0.0]), ["node_1"])
    payload = {
      "trigger_checks": observer.trigger_checks,
      "focal_points": observer.focal_points,
      "conversation_followup_thoughts": observer.conversation_followup_thoughts,
    }
    encoded = json.dumps(payload)
    self.assertIn("Maria Lopez", encoded)

  def test_observer_records_privacy_safe_insight_attempt_diagnostics(self):
    persona = self._persona("Klaus Mueller")
    server = self._server(persona)
    sentinel = "PRIVATE-INVALID-RESPONSE"
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="PRIVATE-PROMPT"), (
        patch.object(gpt_structure, "GPT_request", return_value=sentinel)):
      with subject._observe_reflection_lifecycle(
          server, {"tick": 12}, "sim") as observer:
        output = reflect_module.run_gpt_prompt_insight_and_guidance(
          persona, "1. private", 5)[0]
    self.assertEqual({}, output)
    self.assertEqual(5, len(observer.insight_attempt_diagnostics))
    encoded = json.dumps(observer.insight_attempt_diagnostics)
    self.assertNotIn(sentinel, encoded)
    self.assertNotIn("PRIVATE-PROMPT", encoded)
    self.assertTrue(all(
      item["actor"] == "Klaus Mueller"
      and item["simulation_step"] == 12
      and item["failure_category"] == "PARSE_FAILURE"
      and item["shape"]["response_line_count"] == 1
      and item["shape"]["canonical_line_match_count"] == 0
      and item["shape"]["citation_line_count"] == 0
      and item["shape"][
        "citation_line_with_post_parenthesis_suffix_count"] == 0
      for item in observer.insight_attempt_diagnostics))

  def test_observer_records_mapping_failure_category_without_masking_error(self):
    persona = self._persona("Klaus Mueller")
    server = self._server(persona)

    def fail_mapping(*args, **kwargs):
      raise reflect_module.ReflectionInsightContractError(
        "no mapping", category="EMPTY_MAPPING")

    with patch.object(
        reflect_module, "generate_insights_and_evidence", fail_mapping):
      with subject._observe_reflection_lifecycle(
          server, {"tick": 13}, "sim") as observer:
        with self.assertRaises(reflect_module.ReflectionInsightContractError):
          reflect_module.generate_insights_and_evidence(persona, [], 5)
    self.assertEqual([{
      "actor": "Klaus Mueller", "simulation_step": 13,
      "failure_category": "EMPTY_MAPPING",
    }], observer.insight_contract_failures)


class ReflectionThoughtPersistenceRoundTripTests(unittest.TestCase):
  """R1REF-A section 9/10: prove a reflection Thought's save -> reload
  round trip via the real AssociativeMemory store (not a fake), and that
  _reflection_round_trip_checks correctly classifies persistence
  eligibility independent of any full simulation run."""

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)

  def tearDown(self):
    self.temporary.cleanup()

  @staticmethod
  def _bootstrap_memory(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "nodes.json").write_text("{}", encoding="utf-8")
    (path / "embeddings.json").write_text("{}", encoding="utf-8")
    (path / "kw_strength.json").write_text(
      json.dumps({"kw_strength_event": {}, "kw_strength_thought": {}}),
      encoding="utf-8")
    write_embedding_manifest(path, LEGACY_ADA_002_MANIFEST)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", LegacyEmbeddingSpaceWarning)
      return AssociativeMemory(str(path))

  @staticmethod
  def _vector(first=0.1, second=0.2):
    return [first, second] + [0.0] * (
      LEGACY_ADA_002_MANIFEST.dimensions - 2)

  def test_reflection_thought_round_trips_through_save_and_reload(self):
    actor = "Maria Lopez"
    source = self._bootstrap_memory(self.root / "source")
    created = datetime.datetime(2026, 1, 1, 8, 0, 0)
    evidence_node = source.add_event(
      created, None, actor, "observed", "the market",
      "Maria observed the market", {"market"}, 6,
      ("Maria observed the market", self._vector(0.1, 0.2)), [])
    thought = source.add_thought(
      created, created + datetime.timedelta(days=30), actor, "reflected on",
      "market trends", "Maria reflected on market trends", {"market"}, 7,
      ("Maria reflected on market trends", self._vector(0.3, 0.4)),
      [evidence_node.node_id])
    record = {
      "actor": actor, "node_id": thought.node_id,
      "node_ref": f"{actor}::{thought.node_id}",
      "poignancy": 7,
      "description_hash": subject._content_hash(
        "Maria reflected on market trends"),
      "evidence_node_ids": [evidence_node.node_id],
    }

    saved_personas_root = self.root / "saved"
    memory_dir = (saved_personas_root / "personas" / actor
                  / "bootstrap_memory" / "associative_memory")
    memory_dir.mkdir(parents=True)
    source.save(str(memory_dir))

    with warnings.catch_warnings():
      warnings.simplefilter("ignore", LegacyEmbeddingSpaceWarning)
      reloaded_memory = AssociativeMemory(str(memory_dir))
    reloaded_persona = SimpleNamespace(a_mem=reloaded_memory)

    checks = subject._reflection_round_trip_checks(
      [record], saved_personas_root, {actor: reloaded_persona})
    self.assertEqual(1, len(checks))
    check = checks[0]
    self.assertTrue(check["persisted"], check)
    self.assertTrue(check["reloaded"], check)
    self.assertTrue(check["poignancy_preserved"], check)
    self.assertTrue(check["description_hash_preserved"], check)
    self.assertTrue(check["evidence_preserved"], check)
    self.assertTrue(check["in_seq_thought_after_reload"], check)
    self.assertTrue(check["embedding_available_after_reload"], check)

  def test_missing_node_reports_not_persisted_honestly(self):
    checks = subject._reflection_round_trip_checks(
      [{"actor": "Maria Lopez", "node_id": "node_99",
        "node_ref": "Maria Lopez::node_99", "poignancy": 5,
        "description_hash": "irrelevant", "evidence_node_ids": []}],
      self.root / "nowhere", {})
    self.assertEqual(1, len(checks))
    self.assertFalse(checks[0]["persisted"])
    self.assertFalse(checks[0]["reloaded"])


class ReflectionLifecycleVerdictTests(unittest.TestCase):
  @staticmethod
  def _section(**overrides):
    section = {
      "observation_enabled": True,
      "natural_trigger_observed": False,
      "reflection_thoughts_created": [],
      "lineage_verified": False,
      "reflection_persisted": False,
      "reflection_reloaded": False,
      "reflection_retrieval_eligible_after_reload": False,
    }
    section.update(overrides)
    return section

  def test_disabled_observation_yields_no_verdict(self):
    section = self._section(observation_enabled=False)
    self.assertIsNone(subject._reflection_lifecycle_verdict(
      section, run_error=None, run_success=True))

  def test_run_error_yields_blocked(self):
    section = self._section()
    self.assertEqual(
      subject.R1REF_A_BLOCKED_VERDICT,
      subject._reflection_lifecycle_verdict(
        section, run_error=RuntimeError("x"), run_success=False))

  def test_no_natural_trigger_yields_path_not_reached(self):
    section = self._section()
    self.assertEqual(
      subject.R1REF_A_PATH_NOT_REACHED_VERDICT,
      subject._reflection_lifecycle_verdict(
        section, run_error=None, run_success=True))

  def test_full_chain_yields_passed(self):
    section = self._section(
      natural_trigger_observed=True,
      reflection_thoughts_created=[{"node_id": "node_1"}],
      lineage_verified=True, reflection_persisted=True,
      reflection_reloaded=True,
      reflection_retrieval_eligible_after_reload=True)
    self.assertEqual(
      subject.R1REF_A_PASSED_VERDICT,
      subject._reflection_lifecycle_verdict(
        section, run_error=None, run_success=True))

  def test_trigger_fired_but_chain_incomplete_yields_blocked(self):
    section = self._section(
      natural_trigger_observed=True,
      reflection_thoughts_created=[{"node_id": "node_1"}],
      lineage_verified=True, reflection_persisted=True,
      reflection_reloaded=False,
      reflection_retrieval_eligible_after_reload=False)
    self.assertEqual(
      subject.R1REF_A_BLOCKED_VERDICT,
      subject._reflection_lifecycle_verdict(
        section, run_error=None, run_success=True))


class ReflectionLifecycleIntegrationTests(unittest.TestCase):
  """Full offline (FakeModernChatAdapter, zero live provider calls) proof
  that the --observe-reflection-lifecycle wiring integrates cleanly into
  run_modern_smallville without ever naturally reaching the reflection
  threshold in one tick -- an honest PATH_NOT_REACHED, not a forced PASS."""

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.runtime_root = Path(self.temporary.name) / "live-runs"

  def tearDown(self):
    self.temporary.cleanup()

  def test_disabled_by_default_and_absent_from_behavior(self):
    adapter = ModernTickFakeAdapter()
    config = subject.ModernRunConfig(
      run_name="reflection-observer-off", ticks=1,
      cost_ceiling_usd=Decimal("0.03"))
    self.assertFalse(config.observe_reflection_lifecycle)
    result = subject.run_modern_smallville(
      config, adapter=adapter, runtime_root=self.runtime_root)
    self.assertEqual("MODERN_SMALLVILLE_HEADLESS_RUN_PASSED", result.verdict)
    report = subject._read_json(result.run_directory / "report.json")
    section = report["reflection_lifecycle"]
    self.assertFalse(section["observation_enabled"])
    self.assertEqual([], section["trigger_checks"])
    self.assertEqual([], section["reflection_thoughts_created"])
    self.assertIsNone(section["verdict"])

  def test_enabled_flag_survives_one_tick_with_path_not_reached(self):
    adapter = ModernTickFakeAdapter()
    config = subject.ModernRunConfig(
      run_name="reflection-observer-on", ticks=1,
      cost_ceiling_usd=Decimal("0.03"),
      observe_reflection_lifecycle=True)
    result = subject.run_modern_smallville(
      config, adapter=adapter, runtime_root=self.runtime_root)
    self.assertEqual("MODERN_SMALLVILLE_HEADLESS_RUN_PASSED", result.verdict)
    report = subject._read_json(result.run_directory / "report.json")
    section = report["reflection_lifecycle"]
    self.assertTrue(section["observation_enabled"])
    self.assertFalse(section["natural_trigger_observed"])
    self.assertEqual([], section["reflection_thoughts_created"])
    self.assertEqual([], section["insight_attempt_diagnostics"])
    self.assertEqual([], section["insight_contract_failures"])
    self.assertEqual(
      subject.R1REF_A_PATH_NOT_REACHED_VERDICT, section["verdict"])
    self.assertIn(
      "Isabella Rodriguez", section["importance_trigger_max_by_actor"])

  def test_cli_wires_flag_for_run_and_resume(self):
    parser = subject.build_parser()
    run_args = parser.parse_args(["run", "--observe-reflection-lifecycle"])
    self.assertTrue(run_args.observe_reflection_lifecycle)
    resume_args = parser.parse_args(
      ["resume", "--from", "x", "--observe-reflection-lifecycle"])
    self.assertTrue(resume_args.observe_reflection_lifecycle)


class CausalSocialMemoryTests(unittest.TestCase):
  @staticmethod
  def _nodes():
    created = datetime.datetime(2023, 2, 13, 10, 0, 10)
    return {
      "node_1": SimpleNamespace(
        node_id="node_1", type="chat", created=created,
        embedding_key="conversation summary", object="Klaus Mueller",
        filling=[["Maria Lopez", "content omitted"]],
        last_accessed=created),
      "node_2": SimpleNamespace(
        node_id="node_2", type="event", created=created,
        embedding_key="Maria chatted with Klaus", filling=["node_1"],
        last_accessed=created),
      "node_3": SimpleNamespace(
        node_id="node_3", type="thought", created=created,
        embedding_key="planning memory", filling=["node_1"],
        last_accessed=created),
    }

  @classmethod
  def _bilateral_nodes(cls):
    maria = cls._nodes()
    klaus = cls._nodes()
    klaus["node_1"].object = "Maria Lopez"
    return {"Maria Lopez": maria, "Klaus Mueller": klaus}

  def test_chat_to_event_lineage_is_content_free(self):
    snapshot = subject._social_memory_actor_snapshot(
      "Maria Lopez", self._nodes())
    event = snapshot["derived_event_nodes"][0]
    self.assertEqual("node_2", event["node_id"])
    self.assertEqual(["node_1"], event["source_chat_node_ids"])
    self.assertEqual("filling", event["lineage_field"])
    self.assertNotIn("embedding_key", event)

  def test_chat_to_thought_evidence_lineage_is_recognized(self):
    snapshot = subject._social_memory_actor_snapshot(
      "Maria Lopez", self._nodes())
    thought = snapshot["derived_thought_nodes"][0]
    self.assertEqual("node_3", thought["node_id"])
    self.assertEqual(["node_1"], thought["source_chat_node_ids"])
    self.assertEqual("evidence", thought["lineage_field"])

  def test_hydration_identity_and_lineage_are_compared(self):
    before = subject._social_memory_snapshot({
      "Maria Lopez": self._nodes(), "Klaus Mueller": self._nodes()})
    hydrated = subject._social_memory_snapshot({
      "Maria Lopez": self._nodes(), "Klaus Mueller": self._nodes()})
    checks = subject._compare_social_memory_hydration(before, hydrated)
    self.assertTrue(checks["raw_social_memory_persisted"])
    self.assertTrue(checks["derived_social_memory_persisted"])
    self.assertTrue(checks["lineage_preserved"])

    broken_nodes = self._nodes()
    broken_nodes["node_2"].filling = ["node_999"]
    broken = subject._social_memory_snapshot({
      "Maria Lopez": broken_nodes, "Klaus Mueller": self._nodes()})
    self.assertFalse(subject._compare_social_memory_hydration(
      before, broken)["lineage_preserved"])

  def test_retrieval_observer_records_without_changing_return(self):
    import persona.cognitive_modules.converse as converse_module
    returned = {"Klaus Mueller": [self._nodes()["node_2"]]}
    persona = SimpleNamespace(
      name="Maria Lopez", perceive=lambda maze: [],
      a_mem=SimpleNamespace(get_last_chat=lambda target: False))
    server = SimpleNamespace(personas={"Maria Lopez": persona})
    state = {"tick": 5}

    def original_retrieve(unused_persona, unused_focal, unused_count=30):
      return returned

    with patch.object(converse_module, "new_retrieve", original_retrieve):
      observer = subject._ConversationObserver(
        server, state, "simulation",
        pre_resume_derived_node_refs=("Maria Lopez::node_2",)).install()
      try:
        actual = converse_module.new_retrieve(
          persona, ["Klaus Mueller"], 50)
      finally:
        observer.restore()
    self.assertIs(returned, actual)
    self.assertEqual(["node_2"], observer.retrieval_events[0][
      "retrieved_node_ids"])
    self.assertEqual(["Maria Lopez::node_2"], observer.retrieval_events[0][
      "retrieved_pre_resume_derived_social_node_refs"])

  def test_relationship_observer_preserves_arguments_and_output(self):
    import persona.cognitive_modules.converse as converse_module
    returned = {"Klaus Mueller": [self._nodes()["node_2"]]}
    maria = SimpleNamespace(
      name="Maria Lopez", perceive=lambda maze: [],
      a_mem=SimpleNamespace(get_last_chat=lambda target: False))
    klaus = SimpleNamespace(
      name="Klaus Mueller", perceive=lambda maze: [],
      a_mem=SimpleNamespace(get_last_chat=lambda target: False))
    server = SimpleNamespace(personas={
      "Maria Lopez": maria, "Klaus Mueller": klaus})
    calls = []
    sentinel = object()

    def original_relationship(actor, target, retrieved):
      calls.append((actor, target, retrieved))
      return sentinel

    with patch.object(
        converse_module, "generate_summarize_agent_relationship",
        original_relationship):
      observer = subject._ConversationObserver(
        server, {"tick": 5}, "simulation",
        pre_resume_derived_node_refs=("Maria Lopez::node_2",)).install()
      try:
        actual = converse_module.generate_summarize_agent_relationship(
          maria, klaus, returned)
      finally:
        observer.restore()
    self.assertIs(sentinel, actual)
    self.assertEqual((maria, klaus, returned), calls[0])
    event = observer.relationship_events[0]
    self.assertEqual(["node_2"], event["input_node_ids"])
    self.assertEqual(
      ["Maria Lopez::node_2"],
      event["pre_resume_derived_social_node_refs_present"])

  def test_positive_causal_classifier_requires_same_derived_node(self):
    result = subject._classify_causal_social_memory(
      raw_social_memory_persisted=True,
      derived_social_memory_persisted=True,
      lineage_preserved_after_hydration=True,
      pre_resume_derived_node_ids=("Maria Lopez::node_2",),
      retrieved_node_ids=("Maria Lopez::node_2",),
      consumed_node_ids=("Maria Lopez::node_2",))
    self.assertTrue(result["social_memory_causally_reused"])
    self.assertTrue(result["causal_link_verified"])

  def test_negative_causal_classifier_cases_fail_closed(self):
    valid = {
      "raw_social_memory_persisted": True,
      "derived_social_memory_persisted": True,
      "lineage_preserved_after_hydration": True,
      "pre_resume_derived_node_ids": ("Maria Lopez::node_2",),
      "retrieved_node_ids": ("Maria Lopez::node_2",),
      "consumed_node_ids": ("Maria Lopez::node_2",),
    }
    cases = {
      "derived missing": {"derived_social_memory_persisted": False},
      "not retrieved": {"retrieved_node_ids": ()},
      "wrong derived": {"retrieved_node_ids": ("Maria Lopez::node_9",)},
      "not consumed": {"consumed_node_ids": ()},
      "broken lineage": {"lineage_preserved_after_hydration": False},
    }
    for name, changed in cases.items():
      with self.subTest(name=name):
        values = {**valid, **changed}
        self.assertFalse(subject._classify_causal_social_memory(
          **values)["causal_link_verified"])

  def test_bilateral_lineage_requires_both_actors(self):
    snapshot = subject._social_memory_snapshot({
      "Maria Lopez": self._nodes()})
    self.assertFalse(snapshot["bilateral_chat_lineage_present"])

  def test_bilateral_lineage_requires_derived_event(self):
    actors = self._bilateral_nodes()
    del actors["Maria Lopez"]["node_2"]
    snapshot = subject._social_memory_snapshot(actors)
    self.assertFalse(snapshot["bilateral_chat_lineage_present"])

  def test_bilateral_lineage_requires_event_to_reference_chat(self):
    actors = self._bilateral_nodes()
    actors["Maria Lopez"]["node_2"].filling = ["node_999"]
    snapshot = subject._social_memory_snapshot(actors)
    self.assertFalse(snapshot["bilateral_chat_lineage_present"])

  def test_actor_local_node_id_collisions_preserve_bilateral_lineage(self):
    snapshot = subject._social_memory_snapshot(self._bilateral_nodes())
    self.assertTrue(snapshot["bilateral_chat_lineage_present"])
    self.assertEqual(
      ["Klaus Mueller::node_2", "Maria Lopez::node_2"],
      sorted(
        item["node_ref"]
        for actor in snapshot["actors"].values()
        for item in actor["derived_event_nodes"]))

  def test_bilateral_lineage_requires_expected_chat_participant(self):
    actors = self._bilateral_nodes()
    actors["Klaus Mueller"]["node_1"].object = "Isabella Rodriguez"
    snapshot = subject._social_memory_snapshot(actors)
    self.assertFalse(snapshot["bilateral_chat_lineage_present"])


class ModernRunnerOfflineTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.runtime_root = Path(self.temporary.name) / "live-runs"

  def tearDown(self):
    self.temporary.cleanup()

  def _create_persisted_run(self, name="resume-source"):
    return subject.run_modern_smallville(
      subject.ModernRunConfig(
        run_name=name, ticks=5, cost_ceiling_usd=Decimal("0.05")),
      adapter=ModernTickFakeAdapter(), runtime_root=self.runtime_root)

  def _create_causal_source(self, name):
    return subject.run_modern_smallville(
      subject.ModernRunConfig(
        run_name=name, ticks=2,
        cognitive_actors=subject.VISIBLE_ACTORS, passive_actors=(),
        controlled_proximity=True, cost_ceiling_usd=Decimal("0.05")),
      adapter=NaturalConversationFakeAdapter(), runtime_root=self.runtime_root)

  def _prepare_causal_source(self, source_result, continuation_name):
    return subject._prepare_resume_context(
      subject.ModernResumeConfig(
        source_run=source_result.run_directory,
        run_name=continuation_name,
        observe_causal_social_memory=True),
      self.runtime_root)

  def test_two_real_ticks_save_and_reload_without_network(self):
    adapter = ModernTickFakeAdapter()
    network_calls = []

    def reject_network(*args, **kwargs):
      del args, kwargs
      network_calls.append("attempt")
      raise AssertionError("network is forbidden")

    config = subject.ModernRunConfig(
      run_name="offline-two-ticks", ticks=2,
      cost_ceiling_usd=Decimal("0.03"))
    with patch("socket.create_connection", side_effect=reject_network), \
        patch.object(socket.socket, "connect", reject_network):
      result = subject.run_modern_smallville(
        config, adapter=adapter, runtime_root=self.runtime_root)

    self.assertEqual(
      "MODERN_SMALLVILLE_HEADLESS_RUN_PASSED", result.verdict,
      repr(result) + " calls=" + repr(adapter.calls))
    self.assertEqual(2, result.completed_ticks)
    self.assertEqual(2, result.movement_count)
    self.assertEqual(result.initial_step + 2, result.final_step)
    self.assertEqual(
      result.initial_time + datetime.timedelta(seconds=20),
      result.final_time)
    self.assertTrue(result.save_passed)
    self.assertTrue(result.reload_passed)
    self.assertEqual(0, result.passive_provider_calls)
    self.assertEqual(0, result.passive_memory_mutations)
    self.assertEqual({
      "Isabella Rodriguez": 2, "Maria Lopez": 0, "Klaus Mueller": 0,
    }, dict(result.actor_move_counts))
    self.assertEqual([], network_calls)
    self.assertTrue((result.run_directory / "status.json").is_file())
    self.assertTrue((result.run_directory / "report.json").is_file())

  def test_existing_run_directory_fails_closed(self):
    existing = self.runtime_root / "collision"
    existing.mkdir(parents=True)
    with self.assertRaises(subject.ModernRunConfigurationError):
      subject.run_modern_smallville(
        subject.ModernRunConfig(run_name="collision"),
        adapter=controlled_replay.R1TDeterministicFakeAdapter(),
        runtime_root=self.runtime_root)

  def test_all_three_actors_run_one_real_tick_in_isolation(self):
    adapter = ModernTickFakeAdapter()
    network_calls = []

    def reject_network(*args, **kwargs):
      del args, kwargs
      network_calls.append("attempt")
      raise AssertionError("network is forbidden")

    config = subject.ModernRunConfig(
      run_name="offline-three-cognitive", ticks=1,
      cognitive_actors=subject.VISIBLE_ACTORS, passive_actors=())
    with patch("socket.create_connection", side_effect=reject_network), \
        patch.object(socket.socket, "connect", reject_network):
      result = subject.run_modern_smallville(
        config, adapter=adapter, runtime_root=self.runtime_root)

    self.assertEqual(
      "R1M3_A_THREE_COGNITIVE_ACTORS_ONE_TICK_PASSED", result.verdict,
      repr(result) + " calls=" + repr(adapter.calls))
    self.assertEqual(subject.VISIBLE_ACTORS, result.cognitive_actors)
    self.assertEqual((), result.passive_actors)
    self.assertEqual({name: 1 for name in subject.VISIBLE_ACTORS},
                     dict(result.actor_move_counts))
    self.assertEqual([], network_calls)
    report = subject._read_json(result.run_directory / "report.json")
    self.assertTrue(report["multi_actor_isolation"]["all_checks_passed"])
    self.assertEqual(set(subject.VISIBLE_ACTORS),
                     set(report["telemetry"]["by_actor"]))
    self.assertEqual(set(subject.VISIBLE_ACTORS),
                     set(report["embedding_stores"]["saved"]))
    for actor in subject.VISIBLE_ACTORS:
      self.assertEqual(
        controlled_replay.MODERN_COMPATIBLE,
        report["embedding_stores"]["saved"][actor]["classification"])
      self.assertGreaterEqual(report["actors"][actor]["after"][
        "memory_node_count"] - report["actors"][actor]["before"][
          "memory_node_count"], 1)

  def test_three_cognitive_actors_preserve_continuity_for_five_ticks(self):
    adapter = ModernTickFakeAdapter()
    network_calls = []

    def reject_network(*args, **kwargs):
      del args, kwargs
      network_calls.append("attempt")
      raise AssertionError("network is forbidden")

    config = subject.ModernRunConfig(
      run_name="offline-three-cognitive-five-ticks", ticks=5,
      cognitive_actors=subject.VISIBLE_ACTORS, passive_actors=(),
      cost_ceiling_usd=Decimal("0.05"))
    with patch("socket.create_connection", side_effect=reject_network), \
        patch.object(socket.socket, "connect", reject_network):
      result = subject.run_modern_smallville(
        config, adapter=adapter, runtime_root=self.runtime_root)

    self.assertEqual(
      "R1M3_B_THREE_COGNITIVE_ACTORS_FIVE_TICKS_PASSED",
      result.verdict, repr(result) + " calls=" + repr(adapter.calls))
    self.assertEqual(5, result.completed_ticks)
    self.assertEqual(5, result.movement_count)
    self.assertEqual({name: 5 for name in subject.VISIBLE_ACTORS},
                     dict(result.actor_move_counts))
    self.assertEqual([], network_calls)
    report = subject._read_json(result.run_directory / "report.json")
    self.assertTrue(report["continuity"]["all_checks_passed"])
    self.assertTrue(report["multi_actor_isolation"]["all_checks_passed"])
    self.assertTrue(report["movement_integrity"]["all_checks_passed"])
    self.assertTrue(report["telemetry"]["attribution_valid"])
    self.assertEqual(5, len(report["tick_progression"]))
    self.assertEqual(5, len(report["telemetry"]["by_tick"]))
    self.assertEqual(5, len(report["movement_integrity"]["frame_hashes"]))
    for name in subject.VISIBLE_ACTORS:
      counts = [
        item["actors"][name]["memory_node_count"]
        for item in report["tick_progression"]]
      self.assertEqual(counts, sorted(counts))
      self.assertEqual(
        report["actors"][name]["after"]["memory_node_count"],
        report["actors"][name]["reload"]["memory_node_count"])
      self.assertEqual(
        report["actors"][name]["after"]["embedding_count"],
        report["actors"][name]["reload"]["embedding_count"])

  def test_controlled_proximity_reaches_natural_bilateral_conversation(self):
    adapter = NaturalConversationFakeAdapter()
    source = subject.SOURCE_ROOT / subject.DEFAULT_SOURCE
    source_hash = subject._tree_sha256(source)
    network_calls = []

    def reject_network(*args, **kwargs):
      del args, kwargs
      network_calls.append("attempt")
      raise AssertionError("network is forbidden")

    config = subject.ModernRunConfig(
      run_name="offline-r1m3c", ticks=2,
      cognitive_actors=subject.VISIBLE_ACTORS, passive_actors=(),
      cost_ceiling_usd=Decimal("0.05"), controlled_proximity=True)
    with patch("socket.create_connection", side_effect=reject_network), \
        patch.object(socket.socket, "connect", reject_network):
      result = subject.run_modern_smallville(
        config, adapter=adapter, runtime_root=self.runtime_root)

    self.assertEqual(
      "R1M3_C_NATURAL_CONVERSATION_AND_BILATERAL_MEMORY_PASSED",
      result.verdict, repr(result) + " calls=" + repr(adapter.calls))
    self.assertEqual([], network_calls)
    self.assertEqual(source_hash, subject._tree_sha256(source))
    self.assertEqual({name: 2 for name in subject.VISIBLE_ACTORS},
                     dict(result.actor_move_counts))
    self.assertEqual(0, result.legacy_fallback_count)
    report = subject._read_json(result.run_directory / "report.json")
    fixture = report["fixture"]
    self.assertTrue(fixture["source_unchanged"])
    self.assertTrue(fixture["validation"]["all_checks_passed"])
    self.assertEqual(1.0, fixture["validation"]["distance"])
    self.assertTrue(fixture["validation"]["same_arena"])
    self.assertTrue(fixture["validation"]["distinct_tiles"])
    self.assertTrue(fixture["validation"]["within_perception_range"])
    for actor in subject.VISIBLE_ACTORS:
      checks = fixture["validation"]["actors"][actor]
      self.assertTrue(checks["walkable"])
      self.assertTrue(checks["awake"])
      self.assertTrue(checks["action_valid"])
      self.assertTrue(checks["schedule_valid"])
    interaction = report["interaction"]
    self.assertTrue(interaction["encounter_gate"]["bilateral"])
    self.assertTrue(interaction["reaction_gate"]["reached"])
    self.assertEqual(
      "CHAT", interaction["reaction_gate"]["events"][0][
        "decision_category"])
    conversation = interaction["conversation_gate"]["conversations"][0]
    self.assertTrue(interaction["conversation_gate"]["valid"])
    self.assertEqual(2, conversation["turn_count"])
    self.assertEqual(list(subject.R1M3C_ACTORS),
                     conversation["speaker_sequence"])
    self.assertEqual("MODEL_END", conversation["termination"])
    self.assertTrue(conversation["distinct_chat_objects"])
    self.assertTrue(interaction["bilateral_memory"]["saved"])
    self.assertTrue(interaction["bilateral_memory"]["reloaded"])
    for actor in subject.R1M3C_ACTORS:
      memory = interaction["bilateral_memory"]["actors"][actor]
      self.assertEqual(0, memory["before"])
      self.assertEqual(1, memory["after"])
      self.assertEqual(memory["after"], memory["reload"])
      self.assertEqual(1, len(memory["new_node_ids"]))
      self.assertTrue(report["embedding_stores"]["saved"][actor][
        "references_valid"])
      causal_actor = report["causal_resume"]["pre_resume"]["actors"][actor]
      self.assertEqual(1, len(causal_actor["chat_nodes"]))
      self.assertEqual(1, len(causal_actor["derived_event_nodes"]))
      self.assertEqual(
        causal_actor["chat_node_ids"],
        causal_actor["derived_event_nodes"][0]["source_chat_node_ids"])
    self.assertTrue(report["causal_resume"]["pre_resume"][
      "bilateral_chat_lineage_present"])
    self.assertTrue(report["causal_resume"][
      "last_accessed_persistence_gap_detected"])
    self.assertTrue(report["continuity"]["all_checks_passed"])
    self.assertTrue(report["multi_actor_isolation"]["all_checks_passed"])
    self.assertEqual(0, report["reload"]["provider_calls"])
    chat_callers = {
      call["caller_id"] for call in adapter.calls
      if call["method"] == "create_chat"}
    self.assertTrue({
      "decide_to_talk", "agent_chat_summarize_relationship",
      "iterative_chat_utterance", "summarize_conversation",
      "chat_poignancy",
    }.issubset(chat_callers))
    self.assertTrue(all(
      call["model"] == "gpt-4o-mini" for call in adapter.calls
      if call["method"] == "create_chat"))

  def test_all_continue_dialogue_stops_at_historical_safety_ceiling(self):
    config = subject.ModernRunConfig(
      run_name="offline-r1m3c-ceiling", ticks=2,
      cognitive_actors=subject.VISIBLE_ACTORS, passive_actors=(),
      cost_ceiling_usd=Decimal("0.05"), controlled_proximity=True)
    result = subject.run_modern_smallville(
      config, adapter=CeilingConversationFakeAdapter(),
      runtime_root=self.runtime_root)
    report = subject._read_json(result.run_directory / "report.json")
    gate = report["interaction"]["conversation_gate"]
    conversation = gate["conversations"][0]
    self.assertEqual(subject.R1M3C_FUNCTIONAL_VERDICT, result.verdict)
    self.assertTrue(gate["committed"])
    self.assertTrue(gate["social_pipeline_functional"])
    self.assertFalse(gate["model_end_observed"])
    self.assertTrue(gate["safety_ceiling_reached"])
    self.assertEqual(16, conversation["turn_count"])
    self.assertEqual("SAFETY_CEILING", conversation["termination"])
    self.assertEqual(
      ["Maria Lopez", "Klaus Mueller"] * 8,
      conversation["speaker_sequence"])

  def test_model_end_with_persistent_memory_is_natural_complete(self):
    self.assertEqual(
      subject.R1M3C_NATURAL_VERDICT,
      subject._classify_r1m3c_conversation(
        conversation_committed=True, model_end_observed=True,
        safety_ceiling_reached=False, bilateral_memory=True,
        bilateral_memory_reloaded=True, memory_integrity_valid=True,
        save_passed=True, reload_passed=True))

  def test_missing_bilateral_memory_is_memory_blocked(self):
    self.assertEqual(
      subject.R1M3C_MEMORY_BLOCKED_VERDICT,
      subject._classify_r1m3c_conversation(
        conversation_committed=True, model_end_observed=False,
        safety_ceiling_reached=True, bilateral_memory=False,
        bilateral_memory_reloaded=False, memory_integrity_valid=True,
        save_passed=True, reload_passed=True))

  def test_reload_memory_loss_is_memory_blocked(self):
    self.assertEqual(
      subject.R1M3C_MEMORY_BLOCKED_VERDICT,
      subject._classify_r1m3c_conversation(
        conversation_committed=True, model_end_observed=False,
        safety_ceiling_reached=True, bilateral_memory=True,
        bilateral_memory_reloaded=False, memory_integrity_valid=True,
        save_passed=True, reload_passed=True))

  def test_uncommitted_conversation_is_memory_blocked(self):
    self.assertEqual(
      subject.R1M3C_MEMORY_BLOCKED_VERDICT,
      subject._classify_r1m3c_conversation(
        conversation_committed=False, model_end_observed=False,
        safety_ceiling_reached=False, bilateral_memory=False,
        bilateral_memory_reloaded=False, memory_integrity_valid=True,
        save_passed=True, reload_passed=True))

  def test_event_poignancy_uses_declared_fail_safe(self):
    result = subject.run_modern_smallville(
      subject.ModernRunConfig(run_name="poignancy-fail-safe"),
      adapter=MalformedPoignancyFakeAdapter(),
      runtime_root=self.runtime_root)
    self.assertEqual(
      "MODERN_SMALLVILLE_HEADLESS_RUN_PASSED", result.verdict, repr(result))
    self.assertEqual(1, result.completed_ticks)

  def test_missing_source_fails_closed(self):
    with self.assertRaises(subject.ModernRunConfigurationError):
      subject.run_modern_smallville(
        subject.ModernRunConfig(
          run_name="missing-source", source_simulation="not-present"),
        adapter=controlled_replay.R1TDeterministicFakeAdapter(),
        runtime_root=self.runtime_root)

  def test_legacy_provider_configuration_fails_closed(self):
    adapter = controlled_replay.R1TDeterministicFakeAdapter()
    with patch.object(
        controlled_replay, "ControlledReplayProviders",
        side_effect=controlled_replay.ControlledReplayLegacyConfigurationError(
          "legacy fallback")):
      with self.assertRaises(
          controlled_replay.ControlledReplayLegacyConfigurationError):
        subject.run_modern_smallville(
          subject.ModernRunConfig(run_name="legacy-rejected"),
          adapter=adapter, runtime_root=self.runtime_root)

  def test_resume_rejects_missing_source_before_provider_use(self):
    self.runtime_root.mkdir(parents=True)
    adapter = ModernTickFakeAdapter()
    with self.assertRaises(subject.ModernRunConfigurationError):
      subject.run_modern_smallville_resume(
        subject.ModernResumeConfig(
          source_run=self.runtime_root / "missing",
          run_name="missing-continuation"),
        adapter=adapter, runtime_root=self.runtime_root)
    self.assertEqual([], adapter.calls)

  def test_resume_preserves_hydrated_state_and_appends_history(self):
    source_result = self._create_persisted_run()
    source_report = subject._read_json(
      source_result.run_directory / "report.json")
    source_simulation = Path(source_report["artifacts"]["simulation"])
    old_movement_hashes = {
      f"{index}.json": subject._file_sha256(
        source_simulation / "movement" / f"{index}.json")
      for index in range(5)}
    old_environment_hashes = {
      f"{index}.json": subject._file_sha256(
        source_simulation / "environment" / f"{index}.json")
      for index in range(6)}

    result = subject.run_modern_smallville_resume(
      subject.ModernResumeConfig(
        source_run=source_result.run_directory,
        run_name="resume-continuation", ticks=5,
        cost_ceiling_usd=Decimal("0.05")),
      adapter=ModernTickFakeAdapter(), runtime_root=self.runtime_root)

    self.assertEqual(subject.R1CLI_A2_A_READY_VERDICT, result.verdict)
    self.assertEqual(5, result.initial_step)
    self.assertEqual(10, result.final_step)
    self.assertEqual(
      source_result.final_time, result.initial_time)
    self.assertEqual(
      result.initial_time + datetime.timedelta(seconds=50),
      result.final_time)
    self.assertEqual(5, result.movement_count)
    self.assertTrue(result.save_passed)
    self.assertTrue(result.reload_passed)
    self.assertEqual(0, result.legacy_fallback_count)

    report = subject._read_json(result.run_directory / "report.json")
    resumed_simulation = Path(report["artifacts"]["simulation"])
    resume = report["resume"]
    self.assertEqual("STANFORD_FORK_COPY", resume["continuation_strategy"])
    self.assertEqual(0, resume["hydration_provider_calls"])
    self.assertTrue(resume["history_preserved"])
    self.assertTrue(resume["cognitive_state_preserved"])
    self.assertTrue(resume["hydration"]["step_retained"])
    self.assertTrue(resume["hydration"]["time_retained"])
    self.assertTrue(resume["hydration"]["all_checks_passed"])
    actor_checks = resume["hydration"]["actors"][subject.COGNITIVE_ACTOR]
    self.assertTrue(actor_checks["all_checks_passed"])
    self.assertTrue(all(actor_checks.values()))

    movement = report["movement_integrity"]
    self.assertEqual(10, movement["saved_frame_count"])
    self.assertEqual(10, movement["reload_frame_count"])
    self.assertEqual(old_movement_hashes, movement["prior_frame_hashes"])
    self.assertEqual(
      {f"{index}.json" for index in range(5, 10)},
      set(movement["frame_hashes"]))
    for name, expected_hash in old_movement_hashes.items():
      self.assertEqual(
        expected_hash,
        subject._file_sha256(resumed_simulation / "movement" / name))
      self.assertEqual(
        expected_hash,
        subject._file_sha256(source_simulation / "movement" / name))
    for name, expected_hash in old_environment_hashes.items():
      self.assertEqual(
        expected_hash,
        subject._file_sha256(resumed_simulation / "environment" / name))
      self.assertEqual(
        expected_hash,
        subject._file_sha256(source_simulation / "environment" / name))

    source_actor = source_report["actors"][subject.COGNITIVE_ACTOR]["after"]
    resumed_actor = report["actors"][subject.COGNITIVE_ACTOR]["before"]
    for field in (
        "scratch_hash", "associative_memory_hash", "spatial_memory_hash",
        "daily_plan_hash", "schedule_hash", "action_hash",
        "memory_node_count", "embedding_count", "chat_count"):
      self.assertEqual(source_actor[field], resumed_actor[field], field)
    planning = report["telemetry"]["by_actor"][
      subject.COGNITIVE_ACTOR]["planning_callers"]
    self.assertEqual(0, planning["wake_up_hour"])
    self.assertEqual(0, planning["daily_plan"])
    self.assertEqual(0, planning["generate_hourly_schedule"])

  def test_causal_resume_hydrates_bilateral_lineage_without_provider_calls(self):
    source_result = self._create_causal_source("causal-source")

    result = subject.run_modern_smallville_resume(
      subject.ModernResumeConfig(
        source_run=source_result.run_directory,
        run_name="causal-continuation", ticks=1,
        cost_ceiling_usd=Decimal("0.05"),
        observe_causal_social_memory=True),
      adapter=ModernTickFakeAdapter(), runtime_root=self.runtime_root)

    self.assertEqual(subject.R1CLI_A2_B_PATH_NOT_REACHED_VERDICT,
                     result.verdict, repr(result))
    report = subject._read_json(result.run_directory / "report.json")
    causal = report["causal_resume"]
    self.assertTrue(causal["observation_enabled"])
    self.assertTrue(causal["hydration"]["chat_nodes_preserved"])
    self.assertTrue(causal["hydration"]["derived_nodes_preserved"])
    self.assertTrue(causal["hydration"]["lineage_preserved"])
    self.assertEqual(0, causal["hydration"]["provider_calls"])
    self.assertTrue(causal["raw_social_memory_persisted"])
    self.assertTrue(causal["derived_social_memory_persisted"])
    self.assertFalse(causal["causal_link_verified"])
    self.assertEqual("NOT_FULLY_GUARANTEED",
                     causal["ranking_state_fidelity"])

  def test_chained_causal_source_uses_final_persisted_lineage(self):
    source = self._create_causal_source("chain-source")
    chained = subject.run_modern_smallville_resume(
      subject.ModernResumeConfig(
        source_run=source.run_directory, run_name="chain-middle", ticks=1,
        cost_ceiling_usd=Decimal("0.05"),
        observe_causal_social_memory=True),
      adapter=ModernTickFakeAdapter(), runtime_root=self.runtime_root)
    report = subject._read_json(chained.run_directory / "report.json")
    self.assertFalse(report["interaction"]["bilateral_memory"]["saved"])
    source_hash = subject._tree_sha256(chained.run_directory)

    context = self._prepare_causal_source(chained, "chain-target")

    self.assertTrue(
      context.causal_social_memory["bilateral_chat_lineage_present"])
    self.assertEqual(source_hash, subject._tree_sha256(chained.run_directory))

  def test_persisted_lineage_overrides_missing_observational_report_section(self):
    source = self._create_causal_source("report-missing-source")
    report_path = source.run_directory / "report.json"
    report = subject._read_json(report_path)
    report.pop("interaction", None)
    subject._write_json(report_path, report)

    context = self._prepare_causal_source(source, "report-missing-target")

    self.assertTrue(
      context.causal_social_memory["bilateral_chat_lineage_present"])

  def test_report_true_does_not_override_invalid_persisted_lineage(self):
    source = self._create_causal_source("invalid-store-source")
    report = subject._read_json(source.run_directory / "report.json")
    self.assertTrue(
      report["interaction"]["bilateral_memory"]["saved"])
    simulation = Path(report["artifacts"]["simulation"])
    nodes_path = (simulation / "personas" / "Klaus Mueller"
                  / "bootstrap_memory" / "associative_memory" / "nodes.json")
    nodes = subject._read_json(nodes_path)
    snapshot = subject._social_memory_actor_snapshot("Klaus Mueller", nodes)
    derived_id = snapshot["derived_event_nodes"][0]["node_id"]
    nodes[derived_id]["filling"] = ["node_missing"]
    subject._write_json(nodes_path, nodes)

    with self.assertRaisesRegex(
        subject.ModernRunConfigurationError,
        "lacks bilateral Chat -> Event lineage"):
      self._prepare_causal_source(source, "invalid-store-target")

  def test_resume_rejects_incompatible_embedding_before_hydration(self):
    source_result = self._create_persisted_run("embedding-source")
    source_report = subject._read_json(
      source_result.run_directory / "report.json")
    source_simulation = Path(source_report["artifacts"]["simulation"])
    manifest_path = (
      source_simulation / "personas" / subject.COGNITIVE_ACTOR
      / "bootstrap_memory" / "associative_memory"
      / "embedding_manifest.json")
    manifest = subject._read_json(manifest_path)
    manifest["dimensions"] = 3072
    subject._write_json(manifest_path, manifest)
    adapter = ModernTickFakeAdapter()

    with self.assertRaisesRegex(
        subject.ModernRunConfigurationError, "embedding store"):
      subject.run_modern_smallville_resume(
        subject.ModernResumeConfig(
          source_run=source_result.run_directory,
          run_name="embedding-continuation"),
        adapter=adapter, runtime_root=self.runtime_root)
    self.assertEqual([], adapter.calls)


class ModernCliTests(unittest.TestCase):
  def test_run_ticks_one_returns_success(self):
    now = datetime.datetime(2026, 8, 7, 11, 37, 0)
    run_dir = Path(".runtime/live-runs") / subject.generate_run_name(now)
    result = subject.ModernRunResult(
      verdict="MODERN_SMALLVILLE_HEADLESS_RUN_PASSED",
      run_directory=run_dir,
      cognitive_actors=(subject.COGNITIVE_ACTOR,),
      passive_actors=subject.PASSIVE_ACTORS,
      completed_ticks=1, movement_count=1,
      initial_step=0, final_step=1, initial_time=now,
      final_time=now + datetime.timedelta(seconds=10),
      logical_calls=1, physical_attempts=1, input_tokens=1,
      output_tokens=1, total_cost_usd=Decimal("0.0001"),
      cost_ceiling_usd=Decimal("0.03"), save_passed=True,
      reload_passed=True,
      actor_move_counts=((subject.COGNITIVE_ACTOR, 1),
                         (subject.PASSIVE_ACTORS[0], 0),
                         (subject.PASSIVE_ACTORS[1], 0)),
      passive_provider_calls=0, passive_memory_mutations=0,
      legacy_fallback_count=0, retry_count=0)
    with patch.object(subject, "run_modern_smallville", return_value=result):
      self.assertEqual(0, subject.main([
        "run", "--ticks", "1", "--name", run_dir.name]))

  def test_cognitive_all_maps_to_three_cognitive_actors(self):
    now = datetime.datetime(2026, 8, 7, 11, 37, 0)
    result = subject.ModernRunResult(
      verdict="MODERN_SMALLVILLE_HEADLESS_RUN_PASSED",
      run_directory=Path(".runtime/live-runs/all"),
      cognitive_actors=subject.VISIBLE_ACTORS, passive_actors=(),
      completed_ticks=1, movement_count=1,
      initial_step=0, final_step=1, initial_time=now,
      final_time=now + datetime.timedelta(seconds=10),
      logical_calls=0, physical_attempts=0, input_tokens=0,
      output_tokens=0, total_cost_usd=Decimal("0"),
      cost_ceiling_usd=Decimal("0.03"), save_passed=True,
      reload_passed=True,
      actor_move_counts=tuple((name, 1) for name in subject.VISIBLE_ACTORS),
      passive_provider_calls=0, passive_memory_mutations=0,
      legacy_fallback_count=0, retry_count=0)
    with patch.object(
        subject, "run_modern_smallville", return_value=result) as runner:
      self.assertEqual(0, subject.main([
        "run", "--ticks", "1", "--name", "all", "--cognitive", "all"]))
    config = runner.call_args.args[0]
    self.assertEqual(subject.VISIBLE_ACTORS, config.cognitive_actors)
    self.assertEqual((), config.passive_actors)

  def test_invalid_ticks_uses_argparse_exit_code_two(self):
    with self.assertRaises(SystemExit) as caught:
      subject.main(["run", "--ticks", "0"])
    self.assertEqual(2, caught.exception.code)

  def test_configuration_error_returns_two(self):
    with patch.object(
        subject, "run_modern_smallville",
        side_effect=subject.ModernRunConfigurationError("invalid")):
      self.assertEqual(2, subject.main([
        "run", "--ticks", "1", "--name", "valid-name"]))

  def test_resume_subcommand_builds_dedicated_config(self):
    now = datetime.datetime(2026, 8, 7, 11, 37, 0)
    result = subject.ModernRunResult(
      verdict=subject.R1CLI_A2_A_READY_VERDICT,
      run_directory=Path(".runtime/live-runs/continued"),
      cognitive_actors=(subject.COGNITIVE_ACTOR,),
      passive_actors=subject.PASSIVE_ACTORS,
      completed_ticks=5, movement_count=5,
      initial_step=5, final_step=10, initial_time=now,
      final_time=now + datetime.timedelta(seconds=50),
      logical_calls=0, physical_attempts=0, input_tokens=0,
      output_tokens=0, total_cost_usd=Decimal("0"),
      cost_ceiling_usd=Decimal("0.05"), save_passed=True,
      reload_passed=True,
      actor_move_counts=((subject.COGNITIVE_ACTOR, 5),
                         (subject.PASSIVE_ACTORS[0], 0),
                         (subject.PASSIVE_ACTORS[1], 0)),
      passive_provider_calls=0, passive_memory_mutations=0,
      legacy_fallback_count=0, retry_count=0)
    with patch.object(
        subject, "run_modern_smallville_resume",
        return_value=result) as runner:
      self.assertEqual(0, subject.main([
        "resume", "--from", ".runtime/live-runs/source",
        "--ticks", "5", "--name", "continued",
        "--cost-ceiling", "0.05",
        "--observe-causal-social-memory"]))
    config = runner.call_args.args[0]
    self.assertIsInstance(config, subject.ModernResumeConfig)
    self.assertEqual(Path(".runtime/live-runs/source"), config.source_run)
    self.assertEqual("continued", config.run_name)
    self.assertEqual(5, config.ticks)
    self.assertTrue(config.observe_causal_social_memory)

  def test_resume_invalid_ticks_uses_argparse_exit_code_two(self):
    with self.assertRaises(SystemExit) as caught:
      subject.main([
        "resume", "--from", ".runtime/live-runs/source", "--ticks", "0"])
    self.assertEqual(2, caught.exception.code)


if __name__ == "__main__":
  unittest.main()
