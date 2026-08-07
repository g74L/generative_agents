import datetime
from decimal import Decimal
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parents[1]
import sys
if str(BACKEND) not in sys.path:
  sys.path.insert(0, str(BACKEND))

import controlled_replay
import modern_smallville as subject


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


class ModernRunnerOfflineTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.runtime_root = Path(self.temporary.name) / "live-runs"

  def tearDown(self):
    self.temporary.cleanup()

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


if __name__ == "__main__":
  unittest.main()
