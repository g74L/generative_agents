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


class ModernRunConfigTests(unittest.TestCase):
  def test_default_policy_is_valid(self):
    config = subject.ModernRunConfig(run_name="offline-valid")
    self.assertEqual(1, config.ticks)
    self.assertEqual((subject.COGNITIVE_ACTOR,), config.cognitive_actors)
    self.assertEqual(subject.VISIBLE_ACTORS, config.visible_actors)

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
      run_directory=run_dir, completed_ticks=1, movement_count=1,
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
