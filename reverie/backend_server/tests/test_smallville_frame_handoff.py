import copy
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
import random
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
  sys.path.insert(0, str(BACKEND))

import controlled_replay
import modern_smallville as subject
from smallville_compatibility import copy_environment_actor_with_movement


class ModernTickFakeAdapter(controlled_replay.R1TDeterministicFakeAdapter):
  @staticmethod
  def response_for_caller(caller_id):
    if caller_id == "event_poignancy":
      return '{"output": "7"}'
    return controlled_replay.R1TDeterministicFakeAdapter.response_for_caller(
      caller_id)


class SmallvilleActorConversionTests(unittest.TestCase):
  def test_inserts_coordinates_and_preserves_legacy_dict_conversion(self):
    cases = (
      ({"maze": "the_ville"}, {"maze": "the_ville", "x": 30, "y": 40}),
      ([], {"x": 30, "y": 40}),
      ([('maze', 'the_ville')],
       {"maze": "the_ville", "x": 30, "y": 40}),
    )
    for environment_record, expected in cases:
      with self.subTest(environment_record=environment_record):
        self.assertEqual(
          expected,
          copy_environment_actor_with_movement(
            environment_record, [30, 40]))

  def test_overwrites_only_xy_with_shallow_copy_without_mutating_inputs(self):
    nested = {"items": ["unchanged"]}
    x_value = {"uncoerced": "x"}
    y_value = ["uncoerced-y"]
    environment_record = {
      "maze": "the_ville", "x": 10, "y": 20,
      "custom": "keep-me", "nested": nested,
    }
    movement = {
      "movement": [x_value, y_value],
      "description": "must not be copied",
      "pronunciatio": "must not be copied",
      "chat": [["A", "must not be copied"]],
    }
    environment_before = copy.deepcopy(environment_record)
    movement_before = copy.deepcopy(movement)
    rng_before = random.getstate()

    converted = copy_environment_actor_with_movement(
      environment_record, movement["movement"])

    self.assertIsNot(converted, environment_record)
    self.assertEqual(environment_before, environment_record)
    self.assertEqual(movement_before, movement)
    self.assertIs(nested, converted["nested"])
    self.assertIs(x_value, converted["x"])
    self.assertIs(y_value, converted["y"])
    self.assertEqual("keep-me", converted["custom"])
    self.assertFalse(
      {"description", "pronunciatio", "chat"}.intersection(converted))
    self.assertEqual(rng_before, random.getstate())

  def test_preserves_native_dict_failures_for_malformed_records(self):
    for environment_record, exception in (
        (None, TypeError), (7, TypeError), ("bad", ValueError), ([1], TypeError)):
      with self.subTest(environment_record=environment_record):
        with self.assertRaises(exception):
          copy_environment_actor_with_movement(environment_record, [1, 2])


class SmallvilleFrameHandoffIntegrationTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.runtime_root = Path(self.temporary.name) / "live-runs"

  def tearDown(self):
    self.temporary.cleanup()

  @staticmethod
  def _block_network(stack, attempts):
    def reject_network(*args, **kwargs):
      del args, kwargs
      attempts.append("attempt")
      raise AssertionError("network is forbidden")

    stack.enter_context(patch("socket.create_connection", reject_network))
    stack.enter_context(patch.object(socket.socket, "connect", reject_network))
    stack.enter_context(patch.object(socket.socket, "connect_ex", reject_network))

  def test_memory_regression_precedes_later_actor_coordinate_error(self):
    original_read = subject._read_json
    original_convert = subject.copy_environment_actor_with_movement
    original_metadata = subject._actor_tick_metadata
    events = []
    network_attempts = []

    def read_with_invalid_second_actor(path):
      value = original_read(path)
      path = Path(path)
      if path.parent.name == "movement" and path.name == "0.json":
        value = copy.deepcopy(value)
        value["persona"][subject.VISIBLE_ACTORS[1]]["movement"] = None
      return value

    def observed_convert(environment_record, coordinate):
      events.append(("convert", tuple(coordinate)))
      return original_convert(environment_record, coordinate)

    def regressed_metadata(persona, coordinate):
      events.append(("cognition", persona.name))
      current = original_metadata(persona, coordinate)
      if persona.name == subject.VISIBLE_ACTORS[0]:
        current["memory_node_count"] = -1
      return current

    config = subject.ModernRunConfig(
      run_name="frame-handoff-ordering", ticks=1,
      cognitive_actors=subject.VISIBLE_ACTORS, passive_actors=())
    with ExitStack() as stack:
      self._block_network(stack, network_attempts)
      stack.enter_context(patch.object(
        subject, "_read_json", side_effect=read_with_invalid_second_actor))
      stack.enter_context(patch.object(
        subject, "copy_environment_actor_with_movement",
        side_effect=observed_convert))
      stack.enter_context(patch.object(
        subject, "_actor_tick_metadata", side_effect=regressed_metadata))
      result = subject.run_modern_smallville(
        config, adapter=ModernTickFakeAdapter(),
        runtime_root=self.runtime_root)

    self.assertEqual("ModernRuntimeInvariantError", result.exception_type)
    self.assertEqual(
      "memory count regressed: actor=Isabella Rodriguez, tick=0",
      result.exception_message)
    self.assertEqual(2, len(events))
    self.assertEqual("convert", events[0][0])
    self.assertEqual(
      ("cognition", subject.VISIBLE_ACTORS[0]), events[1])
    self.assertEqual([], network_attempts)

  def test_multi_actor_conversion_and_cognition_remain_interleaved(self):
    original_convert = subject.copy_environment_actor_with_movement
    original_metadata = subject._actor_tick_metadata
    events = []
    network_attempts = []

    def observed_convert(environment_record, coordinate):
      events.append(("convert", tuple(coordinate)))
      return original_convert(environment_record, coordinate)

    def observed_metadata(persona, coordinate):
      events.append(("cognition", persona.name))
      return original_metadata(persona, coordinate)

    config = subject.ModernRunConfig(
      run_name="frame-handoff-interleaving", ticks=1,
      cognitive_actors=subject.VISIBLE_ACTORS, passive_actors=())
    with ExitStack() as stack:
      self._block_network(stack, network_attempts)
      stack.enter_context(patch.object(
        subject, "copy_environment_actor_with_movement",
        side_effect=observed_convert))
      stack.enter_context(patch.object(
        subject, "_actor_tick_metadata", side_effect=observed_metadata))
      result = subject.run_modern_smallville(
        config, adapter=ModernTickFakeAdapter(),
        runtime_root=self.runtime_root)

    report = subject._read_json(result.run_directory / "report.json")
    movement_root = Path(report["artifacts"]["simulation"]) / "movement"
    movement = subject._read_json(movement_root / "0.json")
    expected = []
    for name in subject.VISIBLE_ACTORS:
      expected.extend((
        ("convert", tuple(movement["persona"][name]["movement"])),
        ("cognition", name),
      ))
    self.assertEqual(expected, events)
    self.assertIsNone(result.exception_type)
    self.assertEqual([], network_attempts)

  def test_full_handoff_preserves_environment_metadata(self):
    original_read = subject._read_json
    network_attempts = []

    def read_with_transport_metadata(path):
      value = original_read(path)
      path = Path(path)
      if path.parent.name == "environment" and path.name == "0.json":
        value = copy.deepcopy(value)
        value["transport_meta"] = {"weather": "sun"}
        value["Invisible Actor"] = {"x": 999, "y": 998, "keep": True}
        value[subject.COGNITIVE_ACTOR]["custom"] = "keep-me"
      return value

    config = subject.ModernRunConfig(
      run_name="frame-handoff-metadata", ticks=1)
    with ExitStack() as stack:
      self._block_network(stack, network_attempts)
      stack.enter_context(patch.object(
        subject, "_read_json", side_effect=read_with_transport_metadata))
      result = subject.run_modern_smallville(
        config, adapter=ModernTickFakeAdapter(),
        runtime_root=self.runtime_root)

    report = original_read(result.run_directory / "report.json")
    simulation = Path(report["artifacts"]["simulation"])
    environment = original_read(simulation / "environment" / "1.json")
    movement = original_read(simulation / "movement" / "0.json")
    actor = environment[subject.COGNITIVE_ACTOR]
    self.assertEqual({"weather": "sun"}, environment["transport_meta"])
    self.assertEqual(
      {"x": 999, "y": 998, "keep": True},
      environment["Invisible Actor"])
    self.assertEqual("keep-me", actor["custom"])
    self.assertEqual(
      movement["persona"][subject.COGNITIVE_ACTOR]["movement"],
      [actor["x"], actor["y"]])
    for field in ("description", "pronunciatio", "chat"):
      self.assertNotIn(field, actor)
    self.assertEqual([], network_attempts)

  def test_nonzero_resume_indices_and_next_frame_ingestion(self):
    network_attempts = []
    with ExitStack() as stack:
      self._block_network(stack, network_attempts)
      source = subject.run_modern_smallville(
        subject.ModernRunConfig(
          run_name="frame-handoff-index-source", ticks=1,
          cost_ceiling_usd=Decimal("0.03")),
        adapter=ModernTickFakeAdapter(), runtime_root=self.runtime_root)

    reverie_module = subject._load_reverie_module()
    original_start_server = reverie_module.ReverieServer.start_server
    ticks = []

    def observed_start_server(server, count):
      step_before = server.step
      result = original_start_server(server, count)
      ticks.append({
        "step_before": step_before,
        "step_after": server.step,
        "tiles_after_ingestion": dict(server.personas_tile),
      })
      return result

    with ExitStack() as stack:
      self._block_network(stack, network_attempts)
      stack.enter_context(patch.object(
        subject, "_load_reverie_module", return_value=reverie_module))
      stack.enter_context(patch.object(
        reverie_module.ReverieServer, "start_server",
        new=observed_start_server))
      result = subject.run_modern_smallville_resume(
        subject.ModernResumeConfig(
          source_run=source.run_directory,
          run_name="frame-handoff-index-resume", ticks=2,
          cost_ceiling_usd=Decimal("0.03")),
        adapter=ModernTickFakeAdapter(), runtime_root=self.runtime_root)

    report = subject._read_json(result.run_directory / "report.json")
    simulation = Path(report["artifacts"]["simulation"])
    environment_two = subject._read_json(
      simulation / "environment" / "2.json")
    self.assertEqual(1, result.initial_step)
    self.assertEqual(3, result.final_step)
    self.assertEqual([1, 2], [tick["step_before"] for tick in ticks])
    self.assertEqual([2, 3], [tick["step_after"] for tick in ticks])
    self.assertTrue((simulation / "movement" / "1.json").is_file())
    self.assertTrue((simulation / "movement" / "2.json").is_file())
    self.assertTrue((simulation / "environment" / "3.json").is_file())
    self.assertEqual({
      name: (record["x"], record["y"])
      for name, record in environment_two.items()
    }, ticks[1]["tiles_after_ingestion"])
    self.assertEqual([], network_attempts)


if __name__ == "__main__":
  unittest.main()
