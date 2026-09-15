import datetime
from contextlib import ExitStack
from decimal import Decimal
import json
import shutil
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
from persona import persona as persona_module
from persona.cognitive_modules import plan as plan_module
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
      return '{"decision":"yes"}'
    if caller_id == "agent_chat_summarize_relationship":
      return '{"relationship": "context"}'
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


class SourceActorRegistryTests(unittest.TestCase):
  START_TIME = datetime.datetime(2023, 2, 13, 5, 55, 0)

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.source = subject.SOURCE_ROOT / "base_the_ville_n25"
    self.names = subject.load_source_actor_registry(self.source)

  def config(self, **changes):
    values = dict(source_simulation="base_the_ville_n25",
                  run_name="source-registry", actor_registry="source",
                  visible_actors=self.names, cognitive_actors=self.names,
                  passive_actors=())
    values.update(changes)
    return subject.ModernRunConfig(**values)

  def test_n25_exact_registry_and_all_cognitive_policy(self):
    meta = subject._read_json(self.source / "reverie" / "meta.json")
    self.assertEqual(25, len(self.names))
    self.assertEqual(tuple(meta["persona_names"]), self.names)
    config = self.config()
    self.assertEqual(self.names, config.visible_actors)
    self.assertEqual(self.names, config.cognitive_actors)
    self.assertEqual((), config.passive_actors)

  def test_source_alone_does_not_opt_in_or_use_provider(self):
    config = subject.ModernRunConfig(source_simulation="base_the_ville_n25")
    self.assertEqual(subject.VISIBLE_ACTORS, config.visible_actors)
    with patch.object(subject, "_default_adapter") as provider:
      with self.assertRaises(subject.ModernRunConfigurationError):
        subject.run_modern_smallville(config, runtime_root=self.root)
    provider.assert_not_called()

  def test_source_policy_rejects_passive_actors_and_controlled_proximity(self):
    for changes in ({"passive_actors": (self.names[-1],)},
                    {"cognitive_actors": self.names[:-1]},
                    {"controlled_proximity": True},
                    {"actor_registry": "default"}):
      with self.subTest(changes=changes), self.assertRaises(
          subject.ModernRunConfigurationError):
        self.config(**changes)

  def test_start_time_requires_source_registry_and_timezone_free_datetime(self):
    invalid = (
      {"start_time": self.START_TIME},
      {"actor_registry": "source", "visible_actors": self.names,
       "cognitive_actors": self.names, "passive_actors": (),
       "start_time": "2023-02-13T05:55:00"},
      {"actor_registry": "source", "visible_actors": self.names,
       "cognitive_actors": self.names, "passive_actors": (),
       "start_time": self.START_TIME.replace(tzinfo=datetime.timezone.utc)},
    )
    for values in invalid:
      with self.subTest(values=values), self.assertRaises(
          subject.ModernRunConfigurationError):
        subject.ModernRunConfig(run_name="bad-start-time", **values)
    with self.assertRaisesRegex(
        subject.ModernRunConfigurationError, "controlled proximity"):
      self.config(start_time=self.START_TIME, controlled_proximity=True)

  def test_start_time_override_changes_only_isolated_world_meta(self):
    isolated = self.root / "isolated"
    shutil.copytree(self.source, isolated)
    source_before = subject._tree_sha256(self.source)
    meta_before = subject._read_json(isolated / "reverie" / "meta.json")
    environment_before = (isolated / "environment" / "0.json").read_bytes()
    file_hashes_before = {
      path.relative_to(isolated).as_posix(): subject._file_sha256(path)
      for path in isolated.rglob("*") if path.is_file()}
    scratch_before = {
      name: (isolated / "personas" / name / "bootstrap_memory"
             / "scratch.json").read_bytes()
      for name in self.names}

    evidence = subject._override_isolated_source_start_time(
      isolated, self.START_TIME)

    meta_after = subject._read_json(isolated / "reverie" / "meta.json")
    expected_meta = dict(meta_before)
    expected_meta["curr_time"] = "February 13, 2023, 05:55:00"
    self.assertEqual(expected_meta, meta_after)
    self.assertEqual(0, meta_after["step"])
    self.assertEqual(self.names, tuple(meta_after["persona_names"]))
    self.assertEqual({
      "requested_start_time": self.START_TIME,
      "effective_start_time": "February 13, 2023, 05:55:00",
      "step": 0,
    }, evidence)
    self.assertEqual(
      environment_before, (isolated / "environment" / "0.json").read_bytes())
    for name in self.names:
      scratch_path = (isolated / "personas" / name / "bootstrap_memory"
                      / "scratch.json")
      self.assertEqual(scratch_before[name], scratch_path.read_bytes())
      scratch = subject._read_json(scratch_path)
      self.assertIsNone(scratch["curr_time"])
    file_hashes_after = {
      path.relative_to(isolated).as_posix(): subject._file_sha256(path)
      for path in isolated.rglob("*") if path.is_file()}
    self.assertEqual(file_hashes_before.keys(), file_hashes_after.keys())
    self.assertEqual(
      ["reverie/meta.json"],
      [name for name in file_hashes_before
       if file_hashes_before[name] != file_hashes_after[name]])
    self.assertEqual(source_before, subject._tree_sha256(self.source))

  def test_start_time_cli_is_strict_and_not_available_to_resume(self):
    parser = subject.build_parser()
    args = parser.parse_args([
      "run", "--source", "base_the_ville_n25",
      "--actor-registry", "source",
      "--start-time", "2023-02-13T05:55:00"])
    self.assertEqual(self.START_TIME, args.start_time)
    for value in ("invalid", "2023-02-30T05:55:00",
                  "2023-02-13", "2023-02-13T05:55:00+01:00"):
      with self.subTest(value=value), self.assertRaises(SystemExit):
        parser.parse_args(["run", "--start-time", value])
    with self.assertRaises(SystemExit):
      parser.parse_args([
        "resume", "--from", "run", "--start-time",
        "2023-02-13T05:55:00"])

  def test_invalid_start_time_fails_before_provider_or_source_mutation(self):
    before = subject._tree_sha256(self.source)
    with patch.object(subject, "_default_adapter") as provider, \
        self.assertRaises(SystemExit):
      subject.main([
        "run", "--source", "base_the_ville_n25",
        "--actor-registry", "source", "--start-time", "invalid"])
    provider.assert_not_called()
    self.assertEqual(before, subject._tree_sha256(self.source))

  def test_start_time_mode_conflicts_fail_before_runtime(self):
    cases = (
      ["run", "--start-time", "2023-02-13T05:55:00"],
      ["run", "--source", "base_the_ville_n25",
       "--actor-registry", "source", "--controlled-proximity",
       "--start-time", "2023-02-13T05:55:00"],
    )
    for args in cases:
      with self.subTest(args=args), patch.object(
          subject, "run_modern_smallville") as runner:
        self.assertEqual(2, subject.main(args))
        runner.assert_not_called()

  def test_invalid_source_names_fail_closed(self):
    meta = subject._read_json(self.source / "reverie" / "meta.json")
    for names in ([], None, "Latoya Williams", [""], [None], [[]],
                  ["../outside"], ["  "], ["Latoya Williams"] * 2):
      with self.subTest(names=names), self.assertRaises(
          subject.ModernRunConfigurationError):
        subject.load_source_actor_registry(self.source, {**meta, "persona_names": names})

  def test_missing_persona_directory_and_environment_actor_fail_closed(self):
    source = self.root / "source"
    shutil.copytree(self.source, source)
    persona = source / "personas" / self.names[-1]
    renamed = persona.with_name("omitted-persona")
    persona.rename(renamed)
    with self.assertRaisesRegex(subject.ModernRunConfigurationError, "directory"):
      subject.load_source_actor_registry(source)
    renamed.rename(persona)
    path = source / "environment" / "0.json"
    environment = subject._read_json(path)
    del environment[self.names[-1]]
    subject._write_json(path, environment)
    with self.assertRaisesRegex(subject.ModernRunConfigurationError, "environment actor"):
      subject.load_source_actor_registry(source)

  def test_config_order_must_match_source_before_provider_use(self):
    config = self.config(visible_actors=tuple(reversed(self.names)),
                         cognitive_actors=tuple(reversed(self.names)))
    with patch.object(subject, "_default_adapter") as provider:
      with self.assertRaises(subject.ModernRunConfigurationError):
        subject.run_modern_smallville(config, runtime_root=self.root)
    provider.assert_not_called()

  def test_cli_source_opt_in_and_existing_isabella_policy(self):
    for flags, expected, mode in (
        (["--source", "base_the_ville_n25", "--actor-registry", "source"],
         self.names, "source"),
        (["--cognitive", "isabella"], (subject.COGNITIVE_ACTOR,), "default")):
      with self.subTest(mode=mode), patch.object(
          subject, "run_modern_smallville",
          side_effect=subject.ModernRunConfigurationError("capture")) as runner:
        self.assertEqual(2, subject.main(["run", *flags]))
        config = runner.call_args.args[0]
        self.assertEqual(expected, config.cognitive_actors)
        self.assertEqual(mode, config.actor_registry)
    with self.assertRaises(SystemExit):
      subject.build_parser().parse_args([
        "run", "--actor-registry", "source", "--cognitive", "isabella"])

  def test_real_n25_hydration_preflight_and_initial_save_reload(self):
    before = subject._tree_sha256(self.source)
    meta = subject._read_json(self.source / "reverie" / "meta.json")
    environment = subject._read_json(self.source / "environment" / "0.json")
    module = subject._load_reverie_module()
    observed = []

    def inspect_before_tick(server, ticks):
      self.assertEqual(1, ticks)
      self.assertEqual(self.names, tuple(server.personas))
      self.assertEqual(0, server.step)
      self.assertEqual(datetime.datetime(2023, 2, 13), server.curr_time)
      self.assertEqual("Maze", type(server.maze).__name__)
      for name, persona in server.personas.items():
        self.assertIsNone(persona.scratch.curr_time)
        self.assertEqual((environment[name]["x"], environment[name]["y"]),
                         server.personas_tile[name])
      self.assertTrue(all(subject._actor_object_isolation(
        server.personas, self.names, Path(module.fs_storage) / server.sim_code).values()))
      server.save()
      reloaded = module.ReverieServer(server.sim_code, "n25-initial-reload")
      self.assertEqual(self.names, tuple(reloaded.personas))
      self.assertEqual(server.curr_time, reloaded.curr_time)
      self.assertEqual(server.step, reloaded.step)
      for name in self.names:
        self.assertEqual(subject._actor_state_metadata(server.personas[name]),
                         subject._actor_state_metadata(reloaded.personas[name]))
        self.assertIsNone(reloaded.personas[name].scratch.curr_time)
      observed.append(len(reloaded.personas))
      raise RuntimeError("structural probe stops before cognition")

    adapter = ModernTickFakeAdapter()
    with patch.object(subject, "_load_reverie_module", return_value=module), \
        patch.object(module.ReverieServer, "start_server", inspect_before_tick), \
        patch.object(subject, "_bootstrap_isolated_temporal_source") as bootstrap, \
        patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
      result = subject.run_modern_smallville(
        self.config(), adapter=adapter, runtime_root=self.root / "runs")
    self.assertEqual([25], observed)
    self.assertEqual("structural probe stops before cognition", result.exception_message)
    self.assertEqual([], adapter.calls)
    bootstrap.assert_not_called()
    report = subject._read_json(result.run_directory / "report.json")
    self.assertEqual(list(self.names), report["actor_policy"]["visible"])
    self.assertEqual([controlled_replay.EMPTY_BOOTSTRAPPABLE] * 25,
                     report["embedding_preflight"]["before"])
    self.assertEqual([controlled_replay.MODERN_COMPATIBLE] * 25,
                     report["embedding_preflight"]["after"])
    self.assertEqual(list(self.names),
                     report["embedding_preflight"]["bootstrapped_personas"])
    self.assertEqual({name: 0 for name in self.names}, dict(result.actor_move_counts))
    isolated = result.run_directory / "fixture" / "storage" / "source-registry-source"
    self.assertEqual(meta, subject._read_json(isolated / "reverie" / "meta.json"))
    for name in self.names:
      relative = Path("personas") / name / "bootstrap_memory"
      for filename in ("scratch.json", "spatial_memory.json"):
        self.assertEqual((self.source / relative / filename).read_bytes(),
                         (isolated / relative / filename).read_bytes())
    audits = controlled_replay.prepare_isolated_embedding_stores(
      controlled_replay.prepare_isolated_reverie_fixture(
        isolated, result.run_directory / "fixture", self.source, 0), self.names)
    self.assertEqual(25, len(audits.audits_after))
    self.assertEqual((), audits.bootstrapped_personas)
    self.assertTrue(report["fixture"]["source_unchanged"])
    self.assertEqual(before, subject._tree_sha256(self.source))

  def test_real_n25_start_time_preserves_native_pre_move_state_and_reports(self):
    before = subject._tree_sha256(self.source)
    source_meta = subject._read_json(self.source / "reverie" / "meta.json")
    source_environment = (self.source / "environment" / "0.json").read_bytes()
    scratch_before = {
      name: (self.source / "personas" / name / "bootstrap_memory"
             / "scratch.json").read_bytes()
      for name in self.names}
    module = subject._load_reverie_module()
    observed = []

    def inspect_before_tick(server, ticks):
      self.assertEqual(1, ticks)
      self.assertEqual(self.names, tuple(server.personas))
      self.assertEqual(0, server.step)
      self.assertEqual(self.START_TIME, server.curr_time)
      for persona in server.personas.values():
        self.assertIsNone(persona.scratch.curr_time)
      observed.append((server.step, server.curr_time, len(server.personas)))
      raise RuntimeError("start-time probe stops before cognition")

    adapter = ModernTickFakeAdapter()
    with patch.object(subject, "_load_reverie_module", return_value=module), \
        patch.object(module.ReverieServer, "start_server", inspect_before_tick), \
        patch.object(subject, "_bootstrap_isolated_temporal_source") as bootstrap, \
        patch.object(socket.socket, "connect",
                     side_effect=AssertionError("network forbidden")):
      result = subject.run_modern_smallville(
        self.config(start_time=self.START_TIME), adapter=adapter,
        runtime_root=self.root / "runs")

    self.assertEqual([(0, self.START_TIME, 25)], observed)
    self.assertEqual([], adapter.calls)
    bootstrap.assert_not_called()
    self.assertEqual("start-time probe stops before cognition",
                     result.exception_message)
    report = subject._read_json(result.run_directory / "report.json")
    self.assertEqual("February 13, 2023, 05:55:00",
                     report["config"]["start_time"])
    self.assertEqual("February 13, 2023, 05:55:00",
                     report["result"]["initial_time"])
    self.assertEqual("February 13, 2023, 05:55:00",
                     report["fixture"]["seed"]["effective_start_time"])
    self.assertTrue(report["fixture"]["source_unchanged"])

    isolated = (result.run_directory / "fixture" / "storage"
                / "source-registry-source")
    isolated_meta = subject._read_json(isolated / "reverie" / "meta.json")
    expected_meta = dict(source_meta)
    expected_meta["curr_time"] = "February 13, 2023, 05:55:00"
    self.assertEqual(expected_meta, isolated_meta)
    self.assertEqual(
      source_environment, (isolated / "environment" / "0.json").read_bytes())
    for name in self.names:
      scratch_path = (isolated / "personas" / name / "bootstrap_memory"
                      / "scratch.json")
      self.assertEqual(scratch_before[name], scratch_path.read_bytes())
      scratch = subject._read_json(scratch_path)
      for field in (
          "curr_time", "daily_plan_req", "daily_req", "f_daily_schedule",
          "act_address", "act_description", "act_event", "planned_path"):
        self.assertEqual(
          subject._read_json(
            self.source / "personas" / name / "bootstrap_memory"
            / "scratch.json")[field], scratch[field])
    self.assertEqual(before, subject._tree_sha256(self.source))

  def test_source_order_drives_real_tick_reports_reload_and_resume(self):
    # Reuse the established R1V seed on a disposable three-person copy.
    # This exercises the complete harness without inventing N25 fake cognition.
    source = self.root / "ordered-source"
    shutil.copytree(subject.SOURCE_ROOT / subject.DEFAULT_SOURCE, source)
    subject._bootstrap_isolated_temporal_source(source, subject.VISIBLE_ACTORS)
    names = tuple(reversed(subject.VISIBLE_ACTORS))
    path = source / "reverie" / "meta.json"
    meta = subject._read_json(path)
    meta["persona_names"] = list(names)
    subject._write_json(path, meta)
    before = subject._tree_sha256(source)
    with patch.object(subject, "SOURCE_ROOT", self.root):
      result = subject.run_modern_smallville(
        self.config(source_simulation=source.name, visible_actors=names,
                    cognitive_actors=names), adapter=ModernTickFakeAdapter(),
        runtime_root=self.root / "runs")
      self.assertEqual("MODERN_SMALLVILLE_HEADLESS_RUN_PASSED", result.verdict,
                       result.exception_message)
      report = subject._read_json(result.run_directory / "report.json")
      self.assertEqual(names, tuple(name for name, _ in result.actor_move_counts))
      for key in ("continuity", "movement_integrity", "multi_actor_isolation"):
        self.assertTrue(report[key]["all_checks_passed"])
      self.assertTrue(report["telemetry"]["attribution_valid"])
      self.assertEqual(set(names), set(report["tick_progression"][0]["actors"]))
      self.assertEqual(3, report["reload"]["persona_count"])
      resumed = subject.run_modern_smallville_resume(
        subject.ModernResumeConfig(source_run=result.run_directory,
                                   run_name="resumed-source", ticks=1),
        adapter=ModernTickFakeAdapter(), runtime_root=self.root / "runs")
    self.assertEqual(subject.R1CLI_A2_A_READY_VERDICT, resumed.verdict,
                     resumed.exception_message)
    self.assertEqual(names, resumed.cognitive_actors)
    self.assertEqual((), resumed.passive_actors)
    self.assertTrue(resumed.reload_passed)
    self.assertEqual(before, subject._tree_sha256(source))


class ActorStateDailyPlanInvariantTests(unittest.TestCase):
  @staticmethod
  def _persona(daily_plan_req, daily_req, name="Synthetic Actor"):
    return SimpleNamespace(
      name=name,
      scratch=SimpleNamespace(
        name=name, daily_plan_req=daily_plan_req, daily_req=daily_req,
        f_daily_schedule=[["synthetic activity", 1440]],
        act_description="synthetic activity",
        act_address="the Ville:synthetic sector:synthetic arena:object",
        act_event=(name, "is", "active"),
        act_start_time=datetime.datetime(2023, 2, 13), act_duration=60),
      a_mem=SimpleNamespace(
        id_to_node={}, embeddings={}, seq_event=[], seq_thought=[], seq_chat=[]),
      s_mem=SimpleNamespace(tree={}))

  def test_daily_plan_presence_uses_generated_plan_only(self):
    cases = (
      ("", ["generated plan"], True),
      ("optional requirement", [], False),
      ("optional requirement", ["generated plan"], True),
      ("", [], False),
    )
    for daily_plan_req, daily_req, expected in cases:
      with self.subTest(
          daily_plan_req=bool(daily_plan_req), daily_req=bool(daily_req)):
        state = subject._actor_state_metadata(
          self._persona(daily_plan_req, daily_req))
        self.assertIs(expected, state["daily_plan_present"])

  def test_n25_requirement_partition_does_not_control_plan_presence(self):
    source = subject.SOURCE_ROOT / "base_the_ville_n25"
    expected_requirement = {
      "Latoya Williams": False,
      "Rajiv Patel": False,
      "Abigail Chen": False,
      "Adam Smith": False,
      "Carmen Ortiz": True,
    }
    for name, requirement_present in expected_requirement.items():
      scratch = subject._read_json(
        source / "personas" / name / "bootstrap_memory" / "scratch.json")
      self.assertIs(requirement_present, bool(scratch["daily_plan_req"]))
      with self.subTest(name=name, generated_plan="present"):
        self.assertTrue(subject._actor_state_metadata(self._persona(
          scratch["daily_plan_req"], ["generated plan"], name))[
            "daily_plan_present"])
      with self.subTest(name=name, generated_plan="absent"):
        self.assertFalse(subject._actor_state_metadata(self._persona(
          scratch["daily_plan_req"], [], name))["daily_plan_present"])

  def test_first_day_planning_writes_generated_plan_and_preserves_requirement(self):
    persona = self._persona("", [])
    persona.scratch.curr_time = datetime.datetime(2023, 2, 13)
    added_thoughts = []
    persona.a_mem.add_thought = lambda *args: added_thoughts.append(args)
    with patch.object(plan_module, "generate_wake_up_hour", return_value=7), \
        patch.object(plan_module, "generate_first_daily_plan",
                     return_value=["generated plan"]), \
        patch.object(plan_module, "generate_hourly_schedule",
                     return_value=[["synthetic activity", 1440]]), \
        patch.object(plan_module, "get_embedding", return_value=[0.0]):
      plan_module._long_term_planning(persona, "First day")
    self.assertEqual("", persona.scratch.daily_plan_req)
    self.assertEqual(["generated plan"], persona.scratch.daily_req)
    self.assertEqual([["synthetic activity", 1440]],
                     persona.scratch.f_daily_schedule)
    self.assertEqual(persona.scratch.f_daily_schedule,
                     persona.scratch.f_daily_schedule_hourly_org)
    self.assertEqual(1, len(added_thoughts))
    self.assertTrue(subject._actor_state_metadata(persona)[
      "daily_plan_present"])


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

  def test_accounting_diagnostic_round_trips_without_private_content(self):
    diagnostic = subject.AccountingFailureDiagnostic(
      schema_version=1,
      operation="COMPLETION_COMPAT", model="gpt-4o-mini",
      response_model="gpt-4o-mini-2024-07-18",
      caller_id="generate_hourly_schedule", cognitive_category="WORLD_TICK",
      actor_id="Ayesha Khan", simulation_id="synthetic-offline-run",
      simulation_step=0, logical_call_id="logical-safe-id",
      physical_attempt=1, provider_outcome="ERROR",
      provider_error_type=None, normalized_result_type=None,
      normalized_error_type="LLMIncompleteResponseError",
      request_id="req-safe-id", finish_reason="length",
      response_status="incomplete", usage_present=True,
      usage_shape="PARTIAL", input_tokens=20, output_tokens=None,
      cached_input_tokens=None, reasoning_tokens=None,
      usage_validation_category="PARTIAL", pricing_status="PARTIAL",
      failure_stage="PROVIDER_NORMALIZATION",
      original_exception_type=None,
      sanitized_exception_message="estimated total cost is unavailable",
      guard_classification="ACCOUNTING_UNAVAILABLE",
      guard_action="TRIPPED_AND_RAISED")
    error = subject.ReplayCostAccountingUnavailableError(
      diagnostic.operation, diagnostic.model, diagnostic)
    result = self._result()
    result = subject.ModernRunResult(
      **{**result.__dict__,
         "exception_type": type(error).__name__,
         "exception_message": str(error)})
    failure = subject._build_failure_report(
      self._execution_state(), result, error)
    forbidden = (
      "secret prompt", "secret response", "API key", "memory content")
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "report.json"
      subject._write_json(path, {"failure": failure})
      serialized = path.read_text(encoding="utf-8")
      reloaded = subject._read_json(path)
    persisted = reloaded["failure"]["accounting_failure"]
    self.assertEqual("PROVIDER_NORMALIZATION", persisted["failure_stage"])
    self.assertEqual("LLMIncompleteResponseError",
                     persisted["normalized_error_type"])
    self.assertEqual(20, persisted["input_tokens"])
    self.assertIsNone(persisted["output_tokens"])
    for private_text in forbidden:
      self.assertNotIn(private_text, serialized)

    def fail_plan():
      raise error

    state = {"stage": "persona_move", "actor": "Ayesha Khan", "tick": 0}
    with self.assertRaises(subject.ReplayCostAccountingUnavailableError) as caught:
      with subject._observe_persona_move_failure(
          SimpleNamespace(plan=fail_plan), state):
        fail_plan()
    self.assertIs(error, caught.exception)
    combined = subject._build_failure_report(state, result, error)
    self.assertEqual(failure["accounting_failure"], combined["accounting_failure"])
    self.assertEqual({"stage": "PLAN", "function": "persona.plan"},
                     combined["cognitive_failure"])
    self.assertNotIn("cognitive_failure", failure)

  def test_unrelated_failure_does_not_gain_accounting_diagnostic(self):
    failure = subject._build_failure_report(
      self._execution_state(), self._result(), RuntimeError("boom"))
    self.assertNotIn("accounting_failure", failure)


class PersonaMoveFailureDiagnosticTests(unittest.TestCase):
  STAGES = ("PERCEIVE", "RETRIEVE", "PLAN", "REFLECT", "EXECUTE")
  PRIVATE = (
    "private prompt", "private response", "private memory",
    "private retrieved text", "private plan", "private reflection",
    "private dialogue", "private embedding vector", "sk-private-credential",
    "private provider payload")

  def _compare_move(self, failing_stage=None, previous_time=None, now=None):
    # Execute the real, unmodified Persona.move and its public methods. Only
    # module functions are deterministic fakes; no N25 model output is invented.
    persona = persona_module.Persona.__new__(persona_module.Persona)
    persona.name = "Synthetic Actor"
    maze, perceived, retrieved, plan, execution = (object() for _ in range(5))
    personas = {persona.name: persona}
    now = now or datetime.datetime(2023, 2, 13)
    tile = (2, 3)
    error = type("Synthetic" + (failing_stage or "Success").title() + "Error",
                 (RuntimeError,), {})(" ".join(self.PRIVATE))
    calls = []

    def record(stage, *args):
      calls.append((stage, args))
      persona.scratch.mutations.append(stage)
      if stage == failing_stage:
        raise error

    def fake_perceive(actor, world):
      record("PERCEIVE", actor, world)
      return perceived

    def fake_retrieve(actor, events):
      record("RETRIEVE", actor, events)
      return retrieved

    def fake_plan(actor, world, registry, new_day, memories):
      record("PLAN", actor, world, registry, new_day, memories)
      return plan

    def fake_reflect(actor):
      record("REFLECT", actor)

    def fake_execute(actor, world, registry, action):
      record("EXECUTE", actor, world, registry, action)
      return execution

    expected_new_day = (
      "First day" if previous_time is None else
      "New day" if previous_time.date() != now.date() else False)
    expected_calls = [
      ("PERCEIVE", (persona, maze)),
      ("RETRIEVE", (persona, perceived)),
      ("PLAN", (persona, maze, personas, expected_new_day, retrieved)),
      ("REFLECT", (persona,)),
      ("EXECUTE", (persona, maze, personas, plan)),
    ]
    if failing_stage:
      expected_calls = expected_calls[:self.STAGES.index(failing_stage) + 1]
    states = []
    with ExitStack() as stack:
      for stage, function in zip(self.STAGES, (
          fake_perceive, fake_retrieve, fake_plan, fake_reflect, fake_execute)):
        stack.enter_context(patch.object(persona_module, stage.lower(), function))
      for enabled in (False, True):
        calls.clear()
        persona.scratch = SimpleNamespace(
          curr_time=previous_time, curr_tile=None, mutations=[])
        before_keys = set(vars(persona))
        state = {"stage": "persona_move", "actor": persona.name, "tick": 0}
        try:
          with subject._observe_persona_move_failure(persona, state, enabled=enabled):
            value = persona.move(maze, personas, tile, now)
        except Exception as caught:
          self.assertIs(error, caught)
          self.assertIs(type(error), type(caught))
          self.assertIsNone(caught.__cause__)
          self.assertIsNone(caught.__context__)
          self.assertIsNotNone(caught.__traceback__)
        else:
          self.assertIsNone(failing_stage)
          self.assertIs(execution, value)
        self.assertEqual(expected_calls, calls)
        self.assertEqual([stage for stage, _ in expected_calls],
                         persona.scratch.mutations)
        self.assertIs(now, persona.scratch.curr_time)
        self.assertIs(tile, persona.scratch.curr_tile)
        self.assertEqual(before_keys, set(vars(persona)))
        states.append(state)
    self.assertNotIn("cognitive_failure", states[0])
    if failing_stage:
      self.assertEqual({"stage": failing_stage,
                        "function": "persona." + failing_stage.lower()},
                       states[1]["cognitive_failure"])
    else:
      self.assertNotIn("cognitive_failure", states[1])

  def test_perceive_failure_is_transparent(self):
    self._compare_move("PERCEIVE")

  def test_world_0555_with_null_scratch_is_first_day(self):
    self._compare_move(now=datetime.datetime(2023, 2, 13, 5, 55, 0))

  def test_retrieve_failure_is_transparent(self):
    self._compare_move("RETRIEVE")

  def test_plan_failure_is_transparent(self):
    self._compare_move("PLAN")

  def test_reflect_failure_is_transparent(self):
    self._compare_move("REFLECT")

  def test_execute_failure_is_transparent(self):
    self._compare_move("EXECUTE")

  def test_success_preserves_sequence_arguments_result_and_scratch(self):
    for previous in (None, datetime.datetime(2023, 2, 12),
                     datetime.datetime(2023, 2, 13)):
      with self.subTest(previous=previous):
        self._compare_move(previous_time=previous)

  def test_scratch_failure_is_unknown_without_stale_previous_stage(self):
    persona = persona_module.Persona.__new__(persona_module.Persona)
    persona.scratch = SimpleNamespace(curr_time=object())
    state = {"cognitive_failure": {"stage": "EXECUTE"}}
    with self.assertRaises(AttributeError):
      with subject._observe_persona_move_failure(persona, state):
        persona.move(None, {}, (1, 2), datetime.datetime(2023, 2, 13))
    self.assertEqual({"stage": "UNKNOWN", "function": None},
                     state["cognitive_failure"])
    with subject._observe_persona_move_failure(persona, state):
      pass
    self.assertNotIn("cognitive_failure", state)

  def test_outer_public_stage_wins_over_nested_stage_and_chained_cause(self):
    error = TypeError("'NoneType' object is not iterable")
    cause = ValueError("private memory")

    class NestedPersona:
      def plan(self):
        self.retrieve()

      def retrieve(self):
        raise error from cause

    persona = NestedPersona()
    state = {}
    with self.assertRaises(TypeError) as caught:
      with subject._observe_persona_move_failure(persona, state):
        persona.plan()
    self.assertIs(error, caught.exception)
    self.assertIs(cause, caught.exception.__cause__)
    self.assertEqual("PLAN", state["cognitive_failure"]["stage"])

  def test_ambiguous_code_and_observer_lookup_errors_fail_unknown(self):
    error = RuntimeError("private response")

    def shared():
      raise error

    persona = SimpleNamespace(plan=shared, retrieve=shared)
    for broken_lookup in (False, True):
      with self.subTest(broken_lookup=broken_lookup), ExitStack() as stack:
        if broken_lookup:
          stack.enter_context(patch.object(subject, "_persona_cognitive_boundary",
                                          side_effect=ValueError("lookup failed")))
        state = {}
        with self.assertRaises(RuntimeError) as caught:
          with subject._observe_persona_move_failure(persona, state):
            shared()
        self.assertIs(error, caught.exception)
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual({"stage": "UNKNOWN", "function": None},
                         state["cognitive_failure"])

  def test_privacy_never_formats_objects_or_custom_exceptions(self):
    class PrivateObject:
      def __repr__(self):
        raise AssertionError("repr must not run")

    class PrivateError(RuntimeError):
      def __str__(self):
        raise AssertionError("str must not run")

    for error in (PrivateError(PrivateObject()), TypeError(PrivateObject()),
                  TypeError("'NoneType' object is not iterable " + " ".join(self.PRIVATE)),
                  RuntimeError(" ".join(self.PRIVATE) * 100)):
      message = subject._sanitized_cognitive_exception_message(error)
      self.assertLessEqual(len(message), 128)
      for private in self.PRIVATE:
        self.assertNotIn(private, message)
    self.assertEqual("'NoneType' object is not iterable",
                     subject._sanitized_cognitive_exception_message(
                       TypeError("'NoneType' object is not iterable")))

  def test_launcher_serializes_each_stage_without_private_content(self):
    # Real launcher/artifact path on the established three-actor fixture only.
    cases = [(stage, False) for stage in self.STAGES] + [("REFLECT", True)]
    for stage, observe_reflection in cases:
      with self.subTest(stage=stage, reflection=observe_reflection), \
          tempfile.TemporaryDirectory() as tmp:
        error = type("Synthetic" + stage.title() + "Error", (RuntimeError,), {})(
          " ".join(self.PRIVATE) * 100)

        def fail(*args, **kwargs):
          raise error

        with patch.object(persona_module.Persona, stage.lower(), fail), \
            patch.object(socket.socket, "connect",
                         side_effect=AssertionError("network forbidden")):
          result = subject.run_modern_smallville(
            subject.ModernRunConfig(
              run_name="synthetic-stage-" + stage.lower(),
              observe_reflection_lifecycle=observe_reflection),
            adapter=ModernTickFakeAdapter(), runtime_root=Path(tmp))
        report_path = result.run_directory / "report.json"
        report = subject._read_json(report_path)
        failure = report["failure"]
        self.assertEqual("MODERN_SMALLVILLE_HEADLESS_RUN_FAILED", result.verdict)
        self.assertEqual(0, result.completed_ticks)
        self.assertFalse(result.save_passed)
        self.assertEqual("persona_move", failure["stage"])
        self.assertEqual(subject.COGNITIVE_ACTOR, failure["actor"])
        self.assertEqual(0, failure["tick"])
        self.assertEqual(type(error).__name__, failure["exception_type"])
        self.assertEqual({"stage": stage, "function": "persona." + stage.lower()},
                         failure["cognitive_failure"])
        self.assertIsNone(failure["caller"])
        self.assertIsNone(failure["operation"])
        self.assertNotIn("accounting_failure", failure)
        self.assertEqual(result.exception_message, failure["exception_message"])
        for path in (report_path, result.run_directory / "status.json"):
          serialized = path.read_text(encoding="utf-8")
          for private in self.PRIVATE:
            self.assertNotIn(private, serialized)
        self.assertEqual({"stage", "actor", "tick", "exception_type",
                          "exception_message", "caller", "operation",
                          "cognitive_failure"}, set(failure))


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
    self.assertTrue(check["last_accessed_preserved"], check)

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

  @staticmethod
  def _encounter_fixture(observer_scratch, target_scratch,
                         observer_tile=(90, 74), target_tile=(91, 74),
                         include_target_position=True):
    perceived = [SimpleNamespace(
      subject="John Lin", type="event", node_id="node_1")]
    mei = SimpleNamespace(
      name="Mei Lin", scratch=SimpleNamespace(curr_tile=observer_scratch),
      a_mem=SimpleNamespace(
        id_to_node={"node_1": perceived[0]},
        get_last_chat=lambda target: False),
      perceive=lambda maze: perceived)
    john = SimpleNamespace(
      name="John Lin", scratch=SimpleNamespace(curr_tile=target_scratch),
      a_mem=SimpleNamespace(
        id_to_node={}, get_last_chat=lambda target: False),
      perceive=lambda maze: [])
    personas_tile = {"Mei Lin": observer_tile}
    if include_target_position:
      personas_tile["John Lin"] = target_tile
    server = SimpleNamespace(
      personas={"Mei Lin": mei, "John Lin": john},
      personas_tile=personas_tile)
    arena_by_tile = {
      tuple(observer_tile): "the Ville:Lin house:shared bedroom",
      tuple(target_tile): "the Ville:Lin house:shared bedroom",
    }
    maze = SimpleNamespace(
      get_tile_path=lambda tile, level: arena_by_tile[tuple(tile)])
    return server, maze, perceived

  @staticmethod
  def _run_observed_perception(server, maze):
    observer = subject._ConversationObserver(
      server, {"tick": 0}, "simulation").install()
    try:
      returned = server.personas["Mei Lin"].perceive(maze)
    finally:
      observer.restore()
    return observer, returned

  def test_encounter_uses_authoritative_target_when_scratch_uninitialized(self):
    server, maze, unused = self._encounter_fixture((90, 74), None)
    observer, unused = self._run_observed_perception(server, maze)
    self.assertEqual(1.0, observer.encounters[0]["distance"])

  def test_encounter_uses_authoritative_positions_when_both_scratches_none(self):
    server, maze, unused = self._encounter_fixture(
      None, None, observer_tile=(2, 2), target_tile=(5, 6))
    observer, unused = self._run_observed_perception(server, maze)
    self.assertEqual(5.0, observer.encounters[0]["distance"])
    self.assertTrue(observer.encounters[0]["same_arena"])

  def test_encounter_ignores_stale_target_scratch_position(self):
    server, maze, unused = self._encounter_fixture(
      (2, 2), (100, 100), observer_tile=(2, 2), target_tile=(5, 6))
    observer, unused = self._run_observed_perception(server, maze)
    self.assertEqual(5.0, observer.encounters[0]["distance"])

  def test_encounter_semantics_preserved_when_scratch_matches_world(self):
    server, maze, unused = self._encounter_fixture(
      (2, 2), (5, 6), observer_tile=(2, 2), target_tile=(5, 6))
    observer, unused = self._run_observed_perception(server, maze)
    self.assertEqual("Mei Lin", observer.encounters[0]["observer"])
    self.assertEqual("John Lin", observer.encounters[0]["target"])
    self.assertEqual(5.0, observer.encounters[0]["distance"])
    self.assertTrue(observer.encounters[0]["same_arena"])

  def test_encounter_observer_preserves_perception_result_identity(self):
    server, maze, perceived = self._encounter_fixture(None, None)
    unused, returned = self._run_observed_perception(server, maze)
    self.assertIs(perceived, returned)
    self.assertEqual(["node_1"], [node.node_id for node in returned])

  def test_encounter_fails_explicitly_when_authoritative_target_is_missing(self):
    server, maze, unused = self._encounter_fixture(
      None, (91, 74), include_target_position=False)
    observer = subject._ConversationObserver(
      server, {"tick": 0}, "simulation").install()
    try:
      with self.assertRaisesRegex(KeyError, "John Lin"):
        server.personas["Mei Lin"].perceive(maze)
    finally:
      observer.restore()

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

  def test_ranking_fidelity_reporting_uses_schema_and_hydration_evidence(self):
    current = subject._social_memory_snapshot(self._bilateral_nodes())
    self.assertFalse(current["last_accessed_persistence_gap_detected"])
    self.assertEqual(
      "PRESERVED", subject._ranking_state_fidelity(
        current["last_accessed_persistence_gap_detected"]))

    legacy = {
      actor: {
        node_id: {
          key: value for key, value in vars(node).items()
          if key != "last_accessed"
        }
        for node_id, node in nodes.items()
      }
      for actor, nodes in self._bilateral_nodes().items()
    }
    legacy_snapshot = subject._social_memory_snapshot(legacy)
    self.assertTrue(
      legacy_snapshot["last_accessed_persistence_gap_detected"])
    self.assertEqual(
      "NOT_FULLY_GUARANTEED", subject._ranking_state_fidelity(
        legacy_snapshot["last_accessed_persistence_gap_detected"]))

    mixed = self._bilateral_nodes()
    mixed["Maria Lopez"]["node_4"] = SimpleNamespace(
      node_id="node_4", type="event", created=datetime.datetime(
        2023, 2, 13, 9, 0, 0), embedding_key="unrelated event", filling=[])
    self.assertTrue(subject._social_memory_snapshot(mixed)[
      "last_accessed_persistence_gap_detected"])

    hydrated_personas = {
      actor: SimpleNamespace(a_mem=SimpleNamespace(
        id_to_node=nodes, last_accessed_hydration_gap_detected=True))
      for actor, nodes in self._bilateral_nodes().items()
    }
    hydrated = subject._loaded_social_memory_snapshot(hydrated_personas)
    self.assertTrue(hydrated["last_accessed_persistence_gap_detected"])

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
    self.assertIsNone(subject._read_json(
      result.run_directory / "report.json")["failure"])

  def test_actor_state_group_fails_closed_only_when_generated_plan_is_empty(self):
    valid = subject.run_modern_smallville(
      subject.ModernRunConfig(run_name="actor-state-valid"),
      adapter=ModernTickFakeAdapter(), runtime_root=self.runtime_root)
    self.assertEqual("MODERN_SMALLVILLE_HEADLESS_RUN_PASSED", valid.verdict)
    valid_actor = subject._read_json(valid.run_directory / "report.json")[
      "tick_progression"][0]["actors"][subject.COGNITIVE_ACTOR]
    self.assertTrue(all((
      valid_actor["node_ids_valid"], valid_actor["node_ids_unique"],
      valid_actor["embedding_references_valid"],
      valid_actor["orphan_embedding_count"] == 0,
      valid_actor["daily_plan_present"], valid_actor["schedule_length"] > 0,
      valid_actor["current_action_present"],
      valid_actor["current_action_actor_aligned"])))

    original = subject._actor_tick_metadata

    def without_generated_plan(persona, coordinate):
      generated_plan = persona.scratch.daily_req
      persona.scratch.daily_req = []
      try:
        current = original(persona, coordinate)
      finally:
        persona.scratch.daily_req = generated_plan
      self.assertTrue(all((
        current["node_ids_valid"], current["node_ids_unique"],
        current["embedding_references_valid"],
        current["orphan_embedding_count"] == 0,
        current["schedule_length"] > 0, current["current_action_present"],
        current["current_action_actor_aligned"])))
      self.assertFalse(current["daily_plan_present"])
      return current

    with patch.object(subject, "_actor_tick_metadata",
                      side_effect=without_generated_plan):
      invalid = subject.run_modern_smallville(
        subject.ModernRunConfig(run_name="actor-state-invalid"),
        adapter=ModernTickFakeAdapter(), runtime_root=self.runtime_root)
    self.assertEqual("MODERN_SMALLVILLE_HEADLESS_RUN_FAILED", invalid.verdict)
    self.assertEqual(subject.ModernRuntimeInvariantError.__name__,
                     invalid.exception_type)
    self.assertEqual(
      "actor state invariant failed: actor=Isabella Rodriguez, tick=0",
      invalid.exception_message)
    self.assertEqual(0, invalid.completed_ticks)
    self.assertFalse((invalid.run_directory / "fixture" / "storage" /
                      "actor-state-invalid" / "environment" / "1.json").exists())

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
    self.assertFalse(report["causal_resume"][
      "last_accessed_persistence_gap_detected"])
    self.assertIsNone(report["causal_resume"]["ranking_state_fidelity"])
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

  def test_all_continue_dialogue_is_incomplete_and_not_committed(self):
    config = subject.ModernRunConfig(
      run_name="offline-r1m3c-ceiling", ticks=2,
      cognitive_actors=subject.VISIBLE_ACTORS, passive_actors=(),
      cost_ceiling_usd=Decimal("0.05"), controlled_proximity=True)
    result = subject.run_modern_smallville(
      config, adapter=CeilingConversationFakeAdapter(),
      runtime_root=self.runtime_root)
    report = subject._read_json(result.run_directory / "report.json")
    gate = report["interaction"]["conversation_gate"]
    memory = report["interaction"]["bilateral_memory"]
    failure = report["failure"]
    self.assertEqual("R1M3_C_MODERN_CALLER_BLOCKED", result.verdict)
    self.assertEqual("ConversationIncompleteError", result.exception_type)
    self.assertFalse(gate["started"])
    self.assertFalse(gate["committed"])
    self.assertFalse(gate["valid"])
    self.assertFalse(gate["social_pipeline_functional"])
    self.assertFalse(gate["model_end_observed"])
    self.assertFalse(gate["safety_ceiling_reached"])
    self.assertEqual([], gate["conversations"])
    self.assertFalse(memory["saved"])
    self.assertFalse(memory["reloaded"])
    self.assertEqual("ConversationIncompleteError", failure["exception_type"])
    self.assertEqual("persona_move", failure["stage"])
    self.assertEqual(
      {"function": "persona.plan", "stage": "PLAN"},
      failure["cognitive_failure"])

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
    # Historical R1CLI reports predate the explicit registry mode field.
    source_report["config"].pop("actor_registry")
    subject._write_json(source_result.run_directory / "report.json", source_report)
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
    self.assertFalse(causal["last_accessed_persistence_gap_detected"])
    self.assertEqual("PRESERVED",
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
