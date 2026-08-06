import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import datetime
from types import SimpleNamespace
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
REPOSITORY = BACKEND_SERVER.parents[1]
TESTS_DIR = Path(__file__).resolve().parent
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

from controlled_replay import (
  R1TDeterministicFakeAdapter,
  prepare_isolated_embedding_stores,
  prepare_isolated_reverie_fixture,
)
from persona.prompt_template.embedding_runtime import (
  build_modern_embedding_runtime_config, use_embedding_runtime)
import r1v_test_support as support

REVERIE_SPEC = importlib.util.spec_from_file_location(
  "r1v_visible_frame_reverie", BACKEND_SERVER / "reverie.py")
reverie_module = importlib.util.module_from_spec(REVERIE_SPEC)
REVERIE_SPEC.loader.exec_module(reverie_module)

BASELINE = (REPOSITORY / "environment" / "frontend_server" / "storage"
            / "base_the_ville_isabella_maria_klaus")
ACTOR = "Isabella Rodriguez"
PASSIVES = ("Maria Lopez", "Klaus Mueller")
PERSONAS = (ACTOR,) + PASSIVES
FRAME_FIELDS = {"movement", "pronunciatio", "description", "chat"}
EXPECTED_START = datetime.datetime(2023, 2, 13, 5, 55, 0)


def files(root):
  root = Path(root)
  return {
    path.relative_to(root).as_posix(): hashlib.sha256(
      path.read_bytes()).hexdigest()
    for path in root.rglob("*") if path.is_file()
  }


def fake_persona(pronunciatio=None, description=None, chat=None):
  scratch = SimpleNamespace(
    act_pronunciatio=pronunciatio, act_description=description,
    act_address=None, act_event=(None, None, None), act_start_time=None,
    act_duration=None, planned_path=[], chat=chat,
    daily_plan_req=[], f_daily_schedule=[])
  return SimpleNamespace(
    scratch=scratch,
    a_mem=SimpleNamespace(
      id_to_node={}, embeddings={}, seq_event=[], seq_thought=[], seq_chat=[]),
    s_mem=SimpleNamespace(tree={}),
    move=lambda *args: (_ for _ in ()).throw(
      AssertionError("original passive cognition reached")),
  )


class R1VPassiveVisibleFrameUnitTests(unittest.TestCase):
  def test_01_passive_frame_uses_authoritative_coordinates_and_fallbacks(self):
    maria = fake_persona()
    klaus = fake_persona("🙂", "standing")
    originals = {"Maria Lopez": maria.move, "Klaus Mueller": klaus.move}
    server = SimpleNamespace(
      personas={"Maria Lopez": maria, "Klaus Mueller": klaus},
      personas_tile={"Maria Lopez": (10, 20), "Klaus Mueller": (30, 40)})

    before = {name: support.passive_cognitive_fingerprint(persona)
              for name, persona in server.personas.items()}
    with support.install_passive_visible_moves(server, PASSIVES) as controller:
      self.assertEqual(((10, 20), "", "idle"),
                       maria.move(None, None, None, None))
      self.assertEqual(((30, 40), "🙂", "standing"),
                       klaus.move(None, None, None, None))
      self.assertEqual({"Maria Lopez": 1, "Klaus Mueller": 1},
                       dict(controller.frame_emissions))
    after = {name: support.passive_cognitive_fingerprint(persona)
             for name, persona in server.personas.items()}

    self.assertEqual(before, after)
    self.assertIs(originals["Maria Lopez"], maria.move)
    self.assertIs(originals["Klaus Mueller"], klaus.move)

  def test_02_installation_requires_complete_visible_registry(self):
    server = SimpleNamespace(personas={}, personas_tile={})
    with self.assertRaises(KeyError):
      with support.install_passive_visible_moves(server, PASSIVES):
        pass

  def test_03_passive_frame_emission_adds_no_provider_calls(self):
    provider_calls = []
    active = fake_persona()
    maria = fake_persona()
    klaus = fake_persona()

    def active_move(*args):
      del args
      provider_calls.append("active-only")
      return (1, 1), "", "idle"

    active.move = active_move
    server = SimpleNamespace(
      personas={ACTOR: active, "Maria Lopez": maria, "Klaus Mueller": klaus},
      personas_tile={ACTOR: (1, 1), "Maria Lopez": (2, 2),
                     "Klaus Mueller": (3, 3)})
    active.move(None, None, None, None)
    single_agent_calls = len(provider_calls)
    provider_calls.clear()

    with support.install_passive_visible_moves(server, PASSIVES):
      for persona in server.personas.values():
        persona.move(None, None, None, None)

    self.assertEqual(single_agent_calls, len(provider_calls))
    self.assertEqual(1, len(provider_calls))

  def test_04_transition_summary_observes_without_forcing_changes(self):
    unchanged = support.summarize_transitions(
      "sleeping", [73, 14], [
        {"action_description": "sleeping", "coordinate": [73, 14]},
        {"action_description": "sleeping", "coordinate": [73, 14]},
      ])
    self.assertEqual(1, unchanged["distinct_action_descriptions"])
    self.assertIsNone(unchanged["first_transition_tick"])
    self.assertIsNone(unchanged["first_movement_tick"])

    changed = support.summarize_transitions(
      "sleeping", [73, 14], [
        {"action_description": "sleeping", "coordinate": [73, 14]},
        {"action_description": "waking up", "coordinate": [73, 14]},
        {"action_description": "morning routine", "coordinate": [74, 14]},
      ])
    self.assertEqual(1, changed["first_transition_tick"])
    self.assertEqual(2, changed["first_movement_tick"])
    self.assertEqual([74, 14], changed["final_coordinate"])


class R1VRealReverieFrameTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.storage = self.root / "storage"
    self.temp_storage = self.root / "temp_storage"
    self.storage.mkdir()
    self.temp_storage.mkdir()
    self.baseline_before = files(BASELINE)

  def tearDown(self):
    self.temporary.cleanup()

  def test_01_sixty_real_world_frames_keep_passives_non_cognitive(self):
    fork_code = "r1v-offline-source"
    sim_code = "r1v-offline-sixty-frames"
    seed = support.build_synthetic_cognitive_seed(
      BASELINE, self.root / "synthetic-seed")
    source = self.storage / fork_code
    support.bootstrap_temporal_source(seed, source, EXPECTED_START)
    fixture = prepare_isolated_reverie_fixture(
      source, self.root, BASELINE, 0)
    prepare_isolated_embedding_stores(fixture)
    adapter = R1TDeterministicFakeAdapter()
    network_calls = []

    def block_network(*args, **kwargs):
      del args, kwargs
      network_calls.append("blocked")
      raise AssertionError("network is forbidden in R1V offline tests")

    previous_cwd = Path.cwd()
    try:
      os.chdir(BACKEND_SERVER)
      with patch.object(reverie_module, "fs_storage", str(self.storage)), \
          patch.object(reverie_module, "fs_temp_storage",
                       str(self.temp_storage)), \
          patch("socket.create_connection", side_effect=block_network), \
          patch.object(socket.socket, "connect", block_network), \
          use_embedding_runtime(
            build_modern_embedding_runtime_config(), adapter):
        server = reverie_module.ReverieServer(fork_code, sim_code)
        self.assertEqual(0, server.step)
        self.assertEqual(EXPECTED_START, server.curr_time)
        saved = self.storage / sim_code
        active_calls = []
        passive_original_calls = []

        def active_move(maze, personas, curr_tile, curr_time):
          del maze, personas, curr_tile, curr_time
          active_calls.append(server.step)
          return server.personas_tile[ACTOR], "", "idle"

        server.personas[ACTOR].move = active_move
        for name in PASSIVES:
          def forbidden_original(*args, _name=name, **kwargs):
            del args, kwargs
            passive_original_calls.append(_name)
            raise AssertionError("passive original move reached")
          server.personas[name].move = forbidden_original

        passive_before = {
          name: support.passive_cognitive_fingerprint(server.personas[name])
          for name in PASSIVES}
        movement_hashes = []
        with support.install_passive_visible_moves(
            server, PASSIVES) as controller:
          for tick in range(60):
            environment_path = saved / "environment" / f"{tick}.json"
            environment = json.loads(
              environment_path.read_text(encoding="utf-8"))
            server.start_server(1)
            movement_path = saved / "movement" / f"{tick}.json"
            movement = json.loads(
              movement_path.read_text(encoding="utf-8"))

            self.assertEqual(set(environment), set(movement["persona"]))
            self.assertEqual(set(PERSONAS), set(movement["persona"]))
            for name in PERSONAS:
              self.assertEqual(FRAME_FIELDS, set(movement["persona"][name]))
              self.assertEqual(2, len(movement["persona"][name]["movement"]))
            for name in PASSIVES:
              self.assertEqual(
                [environment[name]["x"], environment[name]["y"]],
                movement["persona"][name]["movement"])
              self.assertIsNone(movement["persona"][name]["chat"])

            movement_hashes.append(hashlib.sha256(
              movement_path.read_bytes()).hexdigest())
            next_environment = dict(environment)
            for name in PERSONAS:
              x, y = movement["persona"][name]["movement"]
              next_environment[name] = dict(next_environment[name])
              next_environment[name]["x"] = x
              next_environment[name]["y"] = y
            (saved / "environment" / f"{tick + 1}.json").write_text(
              json.dumps(next_environment, indent=2), encoding="utf-8")

        passive_after = {
          name: support.passive_cognitive_fingerprint(server.personas[name])
          for name in PASSIVES}
    finally:
      os.chdir(previous_cwd)

    self.assertEqual(list(range(60)), active_calls)
    self.assertEqual([], passive_original_calls)
    self.assertEqual(
      {"Maria Lopez": 60, "Klaus Mueller": 60},
      dict(controller.frame_emissions))
    self.assertEqual(passive_before, passive_after)
    self.assertEqual([], adapter.calls)
    self.assertEqual([], network_calls)
    self.assertEqual(60, len(set(movement_hashes)))
    self.assertEqual(60, server.step)
    self.assertEqual(datetime.datetime(2023, 2, 13, 6, 5),
                     server.curr_time)
    saved = self.storage / sim_code
    self.assertEqual(60, len(list((saved / "movement").glob("*.json"))))
    self.assertEqual(61, len(list((saved / "environment").glob("*.json"))))
    self.assertEqual(self.baseline_before, files(BASELINE))


if __name__ == "__main__":
  unittest.main()
