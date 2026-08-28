import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
REPOSITORY = BACKEND_SERVER.parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from controlled_replay import (
  DeterministicReplayFakeAdapter,
  prepare_isolated_embedding_stores,
  prepare_isolated_reverie_fixture,
)
from persona.prompt_template.chat_runtime import (
  build_modern_chat_runtime_config,
  use_modern_chat_runtime,
)
from persona.prompt_template.embedding_runtime import (
  build_modern_embedding_runtime_config,
  use_embedding_runtime,
)
from persona.prompt_template.llm_provider import (
  clear_telemetry,
  reset_embedding_cache,
)


REVERIE_SPEC = importlib.util.spec_from_file_location(
  "r1world_reverie", BACKEND_SERVER / "reverie.py")
reverie_module = importlib.util.module_from_spec(REVERIE_SPEC)
REVERIE_SPEC.loader.exec_module(reverie_module)

BASELINE = (REPOSITORY / "environment" / "frontend_server" / "storage"
            / "base_the_ville_isabella_maria_klaus")
ACTOR_A = "Isabella Rodriguez"
ACTOR_B = "Maria Lopez"
PASSIVE_ACTOR = "Klaus Mueller"
TOASTER_ADDRESS = "the Ville:Dorm for Oak Hill College:kitchen:toaster"
EXPECTED_TOASTER_TILE = (120, 45)
EXPECTED_OBSERVER_TILE = (119, 45)
ACTIVE_PREDICATE = "is"
ACTIVE_OBJECT = "on"
ACTIVE_DESCRIPTION = "on"


def _file_hashes(root):
  root = Path(root)
  return {
    path.relative_to(root).as_posix(): hashlib.sha256(
      path.read_bytes()).hexdigest()
    for path in root.rglob("*") if path.is_file()
  }


def _read_json(path):
  return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
  Path(path).write_text(
    json.dumps(payload, indent=2), encoding="utf-8")


class GroundedWorldEffectPerceptionMemoryTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.storage = self.root / "storage"
    self.temp_storage = self.root / "temp_storage"
    self.storage.mkdir()
    self.temp_storage.mkdir()
    self.baseline_before = _file_hashes(BASELINE)
    clear_telemetry()
    reset_embedding_cache()

  def tearDown(self):
    clear_telemetry()
    reset_embedding_cache()
    self.temporary.cleanup()

  def _prepare_source(self):
    source_code = "r1world-source"
    source = self.storage / source_code
    shutil.copytree(BASELINE, source)

    environment_path = source / "environment" / "0.json"
    environment = _read_json(environment_path)
    environment[ACTOR_A].update(x=EXPECTED_TOASTER_TILE[0],
                                y=EXPECTED_TOASTER_TILE[1])
    environment[ACTOR_B].update(x=EXPECTED_OBSERVER_TILE[0],
                                y=EXPECTED_OBSERVER_TILE[1])
    _write_json(environment_path, environment)

    scratch_path = (source / "personas" / ACTOR_A / "bootstrap_memory"
                    / "scratch.json")
    scratch = _read_json(scratch_path)
    scratch.update({
      "act_address": TOASTER_ADDRESS,
      "act_start_time": "February 13, 2023, 00:00:00",
      "act_duration": 60,
      "act_description": "using the toaster",
      "act_pronunciatio": "",
      "act_event": [ACTOR_A, "is", "using the toaster"],
      "act_obj_description": ACTIVE_DESCRIPTION,
      "act_obj_pronunciatio": "",
      "act_obj_event": ["toaster", ACTIVE_PREDICATE, ACTIVE_OBJECT],
      "act_path_set": True,
      "planned_path": [],
    })
    _write_json(scratch_path, scratch)

    fixture = prepare_isolated_reverie_fixture(
      source, self.root, BASELINE, 0)
    prepare_isolated_embedding_stores(fixture)
    return source_code

  def test_server_grounded_toaster_effect_enters_observer_memory(self):
    source_code = self._prepare_source()
    simulation_code = "r1world-verification"
    adapter = DeterministicReplayFakeAdapter(
      text_responses=('{"output": "4"}',) * 16)
    network_calls = []

    def reject_network(*args, **kwargs):
      del args, kwargs
      network_calls.append("attempt")
      raise AssertionError("network is forbidden in R1WORLD verification")

    previous_cwd = Path.cwd()
    try:
      os.chdir(BACKEND_SERVER)
      with patch.object(reverie_module, "fs_storage", str(self.storage)), \
          patch.object(reverie_module, "fs_temp_storage",
                       str(self.temp_storage)), \
          patch("socket.create_connection", side_effect=reject_network), \
          patch.object(socket.socket, "connect", reject_network), \
          use_embedding_runtime(
            build_modern_embedding_runtime_config(), adapter), \
          use_modern_chat_runtime(
            build_modern_chat_runtime_config(), adapter), \
          redirect_stdout(io.StringIO()):
        server = reverie_module.ReverieServer(source_code, simulation_code)
        actor_a = server.personas[ACTOR_A]
        actor_b = server.personas[ACTOR_B]

        # Fresh repository evidence confirms the nominated coordinates and
        # makes the single-tile, same-arena attentional opportunity explicit.
        toaster_tiles = server.maze.address_tiles[TOASTER_ADDRESS]
        self.assertEqual({EXPECTED_TOASTER_TILE}, toaster_tiles)
        toaster_details = server.maze.access_tile(EXPECTED_TOASTER_TILE)
        observer_details = server.maze.access_tile(EXPECTED_OBSERVER_TILE)
        self.assertEqual("toaster", toaster_details["game_object"])
        self.assertFalse(toaster_details["collision"])
        self.assertFalse(observer_details["collision"])
        self.assertEqual(
          server.maze.get_tile_path(EXPECTED_TOASTER_TILE, "arena"),
          server.maze.get_tile_path(EXPECTED_OBSERVER_TILE, "arena"))
        self.assertLessEqual(
          math.dist(EXPECTED_TOASTER_TILE, EXPECTED_OBSERVER_TILE),
          actor_b.scratch.vision_r)
        self.assertGreaterEqual(actor_b.scratch.att_bandwidth, 6)

        idle_event = (TOASTER_ADDRESS, None, None, None)
        active_event = actor_a.scratch.get_curr_obj_event_and_desc()
        self.assertEqual(
          (TOASTER_ADDRESS, ACTIVE_PREDICATE, ACTIVE_OBJECT,
           ACTIVE_DESCRIPTION), active_event)

        # T0/T1: authoritative world state is idle and Actor B has no matching
        # Event before the server mutation begins.
        events_before = set(toaster_details["events"])
        matching_before = [
          node for node in actor_b.a_mem.seq_event
          if node.spo_summary() == active_event[:3]
        ]
        self.assertIn(idle_event, events_before)
        self.assertNotIn(active_event, events_before)
        self.assertEqual([], matching_before)
        node_ids_before = set(actor_b.a_mem.id_to_node)

        mutation_calls = []
        perceived_nodes = []
        active_seen_before_perception = []
        perception_times = []
        original_add = server.maze.add_event_from_tile
        original_remove = server.maze.remove_event_from_tile

        def observed_add(event, tile):
          mutation_calls.append(("add", event, tile))
          return original_add(event, tile)

        def observed_remove(event, tile):
          mutation_calls.append(("remove", event, tile))
          return original_remove(event, tile)

        def actor_a_move(maze, personas, curr_tile, curr_time):
          del maze, personas, curr_time
          return curr_tile, "", "grounded toaster action"

        def actor_b_move(maze, personas, curr_tile, curr_time):
          del personas
          current_events = set(
            maze.access_tile(EXPECTED_TOASTER_TILE)["events"])
          active_seen_before_perception.append(
            active_event in current_events and idle_event not in current_events)
          actor_b.scratch.curr_tile = curr_tile
          actor_b.scratch.curr_time = curr_time
          perception_times.append(curr_time)
          perceived_nodes.extend(actor_b.perceive(maze))
          return curr_tile, "", "normal perception completed"

        def passive_move(maze, personas, curr_tile, curr_time):
          del maze, personas, curr_time
          return curr_tile, "", "passive"

        actor_a.move = actor_a_move
        actor_b.move = actor_b_move
        server.personas[PASSIVE_ACTOR].move = passive_move

        # T2: only ReverieServer.start_server drives the production Maze
        # mutation. The observed methods delegate unchanged to the real Maze.
        with patch.object(
            server.maze, "add_event_from_tile", side_effect=observed_add), \
            patch.object(
              server.maze, "remove_event_from_tile",
              side_effect=observed_remove):
          server.start_server(1)

        # T3/T4: the active tuple existed before Actor B invoked the normal
        # Persona.perceive entry point and remains authoritative afterward.
        events_after = set(
          server.maze.access_tile(EXPECTED_TOASTER_TILE)["events"])
        self.assertEqual([True], active_seen_before_perception)
        self.assertIn(("add", active_event, EXPECTED_TOASTER_TILE),
                      mutation_calls)
        self.assertIn(("remove", idle_event, EXPECTED_TOASTER_TILE),
                      mutation_calls)
        self.assertLess(
          mutation_calls.index(("add", active_event,
                                EXPECTED_TOASTER_TILE)),
          mutation_calls.index(("remove", idle_event,
                                EXPECTED_TOASTER_TILE)))
        self.assertIn(active_event, events_after)
        self.assertNotIn(idle_event, events_after)

        # T5: normal perception created a new actor-local Event whose S/P/O
        # identity is derived from the actual authoritative active tuple.
        matching_perceived = [
          node for node in perceived_nodes
          if node.spo_summary() == active_event[:3]
        ]
        self.assertEqual(1, len(matching_perceived))
        node = matching_perceived[0]
        self.assertNotIn(node.node_id, node_ids_before)
        self.assertIs(actor_b.a_mem.id_to_node[node.node_id], node)
        self.assertEqual("event", node.type)
        self.assertEqual(active_event[0], node.subject)
        self.assertEqual(active_event[1], node.predicate)
        self.assertEqual(active_event[2], node.object)
        self.assertEqual(
          f"{active_event[0].split(':')[-1]} is {active_event[3]}",
          node.description)
        self.assertEqual(perception_times[0], node.created)
        self.assertEqual(node.created, node.last_accessed)
        self.assertEqual(node.description, node.embedding_key)
        self.assertEqual(4, node.poignancy)
        self.assertIn("toaster", node.keywords)
        self.assertIn(ACTIVE_OBJECT, node.keywords)

        # T6: the node is an ordinary keyword-indexed retrieval candidate.
        retrieved = actor_b.a_mem.retrieve_relevant_events(*active_event[:3])
        self.assertIn(node, retrieved)

    finally:
      os.chdir(previous_cwd)

    methods = [call[0] for call in adapter.calls]
    self.assertEqual(6, methods.count("create_embedding"))
    self.assertEqual(2, methods.count("create_chat"))
    self.assertEqual(8, len(methods))
    self.assertEqual([], network_calls)
    self.assertTrue(adapter.offline_fake)
    self.assertEqual(self.baseline_before, _file_hashes(BASELINE))


if __name__ == "__main__":
  unittest.main()
