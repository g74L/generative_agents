"""Offline authority tests using the real Maze map and legacy Scratch getter."""
import copy
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
import datetime
import itertools
import json
import os
from pathlib import Path
import random
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
  sys.path.insert(0, str(BACKEND))

if __package__:
  from . import test_world_effect_perception as fixture
else:
  import test_world_effect_perception as fixture
from maze import Maze
from persona.memory_structures.scratch import Scratch
from persona.persona import Persona
from persona.prompt_template.modern_openai_provider import ModernOpenAIClientAdapter
from smallville_world_transition import (
  ObjectEffectKind as Kind,
  ObjectEffectRejectionReason as Reason,
  ObjectEffectStatus as Status,
  ObjectEffectTransitionError,
  SmallvilleObjectEffectProposal as Proposal,
  classify_object_effect,
  ground_object_effect,
)

SERVER = fixture.reverie_module
ADDRESS = fixture.TOASTER_ADDRESS
TILE = fixture.EXPECTED_TOASTER_TILE
OTHER_TILE = fixture.EXPECTED_OBSERVER_TILE
ACTIVE = (ADDRESS, "is", "on", "on")
IDLE = (ADDRESS, None, None, None)
UNKNOWN = "the Ville:Dorm for Oak Hill College:kitchen:nonexistent appliance"


class SmallvilleWorldTransitionTests(unittest.TestCase):
  def setUp(self):
    self.stack = ExitStack()
    self.addCleanup(self.stack.close)
    self.network = []

    def reject_network(*args, **kwargs):
      self.network.append("attempt")
      raise AssertionError("world-transition tests must remain offline")

    for owner, name in ((socket, "create_connection"),
                        (socket.socket, "connect"), (socket.socket, "connect_ex")):
      self.stack.enter_context(patch.object(owner, name, reject_network))
    self.provider_guards = [self.stack.enter_context(patch.object(
      ModernOpenAIClientAdapter, method,
      side_effect=AssertionError("no provider work in world transitions")))
      for method in ("create_chat", "create_embedding")]
    previous_cwd = Path.cwd()
    try:
      os.chdir(BACKEND)
      self.maze = Maze("the_ville")
    finally:
      os.chdir(previous_cwd)
    self.assertEqual({TILE}, self.maze.address_tiles[ADDRESS])
    self.assertNotIn(OTHER_TILE, self.maze.address_tiles[ADDRESS])
    self.assertFalse(self.maze.access_tile(OTHER_TILE)["collision"])
    self.assertNotIn(UNKNOWN, self.maze.address_tiles)
    self.scratch = Scratch(str(
      fixture.BASELINE / "personas" / fixture.ACTOR_A
      / "bootstrap_memory" / "scratch.json"))
    self.actor = SimpleNamespace(name=fixture.ACTOR_A, scratch=self.scratch)
    self.server = SERVER.ReverieServer.__new__(SERVER.ReverieServer)
    self.server.maze = self.maze

  def tearDown(self):
    self.assertEqual([], self.network)
    for guard in self.provider_guards:
      guard.assert_not_called()

  def _set_effect(self, event, actor=None):
    actor = actor or self.actor
    address, predicate, value, description = event
    actor.scratch.act_address = address
    actor.scratch.act_obj_event = ("legacy object label", predicate, value)
    actor.scratch.act_obj_description = description
    actor.scratch.planned_path = []

  def _world(self):
    return copy.deepcopy(vars(self.maze))

  def _tick(self, actors=None, new_tiles=None):
    actors = actors or [self.actor]
    new_tiles = new_tiles or [TILE] * len(actors)
    root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
    simulation = root / "transition-test"
    (simulation / "environment").mkdir(parents=True)
    (simulation / "movement").mkdir()
    self.server.sim_code = simulation.name
    self.server.step = 0
    self.server.curr_time = datetime.datetime(2023, 2, 13)
    self.server.sec_per_step = 10
    self.server.server_sleep = 0
    self.server.personas = {actor.name: actor for actor in actors}
    # Deliberately keep Scratch.curr_tile stale: publication uses the frame.
    self.server.personas_tile = {actor.name: OTHER_TILE for actor in actors}
    for actor, tile in zip(actors, new_tiles):
      actor.move = Mock(return_value=(tile, "", "offline actor"))
      self.maze.add_event_from_tile(
        actor.scratch.get_curr_event_and_desc(), OTHER_TILE)
    (simulation / "environment" / "0.json").write_text(json.dumps({
      actor.name: {"x": tile[0], "y": tile[1]}
      for actor, tile in zip(actors, new_tiles)
    }), encoding="utf-8")
    self.stack.enter_context(patch.object(SERVER, "fs_storage", str(root)))
    return simulation

  def test_classification_matrix_and_purity(self):
    cases = [
      ((None, None, None, None), Kind.NO_OBJECT_EFFECT),
      (("", "is", "on", "text"), Kind.NO_OBJECT_EFFECT),
      (("<persona> Maria Lopez", None, None, None), Kind.NO_OBJECT_EFFECT),
      (("<waiting> 119 45", None, None, None), Kind.NO_OBJECT_EFFECT),
      (("the Ville:Dorm for Oak Hill College:kitchen:<random>",
        "is", "on", "text"), Kind.NO_OBJECT_EFFECT),
      ((ADDRESS, None, None, None), Kind.NO_OBJECT_EFFECT),
      ((UNKNOWN, None, None, None), Kind.NO_OBJECT_EFFECT),
      (ACTIVE, Kind.MATERIAL_OBJECT_EFFECT_PROPOSAL),
      ((UNKNOWN, "is", "on", "on"), Kind.MATERIAL_OBJECT_EFFECT_PROPOSAL),
      ((ADDRESS, "", "", ""), Kind.MATERIAL_OBJECT_EFFECT_PROPOSAL),
      (("<persona> Maria Lopez", "is", "on", "on"),
       Kind.MALFORMED_OBJECT_EFFECT_PROPOSAL),
      (("<waiting> 119 45", None, None, "on"),
       Kind.MALFORMED_OBJECT_EFFECT_PROPOSAL),
    ]
    for fields in itertools.product((None, "present"), repeat=3):
      if 0 < sum(field is not None for field in fields) < 3:
        cases.append(((ADDRESS, *fields), Kind.MALFORMED_OBJECT_EFFECT_PROPOSAL))
    world_before = self._world()
    for event, expected in cases:
      with self.subTest(event=event):
        self._set_effect(event)
        actor_before = copy.deepcopy(vars(self.actor))
        rng_before = random.getstate()
        numpy_before = numpy.random.get_state()
        self.assertEqual(expected, classify_object_effect(
          self.scratch.get_curr_obj_event_and_desc()))
        self.assertEqual(vars(actor_before["scratch"]), vars(self.scratch))
        self.assertEqual(actor_before["name"], self.actor.name)
        self.assertEqual(rng_before, random.getstate())
        after = numpy.random.get_state()
        self.assertEqual(numpy_before[0], after[0])
        numpy.testing.assert_array_equal(numpy_before[1], after[1])
        self.assertEqual(numpy_before[2:], after[2:])
    self.assertEqual(world_before, vars(self.maze))

  def test_construction_and_all_grounding_outcomes_are_pure(self):
    self._set_effect(ACTIVE)
    actor_before = copy.deepcopy(vars(self.scratch))
    world_before = self._world()
    rng_before = random.getstate()
    numpy_before = numpy.random.get_state()
    guards = [self.stack.enter_context(patch.object(
      Persona, method, side_effect=AssertionError("no cognition in grounding")))
      for method in ("perceive", "retrieve", "plan", "reflect", "execute")]
    proposal = Proposal.from_event(
      self.actor.name, self.scratch.get_curr_obj_event_and_desc(), TILE)
    self.assertEqual(ACTIVE, proposal.as_event())
    with self.assertRaises(FrozenInstanceError):
      proposal.target_address = UNKNOWN
    for candidate, status, reason in (
        (proposal, Status.ACCEPTED, None),
        (replace(proposal, target_address=UNKNOWN), Status.REJECTED,
         Reason.TARGET_ADDRESS_UNKNOWN),
        (replace(proposal, publication_tile=OTHER_TILE), Status.REJECTED,
         Reason.TARGET_TILE_MISMATCH)):
      with self.subTest(reason=reason):
        decision = ground_object_effect(candidate, self.maze.address_tiles)
        self.assertIs(candidate, decision.proposal)
        self.assertEqual(status, decision.status)
        self.assertEqual(reason, decision.reason)
        self.assertEqual(world_before, vars(self.maze))
    self.assertEqual(actor_before, vars(self.scratch))
    self.assertEqual(rng_before, random.getstate())
    after = numpy.random.get_state()
    numpy.testing.assert_array_equal(numpy_before[1], after[1])
    self.assertEqual(numpy_before[0], after[0])
    self.assertEqual(numpy_before[2:], after[2:])
    for guard in guards:
      guard.assert_not_called()

  def test_grounding_does_not_validate_domain_semantics(self):
    proposal = Proposal(self.actor.name, ADDRESS, "dreams", "impossible",
                        "an intentionally implausible consequence", TILE)
    self.assertEqual(Status.ACCEPTED,
                     ground_object_effect(proposal, self.maze.address_tiles).status)

  def test_only_accepted_commit_changes_world_in_legacy_order(self):
    proposal = Proposal.from_event(self.actor.name, ACTIVE, TILE)
    decision = ground_object_effect(proposal, self.maze.address_tiles)
    self.assertNotIn(ACTIVE, self.maze.access_tile(TILE)["events"])
    self.assertIn(IDLE, self.maze.access_tile(TILE)["events"])
    calls = []

    class Cleanup(dict):
      def __setitem__(self, event, tile):
        calls.append(("cleanup", event, tile))
        super().__setitem__(event, tile)

    cleanup = Cleanup()
    original_add = self.maze.add_event_from_tile
    original_remove = self.maze.remove_event_from_tile

    def add(event, tile):
      calls.append(("add", event, tile))
      return original_add(event, tile)

    def remove(event, tile):
      calls.append(("remove", event, tile))
      return original_remove(event, tile)

    with patch.object(self.maze, "add_event_from_tile", side_effect=add), \
        patch.object(self.maze, "remove_event_from_tile", side_effect=remove):
      self.server._commit_object_effect(decision, cleanup)
    self.assertEqual([("cleanup", ACTIVE, TILE), ("add", ACTIVE, TILE),
                      ("remove", IDLE, TILE)], calls)
    self.assertEqual({ACTIVE: TILE}, cleanup)
    self.assertIn(ACTIVE, self.maze.access_tile(TILE)["events"])
    self.assertNotIn(IDLE, self.maze.access_tile(TILE)["events"])

  def test_rejected_decisions_cannot_commit(self):
    world_before = self._world()
    for target, tile, reason in ((UNKNOWN, TILE, Reason.TARGET_ADDRESS_UNKNOWN),
                                 (ADDRESS, OTHER_TILE, Reason.TARGET_TILE_MISMATCH)):
      proposal = Proposal.from_event(self.actor.name, (target, *ACTIVE[1:]), tile)
      decision = ground_object_effect(proposal, self.maze.address_tiles)
      cleanup = {}
      with self.assertRaises(ObjectEffectTransitionError) as raised:
        self.server._commit_object_effect(decision, cleanup)
      self.assertEqual(reason, raised.exception.reason)
      self.assertEqual({}, cleanup)
      self.assertEqual(world_before, vars(self.maze))

  def test_no_proposal_server_path_preserves_legacy_add_remove_and_cleanup(self):
    cases = (
      (None, None, None, None),
      ("<persona> Maria Lopez", None, None, None),
      ("<waiting> 119 45", None, None, None),
      ("the Ville:Dorm for Oak Hill College:kitchen:<random>", "is", "on", "on"),
      (ADDRESS, None, None, None),
    )
    for event in cases:
      with self.subTest(event=event):
        self._set_effect(event)
        simulation = self._tick()
        # Two cycles are needed to exercise the existing cleanup bookkeeping.
        (simulation / "environment" / "1.json").write_bytes(
          (simulation / "environment" / "0.json").read_bytes())
        legacy_event = self.scratch.get_curr_obj_event_and_desc()
        with patch.object(SERVER, "ground_object_effect") as ground, \
            patch.object(self.server, "_commit_object_effect") as commit, \
            patch.object(self.maze, "add_event_from_tile",
                         wraps=self.maze.add_event_from_tile) as add, \
            patch.object(self.maze, "remove_event_from_tile",
                         wraps=self.maze.remove_event_from_tile) as remove, \
            patch.object(self.maze, "turn_event_from_tile_idle",
                         wraps=self.maze.turn_event_from_tile_idle) as cleanup:
          self.server.start_server(2)
        ground.assert_not_called()
        commit.assert_not_called()
        self.assertEqual(2, sum(call.args == (legacy_event, TILE)
                                for call in add.call_args_list))
        self.assertEqual(2, sum(call.args == ((legacy_event[0], None, None, None), TILE)
                                for call in remove.call_args_list))
        cleanup.assert_called_once_with(legacy_event, TILE)
        self.assertEqual(2, self.server.step)

  def test_invalid_material_stops_before_current_actor_or_object_mutations(self):
    cases = [((UNKNOWN, *ACTIVE[1:]), TILE, Reason.TARGET_ADDRESS_UNKNOWN),
             (ACTIVE, OTHER_TILE, Reason.TARGET_TILE_MISMATCH)]
    for fields in itertools.product((None, "present"), repeat=3):
      if 0 < sum(field is not None for field in fields) < 3:
        cases.append(((ADDRESS, *fields), TILE, Reason.MALFORMED_OBJECT_EFFECT))
    for event, tile, reason in cases:
      with self.subTest(event=event, tile=tile):
        self._set_effect(event)
        simulation = self._tick(new_tiles=[tile])
        world_before = self._world()
        positions_before = dict(self.server.personas_tile)
        with patch.object(SERVER, "ground_object_effect",
                          wraps=ground_object_effect) as ground, \
            patch.object(self.server, "_commit_object_effect") as commit:
          with self.assertRaises(ObjectEffectTransitionError) as raised:
            self.server.start_server(1)
        self.assertEqual(reason, raised.exception.reason)
        self.assertEqual(self.actor.name, raised.exception.source_actor)
        self.assertEqual(event[0], raised.exception.target_address)
        self.assertEqual(0 if reason == Reason.MALFORMED_OBJECT_EFFECT else 1,
                         ground.call_count)
        commit.assert_not_called()
        self.actor.move.assert_not_called()
        self.assertEqual(world_before, vars(self.maze))
        self.assertEqual(positions_before, self.server.personas_tile)
        self.assertEqual(0, self.server.step)
        self.assertFalse((simulation / "movement" / "0.json").exists())
        self.assertIn(IDLE, self.maze.access_tile(TILE)["events"])
        for row in self.maze.tiles:
          for tile_details in row:
            self.assertNotIn(event, tile_details["events"])

  def test_later_rejection_preserves_prior_actor_commit_without_eager_validation(self):
    self._set_effect(ACTIVE)
    later = SimpleNamespace(name=fixture.ACTOR_B, scratch=copy.deepcopy(self.scratch))
    later.scratch.name = later.name
    self._set_effect((UNKNOWN, *ACTIVE[1:]), later)
    self._tick([self.actor, later], [TILE, OTHER_TILE])
    decisions = []

    def observe_ground(proposal, address_tiles):
      decisions.append(proposal.source_actor)
      if proposal.source_actor == later.name:
        self.assertIn(ACTIVE, self.maze.access_tile(TILE)["events"])
        self.assertNotIn(IDLE, self.maze.access_tile(TILE)["events"])
        self.assertEqual(TILE, self.server.personas_tile[self.actor.name])
      return ground_object_effect(proposal, address_tiles)

    with patch.object(SERVER, "ground_object_effect", side_effect=observe_ground):
      with self.assertRaises(ObjectEffectTransitionError) as raised:
        self.server.start_server(1)
    self.assertEqual(Reason.TARGET_ADDRESS_UNKNOWN, raised.exception.reason)
    self.assertEqual([self.actor.name, later.name], decisions)
    self.assertEqual(0, self.server.step)
    self.actor.move.assert_not_called()
    later.move.assert_not_called()

  def test_in_transit_actor_does_not_classify_or_publish_object_effect(self):
    self._set_effect(ACTIVE)
    self.scratch.planned_path = [TILE]
    self._tick(new_tiles=[OTHER_TILE])
    with patch.object(SERVER, "classify_object_effect") as classify, \
        patch.object(SERVER, "ground_object_effect") as ground, \
        patch.object(self.server, "_commit_object_effect") as commit:
      self.server.start_server(1)
    classify.assert_not_called()
    ground.assert_not_called()
    commit.assert_not_called()
    self.assertIn(IDLE, self.maze.access_tile(TILE)["events"])
    self.assertNotIn(ACTIVE, self.maze.access_tile(OTHER_TILE)["events"])


if __name__ == "__main__":
  unittest.main()
