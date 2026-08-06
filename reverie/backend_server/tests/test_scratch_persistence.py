import ast
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
REPOSITORY = BACKEND_SERVER.parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from controlled_replay import (
  R1TDeterministicFakeAdapter,
  prepare_isolated_embedding_stores,
  prepare_isolated_reverie_fixture,
)
from persona.memory_structures.embedding_space import (
  EMBEDDING_MANIFEST_FILENAME,
  read_embedding_manifest,
)
from persona.memory_structures.scratch import Scratch
from persona.persona import Persona
from persona.prompt_template.embedding_runtime import (
  TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
  build_modern_embedding_runtime_config,
  use_embedding_runtime,
)


REVERIE_SPEC = importlib.util.spec_from_file_location(
  "r1t_persistence_reverie", BACKEND_SERVER / "reverie.py")
reverie_module = importlib.util.module_from_spec(REVERIE_SPEC)
REVERIE_SPEC.loader.exec_module(reverie_module)


DATETIME_FORMAT = "%B %d, %Y, %H:%M:%S"
BASELINE = (REPOSITORY / "environment" / "frontend_server" / "storage"
            / "base_the_ville_isabella_maria_klaus")
PERSONAS = ("Isabella Rodriguez", "Maria Lopez", "Klaus Mueller")


def _files(root):
  root = Path(root)
  if not root.exists():
    return {}
  return {
    path.relative_to(root).as_posix(): hashlib.sha256(
      path.read_bytes()).hexdigest()
    for path in root.rglob("*") if path.is_file()
  }


def _scratch_save_required_attributes():
  tree = ast.parse(textwrap.dedent(inspect.getsource(Scratch.save)))
  return tuple(sorted({
    node.attr for node in ast.walk(tree)
    if isinstance(node, ast.Attribute)
    and isinstance(node.value, ast.Name)
    and node.value.id == "self"
  }))


class ScratchPersistenceContractTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.source = (BASELINE / "personas" / "Isabella Rodriguez"
                   / "bootstrap_memory" / "scratch.json")

  def tearDown(self):
    self.temporary.cleanup()

  def copy_scratch(self, name="scratch.json", changes=None):
    data = json.loads(self.source.read_text(encoding="utf-8"))
    if changes:
      data.update(changes)
    target = self.root / name
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target, data

  def test_01_original_failure_expression_and_field_are_deterministic(self):
    scratch = Scratch(str(self.source))

    self.assertIsNone(scratch.curr_time)
    with self.assertRaises(AttributeError) as raised:
      scratch.curr_time.strftime(DATETIME_FORMAT)

    self.assertEqual("NoneType", type(scratch.curr_time).__name__)
    self.assertIn("strftime", str(raised.exception))
    self.assertIn("curr_time", vars(scratch))
    self.assertIn("act_start_time", vars(scratch))
    required = _scratch_save_required_attributes()
    present = tuple(sorted(vars(scratch)))
    diagnostic = {
      "phase": "PERSISTENCE",
      "exception_type": type(raised.exception).__name__,
      "object_type": type(scratch.curr_time).__name__,
      "file": "persona/memory_structures/scratch.py",
      "function": "Scratch.save",
      "historical_line": 251,
      "failing_attribute": "strftime",
      "scratch_field": "curr_time",
      "required_attributes": required,
      "present_attributes": present,
    }
    self.assertEqual("PERSISTENCE", diagnostic["phase"])
    self.assertEqual("AttributeError", diagnostic["exception_type"])
    self.assertEqual("NoneType", diagnostic["object_type"])
    self.assertEqual(251, diagnostic["historical_line"])
    self.assertTrue(set(required).issubset(present))

  def test_02_legacy_nullable_fields_save_and_round_trip_without_loss(self):
    output = self.root / "saved.json"
    original = json.loads(self.source.read_text(encoding="utf-8"))
    scratch = Scratch(str(self.source))

    scratch.save(str(output))
    saved = json.loads(output.read_text(encoding="utf-8"))
    reloaded = Scratch(str(output))

    self.assertEqual(original, saved)
    self.assertIsNone(saved["curr_time"])
    self.assertIsNone(saved["act_start_time"])
    self.assertIsNone(reloaded.curr_time)
    self.assertIsNone(reloaded.act_start_time)
    self.assertEqual(set(original), set(saved))

  def test_03_null_action_start_does_not_erase_loaded_current_time(self):
    serialized = "February 13, 2023, 00:00:00"
    source, original = self.copy_scratch(changes={
      "curr_time": serialized, "act_start_time": None,
    })
    output = self.root / "saved.json"

    scratch = Scratch(str(source))
    scratch.save(str(output))
    reloaded = Scratch(str(output))

    expected = datetime.datetime(2023, 2, 13, 0, 0)
    self.assertEqual(expected, scratch.curr_time)
    self.assertIsNone(scratch.act_start_time)
    self.assertEqual(expected, reloaded.curr_time)
    self.assertIsNone(reloaded.act_start_time)
    self.assertEqual(original, json.loads(output.read_text(encoding="utf-8")))

  def test_04_datetime_fields_keep_the_historical_json_format(self):
    curr_time = "February 13, 2023, 00:00:10"
    act_start_time = "February 13, 2023, 00:00:00"
    source, original = self.copy_scratch(changes={
      "curr_time": curr_time, "act_start_time": act_start_time,
    })
    output = self.root / "saved.json"

    Scratch(str(source)).save(str(output))
    saved = json.loads(output.read_text(encoding="utf-8"))

    self.assertEqual(original, saved)
    self.assertEqual(curr_time, saved["curr_time"])
    self.assertEqual(act_start_time, saved["act_start_time"])
    self.assertEqual(datetime.datetime.strptime(curr_time, DATETIME_FORMAT),
                     Scratch(str(output)).curr_time)

  def test_05_missing_structural_key_still_fails_closed(self):
    source, _ = self.copy_scratch()
    data = json.loads(source.read_text(encoding="utf-8"))
    del data["curr_time"]
    source.write_text(json.dumps(data), encoding="utf-8")

    with self.assertRaises(KeyError) as raised:
      Scratch(str(source))

    self.assertEqual("curr_time", raised.exception.args[0])

  def test_06_persona_save_accepts_an_unstarted_legacy_persona(self):
    persona_root = self.root / "Isabella Rodriguez"
    shutil.copytree(BASELINE / "personas" / "Isabella Rodriguez", persona_root)
    scratch_path = persona_root / "bootstrap_memory" / "scratch.json"
    original = json.loads(scratch_path.read_text(encoding="utf-8"))
    persona = Persona("Isabella Rodriguez", str(persona_root))

    persona.save(str(persona_root / "bootstrap_memory"))

    saved = json.loads(scratch_path.read_text(encoding="utf-8"))
    self.assertEqual(original, saved)
    self.assertIsNone(saved["curr_time"])
    self.assertIsNone(saved["act_start_time"])

  def test_07_persona_save_order_is_multi_file_non_atomic(self):
    persona_root = self.root / "Isabella Rodriguez"
    shutil.copytree(BASELINE / "personas" / "Isabella Rodriguez", persona_root)
    memory = persona_root / "bootstrap_memory"
    scratch_path = memory / "scratch.json"
    scratch_before = scratch_path.read_bytes()
    persona = Persona("Isabella Rodriguez", str(persona_root))
    persona.s_mem.tree = {"persistence_order_marker": {}}
    del persona.scratch.curr_time

    with self.assertRaises(AttributeError):
      persona.save(str(memory))

    self.assertEqual(
      {"persistence_order_marker": {}},
      json.loads((memory / "spatial_memory.json").read_text(encoding="utf-8")))
    self.assertEqual(scratch_before, scratch_path.read_bytes())


class IntegratedScratchPersistenceTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.storage = self.root / "storage"
    self.frontend_temp = self.root / "temp_storage"
    self.storage.mkdir()
    self.frontend_temp.mkdir()
    self.protected_temp = (REPOSITORY / "environment" / "frontend_server"
                           / "temp_storage")
    self.protected_temp_before = _files(self.protected_temp)
    self.baseline_before = _files(BASELINE)

  def tearDown(self):
    self.temporary.cleanup()

  def test_01_post_tick_equivalent_save_and_reload_is_offline_and_contained(self):
    fork_code = "r1t-persist-fork"
    saved_code = "r1t-persist-saved"
    reload_code = "r1t-persist-reloaded"
    fork_root = self.storage / fork_code
    shutil.copytree(BASELINE, fork_root)
    fixture = prepare_isolated_reverie_fixture(
      fork_root, self.root, BASELINE, 0)
    embedding_result = prepare_isolated_embedding_stores(fixture)
    self.assertEqual(set(PERSONAS), set(embedding_result.bootstrapped_personas))

    adapter = R1TDeterministicFakeAdapter()
    network_calls = []

    def block_network(*args, **kwargs):
      network_calls.append("blocked")
      raise AssertionError("network is forbidden in persistence tests")

    previous_cwd = Path.cwd()
    try:
      os.chdir(BACKEND_SERVER)
      with patch.object(reverie_module, "fs_storage", str(self.storage)), \
          patch.object(reverie_module, "fs_temp_storage",
                       str(self.frontend_temp)), \
          patch("socket.create_connection", side_effect=block_network), \
          patch.object(socket.socket, "connect", block_network), \
          use_embedding_runtime(
            build_modern_embedding_runtime_config(), adapter):
        server = reverie_module.ReverieServer(fork_code, saved_code)
        active = server.personas["Isabella Rodriguez"].scratch
        active.curr_time = server.curr_time
        active.act_start_time = server.curr_time

        saved_root = self.storage / saved_code
        movement_path = saved_root / "movement" / "0.json"
        movement_path.write_text(json.dumps({
          "persona": {},
          "meta": {"curr_time": server.curr_time.strftime(DATETIME_FORMAT)},
        }, indent=2), encoding="utf-8")
        movement_digest = hashlib.sha256(movement_path.read_bytes()).hexdigest()
        shutil.copyfile(saved_root / "environment" / "0.json",
                        saved_root / "environment" / "1.json")

        expected_time = server.curr_time + datetime.timedelta(
          seconds=server.sec_per_step)
        server.step = 1
        server.curr_time = expected_time
        server.save()

        saved_meta = json.loads(
          (saved_root / "reverie" / "meta.json").read_text(encoding="utf-8"))
        saved_scratches = {
          name: json.loads((saved_root / "personas" / name
                            / "bootstrap_memory" / "scratch.json").read_text(
                              encoding="utf-8"))
          for name in PERSONAS
        }
        reloaded = reverie_module.ReverieServer(saved_code, reload_code)
    finally:
      os.chdir(previous_cwd)

    self.assertEqual([], adapter.calls)
    self.assertEqual([], network_calls)
    self.assertEqual(1, reloaded.step)
    self.assertEqual(expected_time, reloaded.curr_time)
    self.assertEqual("February 13, 2023, 00:00:10",
                     saved_meta["curr_time"])
    self.assertEqual(1, saved_meta["step"])
    self.assertEqual("February 13, 2023, 00:00:00",
                     saved_scratches["Isabella Rodriguez"]["curr_time"])
    for passive in ("Maria Lopez", "Klaus Mueller"):
      self.assertIsNone(saved_scratches[passive]["curr_time"])
      self.assertIsNone(saved_scratches[passive]["act_start_time"])
      self.assertIsNone(reloaded.personas[passive].scratch.curr_time)
      self.assertIsNone(reloaded.personas[passive].scratch.act_start_time)

    saved_root = self.storage / saved_code
    self.assertEqual(
      movement_digest,
      hashlib.sha256((saved_root / "movement" / "0.json").read_bytes()).hexdigest())
    for name in PERSONAS:
      store = (saved_root / "personas" / name / "bootstrap_memory"
               / "associative_memory")
      self.assertEqual(TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
                       read_embedding_manifest(
                         store / EMBEDDING_MANIFEST_FILENAME))
      self.assertTrue(reloaded.personas[name].a_mem is not None)
      self.assertTrue(reloaded.personas[name].s_mem is not None)
      self.assertTrue(reloaded.personas[name].scratch is not None)

    self.assertEqual(self.baseline_before, _files(BASELINE))
    self.assertEqual(self.protected_temp_before, _files(self.protected_temp))
    for path in self.root.rglob("*"):
      self.assertTrue(path == self.root or self.root in path.parents)


if __name__ == "__main__":
  unittest.main()
