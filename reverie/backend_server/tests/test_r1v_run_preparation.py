import datetime
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))
import r1v_test_support as support


REPOSITORY = TESTS_DIR.parents[2]
BASELINE = (REPOSITORY / "environment" / "frontend_server" / "storage"
            / "base_the_ville_isabella_maria_klaus")
SIM_CODE = "r1v-synthetic-test"
RELOAD_CODE = SIM_CODE + "-reload"
START = datetime.datetime(2023, 2, 13, 5, 55)


class R1VRunPreparationTests(unittest.TestCase):
  def test_01_first_use_is_accepted_in_an_empty_temporary_root(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      run_dir = root / "run"
      storage = root / "storage"
      run_dir.mkdir()
      storage.mkdir()
      support.assert_fresh_run_outputs(
        run_dir, storage, SIM_CODE, RELOAD_CODE)
      self.assertEqual([], list(run_dir.iterdir()))
      self.assertEqual([], list(storage.iterdir()))

  def test_02_second_use_is_rejected_and_temp_reset_is_explicit(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      run_dir = root / "run"
      storage = root / "storage"
      run_dir.mkdir()
      storage.mkdir()
      support.assert_fresh_run_outputs(
        run_dir, storage, SIM_CODE, RELOAD_CODE)
      target = storage / SIM_CODE
      target.mkdir()
      marker = target / "synthetic-marker.txt"
      marker.write_text("preserve", encoding="utf-8")

      with self.assertRaisesRegex(
          RuntimeError, "R1V_RUN_OUTPUT_ALREADY_EXISTS"):
        support.assert_fresh_run_outputs(
          run_dir, storage, SIM_CODE, RELOAD_CODE)
      self.assertEqual("preserve", marker.read_text(encoding="utf-8"))

      marker.unlink()
      target.rmdir()
      support.assert_fresh_run_outputs(
        run_dir, storage, SIM_CODE, RELOAD_CODE)

  def test_03_each_durable_output_fails_closed_without_overwrite(self):
    for name in support.RUN_OUTPUT_NAMES:
      with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = root / "run"
        storage = root / "storage"
        run_dir.mkdir()
        storage.mkdir()
        output = run_dir / name
        if name == "tick-reports":
          output.mkdir()
          marker = output / "synthetic-marker.txt"
        else:
          marker = output
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(
            RuntimeError, "R1V_RUN_OUTPUT_ALREADY_EXISTS"):
          support.assert_fresh_run_outputs(
            run_dir, storage, SIM_CODE, RELOAD_CODE)
        self.assertEqual("preserve", marker.read_text(encoding="utf-8"))

  def test_04_reload_target_also_fails_closed(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      run_dir = root / "run"
      storage = root / "storage"
      run_dir.mkdir()
      storage.mkdir()
      reload_target = storage / RELOAD_CODE
      reload_target.mkdir()
      with self.assertRaisesRegex(
          RuntimeError, f"simulation:{RELOAD_CODE}"):
        support.assert_fresh_run_outputs(
          run_dir, storage, SIM_CODE, RELOAD_CODE)
      self.assertTrue(reload_target.is_dir())

  def test_05_temporary_preparation_leaves_no_files_after_cleanup(self):
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    run_dir = root / "run"
    storage = root / "storage"
    run_dir.mkdir()
    storage.mkdir()
    support.assert_fresh_run_outputs(
      run_dir, storage, SIM_CODE, RELOAD_CODE)
    temporary.cleanup()
    self.assertFalse(root.exists())

  def test_06_synthetic_source_is_clean_and_temporally_coherent(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      seed = support.build_synthetic_cognitive_seed(
        BASELINE, root / "seed")
      source = root / "source"
      support.bootstrap_temporal_source(seed, source, START)
      meta = support.read_json(source / "reverie" / "meta.json")
      scratch = support.read_json(
        source / "personas" / support.ACTOR / "bootstrap_memory"
        / "scratch.json")
      self.assertEqual(0, meta["step"])
      self.assertEqual("February 13, 2023, 05:55:00", meta["curr_time"])
      self.assertEqual(10, meta["sec_per_step"])
      self.assertEqual(["0.json"], [
        path.name for path in (source / "environment").glob("*.json")])
      self.assertEqual([], list((source / "movement").glob("*.json")))
      self.assertEqual("sleeping", scratch["act_description"])
      self.assertEqual(360, scratch["act_duration"])
      self.assertEqual([], scratch["planned_path"])

  def test_07_preparation_is_offline_and_has_no_runtime_dependency(self):
    network_calls = []

    def block_network(*args, **kwargs):
      del args, kwargs
      network_calls.append("blocked")
      raise AssertionError("network is forbidden during R1V preparation")

    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      with patch("socket.create_connection", side_effect=block_network), \
          patch.object(socket.socket, "connect", block_network):
        seed = support.build_synthetic_cognitive_seed(
          BASELINE, root / "seed")
        support.bootstrap_temporal_source(seed, root / "source", START)
      self.assertTrue((root / "source" / "environment" / "0.json").is_file())
    self.assertEqual([], network_calls)
    helper_path = Path(support.__file__).resolve()
    self.assertEqual("tests", helper_path.parent.name)


if __name__ == "__main__":
  unittest.main()
