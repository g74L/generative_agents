import datetime
import hashlib
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
REPOSITORY = BACKEND_SERVER.parents[1]
TESTS_DIR = Path(__file__).resolve().parent
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))
from persona.memory_structures.scratch import Scratch
import r1v_test_support as support


BASELINE = (REPOSITORY / "environment" / "frontend_server" / "storage"
            / "base_the_ville_isabella_maria_klaus")
START = datetime.datetime(2023, 2, 13, 5, 55)
SERIALIZED_START = "February 13, 2023, 05:55:00"


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class R1VTemporalBootstrapTests(unittest.TestCase):
  def test_01_bootstrap_preserves_synthetic_state_and_realigns_time(self):
    network_calls = []

    def block_network(*args, **kwargs):
      del args, kwargs
      network_calls.append("blocked")
      raise AssertionError("network is forbidden during R1V preparation")

    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      seed = support.build_synthetic_cognitive_seed(BASELINE, root / "seed")
      source = root / "source"
      seed_nodes = digest(
        seed / "personas" / support.ACTOR / "bootstrap_memory"
        / "associative_memory" / "nodes.json")
      seed_embeddings = digest(
        seed / "personas" / support.ACTOR / "bootstrap_memory"
        / "associative_memory" / "embeddings.json")
      seed_scratch = support.read_json(
        seed / "personas" / support.ACTOR / "bootstrap_memory"
        / "scratch.json")
      with patch("socket.create_connection", side_effect=block_network), \
          patch.object(socket.socket, "connect", block_network):
        result = support.bootstrap_temporal_source(seed, source, START)

      meta = support.read_json(source / "reverie" / "meta.json")
      self.assertEqual(SERIALIZED_START, meta["curr_time"])
      self.assertEqual(0, meta["step"])
      self.assertEqual(["0.json"], [
        path.name for path in (source / "environment").glob("*.json")])
      self.assertEqual([], list((source / "movement").glob("*.json")))
      for name in support.PERSONAS:
        scratch = support.read_json(
          source / "personas" / name / "bootstrap_memory" / "scratch.json")
        self.assertEqual(SERIALIZED_START, scratch["curr_time"])

      actor = support.read_json(
        source / "personas" / support.ACTOR / "bootstrap_memory"
        / "scratch.json")
      self.assertEqual("sleeping", actor["act_description"])
      self.assertTrue(actor["act_address"].endswith(":bed"))
      self.assertEqual(360, actor["act_duration"])
      self.assertEqual("February 13, 2023, 00:00:00",
                       actor["act_start_time"])
      self.assertEqual([], actor["planned_path"])
      self.assertEqual(seed_scratch["daily_req"], actor["daily_req"])
      self.assertEqual(
        seed_scratch["f_daily_schedule"], actor["f_daily_schedule"])
      self.assertEqual(seed_nodes, digest(
        source / "personas" / support.ACTOR / "bootstrap_memory"
        / "associative_memory" / "nodes.json"))
      self.assertEqual(seed_embeddings, digest(
        source / "personas" / support.ACTOR / "bootstrap_memory"
        / "associative_memory" / "embeddings.json"))
      self.assertEqual("February 13, 2023, 06:00:00",
                       result["sleep_end_time"])
      self.assertEqual([], network_calls)

  def test_02_sleep_duration_uses_minutes_and_finishes_at_0600(self):
    start = datetime.datetime.strptime(
      "February 13, 2023, 00:00:00", support.DATE_FORMAT)
    end = start + datetime.timedelta(minutes=360)
    self.assertEqual(datetime.datetime(2023, 2, 13, 6, 0), end)
    self.assertLess(START, end)
    self.assertEqual(datetime.timedelta(minutes=5), end - START)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      seed = support.build_synthetic_cognitive_seed(BASELINE, root / "seed")
      scratch_path = (seed / "personas" / support.ACTOR
                      / "bootstrap_memory" / "scratch.json")
      loaded = Scratch(str(scratch_path))
      loaded.curr_time = START
      self.assertFalse(loaded.act_check_finished())
      loaded.curr_time = end
      self.assertTrue(loaded.act_check_finished())


if __name__ == "__main__":
  unittest.main()
