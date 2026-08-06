import datetime
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

TESTS_DIR = Path(__file__).resolve().parent
REPOSITORY = TESTS_DIR.parents[2]
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))
import r1v_test_support as support


class R1VFailureObservabilityTests(unittest.TestCase):
  def test_01_exception_message_and_context_are_persisted(self):
    try:
      raise IndexError("synthetic index failure")
    except IndexError as error:
      synthetic_error = error
    server = SimpleNamespace(
      step=30, curr_time=datetime.datetime(2023, 2, 13, 6, 0))
    report = support.build_failure_observability(
      synthetic_error, 30, server, [object()] * 30,
      ["TICK_29_COMPLETED"], REPOSITORY)

    self.assertEqual("IndexError", report["exception_type"])
    self.assertEqual("synthetic index failure", report["exception_message"])
    self.assertIn("IndexError", report["exception_repr"])
    self.assertTrue(report["traceback"])
    self.assertEqual(
      {"file", "function", "line"}, set(report["traceback"][-1]))
    self.assertFalse(Path(report["traceback"][-1]["file"]).is_absolute())
    self.assertEqual({
      "tick": 30,
      "step": 30,
      "curr_time": datetime.datetime(2023, 2, 13, 6, 0),
      "completed_ticks": 30,
      "last_completed_checkpoint": "TICK_29_COMPLETED",
    }, report["failure_context"])

  def test_02_success_path_is_additive_and_null(self):
    report = support.build_failure_observability(
      None, 59, SimpleNamespace(step=60),
      [object()] * 60, ["SIXTY_TICK_RUN_COMPLETED"])
    self.assertIsNone(report["exception_type"])
    self.assertIsNone(report["exception_message"])
    self.assertIsNone(report["exception_repr"])
    self.assertEqual([], report["traceback"])
    self.assertEqual({
      "tick": None,
      "step": None,
      "curr_time": None,
      "completed_ticks": 60,
      "last_completed_checkpoint": "SIXTY_TICK_RUN_COMPLETED",
    }, report["failure_context"])

  def test_03_atomic_json_serializes_datetime(self):
    report = support.build_failure_observability(
      IndexError("synthetic index failure"), 30,
      SimpleNamespace(
        step=30, curr_time=datetime.datetime(2023, 2, 13, 6, 0)),
      [object()] * 30, ["TICK_29_COMPLETED"])
    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary) / "report.json"
      support.atomic_json(path, report)
      persisted = json.loads(path.read_text(encoding="utf-8"))
    self.assertEqual("February 13, 2023, 06:00:00",
                     persisted["failure_context"]["curr_time"])
    self.assertEqual("synthetic index failure",
                     persisted["exception_message"])

  def test_04_failure_evidence_has_no_sensitive_dump_keys(self):
    report = support.build_failure_observability(
      IndexError("synthetic index failure"), 30,
      SimpleNamespace(step=30, curr_time=None), [], [])
    serialized = json.dumps(support.safe(report), sort_keys=True)
    for forbidden in (
        "prompt", "raw_response", "api_key", "embedding_vector",
        "memory_text", "locals"):
      self.assertNotIn(forbidden, serialized.lower())

  def test_05_versioned_helper_never_inspects_frame_locals(self):
    source = inspect.getsource(support)
    self.assertNotIn("f_locals", source)
    self.assertNotIn("locals()", source)


if __name__ == "__main__":
  unittest.main()
