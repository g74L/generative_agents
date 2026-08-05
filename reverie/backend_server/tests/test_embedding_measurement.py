from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.cognitive_modules import plan as plan_module
from persona.cognitive_modules import reflect as reflect_module
from persona.prompt_template import gpt_structure
from persona.prompt_template.llm_provider import (
  CONVERSATION,
  DISABLED,
  EMBEDDING,
  HIT,
  MISS,
  PLANNING,
  REFLECTION,
  RETRIEVAL,
  UNSPECIFIED,
  FakeProvider,
  clear_embedding_cache,
  clear_telemetry,
  embedding_call_context,
  get_embedding_call_category,
  get_embedding_logical_events,
  get_embedding_measurement_snapshot,
  get_telemetry,
  reset_embedding_cache,
  reset_embedding_cache_statistics,
  reset_embedding_measurement_all,
  set_embedding_cache_capacity,
  set_embedding_cache_enabled,
  use_provider,
)


class EmbeddingMeasurementTests(unittest.TestCase):
  def setUp(self):
    clear_telemetry()
    reset_embedding_cache()
    self.fake = FakeProvider("measurement-fake")
    self.provider_context = use_provider(self.fake)
    self.provider_context.__enter__()
    self.network_patch = patch.object(
      gpt_structure.openai.Embedding, "create",
      side_effect=AssertionError("network embedding transport used"))
    self.network_patch.start()

  def tearDown(self):
    self.network_patch.stop()
    self.provider_context.__exit__(None, None, None)
    clear_telemetry()
    reset_embedding_cache()

  def category_stats(self, category):
    return get_embedding_measurement_snapshot()["by_category"][category]

  def test_embedding_measurement_default_category(self):
    self.fake.queue_embedding_response([1.0])

    self.assertEqual([1.0], gpt_structure.get_embedding("default"))

    event = get_embedding_logical_events()[0]
    self.assertEqual(UNSPECIFIED, event.category)
    self.assertEqual(MISS, event.cache_outcome)
    self.assertEqual({
      "logical_requests": 1,
      "physical_attempts": 1,
      "cache_hits": 0,
      "cache_misses": 1,
      "cache_hit_rate": 0.0,
      "evictions": 0,
    }, self.category_stats(UNSPECIFIED))

  def test_embedding_measurement_retrieval_category_hit_and_miss(self):
    self.fake.queue_embedding_response([1.0])

    with embedding_call_context(RETRIEVAL):
      first = gpt_structure.get_embedding("retrieval duplicate")
      second = gpt_structure.get_embedding("retrieval duplicate")

    self.assertEqual(first, second)
    self.assertEqual([MISS, HIT],
                     [event.cache_outcome
                      for event in get_embedding_logical_events()])
    self.assertEqual({
      "logical_requests": 2,
      "physical_attempts": 1,
      "cache_hits": 1,
      "cache_misses": 1,
      "cache_hit_rate": 0.5,
      "evictions": 0,
    }, self.category_stats(RETRIEVAL))

  def test_embedding_measurement_nested_context_restores_parent(self):
    for vector in ([1.0], [2.0], [3.0], [4.0]):
      self.fake.queue_embedding_response(vector)

    with embedding_call_context(PLANNING):
      gpt_structure.get_embedding("A")
      with embedding_call_context(RETRIEVAL):
        gpt_structure.get_embedding("B")
      gpt_structure.get_embedding("C")
    gpt_structure.get_embedding("D")

    self.assertEqual(
      [PLANNING, RETRIEVAL, PLANNING, UNSPECIFIED],
      [event.category for event in get_embedding_logical_events()])
    self.assertEqual(UNSPECIFIED, get_embedding_call_category())

  def test_embedding_measurement_context_restores_after_exception(self):
    with self.assertRaisesRegex(RuntimeError, "context failure"):
      with embedding_call_context(REFLECTION):
        self.assertEqual(REFLECTION, get_embedding_call_category())
        raise RuntimeError("context failure")

    self.assertEqual(UNSPECIFIED, get_embedding_call_category())

  def test_embedding_measurement_cross_category_hit(self):
    self.fake.queue_embedding_response([1.0])

    with embedding_call_context(RETRIEVAL):
      first = gpt_structure.get_embedding("shared")
    with embedding_call_context(CONVERSATION):
      second = gpt_structure.get_embedding("shared")

    self.assertEqual(first, second)
    events = get_embedding_logical_events()
    self.assertEqual([RETRIEVAL, CONVERSATION],
                     [event.category for event in events])
    self.assertEqual([MISS, HIT], [event.cache_outcome for event in events])
    self.assertEqual(events[0].cache_key_fingerprint,
                     events[1].cache_key_fingerprint)
    self.assertEqual(1, len(self.fake.calls))
    self.assertEqual(1, self.category_stats(RETRIEVAL)["cache_misses"])
    self.assertEqual(1, self.category_stats(CONVERSATION)["cache_hits"])

  def test_embedding_measurement_disabled_cache(self):
    set_embedding_cache_enabled(False)
    self.fake.queue_embedding_response([1.0])
    self.fake.queue_embedding_response([2.0])

    with embedding_call_context(CONVERSATION):
      gpt_structure.get_embedding("same")
      gpt_structure.get_embedding("same")

    self.assertEqual([DISABLED, DISABLED],
                     [event.cache_outcome
                      for event in get_embedding_logical_events()])
    stats = self.category_stats(CONVERSATION)
    self.assertEqual(2, stats["logical_requests"])
    self.assertEqual(2, stats["physical_attempts"])
    self.assertEqual(0, stats["cache_hits"])
    self.assertEqual(2, stats["cache_misses"])

  def test_embedding_measurement_error_is_not_cached(self):
    self.fake.queue_error(EMBEDDING, RuntimeError("reflection failure"))
    self.fake.queue_embedding_response([2.0])

    with embedding_call_context(REFLECTION):
      with self.assertRaisesRegex(RuntimeError, "reflection failure"):
        gpt_structure.get_embedding("reflection text")
      result = gpt_structure.get_embedding("reflection text")

    self.assertEqual([2.0], result)
    stats = self.category_stats(REFLECTION)
    self.assertEqual(2, stats["logical_requests"])
    self.assertEqual(2, stats["physical_attempts"])
    self.assertEqual(0, stats["cache_hits"])
    self.assertEqual(2, stats["cache_misses"])

  def test_embedding_measurement_reset_statistics_preserves_entries(self):
    self.fake.queue_embedding_response([1.0])
    with embedding_call_context(RETRIEVAL):
      gpt_structure.get_embedding("preserved")

    reset_embedding_cache_statistics()
    with embedding_call_context(CONVERSATION):
      result = gpt_structure.get_embedding("preserved")

    self.assertEqual([1.0], result)
    snapshot = get_embedding_measurement_snapshot()
    self.assertEqual(1, snapshot["global"]["logical_embedding_requests"])
    self.assertEqual(0, snapshot["global"]["physical_embedding_attempts"])
    self.assertEqual(1, snapshot["global"]["cache_hits"])
    self.assertEqual(1, snapshot["global"]["cache_entries"])
    self.assertEqual(1, snapshot["by_category"][CONVERSATION]["cache_hits"])

  def test_embedding_measurement_clear_cache_preserves_statistics(self):
    self.fake.queue_embedding_response([1.0])
    self.fake.queue_embedding_response([2.0])
    with embedding_call_context(RETRIEVAL):
      gpt_structure.get_embedding("cleared")
      clear_embedding_cache()
      result = gpt_structure.get_embedding("cleared")

    self.assertEqual([2.0], result)
    stats = self.category_stats(RETRIEVAL)
    self.assertEqual(2, stats["logical_requests"])
    self.assertEqual(2, stats["physical_attempts"])
    self.assertEqual(2, stats["cache_misses"])

  def test_embedding_measurement_reset_all(self):
    set_embedding_cache_capacity(2)
    set_embedding_cache_enabled(False)
    self.fake.queue_embedding_response([1.0])
    with embedding_call_context(PLANNING):
      gpt_structure.get_embedding("reset all")

    reset_embedding_measurement_all()

    snapshot = get_embedding_measurement_snapshot()
    self.assertEqual(0, snapshot["global"]["logical_embedding_requests"])
    self.assertEqual(0, snapshot["global"]["physical_embedding_attempts"])
    self.assertEqual(0, snapshot["global"]["cache_entries"])
    self.assertEqual(1024, snapshot["global"]["capacity"])
    self.assertTrue(snapshot["global"]["enabled"])
    self.assertEqual(UNSPECIFIED, get_embedding_call_category())
    self.assertEqual((), get_embedding_logical_events())
    self.assertEqual((), get_telemetry())

  def test_embedding_measurement_reflection_minimal_runtime_context(self):
    self.fake.queue_embedding_response([1.0])
    persona = SimpleNamespace(scratch=SimpleNamespace(
      importance_trigger_max=150,
      importance_trigger_curr=0,
      importance_ele_n=1,
      chatting_end_time=None,
    ))

    def minimal_reflection(_persona):
      gpt_structure.get_embedding("minimal reflection")

    with patch.object(reflect_module, "reflection_trigger", return_value=True), \
         patch.object(reflect_module, "run_reflect", side_effect=minimal_reflection):
      reflect_module.reflect(persona)

    self.assertEqual([REFLECTION],
                     [event.category
                      for event in get_embedding_logical_events()])

  def test_embedding_measurement_planning_minimal_runtime_context(self):
    self.fake.queue_embedding_response([1.0])

    def act_check_finished():
      gpt_structure.get_embedding("minimal planning")
      return False

    scratch = SimpleNamespace(
      act_check_finished=act_check_finished,
      act_event=("Alice", "is", "reading"),
      act_address="world:sector:arena:object",
      chatting_with=None,
      chat=None,
      chatting_end_time=None,
      chatting_with_buffer={},
    )
    persona = SimpleNamespace(scratch=scratch)

    result = plan_module.plan(persona, None, {}, False, {})

    self.assertEqual("world:sector:arena:object", result)
    self.assertEqual([PLANNING],
                     [event.category
                      for event in get_embedding_logical_events()])

  def test_embedding_measurement_synthetic_offline_workload(self):
    set_embedding_cache_capacity(3)
    for vector in ([1.0], [2.0], [3.0], [4.0], [5.0]):
      self.fake.queue_embedding_response(vector)

    with embedding_call_context(RETRIEVAL):
      gpt_structure.get_embedding("retrieval repeated")
      gpt_structure.get_embedding("retrieval repeated")
    with embedding_call_context(CONVERSATION):
      gpt_structure.get_embedding("retrieval repeated")
      gpt_structure.get_embedding("conversation one")
      gpt_structure.get_embedding("conversation one")
      gpt_structure.get_embedding("conversation two")
    with embedding_call_context(REFLECTION):
      gpt_structure.get_embedding("reflection synthetic")
    with embedding_call_context(PLANNING):
      gpt_structure.get_embedding("planning synthetic")

    snapshot = get_embedding_measurement_snapshot()
    self.assertEqual({
      "logical_embedding_requests": 8,
      "physical_embedding_attempts": 5,
      "cache_hits": 3,
      "cache_misses": 5,
      "cache_entries": 3,
      "evictions": 2,
      "enabled": True,
      "capacity": 3,
    }, snapshot["global"])
    self.assertEqual(0.5,
                     snapshot["by_category"][RETRIEVAL]["cache_hit_rate"])
    self.assertEqual(0.5,
                     snapshot["by_category"][CONVERSATION]["cache_hit_rate"])
    self.assertEqual(1,
                     snapshot["by_category"][RETRIEVAL]["evictions"])
    self.assertEqual(1,
                     snapshot["by_category"][CONVERSATION]["evictions"])

  def test_embedding_measurement_snapshot_contains_no_sensitive_content(self):
    sensitive_text = "private memory and conversation sk-test-secret"
    self.fake.queue_embedding_response([1.0])
    with embedding_call_context(RETRIEVAL):
      gpt_structure.get_embedding(sensitive_text)

    snapshot_text = repr(get_embedding_measurement_snapshot())

    self.assertNotIn(sensitive_text, snapshot_text)
    self.assertNotIn("sk-test-secret", snapshot_text)
    self.assertNotIn("prompt", snapshot_text.lower())
    self.assertNotIn("messages", snapshot_text.lower())
    self.assertNotIn("normalized_text", snapshot_text.lower())


if __name__ == "__main__":
  unittest.main()
