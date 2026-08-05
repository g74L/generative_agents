from pathlib import Path
import sys
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.prompt_template import gpt_structure
from persona.prompt_template.llm_provider import (
  DISABLED,
  EMBEDDING,
  HIT,
  MISS,
  FakeProvider,
  clear_embedding_cache,
  clear_telemetry,
  get_embedding_cache_stats,
  get_embedding_logical_events,
  get_telemetry,
  reset_embedding_cache,
  set_embedding_cache_capacity,
  set_embedding_cache_enabled,
  use_provider,
)


class EmbeddingCacheTests(unittest.TestCase):
  def setUp(self):
    clear_telemetry()
    reset_embedding_cache()
    self.fake = FakeProvider("fake-primary")
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

  def assert_stats(self, *, logical, physical, hits, misses, entries,
                   evictions=0):
    stats = get_embedding_cache_stats()
    self.assertEqual(logical, stats.logical_embedding_requests)
    self.assertEqual(physical, stats.physical_embedding_attempts)
    self.assertEqual(hits, stats.cache_hits)
    self.assertEqual(misses, stats.cache_misses)
    self.assertEqual(entries, stats.cache_entries)
    self.assertEqual(evictions, stats.evictions)
    self.assertEqual(stats.cache_misses, stats.physical_embedding_attempts)

  def test_embedding_cache_exact_duplicate_hit(self):
    self.fake.queue_embedding_response([0.1, 0.2])

    first = gpt_structure.get_embedding("same")
    second = gpt_structure.get_embedding("same")

    self.assertEqual([0.1, 0.2], first)
    self.assertEqual(first, second)
    self.assertEqual(1, len(self.fake.calls))
    self.assertEqual([MISS, HIT],
                     [event.cache_outcome
                      for event in get_embedding_logical_events()])
    self.assert_stats(logical=2, physical=1, hits=1, misses=1, entries=1)

  def test_embedding_cache_newline_equivalence(self):
    self.fake.queue_embedding_response([1.0])

    first = gpt_structure.get_embedding("line one\nline two")
    second = gpt_structure.get_embedding("line one line two")

    self.assertEqual(first, second)
    self.assertEqual(["line one line two"],
                     self.fake.calls[0].arguments["input"])
    self.assertEqual(1, len(self.fake.calls))
    self.assert_stats(logical=2, physical=1, hits=1, misses=1, entries=1)

  def test_embedding_cache_blank_equivalence_is_not_expanded(self):
    self.fake.queue_embedding_response([2.0])
    self.fake.queue_embedding_response([3.0])

    blank = gpt_structure.get_embedding("")
    explicit = gpt_structure.get_embedding("this is blank")
    whitespace = gpt_structure.get_embedding(" ")

    self.assertEqual(blank, explicit)
    self.assertNotEqual(blank, whitespace)
    self.assertEqual(["this is blank"], self.fake.calls[0].arguments["input"])
    self.assertEqual([" "], self.fake.calls[1].arguments["input"])
    self.assert_stats(logical=3, physical=2, hits=1, misses=2, entries=2)

  def test_embedding_cache_different_texts_are_misses(self):
    self.fake.queue_embedding_response([1.0])
    self.fake.queue_embedding_response([2.0])

    first = gpt_structure.get_embedding("alpha")
    second = gpt_structure.get_embedding("beta")

    self.assertNotEqual(first, second)
    self.assertEqual(2, len(self.fake.calls))
    self.assert_stats(logical=2, physical=2, hits=0, misses=2, entries=2)

  def test_embedding_cache_different_models_are_misses(self):
    self.fake.queue_embedding_response([1.0])
    self.fake.queue_embedding_response([2.0])

    gpt_structure.get_embedding("same", model="embedding-a")
    gpt_structure.get_embedding("same", model="embedding-b")

    self.assertEqual(["embedding-a", "embedding-b"],
                     [call.arguments["model"] for call in self.fake.calls])
    self.assert_stats(logical=2, physical=2, hits=0, misses=2, entries=2)

  def test_embedding_cache_different_providers_are_misses(self):
    second_fake = FakeProvider("fake-secondary")
    self.fake.queue_embedding_response([1.0])
    second_fake.queue_embedding_response([2.0])

    first = gpt_structure.get_embedding("same")
    with use_provider(second_fake):
      second = gpt_structure.get_embedding("same")

    self.assertNotEqual(first, second)
    self.assertEqual(
      ["fake-primary", "fake-secondary"],
      [event.provider_identity for event in get_embedding_logical_events()])
    self.assert_stats(logical=2, physical=2, hits=0, misses=2, entries=2)

  def test_embedding_cache_returns_copy_safe_vectors(self):
    self.fake.queue_embedding_response([1.0, 2.0])

    first = gpt_structure.get_embedding("mutable")
    first[0] = 999.0
    first.append(3.0)
    second = gpt_structure.get_embedding("mutable")
    second[1] = 888.0
    third = gpt_structure.get_embedding("mutable")

    self.assertEqual([1.0, 2.0], third)
    self.assertEqual(1, len(self.fake.calls))
    self.assert_stats(logical=3, physical=1, hits=2, misses=1, entries=1)

  def test_embedding_cache_failure_is_not_cached(self):
    self.fake.queue_error(EMBEDDING, RuntimeError("temporary failure"))
    self.fake.queue_embedding_response([4.0])

    with self.assertRaisesRegex(RuntimeError, "temporary failure"):
      gpt_structure.get_embedding("retry me")
    self.assert_stats(logical=1, physical=1, hits=0, misses=1, entries=0)

    result = gpt_structure.get_embedding("retry me")

    self.assertEqual([4.0], result)
    self.assertEqual(2, len(self.fake.calls))
    self.assert_stats(logical=2, physical=2, hits=0, misses=2, entries=1)
    self.assertEqual(["ERROR", "SUCCESS"],
                     [event.outcome for event in get_telemetry()])

  def test_embedding_cache_invalid_vector_is_not_cached(self):
    self.fake.queue_embedding_response([])
    self.fake.queue_embedding_response([5.0])

    self.assertEqual([], gpt_structure.get_embedding("invalid first"))
    self.assertEqual([5.0], gpt_structure.get_embedding("invalid first"))

    self.assertEqual(2, len(self.fake.calls))
    self.assert_stats(logical=2, physical=2, hits=0, misses=2, entries=1)

  def test_embedding_cache_lru_eviction(self):
    set_embedding_cache_capacity(2)
    for vector in ([1.0], [2.0], [3.0], [4.0]):
      self.fake.queue_embedding_response(vector)

    self.assertEqual([1.0], gpt_structure.get_embedding("A"))
    self.assertEqual([2.0], gpt_structure.get_embedding("B"))
    self.assertEqual([1.0], gpt_structure.get_embedding("A"))
    self.assertEqual([3.0], gpt_structure.get_embedding("C"))
    self.assertEqual([4.0], gpt_structure.get_embedding("B"))

    self.assertEqual(["A", "B", "C", "B"],
                     [call.arguments["input"][0] for call in self.fake.calls])
    self.assert_stats(logical=5, physical=4, hits=1, misses=4,
                      entries=2, evictions=2)

  def test_embedding_cache_clear_and_reset_force_miss(self):
    self.fake.queue_embedding_response([1.0])
    self.fake.queue_embedding_response([2.0])
    self.fake.queue_embedding_response([3.0])

    self.assertEqual([1.0], gpt_structure.get_embedding("reset me"))
    clear_embedding_cache()
    self.assertEqual([2.0], gpt_structure.get_embedding("reset me"))
    reset_embedding_cache()
    self.assertEqual([3.0], gpt_structure.get_embedding("reset me"))

    self.assertEqual(3, len(self.fake.calls))
    self.assert_stats(logical=1, physical=1, hits=0, misses=1, entries=1)

  def test_embedding_cache_disabled_keeps_physical_calls(self):
    set_embedding_cache_enabled(False)
    self.fake.queue_embedding_response([1.0])
    self.fake.queue_embedding_response([2.0])

    first = gpt_structure.get_embedding("same")
    second = gpt_structure.get_embedding("same")

    self.assertEqual([1.0], first)
    self.assertEqual([2.0], second)
    self.assertEqual([DISABLED, DISABLED],
                     [event.cache_outcome
                      for event in get_embedding_logical_events()])
    self.assert_stats(logical=2, physical=2, hits=0, misses=2, entries=0)


if __name__ == "__main__":
  unittest.main()
