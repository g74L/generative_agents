import datetime
from contextlib import redirect_stdout
import io
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.cognitive_modules import retrieve
from persona.memory_structures.associative_memory import AssociativeMemory
from persona.memory_structures.embedding_space import (
  EMBEDDING_MANIFEST_FILENAME,
  LEGACY_ADA_002_MANIFEST,
  EmbeddingManifestError,
  EmbeddingReferenceError,
  EmbeddingSpaceClassification,
  EmbeddingSpaceManifest,
  EmbeddingSpaceMismatchError,
  EmbeddingVectorValidationError,
  LegacyEmbeddingSpaceWarning,
  assert_same_embedding_space,
  get_runtime_embedding_manifest,
  load_embedding_store,
  read_embedding_manifest,
  reset_runtime_embedding_manifest,
  use_runtime_embedding_manifest,
  write_embedding_manifest,
)
from persona.prompt_template import gpt_structure
from persona.prompt_template.llm_provider import (
  EMBEDDING_NORMALIZATION_VERSION,
  EMBEDDING_VERSION,
  FakeProvider,
  clear_telemetry,
  get_embedding_cache_stats,
  get_telemetry,
  reset_embedding_cache,
  use_provider,
)


def _vector(first=1.0, second=0.0, dimensions=1536):
  return [first, second] + [0.0] * (dimensions - 2)


def _node(count, embedding_key, created=None):
  created = created or datetime.datetime(2023, 1, 1, 8, count, 0)
  return {
    "node_count": count,
    "type_count": count,
    "type": "event",
    "depth": 0,
    "created": created.strftime("%Y-%m-%d %H:%M:%S"),
    "expiration": None,
    "subject": "subject",
    "predicate": "is",
    "object": "present",
    "description": f"technical fixture {count}",
    "embedding_key": embedding_key,
    "poignancy": 5,
    "keywords": ["fixture"],
    "filling": [],
  }


class EmbeddingSpaceManifestTests(unittest.TestCase):
  def setUp(self):
    reset_runtime_embedding_manifest()
    reset_embedding_cache()
    clear_telemetry()
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)

  def tearDown(self):
    reset_runtime_embedding_manifest()
    self.temporary.cleanup()

  def make_store(self, canonical=True, embeddings=None, nodes=None,
                 manifest=None, kw_strength=None, namespace=None):
    root = self.root / namespace if namespace else self.root
    if canonical:
      store = root / "bootstrap_memory" / "associative_memory"
    else:
      store = root / "arbitrary_vectors"
    store.mkdir(parents=True, exist_ok=True)
    embeddings = ({"fixture-key": _vector()} if embeddings is None
                  else embeddings)
    nodes = ({"node_1": _node(1, "fixture-key")} if nodes is None else nodes)
    (store / "embeddings.json").write_text(
      json.dumps(embeddings), encoding="utf-8")
    (store / "nodes.json").write_text(json.dumps(nodes), encoding="utf-8")
    kw_strength = (kw_strength if kw_strength is not None else {
      "kw_strength_event": {}, "kw_strength_thought": {}})
    (store / "kw_strength.json").write_text(
      json.dumps(kw_strength), encoding="utf-8")
    if manifest is not None:
      write_embedding_manifest(store, manifest)
    return store

  def changed_manifest(self, **changes):
    values = LEGACY_ADA_002_MANIFEST.to_dict()
    values.update(changes)
    return EmbeddingSpaceManifest.from_dict(values)

  def assert_load_mismatch(self, field, value, dimensions=1536):
    manifest = self.changed_manifest(**{field: value})
    store = self.make_store(
      embeddings={"fixture-key": _vector(dimensions=dimensions)},
      manifest=manifest)
    with self.assertRaisesRegex(EmbeddingSpaceMismatchError, f"field={field}"):
      load_embedding_store(store)

  def test_manifest_valid_round_trip(self):
    store = self.make_store(manifest=LEGACY_ADA_002_MANIFEST)
    self.assertEqual(LEGACY_ADA_002_MANIFEST,
                     read_embedding_manifest(store / EMBEDDING_MANIFEST_FILENAME))
    loaded = load_embedding_store(store)
    self.assertEqual(EmbeddingSpaceClassification.DECLARED,
                     loaded.classification)

  def test_manifest_schema_version_not_supported(self):
    value = LEGACY_ADA_002_MANIFEST.to_dict()
    value["schema_version"] = 2
    with self.assertRaisesRegex(EmbeddingManifestError, "schema_version"):
      EmbeddingSpaceManifest.from_dict(value)

  def test_manifest_missing_field(self):
    value = LEGACY_ADA_002_MANIFEST.to_dict()
    del value["provider"]
    with self.assertRaisesRegex(EmbeddingManifestError, "provider"):
      EmbeddingSpaceManifest.from_dict(value)

  def test_manifest_provider_mismatch(self):
    self.assert_load_mismatch("provider", "different-provider")

  def test_manifest_model_mismatch(self):
    self.assert_load_mismatch("model", "text-embedding-3-small")

  def test_manifest_dimensions_mismatch(self):
    self.assert_load_mismatch("dimensions", 3, dimensions=3)

  def test_manifest_embedding_space_version_mismatch(self):
    self.assert_load_mismatch("embedding_space_version", "different-space-v1")

  def test_manifest_normalization_version_mismatch(self):
    self.assert_load_mismatch("normalization_version", "different-normalization")

  def test_legacy_archive_recognized_in_compatibility_mode(self):
    store = self.make_store()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      loaded = load_embedding_store(store, legacy_assumption_allowed=True)
    self.assertEqual(LEGACY_ADA_002_MANIFEST, loaded.manifest)
    self.assertEqual(EmbeddingSpaceClassification.LEGACY_ASSUMED,
                     loaded.classification)
    self.assertTrue(any(item.category is LegacyEmbeddingSpaceWarning
                        for item in caught))

  def test_legacy_archive_rejected_in_strict_mode(self):
    store = self.make_store()
    with self.assertRaisesRegex(EmbeddingManifestError, "mode=strict"):
      load_embedding_store(store, legacy_assumption_allowed=False)

  def test_unknown_manifestless_archive_rejected(self):
    store = self.make_store(canonical=False)
    with self.assertRaisesRegex(EmbeddingManifestError, "UNKNOWN"):
      load_embedding_store(store, legacy_assumption_allowed=True)

  def test_vector_wrong_dimension_rejected(self):
    store = self.make_store(
      embeddings={"fixture-key": _vector(dimensions=1535)},
      manifest=LEGACY_ADA_002_MANIFEST)
    with self.assertRaisesRegex(EmbeddingVectorValidationError, "dimensions"):
      load_embedding_store(store)

  def test_vector_empty_rejected(self):
    store = self.make_store(embeddings={"fixture-key": []},
                            manifest=LEGACY_ADA_002_MANIFEST)
    with self.assertRaisesRegex(EmbeddingVectorValidationError, "empty"):
      load_embedding_store(store)

  def test_vector_string_rejected_without_conversion(self):
    vector = _vector()
    vector[10] = "0.5"
    store = self.make_store(embeddings={"fixture-key": vector},
                            manifest=LEGACY_ADA_002_MANIFEST)
    with self.assertRaisesRegex(EmbeddingVectorValidationError, "actual=str"):
      load_embedding_store(store)

  def test_vector_boolean_rejected_as_non_numeric(self):
    vector = _vector()
    vector[10] = True
    store = self.make_store(embeddings={"fixture-key": vector},
                            manifest=LEGACY_ADA_002_MANIFEST)
    with self.assertRaisesRegex(EmbeddingVectorValidationError, "actual=bool"):
      load_embedding_store(store)

  def test_vector_nan_rejected(self):
    vector = _vector()
    vector[10] = math.nan
    store = self.make_store(embeddings={"fixture-key": vector},
                            manifest=LEGACY_ADA_002_MANIFEST)
    with self.assertRaisesRegex(EmbeddingVectorValidationError, "non-finite"):
      load_embedding_store(store)

  def test_vector_infinity_rejected(self):
    vector = _vector()
    vector[10] = math.inf
    store = self.make_store(embeddings={"fixture-key": vector},
                            manifest=LEGACY_ADA_002_MANIFEST)
    with self.assertRaisesRegex(EmbeddingVectorValidationError, "non-finite"):
      load_embedding_store(store)

  def test_vector_zero_norm_rejected(self):
    store = self.make_store(embeddings={"fixture-key": [0.0] * 1536},
                            manifest=LEGACY_ADA_002_MANIFEST)
    with self.assertRaisesRegex(EmbeddingVectorValidationError, "zero"):
      load_embedding_store(store)

  def test_missing_node_reference_rejected(self):
    store = self.make_store(
      nodes={"node_1": _node(1, "missing-key")},
      manifest=LEGACY_ADA_002_MANIFEST)
    with self.assertRaisesRegex(EmbeddingReferenceError, "actual=missing"):
      load_embedding_store(store)

  def test_orphan_embedding_rejected_by_v0_policy(self):
    store = self.make_store(
      embeddings={"fixture-key": _vector(), "orphan-key": _vector(0.0, 1.0)},
      manifest=LEGACY_ADA_002_MANIFEST)
    with self.assertRaisesRegex(EmbeddingReferenceError, "orphan_embeddings"):
      load_embedding_store(store)

  def test_ada_and_embedding_3_small_same_dimension_are_rejected(self):
    modern = self.changed_manifest(
      model="text-embedding-3-small",
      embedding_space_version="openai-embedding-3-small-v1")
    with self.assertRaises(EmbeddingSpaceMismatchError):
      assert_same_embedding_space(LEGACY_ADA_002_MANIFEST, modern)

  def test_ada_and_embedding_3_large_3072_are_rejected(self):
    modern = self.changed_manifest(
      model="text-embedding-3-large", dimensions=3072,
      embedding_space_version="openai-embedding-3-large-v1")
    with self.assertRaises(EmbeddingSpaceMismatchError):
      assert_same_embedding_space(LEGACY_ADA_002_MANIFEST, modern)

  def test_runtime_model_incompatible_with_manifest_is_blocked_pre_transport(self):
    fake = FakeProvider()
    fake.embedding_space_provider = "openai"
    fake.queue_embedding_response(_vector())
    with use_provider(fake):
      with self.assertRaisesRegex(EmbeddingSpaceMismatchError, "field=model"):
        gpt_structure.get_embedding("technical fixture", model="text-embedding-3-small")
    self.assertEqual([], fake.calls)

  def test_same_manifest_is_accepted(self):
    self.assertIsNone(assert_same_embedding_space(
      LEGACY_ADA_002_MANIFEST, LEGACY_ADA_002_MANIFEST))

  def test_manifest_and_cache_versions_have_explicit_distinct_semantics(self):
    self.assertEqual(EMBEDDING_NORMALIZATION_VERSION,
                     LEGACY_ADA_002_MANIFEST.normalization_version)
    self.assertEqual("legacy-embedding-v0", EMBEDDING_VERSION)
    self.assertNotEqual(EMBEDDING_VERSION,
                        LEGACY_ADA_002_MANIFEST.embedding_space_version)

  def test_legacy_load_does_not_rewrite_archive(self):
    store = self.make_store()
    before = {path.name: path.read_bytes() for path in store.iterdir()}
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", LegacyEmbeddingSpaceWarning)
      load_embedding_store(store)
    after = {path.name: path.read_bytes() for path in store.iterdir()}
    self.assertEqual(before, after)
    self.assertNotIn(EMBEDDING_MANIFEST_FILENAME, after)

  def test_save_writes_manifest_without_changing_vectors(self):
    source = self.make_store()
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", LegacyEmbeddingSpaceWarning)
      memory = AssociativeMemory(str(source))
    original_vectors = json.loads((source / "embeddings.json").read_text())
    target = self.root / "saved" / "associative_memory"
    target.mkdir(parents=True)
    memory.save(str(target))
    self.assertEqual(original_vectors,
                     json.loads((target / "embeddings.json").read_text()))
    self.assertEqual(LEGACY_ADA_002_MANIFEST,
                     read_embedding_manifest(target / EMBEDDING_MANIFEST_FILENAME))

  def test_associative_memory_accepts_configured_modern_store(self):
    cases = (
      ("small", self.changed_manifest(
        model="text-embedding-3-small",
        embedding_space_version="openai-embedding-3-small-v1")),
      ("large", self.changed_manifest(
        model="text-embedding-3-large", dimensions=3072,
        embedding_space_version="openai-embedding-3-large-v1")),
    )
    for namespace, manifest in cases:
      with self.subTest(model=manifest.model):
        store = self.make_store(
          embeddings={"fixture-key": _vector(dimensions=manifest.dimensions)},
          manifest=manifest, namespace=namespace)
        memory = AssociativeMemory(
          str(store), runtime_embedding_manifest=manifest)
        self.assertEqual(manifest, memory.embedding_space_manifest)
        self.assertEqual(manifest, memory.runtime_embedding_manifest)
        self.assertEqual(manifest, get_runtime_embedding_manifest())

  def test_associative_memory_rejects_runtime_store_mismatch(self):
    modern = self.changed_manifest(
      model="text-embedding-3-small",
      embedding_space_version="openai-embedding-3-small-v1")
    ada_store = self.make_store(
      manifest=LEGACY_ADA_002_MANIFEST, namespace="ada")
    with self.assertRaises(EmbeddingSpaceMismatchError):
      AssociativeMemory(
        str(ada_store), runtime_embedding_manifest=modern)
    self.assertEqual(LEGACY_ADA_002_MANIFEST,
                     get_runtime_embedding_manifest())

    modern_store = self.make_store(
      manifest=modern, namespace="modern")
    with self.assertRaises(EmbeddingSpaceMismatchError):
      AssociativeMemory(str(modern_store))

  def test_runtime_manifest_is_resolved_at_call_time(self):
    modern = self.changed_manifest(
      model="text-embedding-3-small",
      embedding_space_version="openai-embedding-3-small-v1")
    store = self.make_store(manifest=modern)
    with self.assertRaises(EmbeddingSpaceMismatchError):
      load_embedding_store(store)
    with use_runtime_embedding_manifest(modern):
      self.assertEqual(
        modern, load_embedding_store(store).manifest)
    with self.assertRaises(EmbeddingSpaceMismatchError):
      load_embedding_store(store)

  def test_runtime_embedding_config_has_single_ada_default(self):
    self.assertEqual(LEGACY_ADA_002_MANIFEST,
                     get_runtime_embedding_manifest())
    self.assertIsNone(inspect.signature(load_embedding_store).parameters[
      "runtime_manifest"].default)
    self.assertIsNone(inspect.signature(gpt_structure.get_embedding).parameters[
      "model"].default)
    runtime_files = (
      BACKEND_SERVER / "persona/memory_structures/embedding_space.py",
      BACKEND_SERVER / "persona/memory_structures/associative_memory.py",
      BACKEND_SERVER / "persona/prompt_template/llm_provider.py",
      BACKEND_SERVER / "persona/prompt_template/gpt_structure.py",
    )
    self.assertEqual(1, sum(
      path.read_text(encoding="utf-8").count("text-embedding-ada-002")
      for path in runtime_files))

  def test_legacy_load_save_preserves_empty_kw_strength(self):
    source = self.make_store()
    expected = json.loads((source / "kw_strength.json").read_text())
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", LegacyEmbeddingSpaceWarning)
      memory = AssociativeMemory(str(source))
    target = self.root / "empty-strength-save" / "associative_memory"
    target.mkdir(parents=True)
    memory.save(str(target))
    self.assertEqual(expected,
                     json.loads((target / "kw_strength.json").read_text()))

  def test_legacy_load_save_preserves_nodes_embeddings_and_strengths(self):
    strengths = {
      "kw_strength_event": {"fixture": 7},
      "kw_strength_thought": {"reflection": 3},
    }
    source = self.make_store(kw_strength=strengths)
    expected = {
      name: json.loads((source / name).read_text())
      for name in ("embeddings.json", "nodes.json", "kw_strength.json")
    }
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", LegacyEmbeddingSpaceWarning)
      memory = AssociativeMemory(str(source))
    target = self.root / "complete-save" / "associative_memory"
    target.mkdir(parents=True)
    memory.save(str(target))
    for name, expected_data in expected.items():
      self.assertEqual(expected_data,
                       json.loads((target / name).read_text()), name)
    self.assertEqual(memory.embedding_space_manifest,
                     read_embedding_manifest(target / EMBEDDING_MANIFEST_FILENAME))

  def test_unknown_store_is_not_promoted_on_save(self):
    store = self.make_store(canonical=False)
    with self.assertRaises(EmbeddingManifestError):
      AssociativeMemory(str(store))
    self.assertFalse((store / EMBEDDING_MANIFEST_FILENAME).exists())

  def test_strict_mode_still_rejects_manifestless_store(self):
    store = self.make_store()
    with self.assertRaises(EmbeddingManifestError):
      AssociativeMemory(str(store), legacy_assumption_allowed=False)
    self.assertFalse((store / EMBEDDING_MANIFEST_FILENAME).exists())

  def test_runtime_mismatch_is_blocked_before_cache_and_transport(self):
    fake = FakeProvider()
    fake.embedding_space_provider = "openai"
    fake.queue_embedding_response(_vector())
    before = get_embedding_cache_stats()
    with use_provider(fake):
      with self.assertRaises(EmbeddingSpaceMismatchError):
        gpt_structure.get_embedding(
          "technical fixture", model="text-embedding-3-small")
    after = get_embedding_cache_stats()
    self.assertEqual(before, after)
    self.assertEqual((), get_telemetry())
    self.assertEqual([], fake.calls)

  def test_modern_runtime_configuration_does_not_require_cognitive_changes(self):
    modern = self.changed_manifest(
      model="text-embedding-3-small",
      embedding_space_version="openai-embedding-3-small-v1")
    store = self.make_store(manifest=modern)
    memory = AssociativeMemory(
      str(store), runtime_embedding_manifest=modern)
    fake = FakeProvider()
    fake.embedding_space_provider = "openai"
    fake.queue_embedding_response(_vector())
    with use_provider(fake):
      vector = gpt_structure.get_embedding("technical fixture")
    self.assertEqual(_vector(), vector)
    self.assertEqual(modern, memory.runtime_embedding_manifest)
    self.assertEqual("text-embedding-3-small",
                     fake.calls[0].arguments["model"])

  def test_golden_legacy_load_and_retrieval_preserves_ranking(self):
    embeddings = {
      "first-key": _vector(1.0, 0.0),
      "second-key": _vector(0.0, 1.0),
    }
    nodes = {
      "node_1": _node(1, "first-key"),
      "node_2": _node(2, "second-key"),
    }
    store = self.make_store(embeddings=embeddings, nodes=nodes)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", LegacyEmbeddingSpaceWarning)
      memory = AssociativeMemory(str(store))
    original_vectors = json.loads(json.dumps(memory.embeddings))
    original_access = memory.id_to_node["node_1"].last_accessed
    current_time = datetime.datetime(2023, 1, 2, 9, 0, 0)
    persona = SimpleNamespace(
      a_mem=memory,
      scratch=SimpleNamespace(
        curr_time=current_time, recency_decay=0.99,
        recency_w=1, relevance_w=1, importance_w=1))
    fake = FakeProvider()
    fake.queue_embedding_response(_vector(1.0, 0.0))
    with use_provider(fake):
      with redirect_stdout(io.StringIO()):
        result = retrieve.new_retrieve(persona, ["fixture focus"], 1)
    self.assertEqual("node_1", result["fixture focus"][0].node_id)
    self.assertEqual(current_time, memory.id_to_node["node_1"].last_accessed)
    self.assertNotEqual(original_access, memory.id_to_node["node_1"].last_accessed)
    self.assertEqual(original_vectors, memory.embeddings)
    self.assertEqual(1, len(fake.calls))

  def test_validation_errors_do_not_expose_embedding_source_text(self):
    secret = "SECRET MEMORY CONTENT MUST NOT LEAK"
    store = self.make_store(
      embeddings={secret: [0.0] * 1536},
      nodes={"node_1": _node(1, secret)},
      manifest=LEGACY_ADA_002_MANIFEST)
    with self.assertRaises(EmbeddingVectorValidationError) as caught:
      load_embedding_store(store)
    self.assertNotIn(secret, str(caught.exception))


if __name__ == "__main__":
  unittest.main()
