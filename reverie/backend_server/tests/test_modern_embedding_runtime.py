import json
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.memory_structures.embedding_space import (
  EMBEDDING_MANIFEST_FILENAME,
  EmbeddingManifestError,
  EmbeddingSpaceClassification,
  EmbeddingSpaceManifest,
  EmbeddingSpaceMismatchError,
  EmbeddingVectorValidationError,
  LEGACY_ADA_002_MANIFEST,
  get_runtime_embedding_manifest,
  reset_runtime_embedding_manifest,
  use_runtime_embedding_manifest,
)
from persona.prompt_template import embedding_runtime
from persona.prompt_template.cost_ledger import (
  ModelPricing,
  PricingSnapshot,
  TokenAggregate,
  build_cost_ledger_records,
  cost_ledger_summary_to_dict,
  summarize_cost_ledger,
)
from persona.prompt_template.embedding_runtime import (
  TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
  TEXT_EMBEDDING_3_SMALL_MODEL,
  EmbeddingRuntimeConfig,
  NewEmbeddingStoreRequiredError,
  build_legacy_embedding_runtime_config,
  build_modern_embedding_runtime_config,
  use_embedding_runtime,
  validate_embedding_store_for_runtime,
)
from persona.prompt_template.llm_provider import (
  CHAT,
  COMPLETION,
  EMBEDDING,
  ERROR,
  HIT,
  MISS,
  MODERN_OPENAI,
  REFLECTION,
  SUCCESS,
  FakeProvider,
  OpenAILegacyProvider,
  _invoke,
  _validate_runtime_embedding_vector,
  chat_completion,
  clear_telemetry,
  create_llm_provider,
  embedding,
  embedding_call_context,
  get_embedding_cache_stats,
  get_embedding_logical_events,
  get_provider,
  get_telemetry,
  reset_embedding_measurement_all,
  reset_provider,
  text_completion,
  use_provider,
)
from persona.prompt_template.llm_provider_config import (
  DEFAULT_LLM_PROVIDER_CONFIG,
  LEGACY_OPENAI,
  LEGACY_SDK_MODE,
  LEGACY_TRANSPORT,
  MODERN_SDK_MODE,
  MODERN_TRANSPORT,
  LLMProviderConfig,
  get_llm_provider_config,
  reset_llm_provider_config,
)
from persona.prompt_template.modern_openai_provider import (
  LLMTimeoutError,
  ModernOpenAIProvider,
  NormalizedEmbeddingResponse,
  NormalizedUsage,
)


def _vector(size=1536, value=0.25):
  return [value] * size


class _RecordingAdapter:
  def __init__(self, results, input_tokens=500, request_id="req-modern"):
    self.results = list(results)
    self.input_tokens = input_tokens
    self.request_id = request_id
    self.calls = []

  def create_embedding(self, *, model, input):
    self.calls.append({"model": model, "input": list(input)})
    if not self.results:
      raise AssertionError("No modern embedding result configured")
    result = self.results.pop(0)
    if isinstance(result, Exception):
      raise result
    return NormalizedEmbeddingResponse(
      vector=tuple(result), model=model, request_id=self.request_id,
      usage=NormalizedUsage(input_tokens=self.input_tokens))


class _FailingMetadataProvider:
  provider_identity = "metadata-spy"
  embedding_space_provider = "openai"
  provider_kind = "METADATA_SPY"
  transport_kind = "IN_MEMORY_SPY"

  def __init__(self):
    self.consume_calls = 0
    self.pending_metadata = object()

  def consume_response_metadata(self):
    self.consume_calls += 1
    value = self.pending_metadata
    self.pending_metadata = None
    return value

  def chat_completion(self, **kwargs):
    raise RuntimeError("synthetic chat failure")

  def text_completion(self, **kwargs):
    raise RuntimeError("synthetic completion failure")


class ModernEmbeddingRuntimeTests(unittest.TestCase):
  def setUp(self):
    reset_provider()
    reset_llm_provider_config()
    reset_runtime_embedding_manifest()
    reset_embedding_measurement_all()
    clear_telemetry()

  def tearDown(self):
    reset_provider()
    reset_llm_provider_config()
    reset_runtime_embedding_manifest()
    reset_embedding_measurement_all()
    clear_telemetry()

  def modern_call(self, vector=None, text="modern memory", **adapter_kwargs):
    adapter = _RecordingAdapter(
      [vector if vector is not None else _vector()], **adapter_kwargs)
    config = build_modern_embedding_runtime_config()
    with use_embedding_runtime(config, adapter):
      result = self.get_embedding(text)
    return result, adapter

  def get_embedding(self, text, model=None):
    selected_model = model or get_runtime_embedding_manifest().model
    return embedding(
      input=[text], model=selected_model)["data"][0]["embedding"]

  def isolated_modules(self, discover=False):
    legacy_wrapper = "gpt_" + "structure"
    if discover:
      action = (
        "import unittest\n"
        f"unittest.TestLoader().discover({str(BACKEND_SERVER / 'tests')!r}, "
        "pattern='test_modern_embedding_runtime.py')\n")
    else:
      action = (
        "import importlib\n"
        "importlib.import_module('persona.prompt_template.embedding_runtime')\n"
        "importlib.import_module('persona.prompt_template.llm_provider')\n")
    script = (
      "import json, socket, sys\n"
      f"sys.path.insert(0, {str(BACKEND_SERVER)!r})\n"
      "def blocked(*args, **kwargs): raise AssertionError('network reached')\n"
      "socket.getaddrinfo = blocked\n"
      "socket.create_connection = blocked\n"
      + action
      + "print(json.dumps(sorted(sys.modules)))\n")
    completed = subprocess.run(
      [sys.executable, "-I", "-c", script], check=True,
      capture_output=True, text=True, timeout=30)
    modules = json.loads(completed.stdout)
    return modules, legacy_wrapper

  def write_store(self, root, manifest=None, legacy_layout=False):
    path = Path(root)
    if legacy_layout:
      path = path / "bootstrap_memory" / "associative_memory"
    path.mkdir(parents=True, exist_ok=True)
    (path / "embeddings.json").write_text("{}", encoding="utf-8")
    (path / "nodes.json").write_text("{}", encoding="utf-8")
    (path / "kw_strength.json").write_text(json.dumps({
      "kw_strength_event": {}, "kw_strength_thought": {}}), encoding="utf-8")
    if manifest is not None:
      (path / EMBEDDING_MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8")
    return path

  def test_01_modern_manifest_is_canonical(self):
    self.assertEqual({
      "schema_version": 1, "provider": "openai",
      "model": "text-embedding-3-small", "dimensions": 1536,
      "embedding_space_version":
        "openai-text-embedding-3-small-1536-v1",
      "normalization_version": "newline-and-blank-v0",
    }, TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.to_dict())

  def test_02_manifest_serialization_is_deterministic(self):
    first = json.dumps(
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.to_dict(), sort_keys=True)
    second = json.dumps(
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.to_dict(), sort_keys=True)
    self.assertEqual(first, second)

  def test_03_manifest_round_trip(self):
    value = TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.to_dict()
    self.assertEqual(
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
      EmbeddingSpaceManifest.from_dict(value))

  def test_04_modern_manifest_is_distinct_from_ada(self):
    self.assertNotEqual(
      LEGACY_ADA_002_MANIFEST, TEXT_EMBEDDING_3_SMALL_1536_MANIFEST)
    self.assertEqual(
      LEGACY_ADA_002_MANIFEST.dimensions,
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.dimensions)

  def test_05_same_dimensions_do_not_make_spaces_compatible(self):
    with self.assertRaises(EmbeddingSpaceMismatchError):
      EmbeddingRuntimeConfig(
        LLMProviderConfig(), TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
        TEXT_EMBEDDING_3_SMALL_MODEL)

  def test_06_manifest_model_mismatch_is_rejected(self):
    incompatible = replace(
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST, model="other")
    with self.assertRaises(EmbeddingSpaceMismatchError):
      build_modern_embedding_runtime_config(
        embedding_manifest=incompatible, embedding_model="other")

  def test_07_manifest_dimension_mismatch_is_rejected(self):
    incompatible = replace(
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST, dimensions=1535)
    with self.assertRaises(EmbeddingSpaceMismatchError):
      build_modern_embedding_runtime_config(
        embedding_manifest=incompatible)

  def test_08_manifest_normalization_mismatch_is_rejected(self):
    incompatible = replace(
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
      normalization_version="other")
    with self.assertRaises(EmbeddingSpaceMismatchError):
      build_modern_embedding_runtime_config(
        embedding_manifest=incompatible)

  def test_09_modern_runtime_config_is_canonical(self):
    config = build_modern_embedding_runtime_config()
    self.assertEqual((MODERN_OPENAI, MODERN_TRANSPORT, MODERN_SDK_MODE), (
      config.provider_config.provider_kind,
      config.provider_config.transport_kind,
      config.provider_config.sdk_mode))
    self.assertIs(
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST, config.embedding_manifest)

  def test_10_modern_provider_with_ada_manifest_is_rejected(self):
    with self.assertRaises(ValueError):
      build_modern_embedding_runtime_config(
        embedding_manifest=LEGACY_ADA_002_MANIFEST,
        embedding_model=LEGACY_ADA_002_MANIFEST.model)

  def test_11_legacy_provider_with_modern_manifest_is_rejected(self):
    with self.assertRaises(EmbeddingSpaceMismatchError):
      EmbeddingRuntimeConfig(
        LLMProviderConfig(), TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
        TEXT_EMBEDDING_3_SMALL_MODEL)

  def test_12_transport_mismatch_is_rejected(self):
    with self.assertRaises(ValueError):
      LLMProviderConfig(
        provider_kind=MODERN_OPENAI, transport_kind=LEGACY_TRANSPORT,
        sdk_mode=MODERN_SDK_MODE)

  def test_13_runtime_model_mismatch_is_rejected(self):
    with self.assertRaises(ValueError):
      build_modern_embedding_runtime_config(embedding_model="other")

  def test_14_provider_factory_selects_modern(self):
    adapter = _RecordingAdapter([_vector()])
    provider = create_llm_provider(
      build_modern_embedding_runtime_config().provider_config, adapter)
    self.assertIsInstance(provider, ModernOpenAIProvider)

  def test_15_modern_factory_never_constructs_legacy_provider(self):
    adapter = _RecordingAdapter([_vector()])
    with patch(
        "persona.prompt_template.llm_provider.OpenAILegacyProvider",
        side_effect=AssertionError("legacy provider constructed")) as legacy:
      provider = create_llm_provider(
        build_modern_embedding_runtime_config().provider_config, adapter)
    self.assertIsInstance(provider, ModernOpenAIProvider)
    legacy.assert_not_called()

  def test_16_model_is_forwarded_unchanged(self):
    result, adapter = self.modern_call()
    self.assertEqual(1536, len(result))
    self.assertEqual([{
      "model": TEXT_EMBEDDING_3_SMALL_MODEL,
      "input": ["modern memory"],
    }], adapter.calls)

  def test_17_newline_normalization_is_unchanged(self):
    _, adapter = self.modern_call(text="hello\nworld")
    self.assertEqual(["hello world"], adapter.calls[0]["input"])

  def test_18_blank_normalization_is_unchanged(self):
    _, adapter = self.modern_call(text="")
    self.assertEqual(["this is blank"], adapter.calls[0]["input"])

  def test_19_space_is_not_stripped(self):
    _, adapter = self.modern_call(text=" ")
    self.assertEqual([" "], adapter.calls[0]["input"])

  def test_20_valid_1536_vector_is_accepted(self):
    result, _ = self.modern_call(_vector())
    self.assertEqual(_vector(), result)

  def assert_invalid_vector(self, vector):
    adapter = _RecordingAdapter([vector])
    with self.assertRaises(EmbeddingVectorValidationError):
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(), adapter):
        self.get_embedding("invalid vector")

  def test_21_vector_1535_is_rejected(self):
    self.assert_invalid_vector(_vector(1535))

  def test_22_vector_1537_is_rejected(self):
    self.assert_invalid_vector(_vector(1537))

  def test_23_empty_vector_is_rejected(self):
    self.assert_invalid_vector([])

  def test_24_non_list_vector_is_rejected(self):
    with self.assertRaises(EmbeddingVectorValidationError):
      _validate_runtime_embedding_vector(
        "not-a-list", TEXT_EMBEDDING_3_SMALL_1536_MANIFEST)

  def test_25_string_element_is_rejected(self):
    vector = _vector()
    vector[3] = "bad"
    self.assert_invalid_vector(vector)

  def test_26_bool_element_is_rejected(self):
    vector = _vector()
    vector[3] = True
    self.assert_invalid_vector(vector)

  def test_27_nan_is_rejected(self):
    vector = _vector()
    vector[3] = float("nan")
    self.assert_invalid_vector(vector)

  def test_28_infinity_is_rejected(self):
    vector = _vector()
    vector[3] = float("inf")
    self.assert_invalid_vector(vector)

  def test_29_zero_norm_is_rejected(self):
    self.assert_invalid_vector(_vector(value=0.0))

  def test_30_invalid_vector_is_not_written_to_cache(self):
    adapter = _RecordingAdapter([_vector(1535), _vector()])
    config = build_modern_embedding_runtime_config()
    with use_embedding_runtime(config, adapter):
      with self.assertRaises(EmbeddingVectorValidationError):
        self.get_embedding("same")
      self.assertEqual(_vector(), self.get_embedding("same"))
    stats = get_embedding_cache_stats()
    self.assertEqual((2, 2, 0, 2, 1), (
      stats.logical_embedding_requests, stats.physical_embedding_attempts,
      stats.cache_hits, stats.cache_misses, stats.cache_entries))
    self.assertEqual([ERROR, SUCCESS], [
      event.outcome for event in get_telemetry()])
    self.assertEqual("EmbeddingVectorValidationError",
                     get_telemetry()[0].error_type)

  def test_31_cache_is_isolated_between_ada_and_modern_space(self):
    legacy = FakeProvider("shared-openai", embedding_space_provider="openai")
    modern = FakeProvider("shared-openai", embedding_space_provider="openai")
    legacy.queue_embedding_response([1.0])
    modern.queue_embedding_response(_vector())
    with use_provider(legacy), use_runtime_embedding_manifest(
        LEGACY_ADA_002_MANIFEST):
      self.get_embedding("same text")
      self.get_embedding("same text")
    with use_provider(modern), use_runtime_embedding_manifest(
        TEXT_EMBEDDING_3_SMALL_1536_MANIFEST):
      self.get_embedding("same text")
      self.get_embedding("same text")
    stats = get_embedding_cache_stats()
    self.assertEqual((4, 2, 2, 2, 2), (
      stats.logical_embedding_requests, stats.physical_embedding_attempts,
      stats.cache_hits, stats.cache_misses, stats.cache_entries))
    self.assertEqual([MISS, HIT, MISS, HIT], [
      event.cache_outcome for event in get_embedding_logical_events()])

  def test_32_manifest_request_guard_precedes_cache_and_attempt(self):
    adapter = _RecordingAdapter([_vector()])
    provider = ModernOpenAIProvider(
      build_modern_embedding_runtime_config().provider_config, adapter)
    before = get_embedding_cache_stats()
    with use_provider(provider), use_runtime_embedding_manifest(
        TEXT_EMBEDDING_3_SMALL_1536_MANIFEST):
      with self.assertRaises(EmbeddingSpaceMismatchError):
        embedding(input=["guarded"], model=LEGACY_ADA_002_MANIFEST.model)
    self.assertEqual(before, get_embedding_cache_stats())
    self.assertEqual([], adapter.calls)
    self.assertEqual((), get_telemetry())

  def test_33_store_guard_precedes_provider_factory(self):
    with tempfile.TemporaryDirectory() as root:
      store = self.write_store(root, LEGACY_ADA_002_MANIFEST)
      with patch(
          "persona.prompt_template.llm_provider.create_llm_provider",
          side_effect=AssertionError("provider factory reached")) as factory:
        with patch(
            "persona.prompt_template.embedding_runtime.load_embedding_store",
            side_effect=AssertionError("store vectors opened")) as loader:
          with self.assertRaises(EmbeddingSpaceMismatchError):
            with use_embedding_runtime(
                build_modern_embedding_runtime_config(), store_path=store):
              pass
      factory.assert_not_called()
      loader.assert_not_called()
      self.assertEqual(0, get_embedding_cache_stats().logical_embedding_requests)

  def test_34_declared_ada_store_is_rejected(self):
    with tempfile.TemporaryDirectory() as root:
      store = self.write_store(root, LEGACY_ADA_002_MANIFEST)
      with self.assertRaises(EmbeddingManifestError):
        validate_embedding_store_for_runtime(
          store, build_modern_embedding_runtime_config())

  def test_35_manifestless_legacy_store_is_rejected(self):
    with tempfile.TemporaryDirectory() as root:
      store = self.write_store(root, legacy_layout=True)
      with self.assertRaises(EmbeddingManifestError):
        validate_embedding_store_for_runtime(
          store, build_modern_embedding_runtime_config())

  def test_36_declared_modern_store_is_accepted(self):
    with tempfile.TemporaryDirectory() as root:
      store = self.write_store(
        root, TEXT_EMBEDDING_3_SMALL_1536_MANIFEST)
      loaded = validate_embedding_store_for_runtime(
        store, build_modern_embedding_runtime_config())
      self.assertEqual(
        EmbeddingSpaceClassification.DECLARED, loaded.classification)
      self.assertEqual(TEXT_EMBEDDING_3_SMALL_1536_MANIFEST, loaded.manifest)

  def test_37_unknown_store_is_rejected(self):
    with tempfile.TemporaryDirectory() as root:
      store = self.write_store(root)
      with self.assertRaises(EmbeddingManifestError):
        validate_embedding_store_for_runtime(
          store, build_modern_embedding_runtime_config())

  def test_38_empty_store_requires_explicit_bootstrap(self):
    with tempfile.TemporaryDirectory() as root:
      before = tuple(Path(root).iterdir())
      with self.assertRaises(NewEmbeddingStoreRequiredError):
        validate_embedding_store_for_runtime(
          root, build_modern_embedding_runtime_config())
      self.assertEqual(before, tuple(Path(root).iterdir()))

  def test_39_modern_success_telemetry_is_complete(self):
    adapter = _RecordingAdapter([_vector()], request_id="req-technical")
    with use_embedding_runtime(
        build_modern_embedding_runtime_config(), adapter), \
        embedding_call_context(REFLECTION):
      self.get_embedding("telemetry memory")
    event = get_telemetry()[0]
    logical = get_embedding_logical_events()[0]
    self.assertEqual((
      EMBEDDING, MODERN_OPENAI, MODERN_TRANSPORT,
      TEXT_EMBEDDING_3_SMALL_MODEL, SUCCESS, 1, "req-technical", 500), (
        event.operation, event.provider_kind, event.transport_kind,
        event.model_or_engine, event.outcome, event.physical_attempt,
        event.request_id, event.input_tokens))
    self.assertTrue(event.logical_call_id)
    self.assertGreaterEqual(event.elapsed_seconds, 0)
    self.assertEqual(REFLECTION, logical.category)

  def test_40_modern_failure_is_typed_and_never_falls_back(self):
    adapter = _RecordingAdapter([LLMTimeoutError("synthetic")])
    config = build_modern_embedding_runtime_config()
    with patch.object(
        OpenAILegacyProvider, "embedding",
        side_effect=AssertionError("legacy fallback")) as legacy:
      with self.assertRaises(LLMTimeoutError):
        with use_embedding_runtime(config, adapter):
          self.get_embedding("failure")
    legacy.assert_not_called()
    event = get_telemetry()[0]
    self.assertEqual((ERROR, "LLMTimeoutError", None), (
      event.outcome, event.error_type, event.input_tokens))

  def test_41_cost_ledger_consumes_modern_embedding_telemetry(self):
    with use_embedding_runtime(
        build_modern_embedding_runtime_config(),
        _RecordingAdapter([_vector()], input_tokens=500)), \
        embedding_call_context(REFLECTION):
      self.get_embedding("ledger memory")
    pricing = PricingSnapshot(
      "m3-synthetic", 1, "USD", "2026-08-05",
      (ModelPricing(
        TEXT_EMBEDDING_3_SMALL_MODEL,
        embedding_input_per_million=Decimal("0.5")),), "synthetic")
    records = build_cost_ledger_records(
      get_telemetry(), pricing,
      embedding_logical_events=get_embedding_logical_events())
    summary = summarize_cost_ledger(records)
    record = records[0]
    self.assertEqual((
      "COMPLETE", TEXT_EMBEDDING_3_SMALL_MODEL, MODERN_OPENAI,
      "m3-synthetic", Decimal("0.000250000000")), (
        record.token_usage_status, record.model, record.provider_kind,
        record.pricing_snapshot_id, record.estimated_total_cost_usd))
    self.assertEqual(TokenAggregate(500, 0), summary.totals.input_tokens)
    self.assertEqual([TEXT_EMBEDDING_3_SMALL_MODEL],
                     list(dict(summary.by_model)))
    self.assertEqual([EMBEDDING], list(dict(summary.by_operation)))

  def test_42_modern_runtime_reports_are_privacy_safe(self):
    source = {
      "input": "SECRET_MODERN_MEMORY_SENTINEL",
      "raw_embedding_vector": "SECRET_VECTOR_SENTINEL",
      "authorization": "SECRET_AUTHORIZATION_SENTINEL",
      "api_key": "SECRET_API_KEY_SENTINEL",
      "prompt": "SECRET_PROMPT_SENTINEL",
      "transcript": "SECRET_TRANSCRIPT_SENTINEL",
    }
    adapter = _RecordingAdapter([_vector()], request_id="req-technical")
    adapter.synthetic_source_metadata = source
    with use_embedding_runtime(
        build_modern_embedding_runtime_config(),
        adapter):
      self.get_embedding(source["input"])
    pricing = PricingSnapshot(
      "privacy", 1, "USD", "2026-08-05",
      (ModelPricing(
        TEXT_EMBEDDING_3_SMALL_MODEL,
        embedding_input_per_million=Decimal("1")),), "synthetic")
    report = json.dumps(cost_ledger_summary_to_dict(summarize_cost_ledger(
      build_cost_ledger_records(get_telemetry(), pricing))), sort_keys=True)
    telemetry = json.dumps([asdict(item) for item in get_telemetry()])
    for sentinel in source.values():
      self.assertNotIn(sentinel, report + telemetry)

  def test_43_runtime_context_nests_and_restores(self):
    outer = build_legacy_embedding_runtime_config()
    inner = build_modern_embedding_runtime_config()
    default_provider = get_provider()
    with use_embedding_runtime(outer):
      self.assertIsInstance(get_provider(), OpenAILegacyProvider)
      self.assertEqual(LEGACY_ADA_002_MANIFEST,
                       get_runtime_embedding_manifest())
      with use_embedding_runtime(inner, _RecordingAdapter([_vector()])):
        self.assertIsInstance(get_provider(), ModernOpenAIProvider)
        self.assertEqual(TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
                         get_runtime_embedding_manifest())
      self.assertIsInstance(get_provider(), OpenAILegacyProvider)
      self.assertEqual(LEGACY_ADA_002_MANIFEST,
                       get_runtime_embedding_manifest())
    self.assertIs(default_provider, get_provider())
    self.assertIs(DEFAULT_LLM_PROVIDER_CONFIG, get_llm_provider_config())

  def test_44_default_runtime_remains_legacy(self):
    self.assertIsInstance(get_provider(), OpenAILegacyProvider)
    self.assertIs(DEFAULT_LLM_PROVIDER_CONFIG, get_llm_provider_config())
    self.assertEqual(LEGACY_OPENAI, get_llm_provider_config().provider_kind)
    self.assertEqual(LEGACY_TRANSPORT, get_llm_provider_config().transport_kind)
    self.assertEqual(LEGACY_SDK_MODE, get_llm_provider_config().sdk_mode)
    self.assertEqual(LEGACY_ADA_002_MANIFEST,
                     get_runtime_embedding_manifest())

  def test_45_injected_adapter_prevents_real_sdk_client_creation(self):
    adapter = _RecordingAdapter([_vector()])
    with patch("socket.getaddrinfo",
               side_effect=AssertionError("DNS reached")) as dns, patch(
        "socket.create_connection",
        side_effect=AssertionError("network reached")) as connection, patch(
        "persona.prompt_template.modern_openai_provider._create_modern_sdk_client",
        side_effect=AssertionError("real SDK client created")) as factory:
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(), adapter):
        self.get_embedding("offline")
    factory.assert_not_called()
    dns.assert_not_called()
    connection.assert_not_called()
    self.assertEqual(1, len(adapter.calls))

  def test_46_runtime_module_has_no_persistence_operation(self):
    self.assertFalse(hasattr(embedding_runtime, "write_embedding_manifest"))
    self.assertFalse(hasattr(embedding_runtime, "open"))

  def test_47_clean_runtime_import_does_not_load_legacy_wrapper(self):
    modules, legacy_wrapper = self.isolated_modules()
    self.assertFalse(any(
      name == legacy_wrapper or name.endswith("." + legacy_wrapper)
      for name in modules))

  def test_48_clean_runtime_import_does_not_load_utils(self):
    modules, unused = self.isolated_modules()
    credential_module = "reverie.backend_server." + "utils"
    self.assertFalse(any(
      name in ("utils", credential_module) for name in modules))

  def test_49_isolated_discovery_does_not_load_legacy_modules(self):
    modules, legacy_wrapper = self.isolated_modules(discover=True)
    credential_module = "reverie.backend_server." + "utils"
    self.assertFalse(any(
      name == legacy_wrapper or name.endswith("." + legacy_wrapper)
      or name in ("utils", credential_module)
      for name in modules))

  def test_50_chat_error_does_not_consume_response_metadata(self):
    provider = _FailingMetadataProvider()
    pending = provider.pending_metadata
    with use_provider(provider), self.assertRaisesRegex(
        RuntimeError, "synthetic chat failure"):
      chat_completion(model="chat", messages=[])
    self.assertEqual(0, provider.consume_calls)
    self.assertIs(pending, provider.pending_metadata)
    self.assertEqual((CHAT, ERROR, 1), (
      get_telemetry()[0].operation, get_telemetry()[0].outcome,
      get_telemetry()[0].physical_attempt))

  def test_51_completion_error_does_not_consume_response_metadata(self):
    provider = _FailingMetadataProvider()
    pending = provider.pending_metadata
    with use_provider(provider), self.assertRaisesRegex(
        RuntimeError, "synthetic completion failure"):
      text_completion(
        model="completion", prompt="synthetic", temperature=0,
        max_tokens=1, top_p=1, frequency_penalty=0,
        presence_penalty=0, stream=False, stop=None)
    self.assertEqual(0, provider.consume_calls)
    self.assertIs(pending, provider.pending_metadata)
    self.assertEqual((COMPLETION, ERROR, 1), (
      get_telemetry()[0].operation, get_telemetry()[0].outcome,
      get_telemetry()[0].physical_attempt))

  def test_52_embedding_validation_error_cleans_metadata_once(self):
    adapter = _RecordingAdapter([_vector(1535)])
    config = build_modern_embedding_runtime_config()
    provider = ModernOpenAIProvider(config.provider_config, adapter)
    with patch.object(
        provider, "consume_response_metadata",
        wraps=provider.consume_response_metadata) as consume, use_provider(
          provider), use_runtime_embedding_manifest(
            TEXT_EMBEDDING_3_SMALL_1536_MANIFEST), self.assertRaises(
              EmbeddingVectorValidationError):
      self.get_embedding("invalid metadata")
    consume.assert_called_once_with()
    self.assertEqual(ERROR, get_telemetry()[0].outcome)

  def test_53_recursive_invoke_propagates_accepting_validator_once(self):
    provider = FakeProvider("recursive-success")
    provider.queue_embedding_response([1.0])
    validator_calls = []
    with use_provider(provider):
      result = _invoke(
        EMBEDDING, "embedding", {"input": ["x"], "model": "m"},
        result_validator=lambda value: validator_calls.append(value))
    self.assertEqual(1, len(validator_calls))
    self.assertEqual({"data": [{"embedding": [1.0]}]}, result)
    self.assertEqual(1, len(get_telemetry()))
    self.assertEqual((1, SUCCESS), (
      get_telemetry()[0].physical_attempt, get_telemetry()[0].outcome))
    self.assertEqual(1, get_embedding_cache_stats().physical_embedding_attempts)

  def test_54_recursive_invoke_propagates_rejecting_validator_once(self):
    provider = FakeProvider("recursive-failure")
    provider.queue_embedding_response([1.0])
    validator_calls = []
    def reject(value):
      validator_calls.append(value)
      raise EmbeddingVectorValidationError("synthetic rejection")
    with use_provider(provider), self.assertRaisesRegex(
        EmbeddingVectorValidationError, "synthetic rejection"):
      _invoke(
        EMBEDDING, "embedding", {"input": ["x"], "model": "m"},
        result_validator=reject)
    self.assertEqual(1, len(validator_calls))
    self.assertEqual(1, len(get_telemetry()))
    self.assertEqual((1, ERROR, "EmbeddingVectorValidationError"), (
      get_telemetry()[0].physical_attempt, get_telemetry()[0].outcome,
      get_telemetry()[0].error_type))
    self.assertEqual(1, get_embedding_cache_stats().physical_embedding_attempts)


if __name__ == "__main__":
  unittest.main()
