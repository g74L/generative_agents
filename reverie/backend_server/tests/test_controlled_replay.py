from dataclasses import replace
import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
REPOSITORY = BACKEND_SERVER.parents[1]
VERSIONED_BASELINE = (
  REPOSITORY / "environment" / "frontend_server" / "storage"
  / "base_the_ville_isabella_maria_klaus")
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

import controlled_replay as controlled_replay_module
from controlled_replay import (
  EMPTY_BOOTSTRAPPABLE,
  INCONSISTENT_BLOCKED,
  LEGACY_NONEMPTY_BLOCKED,
  MODERN_COMPATIBLE,
  UNKNOWN_BLOCKED,
  ControlledReplayActorFixture,
  ControlledReplayConfigurationError,
  ControlledReplayEnvironmentFixture,
  IsolatedFixturePreflightError,
  IsolatedEmbeddingPreflightError,
  ControlledReplayLegacyConfigurationError,
  ControlledReplayPathForbiddenError,
  ControlledReplayProfile,
  ControlledReplayProviders,
  DeterministicReplayFakeAdapter,
  R1TDeterministicFakeAdapter,
  R1T_ACT_OBJ_DESC_FAKE_RESPONSE,
  R1T_GENERIC_FAKE_RESPONSE,
  R1T_PRONUNCIATIO_FAKE_RESPONSE,
  R1T_TASK_DECOMP_FAKE_RESPONSE,
  _historical_wrapper_environment,
  ReplayCallCountGuardState,
  ReplayLogicalCallLimitExceededError,
  ReplayPhysicalAttemptLimitExceededError,
  assert_controlled_replay_path_allowed,
  build_controlled_replay_actor_fixture,
  controlled_replay_contexts_are_reset,
  build_sanitized_failure_evidence,
  inspect_isolated_embedding_store,
  prepare_isolated_embedding_stores,
  prepare_isolated_reverie_fixture,
  run_controlled_replay_step,
)
from persona.memory_structures.embedding_space import (
  EMBEDDING_MANIFEST_FILENAME,
  LEGACY_ADA_002_MANIFEST,
  get_runtime_embedding_manifest,
  load_embedding_store,
  read_embedding_manifest,
)
from persona.memory_structures.scratch import Scratch
from persona.prompt_template import gpt_structure
from persona.prompt_template import run_gpt_prompt
from persona.prompt_template.chat_runtime import (
  ModernChatRuntimeConfig,
  build_modern_chat_runtime_config,
  get_modern_chat_runtime_config,
  use_modern_chat_runtime,
)
from persona.prompt_template.completion_runtime import (
  CompletionCompatCallerNotAllowedError,
  LegacyModelInvocationDetectedError,
  ModernCompletionRuntimeConfig,
  assert_no_legacy_model_invocation,
  build_modern_completion_runtime_config,
  get_modern_completion_runtime_config,
  use_modern_completion_runtime,
)
from persona.prompt_template.embedding_runtime import (
  TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
  build_legacy_embedding_runtime_config,
  build_modern_embedding_runtime_config,
  use_embedding_runtime,
)
from persona.prompt_template.cost_ledger import PricingSnapshot
from persona.prompt_template.embedding_store_bootstrap import (
  EmbeddingStoreAlreadyExistsError,
  EmbeddingStoreIncompatibleError,
  EmbeddingStoreUnknownError,
  EmbeddingStoreUnsafePathError,
)
from persona.prompt_template.llm_provider import (
  COMPLETION_COMPAT,
  LLMReplayContext,
  clear_telemetry,
  get_llm_replay_context,
  get_telemetry,
  reset_embedding_measurement_all,
  reset_provider,
  use_llm_replay_context,
)
from persona.prompt_template.llm_provider_config import (
  LLMProviderConfig,
  reset_llm_provider_config,
)
from persona.prompt_template.modern_openai_provider import (
  LLMAuthenticationError,
  LLMConnectionError,
  LLMInvalidRequestError,
  LLMIncompleteResponseError,
  LLMModelNotFoundError,
  LLMTimeoutError,
  LLMUnsupportedParameterError,
  ModernOpenAIClientAdapter,
  NormalizedUsage,
)
from persona.prompt_template.replay_cost_guard import (
  ReplayCostAccountingUnavailableError,
  ReplayCostCeilingExceededError,
  get_replay_cost_guard,
)


def _raise_sanitized_index_error():
  values = []
  prompt = "PRIVATE-PROMPT-MARKER"
  response = "PRIVATE-RESPONSE-MARKER"
  if prompt and response:
    return values[2]


class IsolatedFixturePreflightTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.baseline = self.root / "baseline"
    self.baseline.mkdir()
    (self.baseline / "environment").mkdir()
    (self.baseline / "environment" / "0.json").write_text(
      "{}", encoding="utf-8")
    (self.baseline / "personas").mkdir()
    (self.baseline / "reverie").mkdir()
    self.isolated_parent = self.root / "isolated"
    self.isolated_parent.mkdir()

  def tearDown(self):
    self.temporary.cleanup()

  def copy_fixture(self, name="simulation"):
    target = self.isolated_parent / name
    shutil.copytree(self.baseline, target)
    return target

  def prepare(self, target, **changes):
    values = {
      "simulation_root": target,
      "temporary_root": self.isolated_parent,
      "baseline_root": self.baseline,
      "source_step": 0,
    }
    values.update(changes)
    return prepare_isolated_reverie_fixture(**values)

  def test_01_creates_contained_movement_without_mutating_baseline(self):
    baseline_entries = sorted(
      path.relative_to(self.baseline).as_posix()
      for path in self.baseline.rglob("*"))
    target = self.copy_fixture()

    fixture = self.prepare(target)

    self.assertTrue(fixture.movement_dir.is_dir())
    self.assertIn(fixture.simulation_root, fixture.movement_dir.parents)
    self.assertFalse((self.baseline / "movement").exists())
    self.assertEqual(baseline_entries, sorted(
      path.relative_to(self.baseline).as_posix()
      for path in self.baseline.rglob("*")))

  def test_02_preflight_is_idempotent(self):
    target = self.copy_fixture()
    first = self.prepare(target)
    marker = first.movement_dir / "existing.json"
    marker.write_text("{}", encoding="utf-8")

    second = self.prepare(target)

    self.assertEqual(first, second)
    self.assertTrue(marker.is_file())

  def test_03_missing_required_directories_fail_before_movement_creation(self):
    cases = (
      ("environment", "ENVIRONMENT_DIR_MISSING"),
      ("personas", "PERSONAS_DIR_MISSING"),
      ("reverie", "REVERIE_DIR_MISSING"),
    )
    for index, (directory, reason_code) in enumerate(cases):
      with self.subTest(directory=directory):
        target = self.copy_fixture(f"missing-{index}")
        shutil.rmtree(target / directory)
        with self.assertRaises(IsolatedFixturePreflightError) as raised:
          self.prepare(target)
        self.assertEqual(reason_code, raised.exception.reason_code)
        self.assertFalse((target / "movement").exists())

  def test_04_missing_source_environment_step_fails_before_mutation(self):
    target = self.copy_fixture()
    (target / "environment" / "0.json").unlink()

    with self.assertRaises(IsolatedFixturePreflightError) as raised:
      self.prepare(target)

    self.assertEqual("SOURCE_ENVIRONMENT_STEP_MISSING",
                     raised.exception.reason_code)
    self.assertFalse((target / "movement").exists())

  def test_05_external_movement_target_is_rejected(self):
    target = self.copy_fixture()
    external = self.root / "external-movement"

    with self.assertRaises(IsolatedFixturePreflightError) as raised:
      self.prepare(target, movement_dir=external)

    self.assertEqual("FIXTURE_PATH_OUTSIDE_SIMULATION",
                     raised.exception.reason_code)
    self.assertFalse(external.exists())

  def test_06_protected_repository_storage_is_rejected_as_output(self):
    protected_storage = (
      REPOSITORY / "environment" / "frontend_server" / "storage")
    existing_simulations = [path for path in protected_storage.iterdir()
                            if path.is_dir()]
    self.assertTrue(existing_simulations)

    with self.assertRaises(IsolatedFixturePreflightError) as raised:
      prepare_isolated_reverie_fixture(
        simulation_root=existing_simulations[0],
        temporary_root=protected_storage,
        baseline_root=REPOSITORY,
        source_step=0,
      )

    self.assertEqual("PROTECTED_STORAGE_TARGET", raised.exception.reason_code)

  def test_07_path_traversal_outside_temporary_root_is_rejected(self):
    outside = self.root / "outside"
    shutil.copytree(self.baseline, outside)
    traversing = self.isolated_parent / ".." / "outside"

    with self.assertRaises(IsolatedFixturePreflightError) as raised:
      self.prepare(traversing)

    self.assertEqual("SIMULATION_OUTSIDE_TEMPORARY_ROOT",
                     raised.exception.reason_code)
    self.assertFalse((outside / "movement").exists())

  def test_08_structural_failure_precedes_provider_and_persona_move(self):
    target = self.copy_fixture()
    shutil.rmtree(target / "personas")

    with patch("persona.persona.Persona.move") as move, patch.object(
        ModernOpenAIClientAdapter, "create_chat") as provider:
      with self.assertRaises(IsolatedFixturePreflightError):
        self.prepare(target)

    move.assert_not_called()
    provider.assert_not_called()
    self.assertFalse((target / "movement").exists())

  def test_09_valid_fixture_passes_all_structural_checks(self):
    target = self.copy_fixture()

    fixture = self.prepare(target)

    self.assertEqual((target / "environment" / "0.json").resolve(),
                     fixture.environment_step_path)
    self.assertEqual((target / "personas").resolve(), fixture.personas_dir)
    self.assertEqual((target / "reverie").resolve(), fixture.reverie_dir)
    self.assertTrue(os.access(fixture.movement_dir, os.W_OK))

  def test_10_real_baseline_copy_is_prepared_byte_for_byte_safely(self):
    real_baseline = VERSIONED_BASELINE
    before = {
      path.relative_to(real_baseline).as_posix(): hashlib.sha256(
        path.read_bytes()).hexdigest()
      for path in real_baseline.rglob("*") if path.is_file()
    }
    self.assertFalse((real_baseline / "movement").exists())
    target = self.isolated_parent / "real-baseline-copy"
    shutil.copytree(real_baseline, target)

    fixture = prepare_isolated_reverie_fixture(
      simulation_root=target,
      temporary_root=self.isolated_parent,
      baseline_root=real_baseline,
      source_step=0,
    )

    after = {
      path.relative_to(real_baseline).as_posix(): hashlib.sha256(
        path.read_bytes()).hexdigest()
      for path in real_baseline.rglob("*") if path.is_file()
    }
    self.assertTrue(fixture.movement_dir.is_dir())
    self.assertFalse((real_baseline / "movement").exists())
    self.assertEqual(before, after)


class IsolatedEmbeddingPreflightTests(unittest.TestCase):
  PERSONAS = ("Isabella Rodriguez", "Maria Lopez", "Klaus Mueller")

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.baseline = self.root / "baseline"
    self.baseline.mkdir()
    (self.baseline / "environment").mkdir()
    (self.baseline / "environment" / "0.json").write_text(
      "{}", encoding="utf-8")
    (self.baseline / "personas").mkdir()
    (self.baseline / "reverie").mkdir()
    for persona_name in self.PERSONAS:
      store = (self.baseline / "personas" / persona_name
               / "bootstrap_memory" / "associative_memory")
      store.mkdir(parents=True)
      self.write_store(store)
    self.isolated = self.root / "isolated"
    shutil.copytree(self.baseline, self.isolated)
    self.fixture = prepare_isolated_reverie_fixture(
      simulation_root=self.isolated,
      temporary_root=self.root,
      baseline_root=self.baseline,
      source_step=0,
    )

  def tearDown(self):
    self.temporary.cleanup()

  def write_store(self, store, embeddings=None, nodes=None, manifest=None,
                  keyword_strength=None):
    embeddings = {} if embeddings is None else embeddings
    nodes = {} if nodes is None else nodes
    keyword_strength = keyword_strength or {
      "kw_strength_event": {}, "kw_strength_thought": {}}
    (store / "embeddings.json").write_text(
      json.dumps(embeddings), encoding="utf-8")
    (store / "nodes.json").write_text(
      json.dumps(nodes), encoding="utf-8")
    (store / "kw_strength.json").write_text(
      json.dumps(keyword_strength), encoding="utf-8")
    manifest_path = store / EMBEDDING_MANIFEST_FILENAME
    if manifest is None:
      if manifest_path.exists():
        manifest_path.unlink()
    else:
      manifest_path.write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8")

  def store(self, persona_name, root=None):
    root = root or self.isolated
    return (root / "personas" / persona_name / "bootstrap_memory"
            / "associative_memory")

  def node(self, embedding_key="key"):
    return {
      "node_count": 1, "type_count": 1, "type": "event", "depth": 0,
      "created": "2023-02-13 00:00:00", "expiration": None,
      "subject": "subject", "predicate": "is", "object": "idle",
      "description": "description", "embedding_key": embedding_key,
      "poignancy": 1, "keywords": [], "filling": [],
    }

  def audit(self, persona_name="Isabella Rodriguez"):
    return inspect_isolated_embedding_store(
      persona_name, self.store(persona_name), self.isolated)

  def test_01_empty_legacy_store_is_bootstrappable_with_structural_metadata(self):
    audit = self.audit()

    self.assertEqual(EMPTY_BOOTSTRAPPABLE, audit.classification)
    self.assertEqual("LEGACY_ASSUMED", audit.source_classification)
    self.assertFalse(audit.manifest_present)
    self.assertEqual(LEGACY_ADA_002_MANIFEST.model, audit.model)
    self.assertEqual((0, 0, 0, 0, 0), (
      audit.embedding_count, audit.node_count,
      audit.embedding_reference_count, audit.orphan_embedding_count,
      audit.nonempty_vector_count))
    self.assertEqual((), audit.observed_dimensions)
    self.assertFalse(audit.internal_mismatch)

  def test_02_empty_stores_use_canonical_bootstrap_and_become_loadable(self):
    original = controlled_replay_module.bootstrap_modern_embedding_store
    with patch.object(
        controlled_replay_module, "bootstrap_modern_embedding_store",
        wraps=original) as bootstrap:
      result = prepare_isolated_embedding_stores(self.fixture)

    self.assertEqual(3, bootstrap.call_count)
    self.assertEqual(self.PERSONAS, result.bootstrapped_personas)
    self.assertTrue(all(a.classification == MODERN_COMPATIBLE
                        for a in result.audits_after))
    for persona_name in self.PERSONAS:
      store = self.store(persona_name)
      manifest = read_embedding_manifest(
        store / EMBEDDING_MANIFEST_FILENAME)
      self.assertEqual(TEXT_EMBEDDING_3_SMALL_1536_MANIFEST, manifest)
      loaded = load_embedding_store(
        store, legacy_assumption_allowed=False,
        runtime_manifest=TEXT_EMBEDDING_3_SMALL_1536_MANIFEST)
      self.assertEqual({}, loaded.embeddings)
      self.assertEqual({}, loaded.nodes)

  def test_03_nonempty_legacy_embedding_blocks_without_relabeling(self):
    store = self.store("Isabella Rodriguez")
    vector = [1.0] + [0.0] * 1535
    node = self.node()
    self.write_store(store, {"key": vector}, {"node_1": node})
    before = snapshot = {
      path.name: hashlib.sha256(path.read_bytes()).hexdigest()
      for path in store.iterdir() if path.is_file()}

    audit = self.audit()
    with self.assertRaises(IsolatedEmbeddingPreflightError) as raised:
      prepare_isolated_embedding_stores(self.fixture)

    self.assertEqual(LEGACY_NONEMPTY_BLOCKED, audit.classification)
    self.assertEqual("EMBEDDING_STORE_MIGRATION_REQUIRED",
                     raised.exception.reason_code)
    self.assertFalse((store / EMBEDDING_MANIFEST_FILENAME).exists())
    self.assertEqual(snapshot, {
      path.name: hashlib.sha256(path.read_bytes()).hexdigest()
      for path in store.iterdir() if path.is_file()})

  def test_04_declared_ada_nonempty_manifest_is_never_rewritten(self):
    store = self.store("Isabella Rodriguez")
    vector = [1.0] + [0.0] * 1535
    self.write_store(
      store, {"key": vector}, {"node_1": self.node()},
      LEGACY_ADA_002_MANIFEST)
    manifest_before = (store / EMBEDDING_MANIFEST_FILENAME).read_bytes()

    with self.assertRaises(IsolatedEmbeddingPreflightError):
      prepare_isolated_embedding_stores(self.fixture)

    self.assertEqual(
      manifest_before, (store / EMBEDDING_MANIFEST_FILENAME).read_bytes())
    self.assertEqual(LEGACY_ADA_002_MANIFEST, read_embedding_manifest(
      store / EMBEDDING_MANIFEST_FILENAME))

  def test_05_missing_embedding_reference_is_inconsistent(self):
    store = self.store("Isabella Rodriguez")
    self.write_store(store, {}, {"node_1": {"embedding_key": "missing"}})

    audit = self.audit()
    with self.assertRaises(IsolatedEmbeddingPreflightError) as raised:
      prepare_isolated_embedding_stores(self.fixture)

    self.assertEqual(INCONSISTENT_BLOCKED, audit.classification)
    self.assertEqual("EMBEDDING_FIXTURE_INCONSISTENT",
                     raised.exception.reason_code)
    self.assertTrue(audit.internal_mismatch)

  def test_06_unknown_store_fails_closed(self):
    store = self.store("Isabella Rodriguez")
    (store / "kw_strength.json").unlink()

    audit = self.audit()

    self.assertEqual(UNKNOWN_BLOCKED, audit.classification)
    with self.assertRaises(IsolatedEmbeddingPreflightError) as raised:
      prepare_isolated_embedding_stores(self.fixture)
    self.assertEqual("EMBEDDING_STORE_UNKNOWN", raised.exception.reason_code)

  def test_07_all_personas_are_audited_before_constructor_provider_or_move(self):
    blocked_store = self.store("Klaus Mueller")
    vector = [1.0] + [0.0] * 1535
    self.write_store(
      blocked_store, {"key": vector}, {"node_1": self.node()})
    first_store = self.store("Isabella Rodriguez")
    first_before = {path.name: path.read_bytes() for path in first_store.iterdir()}
    constructor = Mock()

    def construct_only_after_preflight():
      prepare_isolated_embedding_stores(self.fixture)
      constructor()

    with patch("persona.persona.Persona.move") as move, patch.object(
          ModernOpenAIClientAdapter, "create_chat") as provider:
      with self.assertRaises(IsolatedEmbeddingPreflightError) as raised:
        construct_only_after_preflight()

    self.assertEqual(3, len(raised.exception.audits))
    constructor.assert_not_called()
    move.assert_not_called()
    provider.assert_not_called()
    self.assertEqual(first_before, {
      path.name: path.read_bytes() for path in first_store.iterdir()})

  def test_08_modernization_is_contained_and_idempotent(self):
    first = prepare_isolated_embedding_stores(self.fixture)
    first_snapshot = {
      path.relative_to(self.isolated).as_posix(): path.read_bytes()
      for path in self.isolated.rglob("*") if path.is_file()}

    second = prepare_isolated_embedding_stores(self.fixture)

    self.assertEqual((), second.bootstrapped_personas)
    self.assertEqual(first_snapshot, {
      path.relative_to(self.isolated).as_posix(): path.read_bytes()
      for path in self.isolated.rglob("*") if path.is_file()})
    self.assertTrue(all(self.isolated in audit.store_path.parents
                        for audit in first.audits_after))
    self.assertFalse(any(path.name.endswith("r1t_empty_backup")
                         for path in self.root.rglob("*")))

  def test_09_real_baseline_is_unchanged_by_copy_modernization(self):
    real_baseline = VERSIONED_BASELINE
    before = {
      path.relative_to(real_baseline).as_posix(): path.read_bytes()
      for path in real_baseline.rglob("*") if path.is_file()}
    real_copy = self.root / "real-copy"
    shutil.copytree(real_baseline, real_copy)
    fixture = prepare_isolated_reverie_fixture(
      real_copy, self.root, real_baseline, 0)

    result = prepare_isolated_embedding_stores(fixture)

    self.assertTrue(all(a.classification == EMPTY_BOOTSTRAPPABLE
                        for a in result.audits_before))
    self.assertEqual(before, {
      path.relative_to(real_baseline).as_posix(): path.read_bytes()
      for path in real_baseline.rglob("*") if path.is_file()})

  def test_10_existing_empty_modern_store_is_compatible_without_mutation(self):
    for persona_name in self.PERSONAS:
      self.write_store(
        self.store(persona_name), manifest=TEXT_EMBEDDING_3_SMALL_1536_MANIFEST)
    before = {path.relative_to(self.isolated).as_posix(): path.read_bytes()
              for path in self.isolated.rglob("*") if path.is_file()}

    result = prepare_isolated_embedding_stores(self.fixture)

    self.assertEqual((), result.bootstrapped_personas)
    self.assertTrue(all(a.classification == MODERN_COMPATIBLE
                        for a in result.audits_before))
    self.assertEqual(before, {
      path.relative_to(self.isolated).as_posix(): path.read_bytes()
      for path in self.isolated.rglob("*") if path.is_file()})


class SanitizedFailureEvidenceTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.temp_root = Path(self.temporary.name)
    self.event = SimpleNamespace(
      caller_id="act_obj_desc", operation="CHAT", outcome="SUCCESS",
      model_or_engine="gpt-4o-mini", response_model="gpt-4o-mini",
      physical_attempt=1, error_type=None)
    self.call_guard = SimpleNamespace(logical_calls=1, physical_attempts=1)
    snapshot = SimpleNamespace(
      accumulated_cost=Decimal("0.000001"), ceiling=Decimal("0.01"),
      remaining_cost=Decimal("0.009999"), tripped=False,
      logical_calls=1, physical_attempts=1,
      cost_by_operation=(("CHAT", Decimal("0.000001")),))
    self.cost_guard = Mock()
    self.cost_guard.snapshot.return_value = snapshot
    self.cost_guard.records.return_value = (object(),)

  def tearDown(self):
    self.temporary.cleanup()

  def capture(self):
    try:
      _raise_sanitized_index_error()
    except IndexError as error:
      return build_sanitized_failure_evidence(
        error=error, repository_root=REPOSITORY,
        temporary_root=self.temp_root, phase="PLANNING",
        current_module="plan", telemetry=(self.event,),
        call_guard=self.call_guard, cost_guard=self.cost_guard,
        provider_call_count=1, network_call_count=0,
        current_caller="act_obj_desc")
    self.fail("IndexError was not raised")

  def test_01_traceback_preserves_relative_file_function_line_and_phase(self):
    evidence = self.capture()
    failing = evidence["traceback"][-1]

    self.assertEqual("IndexError", evidence["exception_type"])
    self.assertEqual("PLAN_INDEX_ERROR", evidence["reason_code"])
    self.assertEqual("reverie/backend_server/tests/test_controlled_replay.py",
                     failing["file"])
    self.assertEqual("_raise_sanitized_index_error", failing["function"])
    self.assertIsInstance(failing["line"], int)
    self.assertEqual("PLANNING", evidence["failure_phase"])
    self.assertEqual("plan", evidence["current_module"])

  def test_02_traceback_excludes_paths_content_and_locals(self):
    serialized = json.dumps(self.capture())

    self.assertNotIn(str(self.temp_root), serialized)
    self.assertNotIn("PRIVATE-PROMPT-MARKER", serialized)
    self.assertNotIn("PRIVATE-RESPONSE-MARKER", serialized)
    self.assertNotIn("f_locals", serialized)
    self.assertNotIn("source_line", serialized)

  def test_03_runtime_attribution_and_failure_snapshots_are_complete(self):
    evidence = self.capture()

    self.assertEqual("act_obj_desc", evidence["current_caller"])
    self.assertEqual("act_obj_desc", evidence["last_completed_caller"])
    self.assertEqual("CHAT", evidence["operation"])
    self.assertEqual(1, evidence["telemetry_events"])
    self.assertEqual(1, evidence["ledger_records"])
    self.assertEqual((1, 1, 0), (
      evidence["logical_calls"], evidence["physical_attempts"],
      evidence["retry_count"]))
    self.assertEqual("0.000001", evidence["cost"]["accumulated"])
    self.assertEqual("0.01", evidence["cost"]["ceiling"])
    self.assertFalse(evidence["cost"]["tripped"])
    self.assertEqual(1, evidence["provider_calls"])
    self.assertEqual(0, evidence["network_calls"])

  def test_04_collection_diagnostic_is_structural_and_content_free(self):
    collection = self.capture()["collection"]

    self.assertEqual({
      "type": "list", "length": 0, "requested_index": 2,
      "empty": True, "element_type_names": (), "source": "values",
    }, collection)
    serialized = json.dumps(collection)
    self.assertNotIn("PRIVATE", serialized)
    self.assertNotIn("[2]", serialized)

  def test_05_diagnostic_introduces_no_provider_call_or_storage_mutation(self):
    adapter = DeterministicReplayFakeAdapter()
    protected = VERSIONED_BASELINE
    before = {
      path.relative_to(protected).as_posix(): hashlib.sha256(
        path.read_bytes()).hexdigest()
      for path in protected.rglob("*") if path.is_file()}

    evidence = self.capture()

    self.assertEqual([], adapter.calls)
    self.assertEqual(1, evidence["provider_calls"])
    self.assertEqual(before, {
      path.relative_to(protected).as_posix(): hashlib.sha256(
        path.read_bytes()).hexdigest()
      for path in protected.rglob("*") if path.is_file()})

  def test_06_runtime_contexts_teardown_after_index_error(self):
    adapter = DeterministicReplayFakeAdapter()
    try:
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(), adapter):
        with use_modern_chat_runtime(
            build_modern_chat_runtime_config(), adapter):
          with use_modern_completion_runtime(
              build_modern_completion_runtime_config(), adapter):
            _raise_sanitized_index_error()
    except IndexError:
      pass

    self.assertTrue(controlled_replay_contexts_are_reset())
    self.assertEqual([], adapter.calls)


class TaskDecompFakeContractTests(unittest.TestCase):
  def setUp(self):
    reset_provider()
    reset_llm_provider_config()
    clear_telemetry()
    baseline = VERSIONED_BASELINE
    actor_fixture = build_controlled_replay_actor_fixture(
      "Isabella Rodriguez",
      baseline / "personas" / "Isabella Rodriguez" / "bootstrap_memory"
      / "scratch.json",
      baseline / "reverie" / "meta.json")
    self.persona = SimpleNamespace(
      name="Isabella Rodriguez", scratch=actor_fixture.persona.scratch)
    self.assertIsInstance(self.persona.name, str)
    self.assertTrue(self.persona.name)
    self.assertIs(self.persona.scratch, actor_fixture.persona.scratch)
    self.persona.scratch.f_daily_schedule_hourly_org = [
      ["working", 60], ["resting", 60]]
    self.context = LLMReplayContext(
      cognitive_category="PLANNING", actor_id="Isabella Rodriguez",
      simulation_id="task-decomp-micro", simulation_step=0)

  def tearDown(self):
    reset_provider()
    reset_llm_provider_config()
    clear_telemetry()

  def invoke(self, adapter):
    network_calls = []
    def block_network(*args, **kwargs):
      network_calls.append("blocked")
      raise AssertionError("network is forbidden in task_decomp micro test")
    with patch("socket.create_connection", side_effect=block_network), patch.object(
        socket.socket, "connect", block_network):
      with use_modern_completion_runtime(
          build_modern_completion_runtime_config(), adapter):
        with use_llm_replay_context(self.context):
          with _historical_wrapper_environment():
            result = run_gpt_prompt.run_gpt_prompt_task_decomp(
              self.persona, "working", 60)
    return result, get_telemetry(), network_calls

  def test_01_dedicated_response_crosses_real_historical_contract_once(self):
    adapter = R1TDeterministicFakeAdapter()

    result, events, network_calls = self.invoke(adapter)

    parsed, historical_metadata = result
    self.assertIsInstance(result, tuple)
    self.assertEqual(2, len(result))
    self.assertIsInstance(historical_metadata, list)
    self.assertTrue(parsed)
    for entry in parsed:
      self.assertIsInstance(entry, list)
      self.assertEqual(2, len(entry))
      self.assertIsInstance(entry[0], str)
      self.assertIsInstance(entry[1], int)
      self.assertGreater(entry[1], 0)
    self.assertNotEqual(["asleep"], parsed)
    self.assertEqual(1, len(adapter.calls))
    self.assertEqual("task_decomp", adapter.calls[0]["caller_id"])
    self.assertEqual("gpt-4o-mini", adapter.calls[0]["model"])
    self.assertEqual(1, len(events))
    self.assertEqual(1, len({event.logical_call_id for event in events}))
    self.assertEqual(1, events[0].physical_attempt)
    self.assertEqual("task_decomp", events[0].caller_id)
    self.assertEqual(COMPLETION_COMPAT, events[0].operation)
    self.assertEqual("SUCCESS", events[0].outcome)
    self.assertEqual([], network_calls)
    telemetry_text = repr(events)
    self.assertNotIn(R1T_TASK_DECOMP_FAKE_RESPONSE, telemetry_text)
    self.assertNotIn("working", telemetry_text)
    self.assertTrue(controlled_replay_contexts_are_reset())

  def test_02_generic_responses_exhaust_retries_into_structured_fail_safe(self):
    adapter = DeterministicReplayFakeAdapter(
      text_responses=(R1T_GENERIC_FAKE_RESPONSE,) * 5)

    result, events, network_calls = self.invoke(adapter)

    self.assertEqual([["working (working)", 60]], result[0])
    self.assertEqual(5, len(events))
    self.assertEqual(5, len(adapter.calls))
    self.assertTrue(all(event.caller_id == "task_decomp" for event in events))
    self.assertEqual([1, 2, 3, 4, 5],
                     [event.physical_attempt for event in events])
    self.assertEqual([], network_calls)
    self.assertTrue(controlled_replay_contexts_are_reset())

  def test_03_only_task_decomp_differs_from_the_existing_fake_matrix(self):
    self.assertEqual(
      R1T_TASK_DECOMP_FAKE_RESPONSE,
      R1TDeterministicFakeAdapter.response_for_caller("task_decomp"))
    self.assertNotEqual(
      R1T_GENERIC_FAKE_RESPONSE,
      R1TDeterministicFakeAdapter.response_for_caller("task_decomp"))
    self.assertEqual(
      R1T_PRONUNCIATIO_FAKE_RESPONSE,
      R1TDeterministicFakeAdapter.response_for_caller("pronunciatio"))
    self.assertEqual(
      R1T_ACT_OBJ_DESC_FAKE_RESPONSE,
      R1TDeterministicFakeAdapter.response_for_caller("act_obj_desc"))
    self.assertEqual(
      R1T_GENERIC_FAKE_RESPONSE,
      R1TDeterministicFakeAdapter.response_for_caller("wake_up_hour"))


class ControlledReplayTests(unittest.TestCase):
  def setUp(self):
    reset_provider()
    reset_llm_provider_config()
    reset_embedding_measurement_all()
    clear_telemetry()
    self.sleep = patch.object(gpt_structure, "temp_sleep")
    self.sleep.start()
    self.temporary = tempfile.TemporaryDirectory()
    self.parent = Path(self.temporary.name)

  def tearDown(self):
    self.temporary.cleanup()
    self.sleep.stop()
    reset_provider()
    reset_llm_provider_config()
    reset_embedding_measurement_all()
    clear_telemetry()

  def profile(self, **changes):
    values = {
      "replay_id": "replay-r0h-001",
      "simulation_id": "simulation-r0h-001",
      "actor_id": "isabella-rodriguez",
      "step": 0,
      "maximum_cost": Decimal("0.01"),
      "maximum_logical_calls": 1,
      "maximum_physical_attempts": 5,
    }
    values.update(changes)
    return ControlledReplayProfile(**values)

  def actor(self, prompt="PRIVATE-ACTOR-CONTEXT-R0H"):
    return ControlledReplayActorFixture(
      actor_id="isabella-rodriguez", persona=object(),
      prompt_input=(prompt,))

  def environment(self):
    return ControlledReplayEnvironmentFixture(
      simulation_id="simulation-r0h-001", step=0)

  def providers(self, responses=("7 am",), **adapter_changes):
    return ControlledReplayProviders(
      DeterministicReplayFakeAdapter(),
      DeterministicReplayFakeAdapter(responses, **adapter_changes),
      DeterministicReplayFakeAdapter(),
    )

  def execute_replay(self, name="modern-store", profile=None, providers=None,
                     actor=None, environment=None):
    return run_controlled_replay_step(
      profile or self.profile(), actor or self.actor(),
      environment or self.environment(), providers or self.providers(),
      self.parent / name)

  def test_01_profile_accepts_canonical_values_and_is_frozen(self):
    profile = self.profile()
    self.assertEqual(Decimal("0.01"), profile.maximum_cost)
    with self.assertRaises(Exception):
      profile.step = 1

  def test_02_profile_rejects_blank_identities(self):
    for field in ("replay_id", "simulation_id", "actor_id"):
      for value in ("", "   ", None):
        with self.subTest(field=field, value=value):
          with self.assertRaises(ControlledReplayConfigurationError):
            self.profile(**{field: value})

  def test_03_profile_rejects_invalid_step(self):
    for value in (-1, True, 1.5, "0"):
      with self.subTest(value=value):
        with self.assertRaises(ControlledReplayConfigurationError):
          self.profile(step=value)

  def test_04_profile_rejects_invalid_cost(self):
    for value in (1, 0.1, True, Decimal("0"), Decimal("-1"),
                  Decimal("NaN"), Decimal("Infinity")):
      with self.subTest(value=repr(value)):
        with self.assertRaises(ControlledReplayConfigurationError):
          self.profile(maximum_cost=value)

  def test_05_profile_rejects_invalid_call_limits(self):
    for field in ("maximum_logical_calls", "maximum_physical_attempts"):
      for value in (0, -1, True, 1.5):
        with self.subTest(field=field, value=value):
          with self.assertRaises(ControlledReplayConfigurationError):
            self.profile(**{field: value})

  def test_06_profile_rejects_conversation_and_reflection(self):
    with self.assertRaises(ControlledReplayConfigurationError):
      self.profile(conversation_enabled=True)
    with self.assertRaises(ControlledReplayConfigurationError):
      self.profile(reflection_enabled=True)

  def test_07_successful_fake_replay_uses_real_historical_parser(self):
    report = self.execute_replay()
    self.assertEqual("SUCCESS", report.status)
    self.assertEqual("planning.wake_up_hour", report.selected_cognitive_path)
    self.assertEqual(1, report.logical_calls)
    self.assertEqual(1, report.physical_attempts)
    self.assertEqual(((COMPLETION_COMPAT, 1),), report.operation_counts)
    self.assertEqual(("gpt-4o-mini",), report.models_requested)
    self.assertEqual(("gpt-4o-mini",), report.models_returned)
    self.assertEqual(0, report.legacy_detections)
    self.assertIsNone(report.primary_error_type)
    self.assertIsNone(report.underlying_provider_error_type)
    self.assertIsNotNone(report.cognitive_output_digest)
    with self.assertRaises(Exception):
      report.primary_error_type = "changed"

  def test_08_all_modern_runtimes_are_composed_and_then_reset(self):
    observations = {}
    original = run_gpt_prompt.safe_generate_response

    def observe(*args, **kwargs):
      observations["chat"] = get_modern_chat_runtime_config()
      observations["completion"] = get_modern_completion_runtime_config()
      observations["manifest"] = get_runtime_embedding_manifest()
      observations["context"] = get_llm_replay_context()
      observations["cost"] = get_replay_cost_guard()
      return original(*args, **kwargs)

    with patch.object(run_gpt_prompt, "safe_generate_response", observe):
      self.execute_replay()
    self.assertIsNotNone(observations["chat"])
    self.assertIsNotNone(observations["completion"])
    self.assertEqual("text-embedding-3-small", observations["manifest"].model)
    self.assertEqual("isabella-rodriguez", observations["context"].actor_id)
    self.assertIsNotNone(observations["cost"])
    self.assertTrue(controlled_replay_contexts_are_reset())

  def test_09_logical_limit_blocks_the_next_logical_call(self):
    guard = ReplayCallCountGuardState(self.profile())
    guard.check_before_attempt("logical-1")
    guard.record_attempt("logical-1")
    with self.assertRaises(ReplayLogicalCallLimitExceededError):
      guard.check_before_attempt("logical-2")
    self.assertEqual(1, guard.physical_attempts)

  def test_10_physical_limit_blocks_pre_provider(self):
    providers = self.providers(("invalid", "invalid", "7 am"))
    with self.assertRaises(ReplayPhysicalAttemptLimitExceededError) as raised:
      self.execute_replay(
        profile=self.profile(maximum_physical_attempts=2),
        providers=providers)
    self.assertEqual(2, len(providers.completion_adapter.calls))
    self.assertEqual(2, raised.exception.controlled_replay_report.physical_attempts)

  def test_11_under_cost_ceiling_succeeds(self):
    report = self.execute_replay()
    self.assertGreater(report.accumulated_cost, Decimal("0"))
    self.assertLess(report.accumulated_cost, report.cost_ceiling)
    self.assertGreater(report.remaining_cost, Decimal("0"))

  def test_12_reaching_cost_ceiling_allows_current_call(self):
    report = self.execute_replay(profile=self.profile(
      maximum_cost=Decimal("0.000004800000")))
    self.assertEqual("SUCCESS", report.status)
    self.assertEqual(report.cost_ceiling, report.accumulated_cost)
    self.assertEqual(Decimal("0"), report.remaining_cost)

  def test_13_exceeding_cost_ceiling_is_typed_and_reported(self):
    with self.assertRaises(ReplayCostCeilingExceededError) as raised:
      self.execute_replay(
        profile=self.profile(maximum_cost=Decimal("0.000001")))
    report = raised.exception.controlled_replay_report
    self.assertEqual("FAILED", report.status)
    self.assertEqual("ReplayCostCeilingExceededError", report.error_type)
    self.assertEqual(1, report.physical_attempts)

  def test_14_legacy_models_hard_fail_without_provider_calls(self):
    adapter = DeterministicReplayFakeAdapter()
    for model in ("text-davinci-003", "gpt-3.5-turbo",
                  "text-embedding-ada-002"):
      with self.subTest(model=model):
        with self.assertRaises(LegacyModelInvocationDetectedError):
          assert_no_legacy_model_invocation(model)
    self.assertEqual([], adapter.calls)
    with self.assertRaises(ValueError):
      ModernCompletionRuntimeConfig(model="text-davinci-003")
    with self.assertRaises(ValueError):
      ModernChatRuntimeConfig(chat_model="gpt-3.5-turbo")

  def test_15_legacy_provider_configuration_hard_fails(self):
    adapter = DeterministicReplayFakeAdapter()
    with self.assertRaises(ControlledReplayLegacyConfigurationError):
      ControlledReplayProviders(
        adapter, adapter, adapter, chat_config=LLMProviderConfig())
    with self.assertRaises(ControlledReplayLegacyConfigurationError):
      ControlledReplayProviders(
        adapter, adapter, adapter,
        embedding_config=build_legacy_embedding_runtime_config())
    self.assertEqual([], adapter.calls)

  def test_16_non_offline_adapter_is_rejected(self):
    with self.assertRaises(ControlledReplayConfigurationError):
      ControlledReplayProviders(object(), object(), object())

  def test_17_deferred_caller_fails_before_semantic_retry(self):
    adapter = DeterministicReplayFakeAdapter()
    with use_modern_completion_runtime(
        build_modern_completion_runtime_config(), adapter):
      with self.assertRaises(CompletionCompatCallerNotAllowedError):
        gpt_structure.safe_generate_response(
          "PRIVATE", {
            "engine": "text-davinci-003", "max_tokens": 5,
            "temperature": 0, "top_p": 1, "stream": False,
            "frequency_penalty": 0, "presence_penalty": 0, "stop": None,
          }, 5, "fallback", lambda value, prompt="": False,
          lambda value, prompt="": value,
          caller_id="create_conversation")
    self.assertEqual([], adapter.calls)

  def test_18_conversation_and_reflection_tripwires(self):
    for caller in ("create_conversation", "extract_keywords",
                   "keyword_to_thoughts", "convo_to_thoughts"):
      with self.subTest(caller=caller):
        with self.assertRaises(ControlledReplayPathForbiddenError):
          assert_controlled_replay_path_allowed(caller)

  def test_19_store_is_bootstrapped_and_manifest_validated(self):
    store = self.parent / "store"
    report = self.execute_replay(name="store")
    self.assertTrue((store / EMBEDDING_MANIFEST_FILENAME).is_file())
    self.assertIn("text-embedding-3-small", report.store_manifest_identity)

  def test_20_existing_store_is_never_overwritten(self):
    self.execute_replay(name="existing")
    with self.assertRaises(EmbeddingStoreAlreadyExistsError):
      self.execute_replay(name="existing")

  def test_21_manifestless_and_ada_stores_are_rejected(self):
    manifestless = self.parent / "manifestless"
    manifestless.mkdir()
    for name, value in (("embeddings.json", "{}"), ("nodes.json", "{}"),
                        ("kw_strength.json", "{}")):
      (manifestless / name).write_text(value, encoding="utf-8")
    with self.assertRaises(EmbeddingStoreUnknownError):
      self.execute_replay(name="manifestless")

    ada = self.parent / "ada"
    ada.mkdir()
    (ada / EMBEDDING_MANIFEST_FILENAME).write_text(
      __import__("json").dumps(LEGACY_ADA_002_MANIFEST.to_dict()),
      encoding="utf-8")
    for name in ("embeddings.json", "nodes.json", "kw_strength.json"):
      (ada / name).write_text("{}", encoding="utf-8")
    with self.assertRaises(EmbeddingStoreIncompatibleError):
      self.execute_replay(name="ada")

  def test_22_protected_runtime_storage_is_rejected_without_write(self):
    protected = (REPOSITORY / "environment" / "frontend_server"
                 / "temp_storage" / "r0h-forbidden-store")
    self.assertFalse(protected.exists())
    with self.assertRaises(EmbeddingStoreUnsafePathError):
      run_controlled_replay_step(
        self.profile(), self.actor(), self.environment(), self.providers(),
        protected)
    self.assertFalse(protected.exists())

  def test_23_attribution_is_preserved_per_attempt_and_ledger_record(self):
    report = self.execute_replay()
    self.assertEqual(1, len(report.telemetry))
    entry = report.telemetry[0]
    ledger = report.ledger[0]
    expected = ("replay-r0h-001", "simulation-r0h-001",
                "isabella-rodriguez", 0, "PLANNING", "wake_up_hour",
                "COMPLETION_COMPAT")
    self.assertEqual(expected, (
      entry.replay_id, entry.simulation_id, entry.actor_id, entry.step,
      entry.cognitive_category, entry.caller_id, entry.operation))
    self.assertEqual(expected, (
      ledger.replay_id, ledger.simulation_id, ledger.actor_id, ledger.step,
      ledger.cognitive_category, ledger.caller_id, ledger.operation))
    self.assertEqual(entry.logical_call_id, ledger.logical_call_id)
    self.assertEqual(entry.physical_attempt, ledger.physical_attempt)

  def test_24_usage_and_cost_are_complete(self):
    report = self.execute_replay()
    self.assertEqual((20, 3, 0, 0), (
      report.input_tokens, report.output_tokens,
      report.cached_tokens, report.reasoning_tokens))
    self.assertEqual(Decimal("0.000004800000"), report.accumulated_cost)
    self.assertEqual((0, 0), (
      report.embedding_cache_hits, report.embedding_cache_misses))

  def test_25_unknown_usage_fails_closed(self):
    providers = self.providers(input_tokens=None, output_tokens=None,
                               cached_input_tokens=None,
                               reasoning_tokens=None)
    with self.assertRaises(ReplayCostAccountingUnavailableError) as raised:
      self.execute_replay(providers=providers)
    report = raised.exception.controlled_replay_report
    self.assertEqual("FAILED", report.status)
    self.assertEqual(
      "ReplayCostAccountingUnavailableError", report.primary_error_type)
    self.assertIsNone(report.underlying_provider_error_type)
    self.assertEqual(1, len(providers.completion_adapter.calls))

  def test_26_provider_error_is_typed_reported_and_stops(self):
    providers = self.providers((LLMTimeoutError("offline timeout"), "7 am"))
    with self.assertRaises(ReplayCostAccountingUnavailableError) as raised:
      self.execute_replay(providers=providers)
    report = raised.exception.controlled_replay_report
    self.assertEqual("FAILED", report.status)
    self.assertEqual(
      "ReplayCostAccountingUnavailableError", report.primary_error_type)
    self.assertEqual("LLMTimeoutError", report.underlying_provider_error_type)
    self.assertIsInstance(raised.exception.__context__, LLMTimeoutError)
    self.assertEqual(1, report.physical_attempts)
    self.assertEqual(1, len(providers.completion_adapter.calls))

  def test_26a_normalized_provider_failure_types_are_preserved(self):
    error_types = (
      LLMAuthenticationError,
      LLMModelNotFoundError,
      LLMInvalidRequestError,
      LLMUnsupportedParameterError,
      LLMConnectionError,
      LLMTimeoutError,
    )
    for index, error_type in enumerate(error_types):
      with self.subTest(error_type=error_type.__name__):
        providers = self.providers((error_type("PRIVATE-UPSTREAM-BODY"),))
        with self.assertRaises(ReplayCostAccountingUnavailableError) as raised:
          self.execute_replay(name=f"typed-provider-{index}", providers=providers)
        report = raised.exception.controlled_replay_report
        self.assertEqual(error_type.__name__,
                         report.underlying_provider_error_type)
        self.assertEqual(1, len(providers.completion_adapter.calls))

  def test_26b_unknown_provider_failure_is_redacted_and_stops(self):
    secret = "PRIVATE-UNKNOWN-PROVIDER-MESSAGE"
    providers = self.providers((RuntimeError(secret), "7 am"))
    with self.assertRaises(ReplayCostAccountingUnavailableError) as raised:
      self.execute_replay(name="unknown-provider", providers=providers)
    report = raised.exception.controlled_replay_report
    self.assertEqual("UNKNOWN_PROVIDER_ERROR",
                     report.underlying_provider_error_type)
    self.assertNotIn(secret, repr(report))
    self.assertNotIn("RuntimeError", repr(report))
    self.assertEqual(1, len(providers.completion_adapter.calls))

  def test_27_parser_failure_preserves_semantic_retry_and_fail_safe(self):
    providers = self.providers(("invalid",) * 5)
    report = self.execute_replay(providers=providers)
    self.assertEqual("SUCCESS", report.status)
    self.assertEqual(5, report.physical_attempts)
    self.assertEqual(4, report.retry_count)
    self.assertEqual(5, len(providers.completion_adapter.calls))
    self.assertIsNone(report.primary_error_type)
    self.assertIsNone(report.underlying_provider_error_type)
    expected_digest = hashlib.sha256(repr(8).encode("utf-8")).hexdigest()
    self.assertEqual(expected_digest, report.cognitive_output_digest)

  def test_27a_parser_exception_is_not_classified_as_provider_failure(self):
    class ControlledParserError(RuntimeError):
      pass

    with patch.object(
        run_gpt_prompt, "run_gpt_prompt_wake_up_hour",
        side_effect=ControlledParserError("PRIVATE-PARSER-CONTENT")):
      with self.assertRaises(ControlledParserError) as raised:
        self.execute_replay(name="parser-error")
    report = raised.exception.controlled_replay_report
    self.assertEqual("ControlledParserError", report.primary_error_type)
    self.assertIsNone(report.underlying_provider_error_type)
    self.assertEqual(0, report.physical_attempts)
    self.assertNotIn("PRIVATE-PARSER-CONTENT", repr(report))

  def test_28_report_and_error_are_content_private(self):
    secret = "PRIVATE-PROMPT-R0H-DO-NOT-EXPORT"
    report = self.execute_replay(actor=self.actor(secret))
    exported = repr(report)
    self.assertNotIn(secret, exported)
    for forbidden in ("prompt", "messages", "raw output", "memory stream",
                      "transcript", "embedding vector", "API key"):
      self.assertNotIn(forbidden, exported)

    providers = self.providers((LLMTimeoutError(secret),))
    with self.assertRaises(ReplayCostAccountingUnavailableError) as raised:
      self.execute_replay(name="error-store", providers=providers,
                          actor=self.actor(secret))
    self.assertNotIn(secret, repr(raised.exception.controlled_replay_report))

  def test_29_context_teardown_occurs_after_exception(self):
    with self.assertRaises(ReplayCostAccountingUnavailableError):
      self.execute_replay(
        providers=self.providers((LLMTimeoutError("offline"),)))
    self.assertTrue(controlled_replay_contexts_are_reset())
    self.assertIsNone(get_replay_cost_guard())
    self.assertEqual(LLMReplayContext(), get_llm_replay_context())

  def test_30_repeatability_excluding_runtime_ids_and_durations(self):
    first = self.execute_replay(name="repeat-one")
    clear_telemetry()
    second = self.execute_replay(name="repeat-two")
    self.assertEqual((first.status, first.operation_counts,
                      first.logical_calls, first.physical_attempts,
                      first.accumulated_cost, first.store_manifest_identity,
                      first.cognitive_output_digest),
                     (second.status, second.operation_counts,
                      second.logical_calls, second.physical_attempts,
                      second.accumulated_cost, second.store_manifest_identity,
                      second.cognitive_output_digest))

  def test_31_no_socket_dns_or_real_client_is_reached(self):
    with patch.object(socket, "socket", side_effect=AssertionError(
        "network forbidden")), patch.object(
          socket, "getaddrinfo", side_effect=AssertionError(
            "DNS forbidden")):
      report = self.execute_replay()
    self.assertEqual("SUCCESS", report.status)

  def test_32_versioned_baseline_is_unchanged(self):
    paths = (VERSIONED_BASELINE,)

    def snapshot(path):
      if not path.exists():
        return None
      targets = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file())
      return tuple((item.relative_to(REPOSITORY).as_posix(),
                    item.stat().st_mtime_ns, item.stat().st_size,
                    hashlib.sha256(item.read_bytes()).hexdigest())
                   for item in targets)

    before = tuple(snapshot(path) for path in paths)
    self.execute_replay()
    after = tuple(snapshot(path) for path in paths)
    self.assertEqual(before, after)

  def test_33_fixture_identity_mismatch_fails_before_store_creation(self):
    target = self.parent / "identity-mismatch"
    with self.assertRaises(ControlledReplayConfigurationError):
      run_controlled_replay_step(
        self.profile(), replace(self.actor(), actor_id="other"),
        self.environment(), self.providers(), target)
    self.assertFalse(target.exists())

  def test_34_default_isolation_and_missing_operation_config(self):
    self.assertIsNone(get_modern_chat_runtime_config())
    self.assertIsNone(get_modern_completion_runtime_config())
    self.assertEqual(LEGACY_ADA_002_MANIFEST,
                     get_runtime_embedding_manifest())
    self.profile()
    providers = self.providers()
    self.assertIsNone(get_modern_chat_runtime_config())
    self.assertIsNone(get_modern_completion_runtime_config())
    self.assertEqual([], providers.chat_adapter.calls)
    with self.assertRaises(ControlledReplayConfigurationError):
      ControlledReplayProviders(
        providers.chat_adapter, providers.completion_adapter,
        providers.embedding_adapter, chat_config=None)

  def test_35_missing_pricing_fails_closed(self):
    pricing = PricingSnapshot(
      "r0h-missing-pricing", 1, "USD", "2026-08-06", (),
      "deliberately empty")
    providers = replace(self.providers(), pricing_snapshot=pricing)
    with self.assertRaises(ReplayCostAccountingUnavailableError) as raised:
      self.execute_replay(providers=providers)
    self.assertEqual("FAILED", raised.exception.controlled_replay_report.status)
    self.assertEqual(1, len(providers.completion_adapter.calls))

  def test_36_live_adapter_requires_explicit_opt_in(self):
    adapter = ModernOpenAIClientAdapter(client=object())
    with self.assertRaises(ControlledReplayConfigurationError):
      ControlledReplayProviders(adapter, adapter, adapter)
    providers = ControlledReplayProviders(
      adapter, adapter, adapter, live_api_enabled=True)
    self.assertTrue(providers.live_api_enabled)

  def test_37_live_fixture_uses_canonical_simulation_time_offline(self):
    meta_path = self.parent / "meta.json"
    meta_path.write_text(json.dumps({
      "curr_time": "February 13, 2023, 00:00:00",
    }), encoding="utf-8")
    scratch = Scratch.__new__(Scratch)
    scratch.name = "Isabella Rodriguez"
    scratch.first_name = "Isabella"
    scratch.age = 34
    scratch.innate = "friendly"
    scratch.learned = "Isabella owns a cafe."
    scratch.currently = "Isabella is preparing for the day."
    scratch.lifestyle = "Isabella wakes around 6am."
    scratch.daily_plan_req = "Isabella plans to work at the cafe."
    scratch.curr_time = None

    with patch("controlled_replay.Scratch", return_value=scratch), patch.object(
        ModernOpenAIClientAdapter, "create_chat") as provider_call:
      fixture = build_controlled_replay_actor_fixture(
        "Isabella Rodriguez", self.parent / "scratch.json", meta_path)

    self.assertEqual(
      datetime.datetime(2023, 2, 13, 0, 0),
      fixture.persona.scratch.curr_time)
    self.assertIn("Current Date: Monday February 13", fixture.prompt_input[0])
    provider_call.assert_not_called()
    self.assertEqual((), get_telemetry())

  def test_38_live_fixture_rejects_unavailable_simulation_metadata(self):
    with self.assertRaisesRegex(
        ControlledReplayConfigurationError,
        "simulation metadata is unavailable"):
      build_controlled_replay_actor_fixture(
        "Isabella Rodriguez", self.parent / "scratch.json",
        self.parent / "missing-meta.json")

  def test_39_live_fixture_rejects_malformed_simulation_time(self):
    meta_path = self.parent / "meta.json"
    meta_path.write_text(
      json.dumps({"curr_time": "not-a-datetime"}), encoding="utf-8")
    with self.assertRaisesRegex(
        ControlledReplayConfigurationError,
        "simulation metadata curr_time is malformed"):
      build_controlled_replay_actor_fixture(
        "Isabella Rodriguez", self.parent / "scratch.json", meta_path)

  def test_40_incomplete_response_with_usage_is_costed_and_reported(self):
    error = LLMIncompleteResponseError(
      "Modern chat response is incomplete",
      request_id="req_incomplete",
      response_model="gpt-4o-mini-2024-07-18",
      response_status="completed",
      finish_reason="length",
      usage=NormalizedUsage(20, 3, 0, 0),
    )
    with self.assertRaises(LLMIncompleteResponseError) as raised:
      self.execute_replay(
        name="incomplete-with-usage", providers=self.providers((error,)))
    report = raised.exception.controlled_replay_report
    self.assertEqual("LLMIncompleteResponseError", report.primary_error_type)
    self.assertEqual(
      "LLMIncompleteResponseError", report.underlying_provider_error_type)
    self.assertEqual((20, 3), (report.input_tokens, report.output_tokens))
    self.assertEqual(Decimal("0.000004800000"), report.accumulated_cost)
    self.assertEqual("gpt-4o-mini-2024-07-18", report.models_returned[0])

  def test_41_incomplete_response_without_usage_fails_cost_closed(self):
    error = LLMIncompleteResponseError(
      "Modern chat response is incomplete", finish_reason="length")
    with self.assertRaises(ReplayCostAccountingUnavailableError) as raised:
      self.execute_replay(
        name="incomplete-without-usage", providers=self.providers((error,)))
    report = raised.exception.controlled_replay_report
    self.assertEqual(
      "ReplayCostAccountingUnavailableError", report.primary_error_type)
    self.assertEqual(
      "LLMIncompleteResponseError", report.underlying_provider_error_type)
    self.assertEqual(Decimal("0"), report.accumulated_cost)


if __name__ == "__main__":
  unittest.main()
