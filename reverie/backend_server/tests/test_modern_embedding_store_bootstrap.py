import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.memory_structures.embedding_space import (
  EMBEDDING_MANIFEST_FILENAME,
  LEGACY_ADA_002_MANIFEST,
  EmbeddingSpaceMismatchError,
  read_embedding_manifest,
  reset_runtime_embedding_manifest,
)
from persona.prompt_template import embedding_store_bootstrap as bootstrap
from persona.prompt_template.embedding_runtime import (
  TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
  build_legacy_embedding_runtime_config,
  build_modern_embedding_runtime_config,
  validate_embedding_store_for_runtime,
)
from persona.prompt_template.llm_provider import (
  clear_telemetry,
  get_embedding_cache_stats,
  get_telemetry,
  reset_embedding_measurement_all,
)
from persona.prompt_template.cost_ledger import (
  PricingSnapshot,
  build_cost_ledger_records,
)


class ModernEmbeddingStoreBootstrapTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    clear_telemetry()
    reset_embedding_measurement_all()

  def tearDown(self):
    self.temporary.cleanup()

  def request(self, name="modern-store", **changes):
    values = {
      "target_path": self.root / name,
      "allowed_parent": self.root,
    }
    values.update(changes)
    return bootstrap.ModernEmbeddingStoreBootstrapRequest(**values)

  def create(self, name="modern-store", **changes):
    request = self.request(name, **changes)
    result = bootstrap.bootstrap_modern_embedding_store(request)
    return Path(result.target_path), result

  def write_json(self, path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

  def seed_memory_files(self, target, manifest=None):
    target.mkdir()
    self.write_json(target / bootstrap.EMBEDDINGS_FILENAME, {})
    self.write_json(target / bootstrap.NODES_FILENAME, {})
    self.write_json(target / bootstrap.KEYWORD_STRENGTH_FILENAME, {
      "kw_strength_event": {}, "kw_strength_thought": {}})
    if manifest is not None:
      self.write_json(
        target / EMBEDDING_MANIFEST_FILENAME, manifest.to_dict())

  def temporary_bootstrap_entries(self):
    return tuple(path for path in self.root.iterdir()
                 if bootstrap._TEMP_MARKER in path.name)

  def assert_protected_request_rejected(self, request):
    with patch.object(
        tempfile, "mkdtemp",
        side_effect=AssertionError("staging created")) as staging, patch.object(
          bootstrap, "_write_store_files",
          side_effect=AssertionError("files written")) as writer, patch.object(
            bootstrap, "_rename_staged_directory",
            side_effect=AssertionError("rename reached")) as rename, patch.object(
              bootstrap, "_classify_existing_target",
              side_effect=AssertionError("classification reached")) as classify, \
              patch.object(
                bootstrap, "validate_embedding_store_for_runtime",
                side_effect=AssertionError("preflight reached")) as preflight, \
              patch.object(os, "rmdir",
                           side_effect=AssertionError("rmdir reached")) as rmdir:
      with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
        bootstrap.bootstrap_modern_embedding_store(request)
    for operation in (staging, writer, rename, classify, preflight, rmdir):
      operation.assert_not_called()

  def test_01_bootstrap_nonexistent_path(self):
    target, result = self.create()
    self.assertTrue(target.is_dir())
    self.assertEqual(target.as_posix(), result.target_path)

  def test_02_modern_manifest_is_written(self):
    target, unused = self.create()
    self.assertTrue((target / EMBEDDING_MANIFEST_FILENAME).is_file())

  def test_03_manifest_is_the_canonical_modern_contract(self):
    target, unused = self.create()
    self.assertEqual(
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
      read_embedding_manifest(target / EMBEDDING_MANIFEST_FILENAME))

  def test_04_manifest_serialization_and_newline_are_deterministic(self):
    target, unused = self.create()
    actual = (target / EMBEDDING_MANIFEST_FILENAME).read_bytes()
    expected = (json.dumps(
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.to_dict(), indent=2)
      + "\n").encode("utf-8")
    self.assertEqual(expected, actual)
    self.assertTrue(actual.endswith(b"\n"))

  def test_05_result_is_immutable(self):
    unused, result = self.create()
    with self.assertRaises(FrozenInstanceError):
      result.bootstrap_version = 2

  def test_06_result_serialization_is_json_safe(self):
    unused, result = self.create()
    value = bootstrap.modern_embedding_store_bootstrap_result_to_dict(result)
    self.assertEqual(value, json.loads(json.dumps(value, sort_keys=True)))
    self.assertIsInstance(value["created_files"], list)

  def test_07_modern_preflight_accepts_bootstrapped_store(self):
    target, unused = self.create()
    loaded = validate_embedding_store_for_runtime(
      target, build_modern_embedding_runtime_config())
    self.assertEqual(TEXT_EMBEDDING_3_SMALL_1536_MANIFEST, loaded.manifest)

  def test_08_legacy_preflight_rejects_bootstrapped_store(self):
    target, unused = self.create()
    with self.assertRaises(EmbeddingSpaceMismatchError):
      validate_embedding_store_for_runtime(
        target, build_legacy_embedding_runtime_config())

  def test_09_second_bootstrap_is_rejected(self):
    target, unused = self.create()
    with self.assertRaises(bootstrap.EmbeddingStoreAlreadyExistsError):
      bootstrap.bootstrap_modern_embedding_store(self.request())
    self.assertTrue(target.is_dir())

  def test_10_existing_empty_directory_is_rejected_by_default(self):
    target = self.root / "empty"
    target.mkdir()
    with self.assertRaises(bootstrap.EmbeddingStoreAlreadyExistsError):
      bootstrap.bootstrap_modern_embedding_store(self.request("empty"))
    self.assertEqual((), tuple(target.iterdir()))

  def test_11_existing_empty_directory_requires_explicit_flag(self):
    target = self.root / "empty-allowed"
    target.mkdir()
    result = bootstrap.bootstrap_modern_embedding_store(self.request(
      "empty-allowed", allow_existing_empty_directory=True))
    self.assertEqual(set(bootstrap.CREATED_FILES), {
      path.name for path in Path(result.target_path).iterdir()})

  def test_12_nonempty_directory_is_rejected(self):
    target = self.root / "nonempty"
    target.mkdir()
    (target / "unrelated.txt").write_text("preserve", encoding="utf-8")
    with self.assertRaises(bootstrap.EmbeddingStoreNotEmptyError):
      bootstrap.bootstrap_modern_embedding_store(self.request("nonempty"))

  def test_13_file_target_is_rejected(self):
    target = self.root / "file-target"
    target.write_text("preserve", encoding="utf-8")
    with self.assertRaises(bootstrap.EmbeddingStoreTargetTypeError):
      bootstrap.bootstrap_modern_embedding_store(self.request("file-target"))

  def test_14_symlink_policy_is_rejected_before_writes(self):
    with patch.object(bootstrap, "_contains_symlink", return_value=True), \
        patch.object(bootstrap, "_write_store_files") as writer:
      with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
        bootstrap.bootstrap_modern_embedding_store(self.request("link"))
    writer.assert_not_called()

  def test_15_existing_ada_store_is_incompatible(self):
    target = self.root / "ada"
    self.seed_memory_files(target, LEGACY_ADA_002_MANIFEST)
    with self.assertRaises(bootstrap.EmbeddingStoreIncompatibleError):
      bootstrap.bootstrap_modern_embedding_store(self.request("ada"))

  def test_16_manifestless_store_is_unknown(self):
    target = self.root / "unknown"
    self.seed_memory_files(target)
    with self.assertRaises(bootstrap.EmbeddingStoreUnknownError):
      bootstrap.bootstrap_modern_embedding_store(self.request("unknown"))

  def test_17_partial_store_is_rejected(self):
    target = self.root / "partial"
    target.mkdir()
    self.write_json(
      target / EMBEDDING_MANIFEST_FILENAME,
      TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.to_dict())
    with self.assertRaises(
        bootstrap.EmbeddingStorePartialInitializationError):
      bootstrap.bootstrap_modern_embedding_store(self.request("partial"))

  def test_18_real_storage_root_is_protected(self):
    target = bootstrap._repository_protected_roots()[0]
    request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
      target, allowed_parent=target.parent)
    with patch.object(bootstrap, "_write_store_files") as writer:
      with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
        bootstrap.bootstrap_modern_embedding_store(request)
    writer.assert_not_called()

  def test_19_real_temp_storage_root_is_protected(self):
    target = bootstrap._repository_protected_roots()[1]
    request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
      target, allowed_parent=target.parent)
    with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
      bootstrap.bootstrap_modern_embedding_store(request)

  def test_20_target_inside_existing_store_is_rejected(self):
    store = self.root / "historical-store"
    store.mkdir()
    self.write_json(store / bootstrap.EMBEDDINGS_FILENAME, {})
    request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
      store / "nested", allowed_parent=store)
    with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
      bootstrap.bootstrap_modern_embedding_store(request)

  def test_21_filesystem_root_is_rejected(self):
    root = Path(Path.cwd().anchor)
    request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
      root, allowed_parent=root)
    with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
      bootstrap.bootstrap_modern_embedding_store(request)

  def test_22_missing_parent_is_rejected(self):
    missing = self.root / "missing-parent"
    request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
      missing / "store", allowed_parent=missing)
    with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
      bootstrap.bootstrap_modern_embedding_store(request)

  def test_23_relative_target_is_normalized_safely(self):
    with patch.object(Path, "cwd", return_value=self.root):
      request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
        Path("relative-store"), allowed_parent=Path("."))
      result = bootstrap.bootstrap_modern_embedding_store(request)
    self.assertEqual((self.root / "relative-store").as_posix(),
                     result.target_path)

  def test_24_second_call_does_not_rewrite_files(self):
    target, unused = self.create()
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in target.iterdir()}
    with self.assertRaises(bootstrap.EmbeddingStoreAlreadyExistsError):
      bootstrap.bootstrap_modern_embedding_store(self.request())
    after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
             for path in target.iterdir()}
    self.assertEqual(before, after)

  def test_25_nonempty_target_is_never_merged(self):
    target = self.root / "no-merge"
    target.mkdir()
    original = target / "original.txt"
    original.write_bytes(b"preserve exactly")
    with self.assertRaises(bootstrap.EmbeddingStoreNotEmptyError):
      bootstrap.bootstrap_modern_embedding_store(self.request("no-merge"))
    self.assertEqual(["original.txt"], [path.name for path in target.iterdir()])
    self.assertEqual(b"preserve exactly", original.read_bytes())

  def test_26_failure_before_write_leaves_no_target(self):
    with patch.object(
        bootstrap, "_write_store_files", side_effect=RuntimeError("injected")):
      with self.assertRaises(bootstrap.EmbeddingStoreAtomicWriteError):
        bootstrap.bootstrap_modern_embedding_store(self.request("before-write"))
    self.assertFalse((self.root / "before-write").exists())

  def test_27_failure_during_write_cleans_staging(self):
    original = bootstrap._write_json_file
    calls = []
    def fail_second(path, value):
      calls.append(path)
      if len(calls) == 2:
        raise OSError("injected")
      original(path, value)
    with patch.object(bootstrap, "_write_json_file", side_effect=fail_second):
      with self.assertRaises(bootstrap.EmbeddingStoreAtomicWriteError):
        bootstrap.bootstrap_modern_embedding_store(self.request("during-write"))
    self.assertFalse((self.root / "during-write").exists())
    self.assertEqual((), self.temporary_bootstrap_entries())

  def test_28_failure_before_validation_leaves_no_target(self):
    with patch.object(
        bootstrap, "_validate_complete_store",
        side_effect=bootstrap.EmbeddingStorePartialInitializationError(
          "injected")):
      with self.assertRaises(
          bootstrap.EmbeddingStorePartialInitializationError):
        bootstrap.bootstrap_modern_embedding_store(
          self.request("before-validation"))
    self.assertFalse((self.root / "before-validation").exists())

  def test_29_failure_before_rename_leaves_no_target(self):
    with patch.object(
        bootstrap, "_rename_staged_directory",
        side_effect=RuntimeError("injected")):
      with self.assertRaises(bootstrap.EmbeddingStoreAtomicWriteError):
        bootstrap.bootstrap_modern_embedding_store(self.request("before-rename"))
    self.assertFalse((self.root / "before-rename").exists())

  def test_30_rename_os_error_is_typed_and_cleaned(self):
    with patch.object(
        bootstrap, "_rename_staged_directory",
        side_effect=OSError("injected")):
      with self.assertRaises(bootstrap.EmbeddingStoreAtomicWriteError):
        bootstrap.bootstrap_modern_embedding_store(self.request("rename-error"))
    self.assertFalse((self.root / "rename-error").exists())
    self.assertEqual((), self.temporary_bootstrap_entries())

  def test_31_concurrent_target_is_preserved(self):
    target = self.root / "concurrent-target"
    def create_competing_target(staging, destination):
      destination.mkdir()
      (destination / "owner.txt").write_text("other", encoding="utf-8")
      raise FileExistsError("injected race")
    with patch.object(
        bootstrap, "_rename_staged_directory",
        side_effect=create_competing_target):
      with self.assertRaises(bootstrap.EmbeddingStoreAlreadyExistsError):
        bootstrap.bootstrap_modern_embedding_store(
          self.request("concurrent-target"))
    self.assertEqual("other", (target / "owner.txt").read_text(encoding="utf-8"))
    self.assertEqual((), self.temporary_bootstrap_entries())

  def test_32_two_concurrent_bootstraps_have_one_winner(self):
    original = bootstrap._rename_staged_directory
    barrier = threading.Barrier(2)
    def synchronized_rename(staging, destination):
      barrier.wait(timeout=10)
      original(staging, destination)
    def run():
      try:
        bootstrap.bootstrap_modern_embedding_store(self.request("race"))
        return "SUCCESS"
      except bootstrap.EmbeddingStoreAlreadyExistsError:
        return "EXISTS"
    with patch.object(
        bootstrap, "_rename_staged_directory",
        side_effect=synchronized_rename), ThreadPoolExecutor(2) as executor:
      futures = (executor.submit(run), executor.submit(run))
      outcomes = sorted(future.result(timeout=20) for future in futures)
    self.assertEqual(["EXISTS", "SUCCESS"], outcomes)
    bootstrap._validate_complete_store(
      self.root / "race", TEXT_EMBEDDING_3_SMALL_1536_MANIFEST)

  def test_33_temporary_directories_are_removed_after_failure(self):
    with patch.object(
        bootstrap, "_write_store_files", side_effect=OSError("injected")):
      with self.assertRaises(bootstrap.EmbeddingStoreAtomicWriteError):
        bootstrap.bootstrap_modern_embedding_store(self.request("cleanup"))
    self.assertEqual((), self.temporary_bootstrap_entries())

  def test_34_created_file_set_is_exact(self):
    target, result = self.create()
    self.assertEqual(bootstrap.CREATED_FILES, result.created_files)
    self.assertEqual(set(bootstrap.CREATED_FILES), {
      path.name for path in target.iterdir()})

  def test_35_no_provider_embedding_sdk_or_network_is_called(self):
    with patch("socket.getaddrinfo",
               side_effect=AssertionError("DNS reached")) as dns, patch(
        "socket.create_connection",
        side_effect=AssertionError("network reached")) as connection, patch(
        "persona.prompt_template.llm_provider.create_llm_provider",
        side_effect=AssertionError("provider reached")) as provider, patch(
        "persona.prompt_template.llm_provider.embedding",
        side_effect=AssertionError("embedding reached")) as embedding:
      self.create("isolated")
    dns.assert_not_called()
    connection.assert_not_called()
    provider.assert_not_called()
    embedding.assert_not_called()

  def test_36_telemetry_cost_inputs_and_attempts_remain_unchanged(self):
    before_events = get_telemetry()
    before_stats = get_embedding_cache_stats()
    pricing = PricingSnapshot("m4-audit", 1, "USD", "stable", (), "synthetic")
    before_ledger = build_cost_ledger_records(before_events, pricing)
    self.create("measurement-free")
    self.assertEqual(before_events, get_telemetry())
    self.assertEqual(before_stats, get_embedding_cache_stats())
    self.assertEqual(
      before_ledger, build_cost_ledger_records(get_telemetry(), pricing))
    self.assertEqual(0, get_embedding_cache_stats().logical_embedding_requests)
    self.assertEqual(0, get_embedding_cache_stats().physical_embedding_attempts)

  def isolated_imported_modules(self):
    script = (
      "import importlib,json,socket,sys\n"
      f"sys.path.insert(0,{str(BACKEND_SERVER)!r})\n"
      "def blocked(*args,**kwargs): raise AssertionError('network')\n"
      "socket.getaddrinfo=blocked\n"
      "socket.create_connection=blocked\n"
      "importlib.import_module('persona.prompt_template.embedding_store_bootstrap')\n"
      "print(json.dumps(sorted(sys.modules)))\n")
    completed = subprocess.run(
      [sys.executable, "-I", "-c", script], check=True,
      capture_output=True, text=True, timeout=30)
    return json.loads(completed.stdout)

  def test_37_import_does_not_load_cognitive_or_simulation_modules(self):
    modules = self.isolated_imported_modules()
    forbidden = {
      "gpt_" + "structure", "associative_" + "memory", "planning",
      "reflection", "conversation", "retrieve", "reverie"}
    self.assertFalse(any(
      name.rsplit(".", 1)[-1] in forbidden for name in modules))

  def test_38_import_does_not_load_utils(self):
    modules = self.isolated_imported_modules()
    credential_module = "reverie.backend_server." + "utils"
    self.assertFalse(any(
      name in ("utils", credential_module) for name in modules))

  def test_39_real_storage_snapshot_is_unchanged_by_guard(self):
    target = bootstrap._repository_protected_roots()[0]
    before = target.stat().st_mtime_ns
    request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
      target, allowed_parent=target.parent)
    with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
      bootstrap.bootstrap_modern_embedding_store(request)
    self.assertEqual(before, target.stat().st_mtime_ns)

  def test_40_two_bootstraps_have_identical_persistent_bytes(self):
    first, first_result = self.create("deterministic-a")
    second, second_result = self.create("deterministic-b")
    first_bytes = {path.name: path.read_bytes() for path in first.iterdir()}
    second_bytes = {path.name: path.read_bytes() for path in second.iterdir()}
    self.assertEqual(first_bytes, second_bytes)
    self.assertEqual(first_result.created_files, second_result.created_files)
    self.assertEqual(first_result.manifest, second_result.manifest)

  def test_41_persistent_files_contain_no_volatile_metadata(self):
    target, unused = self.create("no-volatile")
    combined = b"".join(path.read_bytes() for path in target.iterdir())
    for field in (b"timestamp", b"username", b"hostname", b"git_hash",
                  b"api_key"):
      self.assertNotIn(field, combined.lower())

  def test_42_request_is_immutable_and_rejects_ada(self):
    request = self.request("immutable-request")
    with self.assertRaises(FrozenInstanceError):
      request.bootstrap_version = 2
    with self.assertRaises(bootstrap.EmbeddingStoreIncompatibleError):
      self.request("ada-request", manifest=LEGACY_ADA_002_MANIFEST)

  def test_43_permission_failure_is_typed(self):
    with patch.object(tempfile, "mkdtemp", side_effect=PermissionError("denied")):
      with self.assertRaises(bootstrap.EmbeddingStoreAtomicWriteError):
        bootstrap.bootstrap_modern_embedding_store(self.request("permission"))
    self.assertFalse((self.root / "permission").exists())

  def test_44_malformed_existing_manifest_is_partial(self):
    target = self.root / "malformed"
    target.mkdir()
    (target / EMBEDDING_MANIFEST_FILENAME).write_text("{", encoding="utf-8")
    with self.assertRaises(
        bootstrap.EmbeddingStorePartialInitializationError):
      bootstrap.bootstrap_modern_embedding_store(self.request("malformed"))

  def test_45_complete_store_with_extra_file_is_not_merged(self):
    target, unused = self.create("extra")
    (target / "extra.txt").write_text("preserve", encoding="utf-8")
    with self.assertRaises(bootstrap.EmbeddingStoreNotEmptyError):
      bootstrap.bootstrap_modern_embedding_store(self.request("extra"))
    self.assertEqual("preserve", (target / "extra.txt").read_text(encoding="utf-8"))

  def test_46_parent_traversal_is_rejected(self):
    request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
      Path("safe") / ".." / "escaped", allowed_parent=self.root)
    with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
      bootstrap.bootstrap_modern_embedding_store(request)

  def test_47_dot_target_is_rejected(self):
    request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
      Path("."), allowed_parent=self.root)
    with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
      bootstrap.bootstrap_modern_embedding_store(request)

  def test_48_deep_target_inside_existing_store_is_rejected(self):
    store = self.root / "outer-store"
    allowed_parent = store / "nested-parent"
    allowed_parent.mkdir(parents=True)
    self.write_json(store / bootstrap.EMBEDDINGS_FILENAME, {})
    request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
      allowed_parent / "target", allowed_parent=allowed_parent)
    with self.assertRaises(bootstrap.EmbeddingStoreUnsafePathError):
      bootstrap.bootstrap_modern_embedding_store(request)

  def test_49_storage_root_as_allowed_parent_is_rejected_before_writes(self):
    parent = bootstrap._repository_protected_roots()[0]
    self.assert_protected_request_rejected(
      bootstrap.ModernEmbeddingStoreBootstrapRequest(
        parent / "m4r-never-create", allowed_parent=parent))

  def test_50_temp_storage_root_as_allowed_parent_is_rejected_before_writes(self):
    parent = bootstrap._repository_protected_roots()[1]
    self.assert_protected_request_rejected(
      bootstrap.ModernEmbeddingStoreBootstrapRequest(
        parent / "m4r-never-create", allowed_parent=parent))

  def test_51_storage_descendant_as_allowed_parent_is_rejected(self):
    root = bootstrap._repository_protected_roots()[0]
    parent = root / "m4r-nonexistent-parent"
    self.assert_protected_request_rejected(
      bootstrap.ModernEmbeddingStoreBootstrapRequest(
        parent / "store", allowed_parent=parent))

  def test_52_temp_storage_descendant_as_allowed_parent_is_rejected(self):
    root = bootstrap._repository_protected_roots()[1]
    parent = root / "m4r-nonexistent-parent"
    self.assert_protected_request_rejected(
      bootstrap.ModernEmbeddingStoreBootstrapRequest(
        parent / "store", allowed_parent=parent))

  def test_53_implicit_storage_parent_is_rejected(self):
    parent = bootstrap._repository_protected_roots()[0]
    self.assert_protected_request_rejected(
      bootstrap.ModernEmbeddingStoreBootstrapRequest(
        parent / "m4r-implicit-never-create"))

  def test_54_implicit_temp_storage_parent_is_rejected(self):
    parent = bootstrap._repository_protected_roots()[1]
    self.assert_protected_request_rejected(
      bootstrap.ModernEmbeddingStoreBootstrapRequest(
        parent / "m4r-implicit-never-create"))

  def test_55_relative_storage_parent_is_rejected(self):
    repository = BACKEND_SERVER.parents[1]
    parent = Path("environment/frontend_server/storage")
    with patch.object(Path, "cwd", return_value=repository):
      self.assert_protected_request_rejected(
        bootstrap.ModernEmbeddingStoreBootstrapRequest(
          parent / "m4r-relative-never-create", allowed_parent=parent))

  def test_56_relative_temp_storage_parent_is_rejected(self):
    repository = BACKEND_SERVER.parents[1]
    parent = Path(r"environment\frontend_server\temp_storage")
    with patch.object(Path, "cwd", return_value=repository):
      self.assert_protected_request_rejected(
        bootstrap.ModernEmbeddingStoreBootstrapRequest(
          parent / "m4r-relative-never-create", allowed_parent=parent))

  def test_57_storage_parent_case_variant_is_rejected_on_windows(self):
    parent = Path(str(bootstrap._repository_protected_roots()[0]).swapcase())
    self.assert_protected_request_rejected(
      bootstrap.ModernEmbeddingStoreBootstrapRequest(
        parent / "m4r-case-never-create", allowed_parent=parent))

  def test_58_temp_storage_parent_case_variant_is_rejected_on_windows(self):
    parent = Path(str(bootstrap._repository_protected_roots()[1]).swapcase())
    self.assert_protected_request_rejected(
      bootstrap.ModernEmbeddingStoreBootstrapRequest(
        parent / "m4r-case-never-create", allowed_parent=parent))

  def test_59_storage_parent_forward_slash_variant_is_rejected(self):
    root = bootstrap._repository_protected_roots()[0]
    parent = Path(str(root).replace("\\", "/"))
    self.assert_protected_request_rejected(
      bootstrap.ModernEmbeddingStoreBootstrapRequest(
        parent / "m4r-slash-never-create", allowed_parent=parent))

  def test_60_temp_storage_parent_backslash_variant_is_rejected(self):
    root = bootstrap._repository_protected_roots()[1]
    parent = Path(str(root).replace("/", "\\"))
    self.assert_protected_request_rejected(
      bootstrap.ModernEmbeddingStoreBootstrapRequest(
        parent / "m4r-slash-never-create", allowed_parent=parent))

  def test_61_storage_traversal_variant_is_rejected(self):
    repository = BACKEND_SERVER.parents[1]
    parent = Path("safe-parent/../environment/frontend_server/storage")
    with patch.object(Path, "cwd", return_value=repository):
      self.assert_protected_request_rejected(
        bootstrap.ModernEmbeddingStoreBootstrapRequest(
          parent / "m4r-traversal-never-create", allowed_parent=parent))

  def test_62_temp_storage_traversal_variant_is_rejected(self):
    repository = BACKEND_SERVER.parents[1]
    parent = Path("safe-parent/../environment/frontend_server/temp_storage")
    with patch.object(Path, "cwd", return_value=repository):
      self.assert_protected_request_rejected(
        bootstrap.ModernEmbeddingStoreBootstrapRequest(
          parent / "m4r-traversal-never-create", allowed_parent=parent))

  def assert_symlink_parent_rejected(self, protected_root, link_name):
    link = self.root / link_name
    try:
      os.symlink(protected_root, link, target_is_directory=True)
    except OSError:
      request = bootstrap.ModernEmbeddingStoreBootstrapRequest(
        link / "store", allowed_parent=link)
      with patch.object(bootstrap, "_contains_symlink", return_value=True):
        self.assert_protected_request_rejected(request)
    else:
      self.assert_protected_request_rejected(
        bootstrap.ModernEmbeddingStoreBootstrapRequest(
          link / "store", allowed_parent=link))

  def test_63_symlink_parent_to_storage_is_rejected(self):
    self.assert_symlink_parent_rejected(
      bootstrap._repository_protected_roots()[0], "storage-link")

  def test_64_symlink_parent_to_temp_storage_is_rejected(self):
    self.assert_symlink_parent_rejected(
      bootstrap._repository_protected_roots()[1], "temp-storage-link")

  def test_65_component_comparison_does_not_block_storage_backup(self):
    storage = bootstrap._repository_protected_roots()[0]
    lookalike = Path(str(storage) + "_backup")
    self.assertFalse(bootstrap._is_same_or_descendant(lookalike, storage))

  def test_66_valid_absolute_temporary_parent_still_bootstraps(self):
    target, unused = self.create("m4r-valid-absolute")
    self.assertTrue(target.is_dir())

  def test_67_directory_fsync_abstraction_is_invoked(self):
    with patch.object(
        bootstrap, "_fsync_directory",
        wraps=bootstrap._fsync_directory) as directory_fsync:
      self.create("m4r-directory-fsync")
    self.assertEqual(2, directory_fsync.call_count)

  def test_68_directory_fsync_failure_leaves_no_partial_target(self):
    target = self.root / "m4r-fsync-failure"
    with patch.object(
        bootstrap, "_fsync_directory", side_effect=OSError("injected")):
      with self.assertRaises(bootstrap.EmbeddingStoreAtomicWriteError):
        bootstrap.bootstrap_modern_embedding_store(
          self.request("m4r-fsync-failure"))
    self.assertFalse(target.exists())
    self.assertEqual((), self.temporary_bootstrap_entries())

  def test_69_each_persistent_file_is_fsynced(self):
    with patch.object(os, "fsync", wraps=os.fsync) as file_fsync:
      self.create("m4r-file-fsync")
    self.assertGreaterEqual(file_fsync.call_count, len(bootstrap.CREATED_FILES))

  def test_70_post_validation_failure_leaves_valid_complete_target(self):
    target = self.root / "m4r-post-validation"
    original = bootstrap._validate_complete_store
    calls = []
    def fail_second(store_path, manifest):
      calls.append(store_path)
      if len(calls) == 2:
        raise bootstrap.EmbeddingStorePartialInitializationError(
          "injected post-validation failure")
      return original(store_path, manifest)
    with patch.object(
        bootstrap, "_validate_complete_store", side_effect=fail_second):
      with self.assertRaises(
          bootstrap.EmbeddingStorePartialInitializationError):
        bootstrap.bootstrap_modern_embedding_store(
          self.request("m4r-post-validation"))
    self.assertEqual(set(bootstrap.CREATED_FILES), {
      path.name for path in target.iterdir()})
    for path in target.iterdir():
      json.loads(path.read_text(encoding="utf-8"))
    loaded = original(target, TEXT_EMBEDDING_3_SMALL_1536_MANIFEST)
    self.assertEqual(TEXT_EMBEDDING_3_SMALL_1536_MANIFEST, loaded.manifest)
    from persona.memory_structures.associative_memory import AssociativeMemory
    try:
      memory = AssociativeMemory(
        str(target), legacy_assumption_allowed=False,
        runtime_embedding_manifest=TEXT_EMBEDDING_3_SMALL_1536_MANIFEST)
      self.assertEqual((0, 0, 0, 0), (
        len(memory.id_to_node), len(memory.embeddings),
        len(memory.kw_strength_event), len(memory.kw_strength_thought)))
    finally:
      reset_runtime_embedding_manifest()
    self.assertEqual((), self.temporary_bootstrap_entries())


if __name__ == "__main__":
  unittest.main()
