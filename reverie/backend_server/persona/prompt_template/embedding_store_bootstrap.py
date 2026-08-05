"""Atomic bootstrap for a new, explicitly modern embedding store.

This module creates only the persistent associative-memory container.  It
does not create a simulation, activate a provider, or generate embeddings.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile

from persona.memory_structures.embedding_space import (
  EMBEDDING_MANIFEST_FILENAME,
  EmbeddingManifestError,
  EmbeddingSpaceManifest,
  EmbeddingSpaceMismatchError,
  assert_same_embedding_space,
  read_embedding_manifest,
)
from persona.prompt_template.embedding_runtime import (
  TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
  build_modern_embedding_runtime_config,
  validate_embedding_store_for_runtime,
)


BOOTSTRAP_VERSION = 1
EMBEDDINGS_FILENAME = "embeddings.json"
NODES_FILENAME = "nodes.json"
KEYWORD_STRENGTH_FILENAME = "kw_strength.json"
CREATED_FILES = tuple(sorted((
  EMBEDDING_MANIFEST_FILENAME,
  EMBEDDINGS_FILENAME,
  KEYWORD_STRENGTH_FILENAME,
  NODES_FILENAME,
)))
_STORE_MARKERS = frozenset(CREATED_FILES)
_TEMP_MARKER = ".__bootstrap_tmp_"


class EmbeddingStoreBootstrapError(RuntimeError):
  """Base class for safe modern-store bootstrap failures."""


class EmbeddingStoreAlreadyExistsError(EmbeddingStoreBootstrapError):
  """The requested target already exists or was created concurrently."""


class EmbeddingStoreNotEmptyError(EmbeddingStoreBootstrapError):
  """The target contains unrelated or additional content."""


class EmbeddingStoreIncompatibleError(EmbeddingStoreBootstrapError):
  """The target declares a different embedding space."""


class EmbeddingStoreUnknownError(EmbeddingStoreBootstrapError):
  """The target resembles a store but has no declared embedding space."""


class EmbeddingStorePartialInitializationError(EmbeddingStoreBootstrapError):
  """The target contains an incomplete or malformed store."""


class EmbeddingStoreTargetTypeError(EmbeddingStoreBootstrapError):
  """The target exists but is not a directory."""


class EmbeddingStoreUnsafePathError(EmbeddingStoreBootstrapError):
  """The target or its parent violates the V0 path-safety policy."""


class EmbeddingStoreAtomicWriteError(EmbeddingStoreBootstrapError):
  """The staged store could not be written or installed atomically."""


@dataclass(frozen=True)
class ModernEmbeddingStoreBootstrapRequest:
  target_path: Path
  manifest: EmbeddingSpaceManifest = TEXT_EMBEDDING_3_SMALL_1536_MANIFEST
  bootstrap_version: int = BOOTSTRAP_VERSION
  allow_existing_empty_directory: bool = False
  allowed_parent: Path | None = None

  def __post_init__(self):
    object.__setattr__(self, "target_path", Path(self.target_path))
    if self.allowed_parent is not None:
      object.__setattr__(self, "allowed_parent", Path(self.allowed_parent))
    if not isinstance(self.manifest, EmbeddingSpaceManifest):
      raise EmbeddingStoreIncompatibleError(
        "manifest must be an EmbeddingSpaceManifest")
    try:
      assert_same_embedding_space(
        TEXT_EMBEDDING_3_SMALL_1536_MANIFEST, self.manifest,
        "modern embedding store bootstrap")
    except EmbeddingSpaceMismatchError as error:
      raise EmbeddingStoreIncompatibleError(
        "manifest is not the canonical modern embedding space") from error
    if type(self.bootstrap_version) is not int or self.bootstrap_version != 1:
      raise EmbeddingStoreBootstrapError(
        "bootstrap_version must be the supported version 1")
    if type(self.allow_existing_empty_directory) is not bool:
      raise EmbeddingStoreBootstrapError(
        "allow_existing_empty_directory must be boolean")


@dataclass(frozen=True)
class ModernEmbeddingStoreBootstrapResult:
  target_path: str
  manifest_path: str
  created_files: tuple[str, ...]
  manifest: EmbeddingSpaceManifest
  bootstrap_version: int


def modern_embedding_store_bootstrap_result_to_dict(result):
  """Return a deterministic JSON-safe representation of a bootstrap result."""
  if not isinstance(result, ModernEmbeddingStoreBootstrapResult):
    raise TypeError("result must be ModernEmbeddingStoreBootstrapResult")
  return {
    "target_path": result.target_path,
    "manifest_path": result.manifest_path,
    "created_files": list(result.created_files),
    "manifest": result.manifest.to_dict(),
    "bootstrap_version": result.bootstrap_version,
  }


def _absolute_without_resolving(path):
  path = Path(path).expanduser()
  return path if path.is_absolute() else Path.cwd() / path


def _contains_symlink(path):
  current = _absolute_without_resolving(path)
  while True:
    if current.is_symlink():
      return True
    if current.parent == current:
      return False
    current = current.parent


def _repository_protected_roots():
  repository = Path(__file__).resolve().parents[4]
  frontend = repository / "environment" / "frontend_server"
  return (
    (frontend / "storage").resolve(strict=False),
    (frontend / "temp_storage").resolve(strict=False),
  )


def _is_same_or_descendant(candidate, protected_root):
  """Compare normalized paths by components, case-insensitively on Windows."""
  try:
    candidate.relative_to(protected_root)
  except ValueError:
    return False
  return True


def _has_store_ancestor(target):
  current = target.parent
  while True:
    if any((current / name).exists() for name in _STORE_MARKERS):
      return True
    if current.parent == current:
      return False
    current = current.parent


def _normalize_and_validate_paths(request):
  lexical_dot_target = request.target_path == Path(".")
  lexical_traversal = (
    ".." in request.target_path.parts
    or (request.allowed_parent is not None
        and ".." in request.allowed_parent.parts))
  raw_target = _absolute_without_resolving(request.target_path)
  raw_parent = _absolute_without_resolving(
    request.allowed_parent if request.allowed_parent is not None
    else raw_target.parent)
  if _contains_symlink(raw_target) or _contains_symlink(raw_parent):
    raise EmbeddingStoreUnsafePathError("symlink paths are unsupported in V0")
  target = raw_target.resolve(strict=False)
  allowed_parent = raw_parent.resolve(strict=False)
  protected_roots = _repository_protected_roots()
  if any(_is_same_or_descendant(allowed_parent, protected_root)
         for protected_root in protected_roots):
    raise EmbeddingStoreUnsafePathError(
      "allowed parent is inside a protected runtime storage root")
  if any(_is_same_or_descendant(target, protected_root)
         for protected_root in protected_roots):
    raise EmbeddingStoreUnsafePathError(
      "target is inside a protected runtime storage root")
  if lexical_dot_target or lexical_traversal:
    raise EmbeddingStoreUnsafePathError(
      "dot targets and parent traversal components are unsupported in V0")
  if (not allowed_parent.exists() or not allowed_parent.is_dir()
      or allowed_parent.parent == allowed_parent):
    raise EmbeddingStoreUnsafePathError(
      "allowed parent must be an existing non-root directory")
  if target.parent != allowed_parent or target == allowed_parent:
    raise EmbeddingStoreUnsafePathError(
      "target must be a direct child of the allowed parent")
  if target.parent == target:
    raise EmbeddingStoreUnsafePathError("filesystem roots are not targets")
  if _has_store_ancestor(target):
    raise EmbeddingStoreUnsafePathError(
      "target cannot be nested inside an existing embedding store")
  return target, allowed_parent


def _classify_existing_target(target, request):
  if target.is_symlink():
    raise EmbeddingStoreUnsafePathError("symlink targets are unsupported in V0")
  if not target.is_dir():
    raise EmbeddingStoreTargetTypeError("target exists and is not a directory")
  children = tuple(target.iterdir())
  if not children:
    if request.allow_existing_empty_directory:
      return "EXISTING_EMPTY_ALLOWED"
    raise EmbeddingStoreAlreadyExistsError(
      "existing empty directory requires explicit opt-in")

  names = {item.name for item in children}
  manifest_path = target / EMBEDDING_MANIFEST_FILENAME
  if manifest_path.is_file():
    try:
      manifest = read_embedding_manifest(manifest_path)
    except EmbeddingManifestError as error:
      raise EmbeddingStorePartialInitializationError(
        "existing store has an invalid manifest") from error
    try:
      assert_same_embedding_space(
        request.manifest, manifest, "existing embedding store")
    except EmbeddingSpaceMismatchError as error:
      raise EmbeddingStoreIncompatibleError(
        "existing store declares an incompatible embedding space") from error
    if names == set(CREATED_FILES) and all(item.is_file() for item in children):
      try:
        validate_embedding_store_for_runtime(
          target, build_modern_embedding_runtime_config())
      except Exception as error:
        raise EmbeddingStorePartialInitializationError(
          "existing modern store is incomplete or malformed") from error
      raise EmbeddingStoreAlreadyExistsError(
        "modern embedding store already exists")
    if set(CREATED_FILES).issubset(names):
      raise EmbeddingStoreNotEmptyError(
        "existing modern store contains additional content")
    raise EmbeddingStorePartialInitializationError(
      "existing modern store is partially initialized")

  memory_files = {
    EMBEDDINGS_FILENAME, NODES_FILENAME, KEYWORD_STRENGTH_FILENAME}
  if memory_files.issubset(names):
    raise EmbeddingStoreUnknownError(
      "existing embedding store has no declared manifest")
  if names & _STORE_MARKERS:
    raise EmbeddingStorePartialInitializationError(
      "existing target contains a partial embedding store")
  raise EmbeddingStoreNotEmptyError("existing target is not empty")


def _write_json_file(path, value):
  with open(path, "x", encoding="utf-8", newline="\n") as outfile:
    json.dump(value, outfile, indent=2)
    outfile.write("\n")
    outfile.flush()
    os.fsync(outfile.fileno())


def _write_store_files(staging_path, manifest):
  _write_json_file(
    staging_path / EMBEDDING_MANIFEST_FILENAME, manifest.to_dict())
  _write_json_file(staging_path / EMBEDDINGS_FILENAME, {})
  _write_json_file(staging_path / NODES_FILENAME, {})
  _write_json_file(staging_path / KEYWORD_STRENGTH_FILENAME, {
    "kw_strength_event": {},
    "kw_strength_thought": {},
  })


def _validate_complete_store(store_path, manifest):
  children = tuple(store_path.iterdir())
  if ({item.name for item in children} != set(CREATED_FILES)
      or not all(item.is_file() for item in children)):
    raise EmbeddingStorePartialInitializationError(
      "staged embedding store has an unexpected file set")
  actual = read_embedding_manifest(store_path / EMBEDDING_MANIFEST_FILENAME)
  assert_same_embedding_space(manifest, actual, "staged embedding store")
  loaded = validate_embedding_store_for_runtime(
    store_path, build_modern_embedding_runtime_config())
  assert_same_embedding_space(manifest, loaded.manifest, "preflight result")
  return loaded


def _rename_staged_directory(staging_path, target):
  os.rename(staging_path, target)


def _remove_temporary_directory(staging_path, allowed_parent, target_name):
  if (staging_path.parent == allowed_parent
      and staging_path.name.startswith(f".{target_name}{_TEMP_MARKER}")):
    shutil.rmtree(staging_path, ignore_errors=True)


def _fsync_directory(path):
  if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
    return
  descriptor = None
  try:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
  except OSError:
    pass
  finally:
    if descriptor is not None:
      os.close(descriptor)


def bootstrap_modern_embedding_store(request):
  """Create and atomically install a minimal modern embedding store.

  V0 installs a direct child of an existing, explicitly allowed parent.  On
  Windows, ``os.rename`` supplies the required no-replace directory rename.
  """
  if not isinstance(request, ModernEmbeddingStoreBootstrapRequest):
    raise TypeError("request must be ModernEmbeddingStoreBootstrapRequest")
  target, allowed_parent = _normalize_and_validate_paths(request)
  existing_empty_allowed = False
  if target.exists() or target.is_symlink():
    existing_empty_allowed = (
      _classify_existing_target(target, request) == "EXISTING_EMPTY_ALLOWED")

  prefix = f".{target.name}{_TEMP_MARKER}"
  staging_path = None
  try:
    staging_path = Path(tempfile.mkdtemp(prefix=prefix, dir=allowed_parent))
    _write_store_files(staging_path, request.manifest)
    _validate_complete_store(staging_path, request.manifest)
    _fsync_directory(staging_path)

    if existing_empty_allowed:
      try:
        if (target.is_symlink() or not target.is_dir()
            or any(target.iterdir())):
          _classify_existing_target(target, request)
        os.rmdir(target)
      except EmbeddingStoreBootstrapError:
        raise
      except OSError as error:
        raise EmbeddingStoreAlreadyExistsError(
          "existing empty target changed during bootstrap") from error
    elif target.exists() or target.is_symlink():
      _classify_existing_target(target, request)

    try:
      _rename_staged_directory(staging_path, target)
      staging_path = None
    except OSError as error:
      if target.exists() or target.is_symlink():
        raise EmbeddingStoreAlreadyExistsError(
          "target was created concurrently") from error
      raise EmbeddingStoreAtomicWriteError(
        "atomic store installation failed") from error
    _fsync_directory(allowed_parent)
    _validate_complete_store(target, request.manifest)
  except EmbeddingStoreBootstrapError:
    raise
  except Exception as error:
    raise EmbeddingStoreAtomicWriteError(
      "modern embedding store bootstrap failed") from error
  finally:
    if staging_path is not None:
      _remove_temporary_directory(
        staging_path, allowed_parent, target.name)

  target_text = target.as_posix()
  return ModernEmbeddingStoreBootstrapResult(
    target_path=target_text,
    manifest_path=(target / EMBEDDING_MANIFEST_FILENAME).as_posix(),
    created_files=CREATED_FILES,
    manifest=request.manifest,
    bootstrap_version=request.bootstrap_version,
  )
