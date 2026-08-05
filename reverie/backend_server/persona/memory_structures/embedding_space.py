"""Persistent embedding-space contract and validation for associative memory.

Vector length is not an embedding-space identity.  Persisted vectors may only
be compared when their provider, model, semantic space, text normalization,
and dimensions all match.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
import warnings


EMBEDDING_MANIFEST_FILENAME = "embedding_manifest.json"
SUPPORTED_SCHEMA_VERSION = 1
EMBEDDING_SPACE_FIELDS = (
  "provider",
  "model",
  "dimensions",
  "embedding_space_version",
  "normalization_version",
)


class EmbeddingManifestError(ValueError):
  """The embedding manifest is missing, malformed, or unsupported."""


class EmbeddingSpaceMismatchError(EmbeddingManifestError):
  """Declared and expected embedding spaces are not identical."""


class EmbeddingVectorValidationError(ValueError):
  """A persisted embedding vector violates its declared contract."""


class EmbeddingReferenceError(ValueError):
  """Nodes and persisted embedding keys are inconsistent."""


class LegacyEmbeddingSpaceWarning(UserWarning):
  """A recognized historical store is using a virtual legacy manifest."""


class EmbeddingSpaceClassification(Enum):
  DECLARED = "DECLARED"
  LEGACY_ASSUMED = "LEGACY_ASSUMED"
  UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EmbeddingSpaceManifest:
  schema_version: int
  provider: str
  model: str
  dimensions: int
  embedding_space_version: str
  normalization_version: str

  @classmethod
  def from_dict(cls, value, path="<memory>"):
    if not isinstance(value, dict):
      raise EmbeddingManifestError(
        f"{path}: manifest must be a JSON object; actual={type(value).__name__}")
    required = {"schema_version", *EMBEDDING_SPACE_FIELDS}
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing:
      raise EmbeddingManifestError(
        f"{path}: manifest missing fields; actual={missing}")
    if extra:
      raise EmbeddingManifestError(
        f"{path}: manifest has unsupported fields; actual={extra}")
    if type(value["schema_version"]) is not int:
      raise EmbeddingManifestError(
        f"{path}: field=schema_version expected=int actual="
        f"{type(value['schema_version']).__name__}")
    if value["schema_version"] != SUPPORTED_SCHEMA_VERSION:
      raise EmbeddingManifestError(
        f"{path}: field=schema_version expected={SUPPORTED_SCHEMA_VERSION} "
        f"actual={value['schema_version']}")
    for field in ("provider", "model", "embedding_space_version",
                  "normalization_version"):
      if not isinstance(value[field], str) or not value[field]:
        raise EmbeddingManifestError(
          f"{path}: field={field} expected=non-empty-string "
          f"actual={type(value[field]).__name__}")
    if type(value["dimensions"]) is not int or value["dimensions"] < 1:
      raise EmbeddingManifestError(
        f"{path}: field=dimensions expected=positive-int "
        f"actual={value['dimensions']!r}")
    return cls(**{field: value[field] for field in (
      "schema_version", *EMBEDDING_SPACE_FIELDS)})

  def to_dict(self):
    return {
      "schema_version": self.schema_version,
      "provider": self.provider,
      "model": self.model,
      "dimensions": self.dimensions,
      "embedding_space_version": self.embedding_space_version,
      "normalization_version": self.normalization_version,
    }


LEGACY_ADA_002_MANIFEST = EmbeddingSpaceManifest(
  schema_version=SUPPORTED_SCHEMA_VERSION,
  provider="openai",
  model="text-embedding-ada-002",
  dimensions=1536,
  embedding_space_version="openai-ada-002-v1",
  normalization_version="newline-and-blank-v0",
)
_runtime_embedding_manifest = ContextVar(
  "runtime_embedding_manifest", default=LEGACY_ADA_002_MANIFEST)


def get_runtime_embedding_manifest():
  """Return the canonical embedding configuration for this runtime context."""
  return _runtime_embedding_manifest.get()


def set_runtime_embedding_manifest(manifest):
  """Configure an embedding space and return a token that can restore it."""
  if not isinstance(manifest, EmbeddingSpaceManifest):
    raise TypeError("Runtime embedding manifest must be EmbeddingSpaceManifest")
  return _runtime_embedding_manifest.set(manifest)


def reset_runtime_embedding_manifest(token=None):
  """Restore a prior token, or restore the legacy default for this context."""
  if token is None:
    _runtime_embedding_manifest.set(LEGACY_ADA_002_MANIFEST)
  else:
    _runtime_embedding_manifest.reset(token)


@contextmanager
def use_runtime_embedding_manifest(manifest):
  token = set_runtime_embedding_manifest(manifest)
  try:
    yield manifest
  finally:
    reset_runtime_embedding_manifest(token)


@dataclass(frozen=True)
class LoadedEmbeddingStore:
  manifest: EmbeddingSpaceManifest
  classification: EmbeddingSpaceClassification
  embeddings: dict
  nodes: dict


def _read_json(path, error_type):
  try:
    with open(path, encoding="utf-8") as infile:
      return json.load(infile)
  except (OSError, ValueError) as error:
    raise error_type(
      f"{path}: invalid or unreadable JSON; actual={type(error).__name__}") from error


def read_embedding_manifest(path):
  manifest_path = Path(path)
  return EmbeddingSpaceManifest.from_dict(
    _read_json(manifest_path, EmbeddingManifestError), str(manifest_path))


def write_embedding_manifest(store_path, manifest):
  """Atomically write the manifest file without touching persisted vectors."""
  store_path = Path(store_path)
  manifest_path = store_path / EMBEDDING_MANIFEST_FILENAME
  descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{EMBEDDING_MANIFEST_FILENAME}.", suffix=".tmp",
    dir=str(store_path), text=True)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as outfile:
      json.dump(manifest.to_dict(), outfile, indent=2)
      outfile.write("\n")
      outfile.flush()
      os.fsync(outfile.fileno())
    os.replace(temporary_name, manifest_path)
  except Exception:
    try:
      os.unlink(temporary_name)
    except OSError:
      pass
    raise


def assert_same_embedding_space(expected, actual, path="<memory>"):
  mismatches = []
  for field in EMBEDDING_SPACE_FIELDS:
    expected_value = getattr(expected, field)
    actual_value = getattr(actual, field)
    if expected_value != actual_value:
      mismatches.append(
        f"field={field} expected={expected_value!r} actual={actual_value!r}")
  if mismatches:
    raise EmbeddingSpaceMismatchError(f"{path}: " + "; ".join(mismatches))


def assert_runtime_embedding_request(provider, model, normalization_version,
                                     runtime_manifest=None):
  """Reject an incompatible runtime request before cache lookup or transport."""
  expected = runtime_manifest or get_runtime_embedding_manifest()
  mismatches = []
  if provider != expected.provider:
    mismatches.append(
      f"field=provider expected={expected.provider!r} actual={provider!r}")
  if model != expected.model:
    mismatches.append(
      f"field=model expected={expected.model!r} actual={model!r}")
  if normalization_version != expected.normalization_version:
    mismatches.append(
      "field=normalization_version "
      f"expected={expected.normalization_version!r} "
      f"actual={normalization_version!r}")
  if mismatches:
    raise EmbeddingSpaceMismatchError(
      "runtime embedding request: " + "; ".join(mismatches))


def _recognized_legacy_layout(store_path, embeddings, nodes):
  store_path = Path(store_path)
  if (store_path.name != "associative_memory"
      or store_path.parent.name != "bootstrap_memory"):
    return False
  if not isinstance(embeddings, dict) or not isinstance(nodes, dict):
    return False
  kw_path = store_path / "kw_strength.json"
  if not kw_path.is_file():
    return False
  try:
    keyword_strength = _read_json(kw_path, EmbeddingManifestError)
  except EmbeddingManifestError:
    return False
  if (not isinstance(keyword_strength, dict)
      or set(keyword_strength) != {"kw_strength_event", "kw_strength_thought"}):
    return False
  required_node_fields = {
    "node_count", "type_count", "type", "depth", "created", "expiration",
    "subject", "predicate", "object", "description", "embedding_key",
    "poignancy", "keywords", "filling",
  }
  for node_id, node in nodes.items():
    if (not isinstance(node_id, str) or not node_id.startswith("node_")
        or not isinstance(node, dict)
        or not required_node_fields.issubset(node)):
      return False
  return True


def _key_fingerprint(key):
  return hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]


def _validate_vectors(embeddings, manifest, path):
  if not isinstance(embeddings, dict):
    raise EmbeddingVectorValidationError(
      f"{path}: embeddings container expected=object "
      f"actual={type(embeddings).__name__}")
  for key, vector in embeddings.items():
    fingerprint = _key_fingerprint(key)
    location = f"{path}: embedding_key_sha256={fingerprint}"
    if not isinstance(vector, (list, tuple)):
      raise EmbeddingVectorValidationError(
        f"{location} expected=vector-list actual={type(vector).__name__}")
    if not vector:
      raise EmbeddingVectorValidationError(
        f"{location} expected=non-empty-vector actual=empty")
    if len(vector) != manifest.dimensions:
      raise EmbeddingVectorValidationError(
        f"{location} field=dimensions expected={manifest.dimensions} "
        f"actual={len(vector)}")
    squared_norm = 0.0
    for index, value in enumerate(vector):
      if not isinstance(value, Real) or isinstance(value, bool):
        raise EmbeddingVectorValidationError(
          f"{location} index={index} expected=number "
          f"actual={type(value).__name__}")
      if not math.isfinite(value):
        raise EmbeddingVectorValidationError(
          f"{location} index={index} expected=finite-number actual=non-finite")
      squared_norm += float(value) * float(value)
    if squared_norm == 0.0:
      raise EmbeddingVectorValidationError(
        f"{location} expected=non-zero-norm actual=zero")


def _validate_references(nodes, embeddings, path, reject_orphans=True):
  if not isinstance(nodes, dict):
    raise EmbeddingReferenceError(
      f"{path}: nodes container expected=object actual={type(nodes).__name__}")
  referenced = set()
  for node_id, node in nodes.items():
    if not isinstance(node, dict) or "embedding_key" not in node:
      raise EmbeddingReferenceError(
        f"{path}: node={node_id!r} expected=embedding_key actual=missing")
    embedding_key = node["embedding_key"]
    if embedding_key not in embeddings:
      raise EmbeddingReferenceError(
        f"{path}: node={node_id!r} reference_sha256="
        f"{_key_fingerprint(embedding_key)} expected=present actual=missing")
    referenced.add(embedding_key)
  if reject_orphans:
    orphaned = set(embeddings) - referenced
    if orphaned:
      fingerprints = sorted(_key_fingerprint(key) for key in orphaned)
      raise EmbeddingReferenceError(
        f"{path}: orphan_embeddings expected=0 actual={len(orphaned)} "
        f"key_sha256={fingerprints[:3]}")


def load_embedding_store(store_path, legacy_assumption_allowed=True,
                         runtime_manifest=None,
                         reject_orphans=True):
  """Load and fully validate a store before any node reaches retrieval."""
  runtime_manifest = runtime_manifest or get_runtime_embedding_manifest()
  store_path = Path(store_path)
  embeddings_path = store_path / "embeddings.json"
  nodes_path = store_path / "nodes.json"
  manifest_path = store_path / EMBEDDING_MANIFEST_FILENAME
  embeddings = _read_json(embeddings_path, EmbeddingVectorValidationError)
  nodes = _read_json(nodes_path, EmbeddingReferenceError)

  if manifest_path.is_file():
    manifest = read_embedding_manifest(manifest_path)
    classification = EmbeddingSpaceClassification.DECLARED
  elif (legacy_assumption_allowed
        and _recognized_legacy_layout(store_path, embeddings, nodes)):
    manifest = LEGACY_ADA_002_MANIFEST
    classification = EmbeddingSpaceClassification.LEGACY_ASSUMED
    warnings.warn(
      f"{store_path}: recognized historical associative-memory store; "
      "using virtual legacy Ada manifest without rewriting files",
      LegacyEmbeddingSpaceWarning, stacklevel=2)
  else:
    classification = EmbeddingSpaceClassification.UNKNOWN
    mode = "compatibility" if legacy_assumption_allowed else "strict"
    raise EmbeddingManifestError(
      f"{store_path}: manifest missing and archive classification="
      f"{classification.value}; mode={mode}")

  assert_same_embedding_space(runtime_manifest, manifest, str(manifest_path))
  _validate_vectors(embeddings, manifest, str(embeddings_path))
  _validate_references(nodes, embeddings, str(nodes_path), reject_orphans)
  return LoadedEmbeddingStore(manifest, classification, embeddings, nodes)
