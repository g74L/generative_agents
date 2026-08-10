"""Explicit, default-off runtime configuration for modern embeddings.

The runtime composes the existing provider factory, embedding-space manifest
contract, store loader, cache, and telemetry.  It does not bootstrap stores or
perform persistence.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from persona.memory_structures.embedding_space import (
  EMBEDDING_MANIFEST_FILENAME,
  EmbeddingManifestError,
  EmbeddingSpaceManifest,
  LEGACY_ADA_002_MANIFEST,
  assert_same_embedding_space,
  load_embedding_store,
  read_embedding_manifest,
  use_runtime_embedding_manifest,
)
from persona.prompt_template.chat_runtime import ModernChatCallerNotAllowedError
from persona.prompt_template.llm_provider_config import (
  LEGACY_OPENAI,
  MODERN_OPENAI,
  LLMProviderConfig,
  modern_openai_config,
)


# Embedding calls that have been migrated to explicit caller attribution.
# Each entry is the caller_id already used by the completion/chat request that
# produced the exact text being embedded -- the semantic owner of that text,
# not merely "the most recent caller seen" before the embedding call.
M2_EMBEDDING_CALLER_ALLOWLIST = (
  "planning_thought_on_convo",
  "memo_on_convo",
)


def validate_modern_embedding_caller(caller_id):
  """Fail-closed Embedding caller gate; mirrors the Chat/CompletionCompat
  guards in chat_runtime.py / completion_runtime.py.  Anonymous or unknown
  callers are rejected -- there is no wildcard or implicit fallback."""
  if (not isinstance(caller_id, str) or not caller_id.strip()
      or caller_id not in M2_EMBEDDING_CALLER_ALLOWLIST):
    raise ModernChatCallerNotAllowedError(
      "Modern Embedding caller is missing or not authorized")
  return caller_id


TEXT_EMBEDDING_3_SMALL_MODEL = "text-embedding-3-small"
TEXT_EMBEDDING_3_SMALL_1536_MANIFEST = EmbeddingSpaceManifest(
  schema_version=1,
  provider="openai",
  model=TEXT_EMBEDDING_3_SMALL_MODEL,
  dimensions=1536,
  embedding_space_version="openai-text-embedding-3-small-1536-v1",
  normalization_version="newline-and-blank-v0",
)


class NewEmbeddingStoreRequiredError(EmbeddingManifestError):
  """The selected runtime requires an explicitly bootstrapped store."""


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
  provider_config: LLMProviderConfig
  embedding_manifest: EmbeddingSpaceManifest
  embedding_model: str

  def __post_init__(self):
    if not isinstance(self.provider_config, LLMProviderConfig):
      raise TypeError("provider_config must be LLMProviderConfig")
    if not isinstance(self.embedding_manifest, EmbeddingSpaceManifest):
      raise TypeError("embedding_manifest must be EmbeddingSpaceManifest")
    if not isinstance(self.embedding_model, str) or not self.embedding_model:
      raise ValueError("embedding_model must be a non-empty string")
    if self.embedding_model != self.embedding_manifest.model:
      raise ValueError("embedding_model must match the embedding manifest")
    expected = {
      LEGACY_OPENAI: LEGACY_ADA_002_MANIFEST,
      MODERN_OPENAI: TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
    }.get(self.provider_config.provider_kind)
    if expected is None:
      raise ValueError("Embedding runtime requires a canonical OpenAI provider")
    assert_same_embedding_space(
      expected, self.embedding_manifest, "embedding runtime configuration")


def build_modern_embedding_runtime_config(
    request_timeout_seconds=600.0,
    provider_config: Optional[LLMProviderConfig] = None,
    embedding_manifest=TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
    embedding_model=TEXT_EMBEDDING_3_SMALL_MODEL,
):
  """Build the canonical explicit modern embedding runtime configuration."""
  return EmbeddingRuntimeConfig(
    provider_config=(provider_config or modern_openai_config(
      request_timeout_seconds=request_timeout_seconds)),
    embedding_manifest=embedding_manifest,
    embedding_model=embedding_model,
  )


def build_legacy_embedding_runtime_config():
  """Describe the unchanged legacy default as an explicit runtime."""
  return EmbeddingRuntimeConfig(
    provider_config=LLMProviderConfig(),
    embedding_manifest=LEGACY_ADA_002_MANIFEST,
    embedding_model=LEGACY_ADA_002_MANIFEST.model,
  )


def validate_embedding_store_for_runtime(
    store_path, runtime_config, legacy_assumption_allowed=True):
  """Validate a declared store without creating or modifying any files."""
  if not isinstance(runtime_config, EmbeddingRuntimeConfig):
    raise TypeError("runtime_config must be EmbeddingRuntimeConfig")
  path = Path(store_path)
  required = (path / "embeddings.json", path / "nodes.json")
  if not path.is_dir() or not any(item.exists() for item in required):
    raise NewEmbeddingStoreRequiredError(
      f"{path}: empty embedding store; explicit bootstrap required")
  if runtime_config.provider_config.provider_kind == MODERN_OPENAI:
    manifest_path = path / EMBEDDING_MANIFEST_FILENAME
    if not manifest_path.is_file():
      raise EmbeddingManifestError(
        f"{path}: modern embedding runtime requires a declared manifest")
    declared_manifest = read_embedding_manifest(manifest_path)
    assert_same_embedding_space(
      runtime_config.embedding_manifest, declared_manifest,
      str(manifest_path))
  return load_embedding_store(
    path,
    legacy_assumption_allowed=(
      legacy_assumption_allowed
      and runtime_config.provider_config.provider_kind != MODERN_OPENAI),
    runtime_manifest=runtime_config.embedding_manifest,
  )


@contextmanager
def use_embedding_runtime(
    runtime_config, modern_client_adapter=None, store_path=None,
    legacy_assumption_allowed=True):
  """Scope a coherent provider and manifest, restoring both on exit."""
  if not isinstance(runtime_config, EmbeddingRuntimeConfig):
    raise TypeError("runtime_config must be EmbeddingRuntimeConfig")
  if store_path is not None:
    validate_embedding_store_for_runtime(
      store_path, runtime_config, legacy_assumption_allowed)
  from persona.prompt_template.llm_provider import use_configured_provider
  with use_runtime_embedding_manifest(runtime_config.embedding_manifest):
    with use_configured_provider(
        runtime_config.provider_config, modern_client_adapter) as provider:
      yield provider
