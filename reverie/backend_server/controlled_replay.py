"""Minimal, offline-only orchestration for one controlled Smallville step.

R0H deliberately supports one actor, one step, and the historical wake-up-hour
planning wrapper.  It composes existing modern runtimes and persistence seams;
it does not alter cognition or provide a general replay framework.
"""
import ast
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass, field
import datetime
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from typing import Any, Optional, Tuple
import warnings

from persona.memory_structures.embedding_space import (
  EMBEDDING_MANIFEST_FILENAME,
  EmbeddingManifestError,
  EmbeddingReferenceError,
  EmbeddingSpaceManifest,
  EmbeddingSpaceMismatchError,
  EmbeddingVectorValidationError,
  LEGACY_ADA_002_MANIFEST,
  load_embedding_store,
  read_embedding_manifest,
)
from persona.memory_structures.scratch import Scratch
from persona.prompt_template import run_gpt_prompt
from persona.prompt_template.chat_runtime import (
  M5_CHAT_MODEL,
  ModernChatRuntimeConfig,
  build_modern_chat_runtime_config,
  use_modern_chat_runtime,
)
from persona.prompt_template.completion_runtime import (
  COMPLETION_COMPAT_MODEL,
  FORBIDDEN_MODERN_RUNTIME_MODELS,
  M6_DEFERRED_CALLERS,
  ModernCompletionRuntimeConfig,
  build_modern_completion_runtime_config,
  use_modern_completion_runtime,
)
from persona.prompt_template.cost_ledger import (
  CostLedgerContext,
  ModelPricing,
  PricingSnapshot,
  use_cost_ledger_context,
)
from persona.prompt_template.embedding_runtime import (
  TEXT_EMBEDDING_3_SMALL_1536_MANIFEST,
  TEXT_EMBEDDING_3_SMALL_MODEL,
  EmbeddingRuntimeConfig,
  build_modern_embedding_runtime_config,
  use_embedding_runtime,
)
from persona.prompt_template.embedding_store_bootstrap import (
  ModernEmbeddingStoreBootstrapRequest,
  bootstrap_modern_embedding_store,
)
from persona.prompt_template.llm_provider import (
  CHAT,
  COMPLETION_COMPAT,
  EMBEDDING,
  LLMReplayContext,
  TelemetryEvent,
  get_embedding_cache_stats,
  get_llm_replay_context,
  get_telemetry,
  install_llm_attempt_observer,
  reset_llm_attempt_observer,
  use_llm_replay_context,
)
from persona.prompt_template.llm_provider_config import (
  MODERN_OPENAI,
  MODERN_SDK_MODE,
  MODERN_TRANSPORT,
)
from persona.prompt_template.modern_openai_provider import (
  LLMProviderError,
  ModernOpenAIClientAdapter,
  NormalizedEmbeddingResponse,
  NormalizedTextResponse,
  NormalizedUsage,
)
from persona.prompt_template.replay_cost_guard import (
  ReplayCostAccountingUnavailableError,
  ReplayCostCeiling,
  ReplayCostGuardConfig,
  ReplayCostGuardError,
  ReplayCostGuardState,
  get_replay_cost_guard,
  use_replay_cost_guard,
)


SUCCESS = "SUCCESS"
PLANNING = "PLANNING"
WAKE_UP_CALLER = "wake_up_hour"
WAKE_UP_PATH = "planning.wake_up_hour"
FORBIDDEN_COGNITIVE_CALLERS = frozenset(M6_DEFERRED_CALLERS)
R0H_PRICING_SNAPSHOT = PricingSnapshot(
  snapshot_id="r0h-offline-pricing-v0",
  schema_version=1,
  currency="USD",
  created_at="2026-08-06",
  models=(
    ModelPricing(
      model=M5_CHAT_MODEL,
      input_per_million=Decimal("0.15"),
      cached_input_per_million=Decimal("0.075"),
      output_per_million=Decimal("0.60"),
      effective_from="2026-08-06",
      source_label="R0H deterministic offline fixture"),
    ModelPricing(
      model=TEXT_EMBEDDING_3_SMALL_MODEL,
      embedding_input_per_million=Decimal("0.02"),
      effective_from="2026-08-06",
      source_label="R0H deterministic offline fixture"),
  ),
  source_note="Pinned deterministic prices for the offline R0H harness",
)


class ControlledReplayError(RuntimeError):
  """Base error for safe harness validation and execution failures."""


class ControlledReplayConfigurationError(ControlledReplayError, ValueError):
  pass


class ControlledReplayLegacyConfigurationError(
    ControlledReplayConfigurationError):
  pass


class ControlledReplayPathForbiddenError(ControlledReplayError):
  pass


class IsolatedFixturePreflightError(ControlledReplayConfigurationError):
  """Typed failure raised before an isolated tick can reach cognition."""

  def __init__(self, reason_code, message):
    self.reason_code = reason_code
    super().__init__(message)


@dataclass(frozen=True)
class IsolatedReverieFixture:
  temporary_root: Path
  baseline_root: Path
  simulation_root: Path
  environment_step_path: Path
  personas_dir: Path
  reverie_dir: Path
  movement_dir: Path


MODERN_COMPATIBLE = "MODERN_COMPATIBLE"
EMPTY_BOOTSTRAPPABLE = "EMPTY_BOOTSTRAPPABLE"
LEGACY_NONEMPTY_BLOCKED = "LEGACY_NONEMPTY_BLOCKED"
INCONSISTENT_BLOCKED = "INCONSISTENT_BLOCKED"
UNKNOWN_BLOCKED = "UNKNOWN_BLOCKED"
ISOLATED_FIXTURE_PERSONAS = (
  "Isabella Rodriguez", "Maria Lopez", "Klaus Mueller")


class IsolatedEmbeddingPreflightError(IsolatedFixturePreflightError):
  def __init__(self, reason_code, message, audits=()):
    self.audits = tuple(audits)
    super().__init__(reason_code, message)


@dataclass(frozen=True)
class IsolatedEmbeddingStoreAudit:
  persona_name: str
  store_path: Path
  manifest_present: bool
  source_classification: str
  classification: str
  provider: Optional[str]
  model: Optional[str]
  dimensions: Optional[int]
  embedding_space_version: Optional[str]
  normalization_version: Optional[str]
  embedding_count: int
  node_count: int
  keyword_strength_entries: int
  embedding_reference_count: int
  orphan_embedding_count: int
  nonempty_vector_count: int
  observed_dimensions: Tuple[int, ...]
  internal_mismatch: bool


@dataclass(frozen=True)
class IsolatedEmbeddingPreflightResult:
  audits_before: Tuple[IsolatedEmbeddingStoreAudit, ...]
  audits_after: Tuple[IsolatedEmbeddingStoreAudit, ...]
  bootstrapped_personas: Tuple[str, ...]


def _path_is_within(path, parent):
  return path == parent or parent in path.parents


def prepare_isolated_reverie_fixture(simulation_root, temporary_root,
                                     baseline_root, source_step,
                                     movement_dir=None):
  """Validate an isolated simulation copy and create its movement directory.

  This is intentionally a filesystem-only preflight.  Callers must invoke it
  before constructing ``ReverieServer`` or entering any provider context.
  """
  if type(source_step) is not int or source_step < 0:
    raise IsolatedFixturePreflightError(
      "INVALID_SOURCE_STEP", "source_step must be a non-negative integer")

  temporary_root = Path(temporary_root).resolve()
  baseline_root = Path(baseline_root).resolve()
  simulation_root = Path(simulation_root).resolve()
  repository_root = Path(__file__).resolve().parents[2]
  protected_storage = (
    repository_root / "environment" / "frontend_server" / "storage"
  ).resolve()

  if not temporary_root.is_dir():
    raise IsolatedFixturePreflightError(
      "TEMPORARY_ROOT_MISSING", "isolated temporary root does not exist")
  if not simulation_root.is_dir():
    raise IsolatedFixturePreflightError(
      "SIMULATION_ROOT_MISSING", "isolated simulation root does not exist")
  if not baseline_root.is_dir():
    raise IsolatedFixturePreflightError(
      "BASELINE_ROOT_MISSING", "baseline simulation root does not exist")
  if not _path_is_within(simulation_root, temporary_root):
    raise IsolatedFixturePreflightError(
      "SIMULATION_OUTSIDE_TEMPORARY_ROOT",
      "isolated simulation root resolves outside the temporary root")
  if simulation_root == baseline_root:
    raise IsolatedFixturePreflightError(
      "SIMULATION_IS_BASELINE",
      "isolated simulation root must differ from the baseline")
  if (_path_is_within(simulation_root, protected_storage)
      or _path_is_within(protected_storage, simulation_root)):
    raise IsolatedFixturePreflightError(
      "PROTECTED_STORAGE_TARGET",
      "protected repository storage cannot be an isolated output target")

  environment_dir = (simulation_root / "environment").resolve()
  environment_step_path = (environment_dir / f"{source_step}.json").resolve()
  personas_dir = (simulation_root / "personas").resolve()
  reverie_dir = (simulation_root / "reverie").resolve()
  requested_movement = (Path(movement_dir) if movement_dir is not None
                        else simulation_root / "movement").resolve()
  fixture_paths = (
    environment_dir, environment_step_path, personas_dir, reverie_dir,
    requested_movement,
  )
  if any(not _path_is_within(path, simulation_root) for path in fixture_paths):
    raise IsolatedFixturePreflightError(
      "FIXTURE_PATH_OUTSIDE_SIMULATION",
      "an isolated fixture path resolves outside the simulation root")
  if requested_movement != (simulation_root / "movement").resolve():
    raise IsolatedFixturePreflightError(
      "EXTERNAL_MOVEMENT_TARGET",
      "movement output must be the simulation-local movement directory")

  required_dirs = (
    (environment_dir, "ENVIRONMENT_DIR_MISSING", "environment"),
    (personas_dir, "PERSONAS_DIR_MISSING", "personas"),
    (reverie_dir, "REVERIE_DIR_MISSING", "reverie"),
  )
  for path, reason_code, label in required_dirs:
    if not path.is_dir():
      raise IsolatedFixturePreflightError(
        reason_code, f"isolated {label} directory does not exist")
  if not environment_step_path.is_file():
    raise IsolatedFixturePreflightError(
      "SOURCE_ENVIRONMENT_STEP_MISSING",
      "source environment step does not exist")

  try:
    requested_movement.mkdir(parents=True, exist_ok=True)
  except OSError as exc:
    raise IsolatedFixturePreflightError(
      "MOVEMENT_DIRECTORY_CREATION_FAILED",
      "isolated movement directory could not be created") from exc
  if not requested_movement.is_dir() or not os.access(requested_movement,
                                                       os.W_OK):
    raise IsolatedFixturePreflightError(
      "MOVEMENT_DIRECTORY_NOT_WRITABLE",
      "isolated movement directory is not writable")

  return IsolatedReverieFixture(
    temporary_root=temporary_root,
    baseline_root=baseline_root,
    simulation_root=simulation_root,
    environment_step_path=environment_step_path,
    personas_dir=personas_dir,
    reverie_dir=reverie_dir,
    movement_dir=requested_movement,
  )


def _embedding_audit_failure(persona_name, store_path, classification,
                             manifest=None):
  return IsolatedEmbeddingStoreAudit(
    persona_name=persona_name,
    store_path=store_path,
    manifest_present=(store_path / EMBEDDING_MANIFEST_FILENAME).is_file(),
    source_classification="UNKNOWN",
    classification=classification,
    provider=getattr(manifest, "provider", None),
    model=getattr(manifest, "model", None),
    dimensions=getattr(manifest, "dimensions", None),
    embedding_space_version=getattr(
      manifest, "embedding_space_version", None),
    normalization_version=getattr(manifest, "normalization_version", None),
    embedding_count=0, node_count=0, keyword_strength_entries=0,
    embedding_reference_count=0, orphan_embedding_count=0,
    nonempty_vector_count=0, observed_dimensions=(),
    internal_mismatch=(classification == INCONSISTENT_BLOCKED),
  )


def inspect_isolated_embedding_store(persona_name, store_path,
                                     simulation_root):
  """Return content-free embedding metadata for one isolated Persona store."""
  store_path = Path(store_path).resolve()
  simulation_root = Path(simulation_root).resolve()
  if not _path_is_within(store_path, simulation_root):
    return _embedding_audit_failure(
      persona_name, store_path, UNKNOWN_BLOCKED)
  expected = (simulation_root / "personas" / persona_name
              / "bootstrap_memory" / "associative_memory").resolve()
  if store_path != expected or not store_path.is_dir():
    return _embedding_audit_failure(
      persona_name, store_path, UNKNOWN_BLOCKED)

  try:
    embeddings = json.loads(
      (store_path / "embeddings.json").read_text(encoding="utf-8"))
    nodes = json.loads(
      (store_path / "nodes.json").read_text(encoding="utf-8"))
    keyword_strength = json.loads(
      (store_path / "kw_strength.json").read_text(encoding="utf-8"))
  except (OSError, ValueError, TypeError):
    return _embedding_audit_failure(
      persona_name, store_path, UNKNOWN_BLOCKED)
  if (not isinstance(embeddings, dict) or not isinstance(nodes, dict)
      or not isinstance(keyword_strength, dict)):
    return _embedding_audit_failure(
      persona_name, store_path, UNKNOWN_BLOCKED)

  manifest_path = store_path / EMBEDDING_MANIFEST_FILENAME
  manifest = None
  source_classification = "DECLARED" if manifest_path.is_file() else "UNKNOWN"
  if manifest_path.is_file():
    try:
      manifest = read_embedding_manifest(manifest_path)
    except EmbeddingManifestError:
      return _embedding_audit_failure(
        persona_name, store_path, UNKNOWN_BLOCKED)
  elif (store_path.name == "associative_memory"
        and store_path.parent.name == "bootstrap_memory"
        and set(keyword_strength) == {
          "kw_strength_event", "kw_strength_thought"}):
    manifest = LEGACY_ADA_002_MANIFEST
    source_classification = "LEGACY_ASSUMED"
  else:
    return _embedding_audit_failure(
      persona_name, store_path, UNKNOWN_BLOCKED)

  references = []
  missing_reference = False
  for node in nodes.values():
    if not isinstance(node, dict) or "embedding_key" not in node:
      missing_reference = True
      continue
    references.append(node["embedding_key"])
    if node["embedding_key"] not in embeddings:
      missing_reference = True
  orphan_count = len(set(embeddings) - set(references))
  nonempty_vectors = tuple(
    vector for vector in embeddings.values()
    if isinstance(vector, (list, tuple)) and bool(vector))
  observed_dimensions = tuple(sorted({len(vector)
                                      for vector in nonempty_vectors}))
  keyword_entries = sum(
    len(value) for value in keyword_strength.values()
    if isinstance(value, dict))
  malformed_vector = any(
    not isinstance(vector, (list, tuple)) or not vector
    for vector in embeddings.values())
  internal_mismatch = (
    missing_reference or orphan_count > 0 or malformed_vector
    or any(length != manifest.dimensions for length in observed_dimensions)
    or any(not isinstance(value, dict) for value in keyword_strength.values())
  )

  completely_empty = (
    not embeddings and not nodes and not references
    and not nonempty_vectors and keyword_entries == 0)
  if internal_mismatch:
    classification = INCONSISTENT_BLOCKED
  elif (completely_empty
        and manifest == TEXT_EMBEDDING_3_SMALL_1536_MANIFEST):
    try:
      load_embedding_store(
        store_path, legacy_assumption_allowed=False,
        runtime_manifest=TEXT_EMBEDDING_3_SMALL_1536_MANIFEST)
      classification = MODERN_COMPATIBLE
    except (EmbeddingManifestError, EmbeddingReferenceError,
            EmbeddingSpaceMismatchError, EmbeddingVectorValidationError):
      classification = INCONSISTENT_BLOCKED
  elif completely_empty:
    classification = EMPTY_BOOTSTRAPPABLE
  else:
    try:
      if manifest == TEXT_EMBEDDING_3_SMALL_1536_MANIFEST:
        load_embedding_store(
          store_path, legacy_assumption_allowed=False,
          runtime_manifest=TEXT_EMBEDDING_3_SMALL_1536_MANIFEST)
        classification = MODERN_COMPATIBLE
      elif manifest == LEGACY_ADA_002_MANIFEST:
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          load_embedding_store(
            store_path, legacy_assumption_allowed=True,
            runtime_manifest=LEGACY_ADA_002_MANIFEST)
        classification = LEGACY_NONEMPTY_BLOCKED
      else:
        classification = UNKNOWN_BLOCKED
    except (EmbeddingManifestError, EmbeddingReferenceError,
            EmbeddingSpaceMismatchError, EmbeddingVectorValidationError):
      classification = INCONSISTENT_BLOCKED

  return IsolatedEmbeddingStoreAudit(
    persona_name=persona_name,
    store_path=store_path,
    manifest_present=manifest_path.is_file(),
    source_classification=source_classification,
    classification=classification,
    provider=manifest.provider,
    model=manifest.model,
    dimensions=manifest.dimensions,
    embedding_space_version=manifest.embedding_space_version,
    normalization_version=manifest.normalization_version,
    embedding_count=len(embeddings),
    node_count=len(nodes),
    keyword_strength_entries=keyword_entries,
    embedding_reference_count=len(references),
    orphan_embedding_count=orphan_count,
    nonempty_vector_count=len(nonempty_vectors),
    observed_dimensions=observed_dimensions,
    internal_mismatch=internal_mismatch,
  )


def prepare_isolated_embedding_stores(fixture,
                                      persona_names=ISOLATED_FIXTURE_PERSONAS):
  """Audit all required stores, then modernize only proven-empty copies."""
  if not isinstance(fixture, IsolatedReverieFixture):
    raise TypeError("fixture must be IsolatedReverieFixture")
  if tuple(persona_names) != ISOLATED_FIXTURE_PERSONAS:
    raise IsolatedEmbeddingPreflightError(
      "PERSONA_SET_MISMATCH", "required Persona set does not match")

  audits_before = tuple(inspect_isolated_embedding_store(
    persona_name,
    fixture.personas_dir / persona_name / "bootstrap_memory"
    / "associative_memory",
    fixture.simulation_root,
  ) for persona_name in persona_names)
  blocked = tuple(audit for audit in audits_before if audit.classification in (
    LEGACY_NONEMPTY_BLOCKED, INCONSISTENT_BLOCKED, UNKNOWN_BLOCKED))
  if blocked:
    if any(audit.classification == INCONSISTENT_BLOCKED for audit in blocked):
      reason_code = "EMBEDDING_FIXTURE_INCONSISTENT"
    elif any(audit.classification == LEGACY_NONEMPTY_BLOCKED
             for audit in blocked):
      reason_code = "EMBEDDING_STORE_MIGRATION_REQUIRED"
    else:
      reason_code = "EMBEDDING_STORE_UNKNOWN"
    raise IsolatedEmbeddingPreflightError(
      reason_code, "one or more required embedding stores are blocked",
      audits_before)

  bootstrapped = []
  for audit in audits_before:
    if audit.classification != EMPTY_BOOTSTRAPPABLE:
      continue
    store_path = audit.store_path
    backup_path = store_path.with_name(
      ".associative_memory.r1t_empty_backup")
    if backup_path.exists():
      raise IsolatedEmbeddingPreflightError(
        "EMBEDDING_BOOTSTRAP_BACKUP_EXISTS",
        "temporary embedding bootstrap backup already exists", audits_before)
    store_path.rename(backup_path)
    try:
      bootstrap_modern_embedding_store(
        ModernEmbeddingStoreBootstrapRequest(
          target_path=store_path, allowed_parent=store_path.parent))
    except Exception as error:
      if store_path.exists():
        shutil.rmtree(store_path)
      backup_path.rename(store_path)
      raise IsolatedEmbeddingPreflightError(
        "EMBEDDING_BOOTSTRAP_FAILED",
        "empty isolated embedding store bootstrap failed",
        audits_before) from error
    shutil.rmtree(backup_path)
    bootstrapped.append(audit.persona_name)

  audits_after = tuple(inspect_isolated_embedding_store(
    persona_name,
    fixture.personas_dir / persona_name / "bootstrap_memory"
    / "associative_memory",
    fixture.simulation_root,
  ) for persona_name in persona_names)
  if any(audit.classification != MODERN_COMPATIBLE
         for audit in audits_after):
    raise IsolatedEmbeddingPreflightError(
      "EMBEDDING_BOOTSTRAP_VERIFICATION_FAILED",
      "modernized embedding stores failed final verification", audits_after)
  return IsolatedEmbeddingPreflightResult(
    audits_before=audits_before,
    audits_after=audits_after,
    bootstrapped_personas=tuple(bootstrapped),
  )


def _sanitized_traceback_path(filename, repository_root, temporary_root):
  path = Path(filename).resolve()
  repository_root = Path(repository_root).resolve()
  temporary_root = Path(temporary_root).resolve()
  if _path_is_within(path, repository_root):
    return path.relative_to(repository_root).as_posix()
  if _path_is_within(path, temporary_root):
    return "<temporary>/" + path.relative_to(temporary_root).as_posix()
  return "<external>/" + path.name


def _safe_ast_value(node, frame):
  if isinstance(node, ast.Name):
    return frame.f_locals.get(node.id), node.id
  if isinstance(node, ast.Attribute):
    parent, origin = _safe_ast_value(node.value, frame)
    if parent is None or origin is None or node.attr.startswith("_"):
      return None, None
    try:
      return getattr(parent, node.attr), f"{origin}.{node.attr}"
    except Exception:
      return None, None
  return None, None


def _safe_ast_index(node, frame):
  if isinstance(node, ast.Constant) and type(node.value) is int:
    return node.value
  if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
      and isinstance(node.operand, ast.Constant)
      and type(node.operand.value) is int):
    return -node.operand.value
  if isinstance(node, ast.Name):
    value = frame.f_locals.get(node.id)
    return value if type(value) is int else None
  return None


def _collection_diagnostic_from_traceback(error):
  traceback_node = error.__traceback__
  if traceback_node is None:
    return None
  while traceback_node.tb_next is not None:
    traceback_node = traceback_node.tb_next
  frame = traceback_node.tb_frame
  filename = Path(frame.f_code.co_filename)
  try:
    tree = ast.parse(filename.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, SyntaxError):
    return None
  line = traceback_node.tb_lineno
  candidates = [node for node in ast.walk(tree)
                if isinstance(node, ast.Subscript)
                and node.lineno <= line <= getattr(node, "end_lineno", line)]
  for node in candidates:
    collection, origin = _safe_ast_value(node.value, frame)
    requested_index = _safe_ast_index(node.slice, frame)
    if origin is None or requested_index is None:
      continue
    try:
      length = len(collection)
    except (TypeError, ValueError):
      continue
    out_of_range = (
      requested_index >= length or requested_index < -length
      if length else True)
    if not out_of_range:
      continue
    try:
      element_types = tuple(sorted({type(item).__name__
                                    for item in collection}))
    except TypeError:
      element_types = ()
    return {
      "type": type(collection).__name__,
      "length": length,
      "requested_index": requested_index,
      "empty": length == 0,
      "element_type_names": element_types,
      "source": origin,
    }
  return None


def build_sanitized_failure_evidence(
    error, repository_root, temporary_root, phase, current_module,
    telemetry, call_guard, cost_guard, provider_call_count,
    network_call_count, current_caller=None):
  """Capture content-free diagnostics while replay contexts are still live."""
  if not isinstance(error, BaseException):
    raise TypeError("error must be an exception")
  events = tuple(telemetry)
  frames = []
  traceback_node = error.__traceback__
  while traceback_node is not None:
    frame = traceback_node.tb_frame
    frames.append({
      "file": _sanitized_traceback_path(
        frame.f_code.co_filename, repository_root, temporary_root),
      "function": frame.f_code.co_name,
      "line": traceback_node.tb_lineno,
    })
    traceback_node = traceback_node.tb_next
  last_event = events[-1] if events else None
  successful = [event for event in events
                if getattr(event, "outcome", None) == "SUCCESS"]
  last_completed = successful[-1] if successful else None
  caller_counts = Counter(
    getattr(event, "caller_id", None) or "unattributed" for event in events)
  operation_counts = Counter(
    getattr(event, "operation", "UNKNOWN") for event in events)
  cost_snapshot = cost_guard.snapshot()
  ledger_records = cost_guard.records()
  forbidden = frozenset(FORBIDDEN_MODERN_RUNTIME_MODELS)
  legacy_detections = sum(
    getattr(event, "model_or_engine", None) in forbidden
    or getattr(event, "response_model", None) in forbidden
    for event in events)
  parser_frames = tuple(frame["function"] for frame in frames if any(
    token in frame["function"].lower()
    for token in ("parser", "clean", "validate", "run_gpt_prompt")))
  collection = (_collection_diagnostic_from_traceback(error)
                if isinstance(error, IndexError) else None)

  last_telemetry = None
  if last_event is not None:
    last_telemetry = {
      "caller_id": getattr(last_event, "caller_id", None),
      "operation": getattr(last_event, "operation", None),
      "outcome": getattr(last_event, "outcome", None),
      "requested_model": getattr(last_event, "model_or_engine", None),
      "returned_model": getattr(last_event, "response_model", None),
      "physical_attempt": getattr(last_event, "physical_attempt", None),
      "error_type": getattr(last_event, "error_type", None),
    }
  return {
    "reason_code": "PLAN_INDEX_ERROR" if isinstance(error, IndexError)
                   else "COGNITIVE_FAILURE",
    "exception_type": type(error).__name__,
    "traceback": tuple(frames),
    "failure_phase": phase,
    "current_module": current_module,
    "current_caller": current_caller or (
      getattr(last_event, "caller_id", None) if last_event else None),
    "last_completed_caller": (
      getattr(last_completed, "caller_id", None) if last_completed else None),
    "operation": getattr(last_event, "operation", None) if last_event else None,
    "parser_frames": parser_frames,
    "fallback_used": None,
    "collection": collection,
    "logical_calls": call_guard.logical_calls,
    "physical_attempts": call_guard.physical_attempts,
    "retry_count": max(
      0, call_guard.physical_attempts - call_guard.logical_calls),
    "operation_breakdown": tuple(sorted(operation_counts.items())),
    "caller_breakdown": tuple(sorted(caller_counts.items())),
    "telemetry_events": len(events),
    "last_telemetry": last_telemetry,
    "ledger_records": len(ledger_records),
    "cost": {
      "accumulated": str(cost_snapshot.accumulated_cost),
      "ceiling": str(cost_snapshot.ceiling),
      "remaining": str(cost_snapshot.remaining_cost),
      "tripped": cost_snapshot.tripped,
      "logical_calls": cost_snapshot.logical_calls,
      "physical_attempts": cost_snapshot.physical_attempts,
      "by_operation": tuple(
        (name, str(value)) for name, value in cost_snapshot.cost_by_operation),
    },
    "legacy_detections": legacy_detections,
    "provider_calls": provider_call_count,
    "network_calls": network_call_count,
  }


class ReplayLogicalCallLimitExceededError(ReplayCostGuardError):
  def __init__(self, maximum):
    self.maximum = maximum
    super().__init__("Controlled replay logical-call limit reached")


class ReplayPhysicalAttemptLimitExceededError(ReplayCostGuardError):
  def __init__(self, maximum):
    self.maximum = maximum
    super().__init__("Controlled replay physical-attempt limit reached")


@dataclass(frozen=True)
class ControlledReplayProfile:
  replay_id: str
  simulation_id: str
  actor_id: str
  step: int
  maximum_cost: Decimal
  maximum_logical_calls: int
  maximum_physical_attempts: int
  conversation_enabled: bool = False
  reflection_enabled: bool = False

  def __post_init__(self):
    for name in ("replay_id", "simulation_id", "actor_id"):
      value = getattr(self, name)
      if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ControlledReplayConfigurationError(
          f"{name} must be non-blank text")
    if type(self.step) is not int or self.step < 0:
      raise ControlledReplayConfigurationError(
        "step must be a non-negative integer")
    if (isinstance(self.maximum_cost, bool)
        or not isinstance(self.maximum_cost, Decimal)):
      raise ControlledReplayConfigurationError(
        "maximum_cost must be a Decimal")
    if not self.maximum_cost.is_finite() or self.maximum_cost <= 0:
      raise ControlledReplayConfigurationError(
        "maximum_cost must be finite and positive")
    for name in ("maximum_logical_calls", "maximum_physical_attempts"):
      value = getattr(self, name)
      if type(value) is not int or value <= 0:
        raise ControlledReplayConfigurationError(
          f"{name} must be a positive integer")
    if type(self.conversation_enabled) is not bool:
      raise ControlledReplayConfigurationError(
        "conversation_enabled must be boolean")
    if type(self.reflection_enabled) is not bool:
      raise ControlledReplayConfigurationError(
        "reflection_enabled must be boolean")
    if self.conversation_enabled:
      raise ControlledReplayConfigurationError(
        "conversation is unsupported by controlled replay V0")
    if self.reflection_enabled:
      raise ControlledReplayConfigurationError(
        "reflection is unsupported by controlled replay V0")


@dataclass(frozen=True)
class ControlledReplayActorFixture:
  actor_id: str
  persona: Any
  prompt_input: Tuple[str, ...]

  def __post_init__(self):
    if not isinstance(self.actor_id, str) or not self.actor_id.strip():
      raise ControlledReplayConfigurationError("fixture actor_id is required")
    if self.persona is None:
      raise ControlledReplayConfigurationError("fixture persona is required")
    if (not isinstance(self.prompt_input, tuple) or not self.prompt_input
        or not all(isinstance(item, str) for item in self.prompt_input)):
      raise ControlledReplayConfigurationError(
        "prompt_input must be a non-empty immutable text tuple")


@dataclass(frozen=True)
class ControlledReplayPersonaFixture:
  scratch: Scratch


def build_controlled_replay_actor_fixture(
    actor_id, scratch_path, simulation_meta_path):
  """Build the historical prompt fixture from canonical simulation time."""
  simulation_meta_path = Path(simulation_meta_path)
  try:
    simulation_meta = json.loads(
      simulation_meta_path.read_text(encoding="utf-8"))
  except OSError as error:
    raise ControlledReplayConfigurationError(
      "simulation metadata is unavailable") from error
  except (json.JSONDecodeError, TypeError) as error:
    raise ControlledReplayConfigurationError(
      "simulation metadata is malformed") from error

  if not isinstance(simulation_meta, dict):
    raise ControlledReplayConfigurationError(
      "simulation metadata is malformed")
  serialized_curr_time = simulation_meta.get("curr_time")
  if not isinstance(serialized_curr_time, str) or not serialized_curr_time.strip():
    raise ControlledReplayConfigurationError(
      "simulation metadata curr_time is unavailable")
  try:
    curr_time = datetime.datetime.strptime(
      serialized_curr_time, "%B %d, %Y, %H:%M:%S")
  except ValueError as error:
    raise ControlledReplayConfigurationError(
      "simulation metadata curr_time is malformed") from error

  scratch = Scratch(str(Path(scratch_path)))
  scratch.curr_time = curr_time
  persona = ControlledReplayPersonaFixture(scratch=scratch)
  return ControlledReplayActorFixture(
    actor_id=actor_id,
    persona=persona,
    prompt_input=(
      scratch.get_str_iss(),
      scratch.get_str_lifestyle(),
      scratch.get_str_firstname(),
    ),
  )


@dataclass(frozen=True)
class ControlledReplayEnvironmentFixture:
  simulation_id: str
  step: int

  def __post_init__(self):
    if not isinstance(self.simulation_id, str) or not self.simulation_id.strip():
      raise ControlledReplayConfigurationError(
        "fixture simulation_id is required")
    if type(self.step) is not int or self.step < 0:
      raise ControlledReplayConfigurationError(
        "fixture step must be a non-negative integer")


class DeterministicReplayFakeAdapter:
  """In-memory modern-client adapter with deterministic normalized results."""

  offline_fake = True

  def __init__(self, text_responses=("7 am",), *, input_tokens=20,
               output_tokens=3, cached_input_tokens=0, reasoning_tokens=0,
               embedding_input_tokens=1):
    self._text_responses = list(text_responses)
    self.input_tokens = input_tokens
    self.output_tokens = output_tokens
    self.cached_input_tokens = cached_input_tokens
    self.reasoning_tokens = reasoning_tokens
    self.embedding_input_tokens = embedding_input_tokens
    self.calls = []

  def create_chat(self, **kwargs):
    self.calls.append(("create_chat", dict(kwargs)))
    if not self._text_responses:
      raise AssertionError("No deterministic text response configured")
    response = self._text_responses.pop(0)
    if isinstance(response, BaseException):
      raise response
    return NormalizedTextResponse(
      text=response,
      model=kwargs["model"],
      request_id=f"r0h-text-{len(self.calls)}",
      finish_reason="stop",
      status="completed",
      usage=NormalizedUsage(
        self.input_tokens, self.output_tokens,
        self.cached_input_tokens, self.reasoning_tokens),
    )

  def create_embedding(self, **kwargs):
    self.calls.append(("create_embedding", dict(kwargs)))
    return NormalizedEmbeddingResponse(
      vector=(1.0,) + (0.0,) * 1535,
      model=kwargs["model"],
      request_id=f"r0h-embedding-{len(self.calls)}",
      usage=NormalizedUsage(self.embedding_input_tokens, None),
    )


R1T_GENERIC_FAKE_RESPONSE = "7 am"
R1T_PRONUNCIATIO_FAKE_RESPONSE = "🙂🙂"
R1T_ACT_OBJ_DESC_FAKE_RESPONSE = "state"
R1T_TASK_DECOMP_FAKE_RESPONSE = (
  "preparing for the activity (duration in minutes: 5, minutes left: 0)")


class R1TDeterministicFakeAdapter(DeterministicReplayFakeAdapter):
  """Caller-aware offline matrix used by the isolated full-tick fixture."""

  def __init__(self, *, input_tokens=20, output_tokens=3,
               embedding_input_tokens=1):
    super().__init__(
      text_responses=(), input_tokens=input_tokens,
      output_tokens=output_tokens,
      embedding_input_tokens=embedding_input_tokens)
    self.calls = []

  @staticmethod
  def response_for_caller(caller_id):
    if caller_id == "task_decomp":
      return R1T_TASK_DECOMP_FAKE_RESPONSE
    if caller_id == "pronunciatio":
      return R1T_PRONUNCIATIO_FAKE_RESPONSE
    if caller_id == "act_obj_desc":
      return R1T_ACT_OBJ_DESC_FAKE_RESPONSE
    return R1T_GENERIC_FAKE_RESPONSE

  def create_chat(self, **kwargs):
    caller_id = get_llm_replay_context().caller_id
    response = self.response_for_caller(caller_id)
    self.calls.append({
      "method": "create_chat", "caller_id": caller_id,
      "model": kwargs["model"],
    })
    return NormalizedTextResponse(
      text=response,
      model=kwargs["model"],
      request_id=f"r1t-text-{len(self.calls)}",
      finish_reason="stop",
      status="completed",
      usage=NormalizedUsage(
        self.input_tokens, self.output_tokens,
        self.cached_input_tokens, self.reasoning_tokens),
    )

  def create_embedding(self, **kwargs):
    caller_id = get_llm_replay_context().caller_id
    self.calls.append({
      "method": "create_embedding", "caller_id": caller_id,
      "model": kwargs["model"],
    })
    return NormalizedEmbeddingResponse(
      vector=(1.0,) + (0.0,) * 1535,
      model=kwargs["model"],
      request_id=f"r1t-embedding-{len(self.calls)}",
      usage=NormalizedUsage(self.embedding_input_tokens, None),
    )


@dataclass(frozen=True)
class ControlledReplayProviders:
  chat_adapter: Any
  completion_adapter: Any
  embedding_adapter: Any
  chat_config: ModernChatRuntimeConfig = field(
    default_factory=build_modern_chat_runtime_config)
  completion_config: ModernCompletionRuntimeConfig = field(
    default_factory=build_modern_completion_runtime_config)
  embedding_config: EmbeddingRuntimeConfig = field(
    default_factory=build_modern_embedding_runtime_config)
  pricing_snapshot: PricingSnapshot = R0H_PRICING_SNAPSHOT
  live_api_enabled: bool = False

  def __post_init__(self):
    if type(self.live_api_enabled) is not bool:
      raise ControlledReplayConfigurationError(
        "live_api_enabled must be boolean")
    adapters = (
      ("chat_adapter", self.chat_adapter, "create_chat"),
      ("completion_adapter", self.completion_adapter, "create_chat"),
      ("embedding_adapter", self.embedding_adapter, "create_embedding"),
    )
    for name, adapter, method in adapters:
      offline_fake = getattr(adapter, "offline_fake", False) is True
      authorized_live = (
        self.live_api_enabled
        and isinstance(adapter, ModernOpenAIClientAdapter))
      if ((not offline_fake and not authorized_live)
          or not callable(getattr(adapter, method, None))):
        raise ControlledReplayConfigurationError(
          f"{name} must be an offline fake or explicitly enabled modern adapter")
    _validate_modern_runtime_configuration(self)


@dataclass(frozen=True)
class ControlledReplayTelemetryEntry:
  replay_id: str
  simulation_id: str
  actor_id: str
  step: int
  cognitive_category: str
  caller_id: str
  operation: str
  logical_call_id: str
  physical_attempt: int
  requested_model: str
  returned_model: Optional[str]
  outcome: str
  error_type: Optional[str]


@dataclass(frozen=True)
class ControlledReplayLedgerEntry:
  replay_id: str
  simulation_id: str
  actor_id: str
  step: int
  cognitive_category: str
  caller_id: str
  operation: str
  logical_call_id: str
  physical_attempt: int
  estimated_cost: Optional[Decimal]


@dataclass(frozen=True)
class ControlledReplayReport:
  replay_id: str
  simulation_id: str
  actor_id: str
  step: int
  status: str
  selected_cognitive_path: str
  logical_calls: int
  physical_attempts: int
  operation_counts: Tuple[Tuple[str, int], ...]
  models_requested: Tuple[str, ...]
  models_returned: Tuple[str, ...]
  retry_count: int
  input_tokens: int
  output_tokens: int
  cached_tokens: int
  reasoning_tokens: int
  embedding_cache_hits: int
  embedding_cache_misses: int
  accumulated_cost: Decimal
  cost_ceiling: Decimal
  remaining_cost: Decimal
  legacy_detections: int
  error_type: Optional[str]
  primary_error_type: Optional[str]
  underlying_provider_error_type: Optional[str]
  store_manifest_identity: str
  parser_result_type: Optional[str]
  wake_up_hour: Optional[int]
  cognitive_output_digest: Optional[str]
  telemetry: Tuple[ControlledReplayTelemetryEntry, ...]
  ledger: Tuple[ControlledReplayLedgerEntry, ...]


class ReplayCallCountGuardState:
  def __init__(self, profile):
    self.maximum_logical_calls = profile.maximum_logical_calls
    self.maximum_physical_attempts = profile.maximum_physical_attempts
    self._logical_call_ids = []
    self._physical_attempts = 0

  @property
  def logical_calls(self):
    return len(self._logical_call_ids)

  @property
  def physical_attempts(self):
    return self._physical_attempts

  def check_before_attempt(self, logical_call_id):
    if (logical_call_id not in self._logical_call_ids
        and self.logical_calls >= self.maximum_logical_calls):
      raise ReplayLogicalCallLimitExceededError(
        self.maximum_logical_calls)
    if self.physical_attempts >= self.maximum_physical_attempts:
      raise ReplayPhysicalAttemptLimitExceededError(
        self.maximum_physical_attempts)

  def record_attempt(self, logical_call_id):
    if logical_call_id not in self._logical_call_ids:
      self._logical_call_ids.append(logical_call_id)
    self._physical_attempts += 1


_active_call_count_guard = ContextVar(
  "active_replay_call_count_guard", default=None)


def get_replay_call_count_guard():
  return _active_call_count_guard.get()


@contextmanager
def use_replay_call_count_guard(profile):
  state = ReplayCallCountGuardState(profile)
  token = _active_call_count_guard.set(state)
  try:
    yield state
  finally:
    _active_call_count_guard.reset(token)


class _CombinedReplayObserver:
  def __init__(self, call_guard, cost_guard):
    self.call_guard = call_guard
    self.cost_guard = cost_guard

  def before_attempt(self, **kwargs):
    logical_call_id = kwargs["logical_call_id"]
    self.call_guard.check_before_attempt(logical_call_id)
    self.cost_guard.before_attempt(**kwargs)
    self.call_guard.record_attempt(logical_call_id)

  def after_attempt(self, event):
    self.cost_guard.after_attempt(event)


@contextmanager
def _use_combined_replay_observer(call_guard, cost_guard):
  token = install_llm_attempt_observer(
    _CombinedReplayObserver(call_guard, cost_guard))
  try:
    yield
  finally:
    reset_llm_attempt_observer(token)


@contextmanager
def _historical_wrapper_environment():
  """Resolve historical relative templates and suppress debug prompt output."""
  previous_directory = Path.cwd()
  previous_debug = getattr(run_gpt_prompt, "debug", False)
  backend_server = Path(__file__).resolve().parent
  run_gpt_prompt.debug = False
  try:
    os.chdir(backend_server)
    with redirect_stdout(io.StringIO()):
      yield
  finally:
    os.chdir(previous_directory)
    run_gpt_prompt.debug = previous_debug


def assert_controlled_replay_path_allowed(caller_id):
  if caller_id in FORBIDDEN_COGNITIVE_CALLERS:
    raise ControlledReplayPathForbiddenError(
      "Conversation and reflection callers are forbidden in R0H")
  if caller_id != WAKE_UP_CALLER:
    raise ControlledReplayPathForbiddenError(
      "Controlled replay V0 supports only its selected planning caller")
  return caller_id


def _validate_modern_runtime_configuration(providers):
  configs = (
    providers.chat_config, providers.completion_config,
    getattr(providers.embedding_config, "provider_config", None))
  for config in configs:
    if config is None or not all(hasattr(config, name) for name in (
        "provider_kind", "transport_kind", "sdk_mode")):
      raise ControlledReplayConfigurationError(
        "Every modern operation requires its explicit runtime configuration")
    if (config.provider_kind, config.transport_kind, config.sdk_mode) != (
        MODERN_OPENAI, MODERN_TRANSPORT, MODERN_SDK_MODE):
      raise ControlledReplayLegacyConfigurationError(
        "Controlled replay requires the modern provider and transport")
  if (not isinstance(providers.chat_config, ModernChatRuntimeConfig)
      or not isinstance(
        providers.completion_config, ModernCompletionRuntimeConfig)
      or not isinstance(providers.embedding_config, EmbeddingRuntimeConfig)):
    raise ControlledReplayConfigurationError(
      "Every modern operation requires its canonical runtime config type")
  requested_models = (
    providers.chat_config.chat_model,
    providers.completion_config.model,
    providers.embedding_config.embedding_model,
  )
  if any(model in FORBIDDEN_MODERN_RUNTIME_MODELS
         for model in requested_models):
    raise ControlledReplayLegacyConfigurationError(
      "A legacy model is forbidden in controlled replay")
  if requested_models != (
      M5_CHAT_MODEL, COMPLETION_COMPAT_MODEL,
      TEXT_EMBEDDING_3_SMALL_MODEL):
    raise ControlledReplayConfigurationError(
      "Controlled replay requires all canonical pinned models")
  if not isinstance(providers.pricing_snapshot, PricingSnapshot):
    raise ControlledReplayConfigurationError(
      "pricing_snapshot must be a PricingSnapshot")


def _manifest_identity(manifest: EmbeddingSpaceManifest):
  return ":".join((manifest.provider, manifest.model,
                   str(manifest.dimensions),
                   manifest.embedding_space_version,
                   manifest.normalization_version))


def _trusted_provider_error_type_names():
  pending = list(LLMProviderError.__subclasses__())
  trusted = {LLMProviderError.__name__}
  while pending:
    error_type = pending.pop()
    trusted.add(error_type.__name__)
    pending.extend(error_type.__subclasses__())
  return frozenset(trusted)


TRUSTED_PROVIDER_ERROR_TYPE_NAMES = _trusted_provider_error_type_names()


def _telemetry_entry(profile, event):
  return ControlledReplayTelemetryEntry(
    replay_id=profile.replay_id,
    simulation_id=event.simulation_id,
    actor_id=event.actor_id,
    step=event.simulation_step,
    cognitive_category=event.cognitive_category,
    caller_id=event.caller_id,
    operation=event.operation,
    logical_call_id=event.logical_call_id,
    physical_attempt=event.physical_attempt,
    requested_model=event.model_or_engine,
    returned_model=event.response_model,
    outcome=event.outcome,
    error_type=(
      event.error_type
      if event.error_type in TRUSTED_PROVIDER_ERROR_TYPE_NAMES
      else "UNKNOWN_PROVIDER_ERROR" if event.error_type else None),
  )


def _underlying_provider_error_type(error, events):
  if isinstance(error, LLMProviderError):
    return type(error).__name__
  if (not isinstance(error, ReplayCostAccountingUnavailableError)
      or not any(event.outcome == "ERROR" for event in events)):
    return None
  cause = error.__cause__ or error.__context__
  seen = set()
  while cause is not None and id(cause) not in seen:
    seen.add(id(cause))
    if isinstance(cause, LLMProviderError):
      return type(cause).__name__
    next_cause = cause.__cause__ or cause.__context__
    if next_cause is None:
      return "UNKNOWN_PROVIDER_ERROR"
    cause = next_cause
  return "UNKNOWN_PROVIDER_ERROR"


def _build_report(profile, events, call_guard, cost_guard, manifest,
                  cache_before, cache_after, output=None, error=None):
  cost_snapshot = cost_guard.snapshot()
  records = cost_guard.records()
  counts = {}
  for event in events:
    counts[event.operation] = counts.get(event.operation, 0) + 1
  telemetry = tuple(_telemetry_entry(profile, event) for event in events)
  ledger = tuple(ControlledReplayLedgerEntry(
    replay_id=profile.replay_id,
    simulation_id=record.simulation_id,
    actor_id=record.actor_id,
    step=record.simulation_step,
    cognitive_category=record.cognitive_category,
    caller_id=record.caller_id,
    operation=record.operation,
    logical_call_id=record.logical_call_id,
    physical_attempt=record.attempt,
    estimated_cost=record.estimated_total_cost_usd,
  ) for record in records)
  requested = tuple(dict.fromkeys(event.model_or_engine for event in events))
  returned = tuple(dict.fromkeys(
    event.response_model for event in events if event.response_model))
  legacy_detections = sum(
    event.model_or_engine in FORBIDDEN_MODERN_RUNTIME_MODELS
    or event.response_model in FORBIDDEN_MODERN_RUNTIME_MODELS
    for event in events)
  digest = None if output is None else hashlib.sha256(
    repr(output).encode("utf-8")).hexdigest()
  primary_error_type = None if error is None else type(error).__name__
  return ControlledReplayReport(
    replay_id=profile.replay_id,
    simulation_id=profile.simulation_id,
    actor_id=profile.actor_id,
    step=profile.step,
    status=SUCCESS if error is None else "FAILED",
    selected_cognitive_path=WAKE_UP_PATH,
    logical_calls=call_guard.logical_calls,
    physical_attempts=call_guard.physical_attempts,
    operation_counts=tuple(sorted(counts.items())),
    models_requested=requested,
    models_returned=returned,
    retry_count=max(0, call_guard.physical_attempts - call_guard.logical_calls),
    input_tokens=sum(event.input_tokens or 0 for event in events),
    output_tokens=sum(event.output_tokens or 0 for event in events),
    cached_tokens=sum(event.cached_input_tokens or 0 for event in events),
    reasoning_tokens=sum(event.reasoning_tokens or 0 for event in events),
    embedding_cache_hits=(cache_after.cache_hits - cache_before.cache_hits),
    embedding_cache_misses=(cache_after.cache_misses - cache_before.cache_misses),
    accumulated_cost=cost_snapshot.accumulated_cost,
    cost_ceiling=cost_snapshot.ceiling,
    remaining_cost=cost_snapshot.remaining_cost,
    legacy_detections=legacy_detections,
    error_type=primary_error_type,
    primary_error_type=primary_error_type,
    underlying_provider_error_type=(
      _underlying_provider_error_type(error, events)),
    store_manifest_identity=_manifest_identity(manifest),
    parser_result_type=None if output is None else type(output).__name__,
    wake_up_hour=(output if type(output) is int else None),
    cognitive_output_digest=digest,
    telemetry=telemetry,
    ledger=ledger,
  )


def _validate_fixture_identity(profile, actor_fixture, environment_fixture):
  if not isinstance(actor_fixture, ControlledReplayActorFixture):
    raise TypeError("actor_fixture must be ControlledReplayActorFixture")
  if not isinstance(environment_fixture, ControlledReplayEnvironmentFixture):
    raise TypeError(
      "environment_fixture must be ControlledReplayEnvironmentFixture")
  if actor_fixture.actor_id != profile.actor_id:
    raise ControlledReplayConfigurationError(
      "actor fixture identity does not match the replay profile")
  if (environment_fixture.simulation_id != profile.simulation_id
      or environment_fixture.step != profile.step):
    raise ControlledReplayConfigurationError(
      "environment fixture identity does not match the replay profile")


def run_controlled_replay_step(profile, actor_fixture, environment_fixture,
                               providers, store_path):
  """Run one offline wake-up planning decision and return a safe report."""
  if not isinstance(profile, ControlledReplayProfile):
    raise TypeError("profile must be ControlledReplayProfile")
  if not isinstance(providers, ControlledReplayProviders):
    raise TypeError("providers must be ControlledReplayProviders")
  _validate_fixture_identity(profile, actor_fixture, environment_fixture)
  _validate_modern_runtime_configuration(providers)
  assert_controlled_replay_path_allowed(WAKE_UP_CALLER)

  store_path = Path(store_path)
  bootstrap_result = bootstrap_modern_embedding_store(
    ModernEmbeddingStoreBootstrapRequest(
      target_path=store_path, allowed_parent=store_path.parent))
  manifest = bootstrap_result.manifest
  telemetry_start = len(get_telemetry())
  cache_before = get_embedding_cache_stats()
  replay_context = LLMReplayContext(
    cognitive_category=PLANNING,
    actor_id=profile.actor_id,
    simulation_id=profile.simulation_id,
    simulation_step=profile.step,
  )
  ledger_context = CostLedgerContext(
    simulation_id=profile.simulation_id,
    simulation_step=profile.step,
    actor_id=profile.actor_id,
    cognitive_category=PLANNING,
  )
  cost_config = ReplayCostGuardConfig(
    replay_id=profile.replay_id,
    simulation_id=profile.simulation_id,
    ceiling=ReplayCostCeiling(profile.maximum_cost),
    pricing_snapshot=providers.pricing_snapshot,
  )

  with use_embedding_runtime(
      providers.embedding_config, providers.embedding_adapter,
      store_path=store_path, legacy_assumption_allowed=False):
    with use_modern_chat_runtime(
        providers.chat_config, providers.chat_adapter):
      with use_modern_completion_runtime(
          providers.completion_config, providers.completion_adapter):
        with use_llm_replay_context(replay_context):
          with use_cost_ledger_context(ledger_context):
            with use_replay_call_count_guard(profile) as call_guard:
              with use_replay_cost_guard(cost_config) as cost_guard:
                with _use_combined_replay_observer(call_guard, cost_guard):
                  try:
                    with _historical_wrapper_environment():
                      output = run_gpt_prompt.run_gpt_prompt_wake_up_hour(
                        actor_fixture.persona,
                        test_input=list(actor_fixture.prompt_input))[0]
                  except Exception as error:
                    events = get_telemetry()[telemetry_start:]
                    report = _build_report(
                      profile, events, call_guard, cost_guard, manifest,
                      cache_before, get_embedding_cache_stats(), error=error)
                    try:
                      error.controlled_replay_report = report
                    except Exception:
                      pass
                    raise
                  return _build_report(
                    profile, get_telemetry()[telemetry_start:], call_guard,
                    cost_guard, manifest, cache_before,
                    get_embedding_cache_stats(), output=output)


def controlled_replay_contexts_are_reset():
  """Content-free diagnostic used by the R0H teardown tests."""
  from persona.prompt_template.chat_runtime import get_modern_chat_runtime_config
  from persona.prompt_template.completion_runtime import (
    get_modern_completion_runtime_config,
  )
  from persona.memory_structures.embedding_space import (
    LEGACY_ADA_002_MANIFEST, get_runtime_embedding_manifest,
  )
  return (
    get_modern_chat_runtime_config() is None
    and get_modern_completion_runtime_config() is None
    and get_llm_replay_context() == LLMReplayContext()
    and get_replay_cost_guard() is None
    and get_replay_call_count_guard() is None
    and get_runtime_embedding_manifest() == LEGACY_ADA_002_MANIFEST
  )
