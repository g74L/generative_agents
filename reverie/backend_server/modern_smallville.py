"""Operational headless launcher for the modern Smallville runtime."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Optional


BACKEND_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = BACKEND_ROOT.parents[1]
if str(BACKEND_ROOT) not in sys.path:
  sys.path.insert(0, str(BACKEND_ROOT))

import controlled_replay
from persona.prompt_template import run_gpt_prompt
from persona.prompt_template.chat_runtime import use_modern_chat_runtime
from persona.prompt_template.completion_runtime import (
  use_modern_completion_runtime,
)
from persona.prompt_template.cost_ledger import (
  CostLedgerContext, use_cost_ledger_context,
)
from persona.prompt_template.embedding_runtime import (
  TEXT_EMBEDDING_3_SMALL_1536_MANIFEST, use_embedding_runtime,
)
from persona.prompt_template.llm_provider import (
  LLMReplayContext, clear_telemetry, get_telemetry, reset_provider,
  use_llm_attempt_observer, use_llm_replay_context,
)
from utils import collision_block_id
from persona.prompt_template.modern_openai_provider import (
  LLMIncompleteResponseError, ModernOpenAIClientAdapter,
)
from persona.prompt_template.replay_cost_guard import (
  ReplayCostCeiling, ReplayCostCeilingExceededError,
  ReplayCostGuardAlreadyTrippedError, ReplayCostGuardConfig,
  use_replay_cost_guard,
)


DEFAULT_SOURCE = "base_the_ville_isabella_maria_klaus"
COGNITIVE_ACTOR = "Isabella Rodriguez"
PASSIVE_ACTORS = ("Maria Lopez", "Klaus Mueller")
VISIBLE_ACTORS = (COGNITIVE_ACTOR,) + PASSIVE_ACTORS
SLEEPING_ADDRESSES = {
  "Isabella Rodriguez":
    "the Ville:Isabella Rodriguez's apartment:main room:bed",
  "Maria Lopez": "the Ville:Dorm for Oak Hill College:Maria Lopez's room:bed",
  "Klaus Mueller":
    "the Ville:Dorm for Oak Hill College:Klaus Mueller's room:bed",
}
DEFAULT_COST_CEILING = Decimal("0.03")
RUNTIME_ROOT = REPOSITORY_ROOT / ".runtime" / "live-runs"
SOURCE_ROOT = (REPOSITORY_ROOT / "environment" / "frontend_server"
               / "storage")
DATE_FORMAT = "%B %d, %Y, %H:%M:%S"
CANONICAL_START_TIME = dt.datetime(2023, 2, 13, 5, 55, 0)
R1M3C_START_TIME = dt.datetime(2023, 2, 13, 10, 0, 0)
R1M3C_ACTORS = ("Maria Lopez", "Klaus Mueller")
R1M3C_COORDINATES = {
  "Isabella Rodriguez": (78, 19),
  "Maria Lopez": (117, 49),
  "Klaus Mueller": (118, 49),
}
R1M3C_ACTIONS = {
  "Isabella Rodriguez": (
    "the Ville:Hobbs Cafe:cafe:behind the cafe counter",
    "working at the cafe counter"),
  "Maria Lopez": (
    "the Ville:Dorm for Oak Hill College:common room",
    "taking a short break in the common room"),
  "Klaus Mueller": (
    "the Ville:Dorm for Oak Hill College:common room",
    "reading in the common room"),
}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

EXIT_SUCCESS = 0
EXIT_RUNTIME_FAILURE = 1
EXIT_CONFIGURATION = 2
EXIT_COST_CEILING = 3
EXIT_MODERN_INVARIANT = 4

R1M3C_NATURAL_VERDICT = (
  "R1M3_C_NATURAL_CONVERSATION_AND_BILATERAL_MEMORY_PASSED")
R1M3C_FUNCTIONAL_VERDICT = (
  "R1M3_C_SOCIAL_PIPELINE_FUNCTIONAL_MODEL_END_NOT_OBSERVED")
R1M3C_MEMORY_BLOCKED_VERDICT = "R1M3_C_CONVERSATION_MEMORY_BLOCKED"
R1CLI_A2_A_READY_VERDICT = (
  "R1CLI_A2_A_STRUCTURAL_RESUME_READY_FOR_PROCESS_BOUNDARY_REVIEW")
R1CLI_A2_A_BLOCKED_VERDICT = "R1CLI_A2_A_BLOCKED"
R1CLI_A2_B_READY_VERDICT = (
  "R1CLI_A2_B_CAUSAL_SOCIAL_MEMORY_READY_FOR_LIVE_PROCESS_BOUNDARY_REVIEW")
R1CLI_A2_B_BLOCKED_VERDICT = "R1CLI_A2_B_CAUSAL_SOCIAL_MEMORY_BLOCKED"
R1CLI_A2_B_PATH_NOT_REACHED_VERDICT = (
  "R1CLI_A2_B_CAUSAL_SOCIAL_PATH_NOT_REACHED")

RESUME_SCRATCH_FIELDS = (
  "curr_time", "curr_tile", "daily_plan_req", "daily_req",
  "f_daily_schedule", "f_daily_schedule_hourly_org",
  "act_address", "act_start_time", "act_duration", "act_description",
  "act_event", "act_obj_description", "act_obj_pronunciatio",
  "act_obj_event", "chatting_with", "chat", "chatting_with_buffer",
  "chatting_end_time", "act_path_set", "planned_path",
  "importance_trigger_max", "importance_trigger_curr", "importance_ele_n",
  "thought_count", "name", "first_name", "last_name", "age", "innate",
  "learned", "currently", "lifestyle", "living_area",
)


def _classify_r1m3c_conversation(
    *, conversation_committed, model_end_observed,
    safety_ceiling_reached, bilateral_memory,
    bilateral_memory_reloaded, memory_integrity_valid,
    save_passed, reload_passed):
  persistence_valid = all((
    conversation_committed, bilateral_memory, bilateral_memory_reloaded,
    memory_integrity_valid, save_passed, reload_passed))
  if not persistence_valid:
    return R1M3C_MEMORY_BLOCKED_VERDICT
  if model_end_observed:
    return R1M3C_NATURAL_VERDICT
  if safety_ceiling_reached:
    return R1M3C_FUNCTIONAL_VERDICT
  return R1M3C_MEMORY_BLOCKED_VERDICT


class ModernSmallvilleError(RuntimeError):
  """Base error for the operational launcher."""


class ModernRunConfigurationError(ModernSmallvilleError, ValueError):
  pass


class ModernRuntimeInvariantError(ModernSmallvilleError):
  pass


class ModernDeferredCallerError(ModernRuntimeInvariantError):
  def __init__(self, caller, operation, actor, tick):
    self.caller = caller
    self.operation = operation
    self.actor = actor
    self.tick = tick
    super().__init__(
      f"deferred caller reached: caller={caller}, operation={operation}, "
      f"actor={actor}, tick={tick}")


@dataclass(frozen=True)
class ModernRunConfig:
  source_simulation: str = DEFAULT_SOURCE
  run_name: str = "modern-smallville"
  ticks: int = 1
  tick_seconds: int = 10
  cognitive_actors: tuple[str, ...] = (COGNITIVE_ACTOR,)
  passive_actors: tuple[str, ...] = PASSIVE_ACTORS
  visible_actors: tuple[str, ...] = VISIBLE_ACTORS
  cost_ceiling_usd: Decimal = DEFAULT_COST_CEILING
  controlled_proximity: bool = False

  def __post_init__(self):
    if type(self.ticks) is not int or self.ticks <= 0:
      raise ModernRunConfigurationError("ticks must be greater than zero")
    if type(self.tick_seconds) is not int or self.tick_seconds <= 0:
      raise ModernRunConfigurationError(
        "tick_seconds must be greater than zero")
    if (not isinstance(self.cost_ceiling_usd, Decimal)
        or not self.cost_ceiling_usd.is_finite()
        or self.cost_ceiling_usd <= 0):
      raise ModernRunConfigurationError(
        "cost ceiling must be a finite Decimal greater than zero")
    if not isinstance(self.run_name, str) or not self.run_name.strip():
      raise ModernRunConfigurationError("run name must not be empty")
    if not SAFE_NAME.fullmatch(self.run_name):
      raise ModernRunConfigurationError(
        "run name may contain only letters, digits, dot, underscore and dash")
    if (not isinstance(self.source_simulation, str)
        or not SAFE_NAME.fullmatch(self.source_simulation)):
      raise ModernRunConfigurationError("source simulation name is invalid")
    if (not isinstance(self.cognitive_actors, tuple)
        or not self.cognitive_actors
        or len(set(self.cognitive_actors)) != len(self.cognitive_actors)):
      raise ModernRunConfigurationError(
        "cognitive actors must be a non-empty tuple of unique names")
    if (not isinstance(self.passive_actors, tuple)
        or len(set(self.passive_actors)) != len(self.passive_actors)):
      raise ModernRunConfigurationError(
        "passive actors must be a tuple of unique names")
    if self.visible_actors != VISIBLE_ACTORS:
      raise ModernRunConfigurationError(
        "modern Smallville requires Isabella, Maria and Klaus as visible actors")
    cognitive = set(self.cognitive_actors)
    passive = set(self.passive_actors)
    visible = set(self.visible_actors)
    if not cognitive.issubset(visible) or not passive.issubset(visible):
      raise ModernRunConfigurationError(
        "every cognitive and passive actor must be visible")
    if cognitive & passive:
      raise ModernRunConfigurationError(
        "cognitive and passive actors must not overlap")
    if cognitive | passive != visible:
      raise ModernRunConfigurationError(
        "every visible actor must be cognitive or passive")
    expected_cognitive_order = tuple(
      name for name in self.visible_actors if name in cognitive)
    expected_passive_order = tuple(
      name for name in self.visible_actors if name in passive)
    if (self.cognitive_actors != expected_cognitive_order
        or self.passive_actors != expected_passive_order):
      raise ModernRunConfigurationError(
        "actor policy must preserve the visible actor order")
    if self.tick_seconds != 10:
      raise ModernRunConfigurationError("R1CLI-A requires 10-second ticks")
    if type(self.controlled_proximity) is not bool:
      raise ModernRunConfigurationError(
        "controlled_proximity must be a boolean")
    if self.controlled_proximity:
      if self.cognitive_actors != VISIBLE_ACTORS or self.passive_actors:
        raise ModernRunConfigurationError(
          "R1M3-C requires three cognitive actors and no passive actors")
      if self.ticks > 10:
        raise ModernRunConfigurationError(
          "R1M3-C permits at most ten ticks")


@dataclass(frozen=True)
class ModernResumeConfig:
  source_run: Path
  run_name: str
  ticks: int = 5
  cost_ceiling_usd: Decimal = DEFAULT_COST_CEILING
  observe_causal_social_memory: bool = False

  def __post_init__(self):
    if not isinstance(self.source_run, Path):
      raise ModernRunConfigurationError("resume source must be a Path")
    if type(self.ticks) is not int or self.ticks <= 0:
      raise ModernRunConfigurationError("ticks must be greater than zero")
    if (not isinstance(self.cost_ceiling_usd, Decimal)
        or not self.cost_ceiling_usd.is_finite()
        or self.cost_ceiling_usd <= 0):
      raise ModernRunConfigurationError(
        "cost ceiling must be a finite Decimal greater than zero")
    if not isinstance(self.run_name, str) or not self.run_name.strip():
      raise ModernRunConfigurationError("run name must not be empty")
    if not SAFE_NAME.fullmatch(self.run_name):
      raise ModernRunConfigurationError(
        "run name may contain only letters, digits, dot, underscore and dash")
    if type(self.observe_causal_social_memory) is not bool:
      raise ModernRunConfigurationError(
        "observe_causal_social_memory must be a boolean")


@dataclass(frozen=True)
class _ResumeContext:
  request: ModernResumeConfig
  source_run: Path
  simulation_root: Path
  source_meta: dict[str, Any]
  cognitive_actors: tuple[str, ...]
  passive_actors: tuple[str, ...]
  movement_hashes: dict[str, str]
  environment_hashes: dict[str, str]
  actor_state: dict[str, dict[str, Any]]
  embedding_audits: tuple[Any, ...]
  causal_social_memory: dict[str, Any]


@dataclass(frozen=True)
class ModernRunResult:
  verdict: str
  run_directory: Path
  cognitive_actors: tuple[str, ...]
  passive_actors: tuple[str, ...]
  completed_ticks: int
  movement_count: int
  initial_step: int
  final_step: int
  initial_time: Optional[dt.datetime]
  final_time: Optional[dt.datetime]
  logical_calls: int
  physical_attempts: int
  input_tokens: int
  output_tokens: int
  total_cost_usd: Decimal
  cost_ceiling_usd: Decimal
  save_passed: bool
  reload_passed: bool
  actor_move_counts: tuple[tuple[str, int], ...]
  passive_provider_calls: int
  passive_memory_mutations: int
  legacy_fallback_count: int
  retry_count: int
  exception_type: Optional[str] = None
  exception_message: Optional[str] = None


def generate_run_name(now: Optional[dt.datetime] = None) -> str:
  stamp = now or dt.datetime.now()
  return "modern-smallville-" + stamp.strftime("%Y%m%d-%H%M%S")


def _normalize(value):
  if isinstance(value, Decimal):
    return str(value)
  if isinstance(value, dt.datetime):
    return value.strftime(DATE_FORMAT)
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {str(key): _normalize(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_normalize(item) for item in value]
  return value


def _write_json(path: Path, value: Any) -> None:
  temporary = path.with_name(path.name + ".tmp")
  with temporary.open("w", encoding="utf-8") as stream:
    json.dump(_normalize(value), stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(temporary, path)


def _read_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(65536), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  for child in sorted(item for item in path.rglob("*") if item.is_file()):
    digest.update(child.relative_to(path).as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(_file_sha256(child).encode("ascii"))
    digest.update(b"\n")
  return digest.hexdigest()


def _state_sha256(value: Any) -> str:
  payload = json.dumps(
    _normalize(value), sort_keys=True, separators=(",", ":"),
    ensure_ascii=False)
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scratch_resume_state(scratch_or_mapping) -> dict[str, Any]:
  if isinstance(scratch_or_mapping, dict):
    get_value = scratch_or_mapping.get
  else:
    get_value = lambda field: getattr(scratch_or_mapping, field, None)
  return {
    field: _normalize(get_value(field)) for field in RESUME_SCRATCH_FIELDS}


def _canonical_node_ids(node_ids):
  def key(node_id):
    match = re.fullmatch(r"node_(\d+)", node_id)
    return (0, int(match.group(1))) if match else (1, node_id)
  return sorted(node_ids, key=key)


def _node_field(node, field, default=None):
  if isinstance(node, dict):
    return node.get(field, default)
  return getattr(node, field, default)


def _content_hash(value) -> str:
  return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _node_created_timestamp(value):
  if isinstance(value, dt.datetime):
    return value.strftime("%Y-%m-%d %H:%M:%S")
  return str(value) if value is not None else None


def _social_memory_actor_snapshot(actor_name, nodes) -> dict[str, Any]:
  """Return content-free Chat -> Event/Thought lineage metadata."""
  chat_ids = {
    node_id for node_id, node in nodes.items()
    if _node_field(node, "type") == "chat"}
  chats = []
  derived_events = []
  derived_thoughts = []
  last_accessed_missing = False
  for node_id in _canonical_node_ids(nodes):
    node = nodes[node_id]
    node_type = _node_field(node, "type")
    if node_type not in ("chat", "event", "thought"):
      continue
    filling = _node_field(node, "filling") or []
    if not isinstance(filling, (tuple, list)):
      filling = []
    lineage = [
      item for item in filling
      if isinstance(item, str) and item in chat_ids]
    if node_type != "chat" and not lineage:
      continue
    created = _node_created_timestamp(_node_field(node, "created"))
    metadata = {
      "actor": actor_name,
      "node_id": node_id,
      "node_ref": f"{actor_name}::{node_id}",
      "node_type": node_type,
      "created": created,
      "embedding_key_hash": _content_hash(
        _node_field(node, "embedding_key", "")),
    }
    if node_type == "chat":
      metadata.update({
        "participant": _normalize(_node_field(node, "object")),
        "filling_present": bool(filling),
      })
      chats.append(metadata)
    else:
      metadata.update({
        "source_chat_node_ids": _canonical_node_ids(lineage),
        "lineage_field": "filling" if node_type == "event" else "evidence",
      })
      (derived_events if node_type == "event"
       else derived_thoughts).append(metadata)
    if ((isinstance(node, dict) and "last_accessed" not in node)
        or (not isinstance(node, dict)
            and not hasattr(node, "last_accessed"))):
      last_accessed_missing = True
  return {
    "chat_nodes": chats,
    "derived_event_nodes": derived_events,
    "derived_thought_nodes": derived_thoughts,
    "chat_node_ids": [item["node_id"] for item in chats],
    "derived_node_ids": [
      item["node_id"] for item in derived_events + derived_thoughts],
    "last_accessed_not_serialized": last_accessed_missing,
  }


def _social_memory_snapshot(actor_nodes) -> dict[str, Any]:
  actors = {
    name: _social_memory_actor_snapshot(name, nodes)
    for name, nodes in actor_nodes.items() if name in R1M3C_ACTORS}
  expected_participant = {
    R1M3C_ACTORS[0]: R1M3C_ACTORS[1],
    R1M3C_ACTORS[1]: R1M3C_ACTORS[0],
  }

  def has_chat_event_lineage(actor_name):
    actor = actors.get(actor_name, {})
    chat_ids = {
      item["node_id"] for item in actor.get("chat_nodes", ())
      if item.get("participant") == expected_participant[actor_name]}
    return bool(chat_ids) and any(
      chat_ids.intersection(item.get("source_chat_node_ids", ()))
      for item in actor.get("derived_event_nodes", ()))

  return {
    "actors": actors,
    "chat_node_ids": _canonical_node_ids({
      node_id for actor in actors.values()
      for node_id in actor["chat_node_ids"]}),
    "derived_node_ids": _canonical_node_ids({
      node_id for actor in actors.values()
      for node_id in actor["derived_node_ids"]}),
    "chat_node_refs": sorted(
      item["node_ref"] for actor in actors.values()
      for item in actor["chat_nodes"]),
    "derived_node_refs": sorted(
      item["node_ref"] for actor in actors.values()
      for item in (actor["derived_event_nodes"]
                   + actor["derived_thought_nodes"])),
    "bilateral_chat_lineage_present": (
      set(actors) == set(R1M3C_ACTORS)
      and all(has_chat_event_lineage(name) for name in R1M3C_ACTORS)),
    "last_accessed_persistence_gap_detected": any(
      actor["last_accessed_not_serialized"] for actor in actors.values()),
  }


def _compare_social_memory_hydration(pre_resume, hydrated):
  actor_checks = {}
  for name, before in pre_resume.get("actors", {}).items():
    after = hydrated.get("actors", {}).get(name, {})
    checks = {
      "chat_nodes_preserved": (
        before.get("chat_nodes") == after.get("chat_nodes")),
      "derived_event_nodes_preserved": (
        before.get("derived_event_nodes")
        == after.get("derived_event_nodes")),
      "derived_thought_nodes_preserved": (
        before.get("derived_thought_nodes")
        == after.get("derived_thought_nodes")),
    }
    checks["derived_nodes_preserved"] = all((
      checks["derived_event_nodes_preserved"],
      checks["derived_thought_nodes_preserved"]))
    checks["lineage_preserved"] = all(checks.values())
    actor_checks[name] = checks
  raw_persisted = bool(actor_checks) and all(
    item["chat_nodes_preserved"] for item in actor_checks.values())
  derived_persisted = bool(actor_checks) and all(
    item["derived_nodes_preserved"] for item in actor_checks.values())
  lineage_preserved = bool(actor_checks) and all(
    item["lineage_preserved"] for item in actor_checks.values())
  return {
    "actors": actor_checks,
    "chat_nodes_preserved": raw_persisted,
    "derived_nodes_preserved": derived_persisted,
    "lineage_preserved": lineage_preserved,
    "raw_social_memory_persisted": raw_persisted,
    "derived_social_memory_persisted": derived_persisted,
  }


def _classify_causal_social_memory(
    *, raw_social_memory_persisted, derived_social_memory_persisted,
    lineage_preserved_after_hydration, pre_resume_derived_node_ids,
    retrieved_node_ids, consumed_node_ids):
  expected = set(pre_resume_derived_node_ids)
  retrieved = expected & set(retrieved_node_ids)
  consumed = retrieved & set(consumed_node_ids)
  derived_retrieved = bool(retrieved)
  derived_consumed = bool(consumed)
  causal = all((
    raw_social_memory_persisted,
    derived_social_memory_persisted,
    lineage_preserved_after_hydration,
    derived_retrieved,
    derived_consumed,
  ))
  return {
    "raw_social_memory_persisted": bool(raw_social_memory_persisted),
    "derived_social_memory_persisted": bool(
      derived_social_memory_persisted),
    "lineage_preserved_after_hydration": bool(
      lineage_preserved_after_hydration),
    "derived_social_memory_retrieved": derived_retrieved,
    "derived_social_memory_consumed": derived_consumed,
    "retrieved_pre_resume_derived_social_node_refs": sorted(retrieved),
    "consumed_pre_resume_derived_social_node_refs": sorted(consumed),
    "retrieved_pre_resume_derived_social_node_ids": _canonical_node_ids(
      {item.split("::", 1)[-1] for item in retrieved}),
    "consumed_pre_resume_derived_social_node_ids": _canonical_node_ids(
      {item.split("::", 1)[-1] for item in consumed}),
    "social_memory_causally_reused": causal,
    "causal_link_verified": causal,
  }


def _persisted_actor_resume_metadata(
    simulation_root: Path, actor_name: str) -> dict[str, Any]:
  memory_root = (simulation_root / "personas" / actor_name
                 / "bootstrap_memory")
  required = (
    memory_root / "scratch.json",
    memory_root / "spatial_memory.json",
    memory_root / "associative_memory" / "nodes.json",
    memory_root / "associative_memory" / "kw_strength.json",
    memory_root / "associative_memory" / "embeddings.json",
  )
  if any(not path.is_file() for path in required):
    raise ModernRunConfigurationError(
      f"Persona persistence is incomplete: actor={actor_name}")
  try:
    scratch = _read_json(required[0])
    spatial = _read_json(required[1])
    nodes = _read_json(required[2])
    embeddings = _read_json(required[4])
  except (OSError, ValueError, TypeError) as error:
    raise ModernRunConfigurationError(
      f"Persona persistence is invalid: actor={actor_name}") from error
  if not all(isinstance(value, dict)
             for value in (scratch, spatial, nodes, embeddings)):
    raise ModernRunConfigurationError(
      f"Persona persistence has an invalid shape: actor={actor_name}")
  raw_node_ids = list(nodes)
  node_ids = _canonical_node_ids(raw_node_ids)
  references = {
    node.get("embedding_key") for node in nodes.values()
    if isinstance(node, dict) and node.get("embedding_key") is not None}
  node_types = Counter(
    node.get("type") for node in nodes.values() if isinstance(node, dict))
  return {
    "identity_aligned": scratch.get("name") == actor_name,
    "memory_node_count": len(nodes),
    "event_count": node_types["event"],
    "thought_count": node_types["thought"],
    "chat_count": node_types["chat"],
    "embedding_count": len(embeddings),
    "node_ids": node_ids,
    "node_ids_valid": node_ids == [
      f"node_{index}" for index in range(1, len(node_ids) + 1)],
    "node_ids_unique": len(raw_node_ids) == len(set(raw_node_ids)),
    "embedding_references_valid": references.issubset(set(embeddings)),
    "orphan_embedding_count": len(set(embeddings) - references),
    "scratch_state_hash": _state_sha256(_scratch_resume_state(scratch)),
    "embedding_state_hash": _state_sha256(embeddings),
    "spatial_memory_hash": _state_sha256(spatial),
    "persona_files_hash": _tree_sha256(memory_root),
    "scratch_curr_time": scratch.get("curr_time"),
  }


def _loaded_actor_resume_metadata(persona) -> dict[str, Any]:
  nodes = persona.a_mem.id_to_node
  raw_node_ids = list(nodes)
  node_ids = _canonical_node_ids(raw_node_ids)
  embeddings = persona.a_mem.embeddings
  references = {
    node.embedding_key for node in nodes.values()
    if getattr(node, "embedding_key", None) is not None}
  return {
    "identity_aligned": persona.name == persona.scratch.name,
    "memory_node_count": len(nodes),
    "event_count": len(persona.a_mem.seq_event),
    "thought_count": len(persona.a_mem.seq_thought),
    "chat_count": len(persona.a_mem.seq_chat),
    "embedding_count": len(embeddings),
    "node_ids": node_ids,
    "node_ids_valid": node_ids == [
      f"node_{index}" for index in range(1, len(node_ids) + 1)],
    "node_ids_unique": len(raw_node_ids) == len(set(raw_node_ids)),
    "embedding_references_valid": references.issubset(set(embeddings)),
    "orphan_embedding_count": len(set(embeddings) - references),
    "scratch_state_hash": _state_sha256(
      _scratch_resume_state(persona.scratch)),
    "embedding_state_hash": _state_sha256(embeddings),
    "spatial_memory_hash": _state_sha256(persona.s_mem.tree),
  }


def _persisted_social_memory_snapshot(simulation_root: Path):
  actor_nodes = {}
  for name in R1M3C_ACTORS:
    nodes_path = (simulation_root / "personas" / name / "bootstrap_memory"
                  / "associative_memory" / "nodes.json")
    if nodes_path.is_file():
      nodes = _read_json(nodes_path)
      if isinstance(nodes, dict):
        actor_nodes[name] = nodes
  return _social_memory_snapshot(actor_nodes)


def _loaded_social_memory_snapshot(personas):
  return _social_memory_snapshot({
    name: personas[name].a_mem.id_to_node
    for name in R1M3C_ACTORS if name in personas})


def _resume_actor_hydration_checks(persisted, loaded, copied_hash):
  compared = (
    "identity_aligned", "memory_node_count", "event_count",
    "thought_count", "chat_count", "embedding_count", "node_ids",
    "node_ids_valid", "node_ids_unique", "embedding_references_valid",
    "orphan_embedding_count", "scratch_state_hash",
    "embedding_state_hash", "spatial_memory_hash",
  )
  checks = {field: persisted[field] == loaded[field] for field in compared}
  checks["persona_files_preserved"] = (
    persisted["persona_files_hash"] == copied_hash)
  checks["all_checks_passed"] = all(checks.values())
  return checks


def _bootstrap_isolated_temporal_source(
    source: Path, cognitive_actors: tuple[str, ...]) -> None:
  """Install the R1V-validated sleeping state in a caller-owned copy."""
  meta_path = source / "reverie" / "meta.json"
  meta = _read_json(meta_path)
  meta.update({
    "curr_time": CANONICAL_START_TIME.strftime(DATE_FORMAT),
    "sec_per_step": 10,
    "step": 0,
  })
  meta["persona_names"] = list(VISIBLE_ACTORS)
  _write_json(meta_path, meta)
  environment_dir = source / "environment"
  initial_environment = environment_dir / "0.json"
  if not initial_environment.is_file():
    raise ModernRunConfigurationError(
      "source simulation has no initial environment")
  for path in environment_dir.glob("*.json"):
    if path != initial_environment:
      path.unlink()
  movement_dir = source / "movement"
  if movement_dir.exists():
    shutil.rmtree(movement_dir)
  movement_dir.mkdir()
  midnight = dt.datetime(2023, 2, 13, 0, 0).strftime(DATE_FORMAT)
  current = CANONICAL_START_TIME.strftime(DATE_FORMAT)
  for name in VISIBLE_ACTORS:
    scratch_path = (source / "personas" / name / "bootstrap_memory"
                    / "scratch.json")
    scratch = _read_json(scratch_path)
    scratch["curr_time"] = current
    scratch["daily_req"] = ["sleeping", "daily activities"]
    scratch["daily_plan_req"] = ["sleeping", "daily activities"]
    scratch["f_daily_schedule"] = [
      ["sleeping", 360], ["daily activities", 1080]]
    scratch["f_daily_schedule_hourly_org"] = list(
      scratch["f_daily_schedule"])
    if name in cognitive_actors:
      scratch.update({
        "act_address": SLEEPING_ADDRESSES[name],
        "act_description": "sleeping", "act_duration": 360,
        "act_event": [name, "is", "sleeping"], "act_path_set": False,
        "act_pronunciatio": "", "act_start_time": midnight,
        "planned_path": [],
      })
    else:
      scratch.update({
        "act_address": None, "act_description": None, "act_duration": None,
        "act_event": [name, None, None], "act_path_set": False,
        "act_pronunciatio": None, "act_start_time": None,
        "planned_path": [],
      })
    _write_json(scratch_path, scratch)


def _bootstrap_controlled_proximity_source(source: Path) -> dict[str, Any]:
  """Create the daytime R1M3-C opportunity in an already isolated copy."""
  meta_path = source / "reverie" / "meta.json"
  meta = _read_json(meta_path)
  meta.update({
    "curr_time": R1M3C_START_TIME.strftime(DATE_FORMAT),
    "sec_per_step": 10,
    "step": 0,
    "persona_names": list(VISIBLE_ACTORS),
  })
  _write_json(meta_path, meta)

  environment_dir = source / "environment"
  initial_environment = environment_dir / "0.json"
  if not initial_environment.is_file():
    raise ModernRunConfigurationError(
      "source simulation has no initial environment")
  for path in environment_dir.glob("*.json"):
    if path != initial_environment:
      path.unlink()
  movement_dir = source / "movement"
  if movement_dir.exists():
    shutil.rmtree(movement_dir)
  movement_dir.mkdir()

  environment = _read_json(initial_environment)
  action_start = dt.datetime(2023, 2, 13, 8, 0, 0)
  for name in VISIBLE_ACTORS:
    x, y = R1M3C_COORDINATES[name]
    environment[name] = {"maze": "the_ville", "x": x, "y": y}
    address, description = R1M3C_ACTIONS[name]
    scratch_path = (source / "personas" / name / "bootstrap_memory"
                    / "scratch.json")
    scratch = _read_json(scratch_path)
    schedule = [
      ["sleeping", 480], [description, 240], ["daily activities", 720]]
    scratch.update({
      "curr_time": R1M3C_START_TIME.strftime(DATE_FORMAT),
      "curr_tile": [x, y],
      "daily_req": [description, "daily activities"],
      "daily_plan_req": [description, "daily activities"],
      "f_daily_schedule": schedule,
      "f_daily_schedule_hourly_org": [list(row) for row in schedule],
      "act_address": address,
      "act_description": description,
      "act_duration": 240,
      "act_event": [name, "is", description],
      "act_path_set": True,
      "act_pronunciatio": "",
      "act_start_time": action_start.strftime(DATE_FORMAT),
      "planned_path": [],
      "act_obj_description": None,
      "act_obj_pronunciatio": None,
      "act_obj_event": [None, None, None],
      "chatting_with": None,
      "chat": None,
      "chatting_with_buffer": {},
      "chatting_end_time": None,
    })
    _write_json(scratch_path, scratch)
  _write_json(initial_environment, environment)
  return {
    "time": R1M3C_START_TIME,
    "semantic_location":
      "the Ville:Dorm for Oak Hill College:common room",
    "coordinates": R1M3C_COORDINATES,
    "distance": math.dist(
      R1M3C_COORDINATES[R1M3C_ACTORS[0]],
      R1M3C_COORDINATES[R1M3C_ACTORS[1]]),
  }


def passive_cognitive_fingerprint(persona):
  """Return content-free hashes for state passive rendering must preserve."""
  def safe(item):
    if isinstance(item, dt.datetime):
      return item.isoformat()
    if isinstance(item, dict):
      return {str(k): safe(v) for k, v in sorted(
        item.items(), key=lambda row: str(row[0]))}
    if isinstance(item, (list, tuple)):
      return [safe(v) for v in item]
    if isinstance(item, set):
      return sorted((safe(v) for v in item), key=repr)
    if item is None or isinstance(item, (str, int, float, bool)):
      return item
    if hasattr(item, "__dict__"):
      return safe(vars(item))
    return type(item).__name__

  def digest(value):
    import hashlib
    payload = json.dumps(
      safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

  nodes = getattr(persona.a_mem, "id_to_node", {})
  embeddings = getattr(persona.a_mem, "embeddings", {})
  return {
    "associative_memory": digest({
      "nodes": nodes,
      "embeddings": embeddings,
      "event_sequence": getattr(persona.a_mem, "seq_event", ()),
      "thought_sequence": getattr(persona.a_mem, "seq_thought", ()),
      "chat_sequence": getattr(persona.a_mem, "seq_chat", ()),
    }),
    "spatial_memory": digest(getattr(persona.s_mem, "tree", {})),
    "scratch": digest(vars(persona.scratch)),
    "daily_plan": digest(getattr(persona.scratch, "daily_plan_req", None)),
    "schedule": digest(getattr(persona.scratch, "f_daily_schedule", None)),
    "action": digest({
      name: getattr(persona.scratch, name, None)
      for name in (
        "act_address", "act_description", "act_event", "act_pronunciatio",
        "act_start_time", "act_duration", "planned_path")
    }),
  }


class PassiveVisibleMoveController:
  """Emit stable visible frames without invoking passive cognition."""

  def __init__(self, server, passive_names):
    self.server = server
    self.passive_names = tuple(passive_names)
    self.frame_emissions = Counter()
    self.original_moves = {}

  def install(self):
    if len(set(self.passive_names)) != len(self.passive_names):
      raise ValueError("passive actor names must be unique")
    for name in self.passive_names:
      if name not in self.server.personas or name not in self.server.personas_tile:
        raise KeyError(name)
      persona = self.server.personas[name]
      self.original_moves[name] = persona.move

      def passive_move(maze, personas, curr_tile, curr_time,
                       _name=name, _persona=persona):
        del maze, personas, curr_tile, curr_time
        self.frame_emissions[_name] += 1
        tile = self.server.personas_tile[_name]
        emoji = getattr(_persona.scratch, "act_pronunciatio", None) or ""
        description = getattr(
          _persona.scratch, "act_description", None) or "idle"
        return tuple(tile), str(emoji), str(description)

      persona.move = passive_move
    return self

  def restore(self):
    for name, original in self.original_moves.items():
      self.server.personas[name].move = original


@contextmanager
def install_passive_visible_moves(server, passive_names):
  controller = PassiveVisibleMoveController(server, passive_names).install()
  try:
    yield controller
  finally:
    controller.restore()


@contextmanager
def _use_event_poignancy_fail_safe():
  """Make the declared score-4 fail-safe operational only for this runner."""
  from persona.cognitive_modules import perceive
  original = perceive.run_gpt_prompt_event_poignancy

  def guarded(*args, **kwargs):
    result = original(*args, **kwargs)
    return result if result is not None else (4, ())

  perceive.run_gpt_prompt_event_poignancy = guarded
  try:
    yield
  finally:
    perceive.run_gpt_prompt_event_poignancy = original


class _RuntimeObserver:
  def __init__(self, cost_guard):
    self.cost_guard = cost_guard
    self.logical_ids = set()
    self.physical_attempts = 0

  def before_attempt(self, **kwargs):
    replay_context = kwargs["replay_context"]
    caller = replay_context.caller_id
    if caller in controlled_replay.FORBIDDEN_COGNITIVE_CALLERS:
      raise ModernDeferredCallerError(
        caller, kwargs["operation"], replay_context.actor_id,
        replay_context.simulation_step)
    self.cost_guard.before_attempt(**kwargs)
    self.logical_ids.add(kwargs["logical_call_id"])
    self.physical_attempts += 1

  def after_attempt(self, event):
    self.cost_guard.after_attempt(event)


class _NullOutput:
  def write(self, value):
    return len(value)

  def flush(self):
    return None


def _load_reverie_module():
  module_name = "modern_smallville_reverie_runtime"
  spec = importlib.util.spec_from_file_location(
    module_name, BACKEND_ROOT / "reverie.py")
  if spec is None or spec.loader is None:
    raise ModernRuntimeInvariantError("Reverie runtime module is unavailable")
  reverie_module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(reverie_module)
  return reverie_module


def _default_adapter():
  if not os.getenv("OPENAI_API_KEY"):
    raise ModernRunConfigurationError("OPENAI_API_KEY is not set")
  from openai import OpenAI
  return ModernOpenAIClientAdapter(
    client=OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0))


def _memory_counts(persona):
  return {
    "nodes": len(persona.a_mem.id_to_node),
    "embeddings": len(persona.a_mem.embeddings),
  }


def _actor_state_metadata(persona):
  nodes = persona.a_mem.id_to_node
  node_ids = list(nodes)
  expected_ids = [f"node_{index}" for index in range(1, len(node_ids) + 1)]
  embedding_keys = set(persona.a_mem.embeddings)
  references = {
    node.embedding_key for node in nodes.values()
    if getattr(node, "embedding_key", None) is not None}
  fingerprints = passive_cognitive_fingerprint(persona)
  chat_nodes = tuple(getattr(persona.a_mem, "seq_chat", ()))
  return {
    "identity_aligned": persona.name == persona.scratch.name,
    "memory_node_count": len(nodes),
    "embedding_count": len(embedding_keys),
    "chat_count": len(chat_nodes),
    "chat_node_ids": [node.node_id for node in chat_nodes],
    "node_ids_valid": node_ids == expected_ids,
    "node_ids_unique": len(node_ids) == len(set(node_ids)),
    "embedding_references_valid": references.issubset(embedding_keys),
    "orphan_embedding_count": len(embedding_keys - references),
    "daily_plan_present": bool(persona.scratch.daily_plan_req),
    "schedule_present": bool(persona.scratch.f_daily_schedule),
    "current_action_present": bool(
      persona.scratch.act_description and persona.scratch.act_address
      and persona.scratch.act_event),
    "current_action_actor_aligned": bool(
      persona.scratch.act_event
      and persona.scratch.act_event[0] == persona.name),
    "scratch_hash": fingerprints["scratch"],
    "associative_memory_hash": fingerprints["associative_memory"],
    "spatial_memory_hash": fingerprints["spatial_memory"],
    "daily_plan_hash": fingerprints["daily_plan"],
    "schedule_hash": fingerprints["schedule"],
    "action_hash": fingerprints["action"],
  }


def _actor_tick_metadata(persona, coordinate):
  state = _actor_state_metadata(persona)
  return {
    "memory_node_count": state["memory_node_count"],
    "embedding_count": state["embedding_count"],
    "chat_count": state["chat_count"],
    "chat_node_ids": state["chat_node_ids"],
    "node_ids_valid": state["node_ids_valid"],
    "node_ids_unique": state["node_ids_unique"],
    "embedding_references_valid": state["embedding_references_valid"],
    "orphan_embedding_count": state["orphan_embedding_count"],
    "daily_plan_present": state["daily_plan_present"],
    "schedule_length": len(persona.scratch.f_daily_schedule),
    "current_action_present": state["current_action_present"],
    "current_action_actor_aligned": state["current_action_actor_aligned"],
    "action_start": persona.scratch.act_start_time,
    "action_duration": persona.scratch.act_duration,
    "coordinate": tuple(coordinate),
  }


def _actor_object_isolation(personas, actor_names, storage_root):
  selected = [personas[name] for name in actor_names]
  action_hashes = [
    passive_cognitive_fingerprint(personas[name])["action"]
    for name in actor_names]
  paths = [
    (storage_root / "personas" / name / "bootstrap_memory"
     / "associative_memory").resolve()
    for name in actor_names]
  return {
    "scratch_objects_distinct": len({id(p.scratch) for p in selected})
      == len(selected),
    "associative_memory_objects_distinct": len({id(p.a_mem) for p in selected})
      == len(selected),
    "spatial_memory_objects_distinct": len({id(p.s_mem) for p in selected})
      == len(selected),
    "schedule_objects_distinct": len({id(p.scratch.f_daily_schedule)
                                      for p in selected}) == len(selected),
    "embedding_store_paths_distinct": len(set(paths)) == len(paths),
    "actor_names_aligned": all(
      name == personas[name].name == personas[name].scratch.name
      for name in actor_names),
    "action_actors_aligned": all(
      personas[name].scratch.act_event
      and personas[name].scratch.act_event[0] == name
      for name in actor_names),
    "action_states_distinct": len(set(action_hashes)) == len(action_hashes),
  }


def _embedding_audits(simulation_root, actor_names):
  return {
    name: controlled_replay.inspect_isolated_embedding_store(
      name,
      simulation_root / "personas" / name / "bootstrap_memory"
      / "associative_memory",
      simulation_root)
    for name in actor_names}


def _embedding_audit_metadata(audit):
  return {
    "classification": audit.classification,
    "manifest_present": audit.manifest_present,
    "model": audit.model,
    "dimensions": audit.dimensions,
    "embedding_count": audit.embedding_count,
    "node_count": audit.node_count,
    "references_valid": (
      audit.embedding_reference_count <= audit.embedding_count
      and not audit.internal_mismatch),
    "orphan_embedding_count": audit.orphan_embedding_count,
    "observed_dimensions": audit.observed_dimensions,
    "internal_mismatch": audit.internal_mismatch,
  }


def _validate_controlled_proximity(server) -> dict[str, Any]:
  checks = {}
  for name, expected in R1M3C_COORDINATES.items():
    persona = server.personas[name]
    actual = tuple(server.personas_tile[name])
    tile = server.maze.access_tile(actual)
    checks[name] = {
      "coordinate": actual,
      "expected_coordinate": expected,
      "coordinate_aligned": actual == expected,
      "scratch_aligned": tuple(persona.scratch.curr_tile) == expected,
      "walkable": (
        server.maze.collision_maze[actual[1]][actual[0]]
        != collision_block_id),
      "arena": server.maze.get_tile_path(actual, "arena"),
      "awake": "sleep" not in persona.scratch.act_description.lower(),
      "action_valid": bool(
        persona.scratch.act_address in server.maze.address_tiles
        and persona.scratch.act_event[0] == name
        and persona.scratch.act_start_time
        and persona.scratch.act_duration),
      "schedule_valid": bool(
        persona.scratch.f_daily_schedule
        and persona.scratch.f_daily_schedule_hourly_org
        and sum(row[1] for row in persona.scratch.f_daily_schedule) == 1440
        and sum(row[1] for row in
                persona.scratch.f_daily_schedule_hourly_org) == 1440),
    }
  maria = checks[R1M3C_ACTORS[0]]
  klaus = checks[R1M3C_ACTORS[1]]
  distance = math.dist(maria["coordinate"], klaus["coordinate"])
  same_arena = maria["arena"] == klaus["arena"]
  distinct_tiles = maria["coordinate"] != klaus["coordinate"]
  within_range = distance <= min(
    server.personas[name].scratch.vision_r for name in R1M3C_ACTORS)
  all_valid = all(
    value for actor in checks.values()
    for key, value in actor.items()
    if key in ("coordinate_aligned", "scratch_aligned", "walkable", "awake",
               "action_valid", "schedule_valid"))
  all_valid = all_valid and same_arena and distinct_tiles and within_range
  result = {
    "actors": checks,
    "distance": distance,
    "same_arena": same_arena,
    "distinct_tiles": distinct_tiles,
    "within_perception_range": within_range,
    "perception_range": {
      name: server.personas[name].scratch.vision_r for name in R1M3C_ACTORS},
    "all_checks_passed": all_valid,
  }
  if not all_valid:
    raise ModernRuntimeInvariantError(
      "controlled proximity fixture validation failed")
  return result


class _ConversationObserver:
  """Content-free observation of the existing Stanford interaction path."""

  def __init__(self, server, execution_state, simulation_id,
               pre_resume_derived_node_refs=(), pre_resume_chat_node_refs=()):
    self.server = server
    self.execution_state = execution_state
    self.simulation_id = simulation_id
    self.encounters = []
    self.reactions = []
    self.conversations = []
    self.turn_results = []
    self.retrieval_events = []
    self.relationship_events = []
    self.last_chat_events = []
    self.pre_resume_derived_node_refs = set(pre_resume_derived_node_refs)
    self.pre_resume_chat_node_refs = set(pre_resume_chat_node_refs)
    self._retrieval_result_events = {}
    self._original_perceive = {}
    self._original_last_chat = {}
    self._originals = {}

  @contextmanager
  def _actor_context(self, actor):
    context = LLMReplayContext(
      cognitive_category="CONVERSATION", actor_id=actor,
      simulation_id=self.simulation_id,
      simulation_step=self.execution_state["tick"])
    ledger = CostLedgerContext(
      simulation_id=self.simulation_id,
      simulation_step=self.execution_state["tick"], actor_id=actor,
      cognitive_category="CONVERSATION")
    with (use_llm_replay_context(context), use_cost_ledger_context(ledger)):
      yield

  def install(self):
    plan_module = importlib.import_module(
      "persona.cognitive_modules.plan")
    converse_module = importlib.import_module(
      "persona.cognitive_modules.converse")
    self._originals = {
      "should_react": plan_module._should_react,
      "chat_react": plan_module._chat_react,
      "new_retrieve": converse_module.new_retrieve,
      "relationship": converse_module.generate_summarize_agent_relationship,
      "utterance": converse_module.generate_one_utterance,
    }

    def observed_should_react(persona, retrieved, personas):
      target = retrieved["curr_event"].subject
      result = self._originals["should_react"](
        persona, retrieved, personas)
      if target in personas and target != persona.name:
        category = "NO_REACTION"
        if isinstance(result, str) and result.startswith("chat with"):
          category = "CHAT"
        elif isinstance(result, str) and result.startswith("wait"):
          category = "WAIT"
        elif result:
          category = "OTHER"
        self.reactions.append({
          "tick": self.execution_state["tick"], "actor": persona.name,
          "target": target, "caller": "decide_to_talk",
          "decision_category": category,
        })
      return result

    def observed_chat_react(maze, persona, focused_event, reaction_mode,
                            personas):
      target_name = reaction_mode[9:].strip()
      self.turn_results = []
      result = self._originals["chat_react"](
        maze, persona, focused_event, reaction_mode, personas)
      chat = persona.scratch.chat or []
      payload = json.dumps(chat, ensure_ascii=False, separators=(",", ":"))
      self.conversations.append({
        "conversation_hash": hashlib.sha256(
          payload.encode("utf-8")).hexdigest(),
        "participants": [persona.name, target_name],
        "start_tick": self.execution_state["tick"],
        "end_tick": self.execution_state["tick"],
        "turn_count": len(chat),
        "speaker_sequence": [row[0] for row in chat],
        "termination": (
          "MODEL_END" if self.turn_results
          and self.turn_results[-1]["end"] else "SAFETY_CEILING"),
        "distinct_chat_objects": (
          id(persona.scratch.chat)
          != id(personas[target_name].scratch.chat)),
      })
      return result

    def actor_retrieve(persona, *args, **kwargs):
      with self._actor_context(persona.name):
        result = self._originals["new_retrieve"](persona, *args, **kwargs)
      nodes = {
        node.node_id: node for values in result.values() for node in values}
      node_ids = _canonical_node_ids(nodes)
      node_refs = [f"{persona.name}::{node_id}" for node_id in node_ids]
      focal_points = args[0] if args else kwargs.get("focal_points", ())
      event = {
        "actor": persona.name,
        "simulation_step": self.execution_state["tick"],
        "retrieval_context": "new_retrieve",
        "focal_point_hashes": [
          _content_hash(item) for item in focal_points],
        "retrieved_node_ids": node_ids,
        "retrieved_node_types": [nodes[node_id].type for node_id in node_ids],
        "retrieved_pre_resume_derived_social_node_ids": [
          node_id for node_id, node_ref in zip(node_ids, node_refs)
          if node_ref in self.pre_resume_derived_node_refs],
        "retrieved_pre_resume_derived_social_node_refs": [
          node_ref for node_ref in node_refs
          if node_ref in self.pre_resume_derived_node_refs],
      }
      self.retrieval_events.append(event)
      self._retrieval_result_events[id(result)] = len(self.retrieval_events) - 1
      return result

    def actor_relationship(init_persona, target_persona, retrieved):
      nodes = {
        node.node_id: node for values in retrieved.values() for node in values}
      node_ids = _canonical_node_ids(nodes)
      node_refs = [
        f"{init_persona.name}::{node_id}" for node_id in node_ids]
      event = {
        "actor": init_persona.name,
        "target_actor": target_persona.name,
        "simulation_step": self.execution_state["tick"],
        "input_node_ids": node_ids,
        "input_node_types": [nodes[node_id].type for node_id in node_ids],
        "pre_resume_derived_social_node_ids_present": [
          node_id for node_id, node_ref in zip(node_ids, node_refs)
          if node_ref in self.pre_resume_derived_node_refs],
        "pre_resume_derived_social_node_refs_present": [
          node_ref for node_ref in node_refs
          if node_ref in self.pre_resume_derived_node_refs],
        "source_retrieval_event_index": self._retrieval_result_events.get(
          id(retrieved)),
      }
      self.relationship_events.append(event)
      with self._actor_context(init_persona.name):
        return self._originals["relationship"](
          init_persona, target_persona, retrieved)

    def actor_utterance(maze, init_persona, target_persona, retrieved,
                        curr_chat):
      with self._actor_context(init_persona.name):
        utterance, end = self._originals["utterance"](
          maze, init_persona, target_persona, retrieved, curr_chat)
      self.turn_results.append({"speaker": init_persona.name,
                                "end": bool(end)})
      return utterance, end

    plan_module._should_react = observed_should_react
    plan_module._chat_react = observed_chat_react
    converse_module.new_retrieve = actor_retrieve
    converse_module.generate_summarize_agent_relationship = actor_relationship
    converse_module.generate_one_utterance = actor_utterance

    for name, persona in self.server.personas.items():
      original = persona.perceive
      self._original_perceive[name] = original
      original_last_chat = persona.a_mem.get_last_chat
      self._original_last_chat[name] = original_last_chat

      def observed_last_chat(target_name, *args, _name=name,
                             _original=original_last_chat, **kwargs):
        node = _original(target_name, *args, **kwargs)
        node_id = getattr(node, "node_id", None) if node else None
        node_ref = f"{_name}::{node_id}" if node_id else None
        self.last_chat_events.append({
          "actor": _name, "target_actor": target_name,
          "simulation_step": self.execution_state["tick"],
          "chat_node_id": node_id,
          "chat_node_type": getattr(node, "type", None) if node else None,
          "pre_resume_chat_node_observed": (
            node_ref in self.pre_resume_chat_node_refs),
        })
        return node

      persona.a_mem.get_last_chat = observed_last_chat

      def observed_perceive(maze, _name=name, _persona=persona,
                           _original=original):
        before_ids = set(_persona.a_mem.id_to_node)
        perceived = _original(maze)
        for node in perceived:
          target = node.subject
          if target in self.server.personas and target != _name:
            target_tile = tuple(self.server.personas[target].scratch.curr_tile)
            observer_tile = tuple(_persona.scratch.curr_tile)
            self.encounters.append({
              "tick": self.execution_state["tick"], "observer": _name,
              "target": target,
              "distance": math.dist(observer_tile, target_tile),
              "same_arena": (
                maze.get_tile_path(observer_tile, "arena")
                == maze.get_tile_path(target_tile, "arena")),
              "perception_event_type": node.type,
              "memory_node_delta": len(
                set(_persona.a_mem.id_to_node) - before_ids),
              "node_id": node.node_id,
            })
        return perceived

      persona.perceive = observed_perceive
    return self

  def restore(self):
    plan_module = importlib.import_module("persona.cognitive_modules.plan")
    converse_module = importlib.import_module(
      "persona.cognitive_modules.converse")
    if self._originals:
      plan_module._should_react = self._originals["should_react"]
      plan_module._chat_react = self._originals["chat_react"]
      converse_module.new_retrieve = self._originals["new_retrieve"]
      converse_module.generate_summarize_agent_relationship = self._originals[
        "relationship"]
      converse_module.generate_one_utterance = self._originals["utterance"]
    for name, original in self._original_perceive.items():
      self.server.personas[name].perceive = original
    for name, original in self._original_last_chat.items():
      self.server.personas[name].a_mem.get_last_chat = original


@contextmanager
def _observe_conversations(
    server, execution_state, simulation_id,
    pre_resume_derived_node_refs=(), pre_resume_chat_node_refs=()):
  observer = _ConversationObserver(
    server, execution_state, simulation_id,
    pre_resume_derived_node_refs,
    pre_resume_chat_node_refs).install()
  try:
    yield observer
  finally:
    observer.restore()


def _indexed_history_hashes(root: Path, first: int, stop: int):
  expected = tuple(f"{index}.json" for index in range(first, stop))
  actual = tuple(sorted(
    (path.name for path in root.glob("*.json")),
    key=lambda name: int(Path(name).stem)
    if Path(name).stem.isdigit() else -1))
  if actual != expected:
    raise ModernRunConfigurationError(
      f"persisted history is inconsistent: directory={root.name}")
  return {name: _file_sha256(root / name) for name in expected}


def _prepare_resume_context(
    config: ModernResumeConfig, root: Path) -> _ResumeContext:
  source_run = config.source_run.resolve()
  if source_run.parent != root or not source_run.is_dir():
    raise ModernRunConfigurationError(
      "resume source must be an existing run inside the runtime root")
  if not SAFE_NAME.fullmatch(source_run.name):
    raise ModernRunConfigurationError("resume source name is invalid")
  report_path = source_run / "report.json"
  status_path = source_run / "status.json"
  if not report_path.is_file() or not status_path.is_file():
    raise ModernRunConfigurationError(
      "resume source is missing its report or status")
  try:
    report = _read_json(report_path)
    status = _read_json(status_path)
  except (OSError, ValueError, TypeError) as error:
    raise ModernRunConfigurationError(
      "resume source report or status is corrupt") from error
  if status.get("state") != "PASSED":
    raise ModernRunConfigurationError("resume source did not finish successfully")
  source_result = report.get("result", {})
  source_config = report.get("config", {})
  actor_policy = report.get("actor_policy", {})
  if (source_config.get("run_name") != source_run.name
      or not source_result.get("save_passed")
      or not source_result.get("reload_passed")):
    raise ModernRunConfigurationError(
      "resume source does not contain a validated persisted run")
  cognitive_actors = tuple(actor_policy.get("cognitive", ()))
  passive_actors = tuple(actor_policy.get("passive", ()))
  if (tuple(actor_policy.get("visible", ())) != VISIBLE_ACTORS
      or not cognitive_actors
      or set(cognitive_actors) | set(passive_actors) != set(VISIBLE_ACTORS)
      or set(cognitive_actors) & set(passive_actors)):
    raise ModernRunConfigurationError(
      "resume source actor registry is incompatible")

  simulation_root = source_run / "fixture" / "storage" / source_run.name
  if not simulation_root.is_dir():
    raise ModernRunConfigurationError(
      "resume source persisted simulation is missing")
  meta_path = simulation_root / "reverie" / "meta.json"
  if not meta_path.is_file():
    raise ModernRunConfigurationError("resume source meta is missing")
  try:
    meta = _read_json(meta_path)
    persisted_time = dt.datetime.strptime(meta.get("curr_time", ""), DATE_FORMAT)
  except (OSError, ValueError, TypeError) as error:
    raise ModernRunConfigurationError("resume source meta is corrupt") from error
  step = meta.get("step")
  if type(step) is not int or step <= 0:
    raise ModernRunConfigurationError(
      "resume source step must be a positive integer")
  if meta.get("sec_per_step") != 10:
    raise ModernRunConfigurationError(
      "resume source tick duration is incompatible")
  if tuple(meta.get("persona_names", ())) != VISIBLE_ACTORS:
    raise ModernRunConfigurationError(
      "resume source actor registry is incompatible")
  if (source_result.get("final_step") != step
      or source_result.get("final_time") != persisted_time.strftime(DATE_FORMAT)):
    raise ModernRunConfigurationError(
      "resume source report does not match persisted step/time")

  movement_hashes = _indexed_history_hashes(
    simulation_root / "movement", 0, step)
  environment_hashes = _indexed_history_hashes(
    simulation_root / "environment", 0, step + 1)
  actor_state = {
    name: _persisted_actor_resume_metadata(simulation_root, name)
    for name in cognitive_actors}
  actor_state_issues = {
    name: tuple(key for key, passed in {
      "identity_aligned": state["identity_aligned"],
      "node_ids_valid": state["node_ids_valid"],
      "node_ids_unique": state["node_ids_unique"],
      "embedding_references_valid": state["embedding_references_valid"],
      "no_orphan_embeddings": state["orphan_embedding_count"] == 0,
    }.items() if not passed)
    for name, state in actor_state.items()}
  actor_state_issues = {
    name: issues for name, issues in actor_state_issues.items() if issues}
  if actor_state_issues:
    raise ModernRunConfigurationError(
      "resume source cognitive state is inconsistent: "
      + repr(actor_state_issues))
  embedding_audits = tuple(_embedding_audits(
    simulation_root, VISIBLE_ACTORS).values())
  if any(not all((
      audit.classification == controlled_replay.MODERN_COMPATIBLE,
      audit.manifest_present,
      audit.model == TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.model,
      audit.dimensions == TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.dimensions,
      not audit.internal_mismatch,
      audit.orphan_embedding_count == 0,
  )) for audit in embedding_audits):
    raise ModernRunConfigurationError(
      "resume source embedding store is incompatible")
  causal_social_memory = _persisted_social_memory_snapshot(simulation_root)
  if config.observe_causal_social_memory:
    if (not set(R1M3C_ACTORS).issubset(cognitive_actors)
        or not causal_social_memory["bilateral_chat_lineage_present"]):
      raise ModernRunConfigurationError(
        "causal social resume source lacks bilateral Chat -> Event lineage")
  return _ResumeContext(
    request=config, source_run=source_run,
    simulation_root=simulation_root, source_meta=meta,
    cognitive_actors=cognitive_actors, passive_actors=passive_actors,
    movement_hashes=movement_hashes,
    environment_hashes=environment_hashes, actor_state=actor_state,
    embedding_audits=embedding_audits,
    causal_social_memory=causal_social_memory)


def run_modern_smallville(
    config: ModernRunConfig, *, adapter=None,
    runtime_root: Optional[Path] = None) -> ModernRunResult:
  """Run one isolated modern Smallville and return a compact typed result."""
  if not isinstance(config, ModernRunConfig):
    raise TypeError("config must be ModernRunConfig")
  return _execute_modern_smallville(
    config, adapter=adapter, runtime_root=runtime_root)


def run_modern_smallville_resume(
    config: ModernResumeConfig, *, adapter=None,
    runtime_root: Optional[Path] = None) -> ModernRunResult:
  """Continue a validated persisted run in a new isolated simulation copy."""
  if not isinstance(config, ModernResumeConfig):
    raise TypeError("config must be ModernResumeConfig")
  root = Path(runtime_root or RUNTIME_ROOT).resolve()
  resume = _prepare_resume_context(config, root)
  run_config = ModernRunConfig(
    source_simulation=resume.source_run.name,
    run_name=config.run_name, ticks=config.ticks, tick_seconds=10,
    cognitive_actors=resume.cognitive_actors,
    passive_actors=resume.passive_actors,
    cost_ceiling_usd=config.cost_ceiling_usd,
    controlled_proximity=False)
  return _execute_modern_smallville(
    run_config, adapter=adapter, runtime_root=root, resume=resume)


def _execute_modern_smallville(
    config: ModernRunConfig, *, adapter=None,
    runtime_root: Optional[Path] = None,
    resume: Optional[_ResumeContext] = None) -> ModernRunResult:
  root = Path(runtime_root or RUNTIME_ROOT).resolve()
  source = (resume.simulation_root if resume is not None
            else (SOURCE_ROOT / config.source_simulation).resolve())
  if (not source.is_dir()
      or (resume is None and source.parent != SOURCE_ROOT.resolve())):
    raise ModernRunConfigurationError(
      f"source simulation does not exist: {config.source_simulation}")
  source_meta = _read_json(source / "reverie" / "meta.json")
  source_hash_before = _tree_sha256(source)
  if type(source_meta.get("step")) is not int or source_meta["step"] < 0:
    raise ModernRunConfigurationError("source simulation step is invalid")
  if source_meta.get("sec_per_step") != config.tick_seconds:
    raise ModernRunConfigurationError("source tick duration is incompatible")
  if tuple(source_meta.get("persona_names", ())) != VISIBLE_ACTORS:
    raise ModernRunConfigurationError("source actor registry is incompatible")
  adapter = adapter or _default_adapter()
  live = isinstance(adapter, ModernOpenAIClientAdapter)
  providers = controlled_replay.ControlledReplayProviders(
    adapter, adapter, adapter, live_api_enabled=live)
  root.mkdir(parents=True, exist_ok=True)
  run_dir = root / config.run_name
  try:
    run_dir.mkdir(exist_ok=False)
  except FileExistsError as error:
    raise ModernRunConfigurationError(
      f"run directory already exists: {run_dir}") from error

  fixture_root = run_dir / "fixture"
  storage_root = fixture_root / "storage"
  temp_root = fixture_root / "temp_storage"
  storage_root.mkdir(parents=True)
  temp_root.mkdir()
  source_code = config.run_name + "-source"
  simulation_code = config.run_name
  reload_code = config.run_name + "-reload"
  isolated_source = storage_root / source_code
  shutil.copytree(source, isolated_source)
  fixture_seed = {}
  if resume is None:
    if config.controlled_proximity:
      fixture_seed = _bootstrap_controlled_proximity_source(isolated_source)
    else:
      _bootstrap_isolated_temporal_source(
        isolated_source, config.cognitive_actors)
  meta = _read_json(isolated_source / "reverie" / "meta.json")
  source_step = meta.get("step")
  if type(source_step) is not int or source_step < 0:
    raise ModernRunConfigurationError("source simulation step is invalid")
  if meta.get("sec_per_step") != config.tick_seconds:
    raise ModernRunConfigurationError("source tick duration is incompatible")
  if tuple(meta.get("persona_names", ())) != VISIBLE_ACTORS:
    raise ModernRunConfigurationError("source actor registry is incompatible")
  fixture = controlled_replay.prepare_isolated_reverie_fixture(
    isolated_source, fixture_root, source, source_step)
  if resume is None:
    embedding_preflight = controlled_replay.prepare_isolated_embedding_stores(
      fixture)
  else:
    copied_audits = tuple(_embedding_audits(
      isolated_source, VISIBLE_ACTORS).values())
    if any(audit.classification != controlled_replay.MODERN_COMPATIBLE
           for audit in copied_audits):
      raise ModernRunConfigurationError(
        "resume copy embedding store is incompatible")
    embedding_preflight = controlled_replay.IsolatedEmbeddingPreflightResult(
      audits_before=copied_audits, audits_after=copied_audits,
      bootstrapped_personas=())

  status_path, report_path = run_dir / "status.json", run_dir / "report.json"
  started_at = dt.datetime.now(dt.timezone.utc)
  _write_json(status_path, {
    "run_id": config.run_name, "state": "RUNNING", "completed_ticks": 0,
    "started_at": started_at.isoformat(),
  })

  reverie_module = _load_reverie_module()
  previous_storage = reverie_module.fs_storage
  previous_temp = reverie_module.fs_temp_storage
  previous_cwd = Path.cwd()
  previous_debug = run_gpt_prompt.debug
  server = None
  initial_step = final_step = source_step
  initial_time = final_time = None
  completed = 0
  movement_count = 0
  save_passed = False
  reload_passed = False
  actor_move_counts = Counter()
  actor_move_sequence = []
  passive_mutations = 0
  passive_provider_calls = 0
  legacy_count = 0
  observer = None
  cost_guard = None
  error = None
  reloaded_summary = {}
  passive_before = {}
  actor_state_before = {}
  actor_state_after = {}
  actor_state_reload = {}
  isolation_before = {}
  isolation_after = {}
  isolation_reload = {}
  saved_embedding_metadata = {}
  reload_embedding_metadata = {}
  tick_progression = []
  tick_isolation = []
  continuity = {
    "single_server": False, "same_maze_across_ticks": False,
    "same_personas_across_ticks": False, "sequential_actor_order": False,
  }
  movement_integrity = {}
  movement_integrity_valid = False
  resume_hydration = {}
  hydration_provider_calls = 0
  causal_pre_resume = (
    resume.causal_social_memory if resume is not None else {})
  causal_hydrated = {}
  causal_hydration = {}
  causal_run_snapshot = {}
  controlled_fixture = {}
  interaction_observer = None
  execution_state = {"stage": "initialization", "actor": None,
                     "tick": source_step}

  try:
    reset_provider()
    clear_telemetry()
    os.chdir(BACKEND_ROOT)
    run_gpt_prompt.debug = False
    reverie_module.fs_storage = str(storage_root)
    reverie_module.fs_temp_storage = str(temp_root)
    with redirect_stdout(_NullOutput()):
      with ExitStack() as stack:
        stack.enter_context(use_embedding_runtime(
          providers.embedding_config, adapter,
          legacy_assumption_allowed=False))
        stack.enter_context(use_modern_chat_runtime(
          providers.chat_config, adapter))
        stack.enter_context(use_modern_completion_runtime(
          providers.completion_config, adapter))
        cost_config = ReplayCostGuardConfig(
          replay_id=config.run_name, simulation_id=simulation_code,
          ceiling=ReplayCostCeiling(config.cost_ceiling_usd),
          pricing_snapshot=providers.pricing_snapshot)
        cost_guard = stack.enter_context(use_replay_cost_guard(cost_config))
        observer = _RuntimeObserver(cost_guard)
        stack.enter_context(use_llm_attempt_observer(observer))
        stack.enter_context(_use_event_poignancy_fail_safe())

        calls_before_hydration = len(get_telemetry())
        server = reverie_module.ReverieServer(source_code, simulation_code)
        hydration_provider_calls = (
          len(get_telemetry()) - calls_before_hydration)
        if set(server.personas) != set(VISIBLE_ACTORS):
          raise ModernRuntimeInvariantError("visible actor registry changed")
        initial_step, initial_time = server.step, server.curr_time
        if config.controlled_proximity:
          controlled_fixture = _validate_controlled_proximity(server)
        saved_root = storage_root / simulation_code
        server_identity = id(server)
        maze_identity = id(server.maze)
        persona_identities = {
          name: id(server.personas[name]) for name in VISIBLE_ACTORS}
        expected_actor_order = tuple(server.personas)
        continuity["sequential_actor_order"] = (
          expected_actor_order == VISIBLE_ACTORS)
        actor_state_before = {
          name: _actor_state_metadata(server.personas[name])
          for name in config.cognitive_actors}
        if resume is not None:
          hydration_actors = {}
          for name in config.cognitive_actors:
            loaded = _loaded_actor_resume_metadata(server.personas[name])
            copied_hash = _tree_sha256(
              saved_root / "personas" / name / "bootstrap_memory")
            hydration_actors[name] = _resume_actor_hydration_checks(
              resume.actor_state[name], loaded, copied_hash)
          source_time = dt.datetime.strptime(
            resume.source_meta["curr_time"], DATE_FORMAT)
          resume_hydration = {
            "provider_calls": hydration_provider_calls,
            "step_retained": server.step == resume.source_meta["step"],
            "time_retained": server.curr_time == source_time,
            "actors": hydration_actors,
          }
          resume_hydration["all_checks_passed"] = all((
            hydration_provider_calls == 0,
            resume_hydration["step_retained"],
            resume_hydration["time_retained"],
            all(checks["all_checks_passed"]
                for checks in hydration_actors.values()),
          ))
          if not resume_hydration["all_checks_passed"]:
            raise ModernRuntimeInvariantError(
              "resume cognitive hydration validation failed")
          if resume.request.observe_causal_social_memory:
            causal_hydrated = _loaded_social_memory_snapshot(server.personas)
            causal_hydration = _compare_social_memory_hydration(
              causal_pre_resume, causal_hydrated)
            causal_hydration["provider_calls"] = hydration_provider_calls
            if not all((
                causal_hydration["raw_social_memory_persisted"],
                causal_hydration["derived_social_memory_persisted"],
                causal_hydration["lineage_preserved"],
                hydration_provider_calls == 0)):
              raise ModernRuntimeInvariantError(
                "causal social memory hydration validation failed")
        isolation_before = _actor_object_isolation(
          server.personas, config.cognitive_actors, saved_root)
        passive_before = {
          name: passive_cognitive_fingerprint(server.personas[name])
          for name in config.passive_actors}
        for name, persona in server.personas.items():
          original = persona.move

          def counted_move(*args, _name=name, _original=original, **kwargs):
            actor_move_counts[_name] += 1
            if _name not in config.cognitive_actors:
              return _original(*args, **kwargs)
            actor_move_sequence.append((_name, server.step))
            execution_state.update(
              stage="persona_move", actor=_name, tick=server.step)
            context = LLMReplayContext(
              cognitive_category="WORLD_TICK", actor_id=_name,
              simulation_id=simulation_code, simulation_step=server.step)
            ledger = CostLedgerContext(
              simulation_id=simulation_code, simulation_step=server.step,
              actor_id=_name, cognitive_category="WORLD_TICK")
            with (use_llm_replay_context(context),
                  use_cost_ledger_context(ledger)):
              return _original(*args, **kwargs)

          persona.move = counted_move
        with (install_passive_visible_moves(
                server, config.passive_actors) as passive_controller,
              _observe_conversations(
                server, execution_state,
                simulation_code,
                (causal_pre_resume.get("derived_node_refs", ())
                 if resume is not None
                 and resume.request.observe_causal_social_memory else ()),
                (causal_pre_resume.get("chat_node_refs", ())
                 if resume is not None
                 and resume.request.observe_causal_social_memory else ()),
              ) as interaction_observer):
          previous_tick_state = actor_state_before
          for tick in range(config.ticks):
            step_before, time_before = server.step, server.curr_time
            if id(server) != server_identity:
              raise ModernRuntimeInvariantError(
                "ReverieServer instance changed between ticks")
            if id(server.maze) != maze_identity:
              raise ModernRuntimeInvariantError(
                "Maze instance changed between ticks")
            if any(id(server.personas[name]) != persona_identities[name]
                   for name in VISIBLE_ACTORS):
              raise ModernRuntimeInvariantError(
                "Persona instance changed between ticks")
            execution_state.update(
              stage="world_tick", actor=None, tick=server.step)
            server.start_server(1)
            execution_state.update(
              stage="movement_validation", actor=None, tick=step_before)
            movement_path = saved_root / "movement" / f"{initial_step + tick}.json"
            movement = _read_json(movement_path)
            if set(movement.get("persona", {})) != set(VISIBLE_ACTORS):
              raise ModernRuntimeInvariantError(
                "movement frame does not contain all visible actors")
            environment = _read_json(
              saved_root / "environment" / f"{initial_step + tick}.json")
            tick_actors = {}
            for name in VISIBLE_ACTORS:
              actor_frame = movement["persona"][name]
              coordinate = actor_frame.get("movement")
              if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ModernRuntimeInvariantError(
                  "movement frame contains an invalid coordinate")
              if (not isinstance(actor_frame.get("description"), str)
                  or not isinstance(actor_frame.get("pronunciatio"), str)
                  or "chat" not in actor_frame):
                raise ModernRuntimeInvariantError(
                  "movement frame contains invalid actor metadata")
              environment[name] = dict(environment[name])
              environment[name]["x"], environment[name]["y"] = coordinate
              if name in config.cognitive_actors:
                tick_actors[name] = _actor_tick_metadata(
                  server.personas[name], coordinate)
                previous = previous_tick_state[name]
                current = tick_actors[name]
                if (current["memory_node_count"]
                    < previous["memory_node_count"]):
                  raise ModernRuntimeInvariantError(
                    f"memory count regressed: actor={name}, tick={step_before}")
                if current["embedding_count"] < previous["embedding_count"]:
                  raise ModernRuntimeInvariantError(
                    f"embedding count regressed: actor={name}, tick={step_before}")
                if not all((
                    current["node_ids_valid"], current["node_ids_unique"],
                    current["embedding_references_valid"],
                    current["orphan_embedding_count"] == 0,
                    current["daily_plan_present"],
                    current["schedule_length"] > 0,
                    current["current_action_present"],
                    current["current_action_actor_aligned"])):
                  raise ModernRuntimeInvariantError(
                    f"actor state invariant failed: actor={name}, "
                    f"tick={step_before}")
            _write_json(saved_root / "environment" / f"{server.step}.json",
                        environment)
            current_isolation = _actor_object_isolation(
              server.personas, config.cognitive_actors, saved_root)
            if not all(current_isolation.values()):
              raise ModernRuntimeInvariantError(
                f"actor isolation failed at tick {step_before}")
            tick_isolation.append(current_isolation)
            tick_progression.append({
              "tick": tick, "step_before": step_before,
              "step_after": server.step, "time_before": time_before,
              "time_after": server.curr_time, "actors": tick_actors,
              "movement_sha256": _file_sha256(movement_path),
            })
            previous_tick_state = {
              name: _actor_state_metadata(server.personas[name])
              for name in config.cognitive_actors}
            movement_count += 1
            completed += 1

          passive_after = {
            name: passive_cognitive_fingerprint(server.personas[name])
            for name in config.passive_actors}
          passive_mutations = sum(
            passive_before[name] != passive_after[name]
            for name in config.passive_actors)
          if passive_mutations:
            raise ModernRuntimeInvariantError("passive cognitive state mutated")
          if any(actor_move_counts[name] for name in config.passive_actors):
            raise ModernRuntimeInvariantError("passive Persona.move was invoked")
          if any(passive_controller.frame_emissions[name] != config.ticks
                 for name in config.passive_actors):
            raise ModernRuntimeInvariantError("passive frame emission mismatch")
          if any(actor_move_counts[name] != config.ticks
                 for name in config.cognitive_actors):
            raise ModernRuntimeInvariantError(
              "cognitive Persona.move invocation mismatch")

        actor_state_after = {
          name: _actor_state_metadata(server.personas[name])
          for name in config.cognitive_actors}
        causal_run_snapshot = _loaded_social_memory_snapshot(server.personas)
        isolation_after = _actor_object_isolation(
          server.personas, config.cognitive_actors, saved_root)
        continuity.update({
          "single_server": id(server) == server_identity,
          "same_maze_across_ticks": id(server.maze) == maze_identity,
          "same_personas_across_ticks": all(
            id(server.personas[name]) == persona_identities[name]
            for name in VISIBLE_ACTORS),
          "sequential_actor_order": all(
            tuple(name for name, step in actor_move_sequence if step == tick)
            == config.cognitive_actors
            for tick in range(initial_step, server.step)),
        })
        expected_frames = tuple(
          saved_root / "movement" / f"{initial_step + tick}.json"
          for tick in range(config.ticks))
        frame_hashes_before_save = {
          path.name: _file_sha256(path) for path in expected_frames}
        prior_frame_hashes_before_save = ({
          name: _file_sha256(saved_root / "movement" / name)
          for name in resume.movement_hashes
        } if resume is not None else {})
        prior_environment_hashes_before_save = ({
          name: _file_sha256(saved_root / "environment" / name)
          for name in resume.environment_hashes
        } if resume is not None else {})
        execution_state.update(stage="save", actor=None, tick=server.step)
        server.save()
        save_passed = True
        if resume is None:
          causal_run_snapshot = _persisted_social_memory_snapshot(saved_root)
        frame_hashes_after_save = {
          path.name: _file_sha256(path) for path in expected_frames}
        prior_frame_hashes_after_save = ({
          name: _file_sha256(saved_root / "movement" / name)
          for name in resume.movement_hashes
        } if resume is not None else {})
        prior_environment_hashes_after_save = ({
          name: _file_sha256(saved_root / "environment" / name)
          for name in resume.environment_hashes
        } if resume is not None else {})
        saved_embedding_audits = _embedding_audits(
          saved_root, config.cognitive_actors)
        saved_embedding_metadata = {
          name: _embedding_audit_metadata(audit)
          for name, audit in saved_embedding_audits.items()}
        calls_before_reload = len(get_telemetry())
        execution_state.update(stage="offline_reload", actor=None,
                               tick=server.step)
        reloaded = reverie_module.ReverieServer(simulation_code, reload_code)
        calls_after_reload = len(get_telemetry())
        actor_state_reload = {
          name: _actor_state_metadata(reloaded.personas[name])
          for name in config.cognitive_actors}
        isolation_reload = _actor_object_isolation(
          reloaded.personas, config.cognitive_actors,
          storage_root / reload_code)
        reload_embedding_audits = _embedding_audits(
          storage_root / reload_code, config.cognitive_actors)
        reload_embedding_metadata = {
          name: _embedding_audit_metadata(audit)
          for name, audit in reload_embedding_audits.items()}
        reload_movement_root = storage_root / reload_code / "movement"
        frame_hashes_after_reload = {
          path.name: _file_sha256(path) for path in expected_frames}
        reload_frame_hashes = {
          path.name: _file_sha256(reload_movement_root / path.name)
          for path in expected_frames}
        reload_prior_frame_hashes = ({
          name: _file_sha256(reload_movement_root / name)
          for name in resume.movement_hashes
        } if resume is not None else {})
        reload_prior_environment_hashes = ({
          name: _file_sha256(
            storage_root / reload_code / "environment" / name)
          for name in resume.environment_hashes
        } if resume is not None else {})
        prior_frame_count = (
          len(resume.movement_hashes) if resume is not None else 0)
        expected_total_frames = prior_frame_count + config.ticks
        movement_integrity = {
          "expected_frame_count": config.ticks,
          "expected_total_frame_count": expected_total_frames,
          "saved_frame_count": len(list(
            (saved_root / "movement").glob("*.json"))),
          "reload_frame_count": len(list(reload_movement_root.glob("*.json"))),
          "all_frames_distinct": (
            len(set(frame_hashes_before_save.values())) == config.ticks),
          "unchanged_by_save": (
            frame_hashes_before_save == frame_hashes_after_save),
          "unchanged_by_reload": (
            frame_hashes_before_save == frame_hashes_after_reload),
          "reload_copy_matches": (
            frame_hashes_before_save == reload_frame_hashes),
          "frame_hashes": frame_hashes_before_save,
          "prior_frame_hashes": prior_frame_hashes_before_save,
          "prior_frames_match_source": (
            resume is None or prior_frame_hashes_before_save
            == resume.movement_hashes),
          "prior_frames_unchanged_by_save": (
            prior_frame_hashes_before_save == prior_frame_hashes_after_save),
          "prior_frames_preserved_on_reload": (
            prior_frame_hashes_before_save == reload_prior_frame_hashes),
          "prior_environment_matches_source": (
            resume is None or prior_environment_hashes_before_save
            == resume.environment_hashes),
          "prior_environment_unchanged_by_save": (
            prior_environment_hashes_before_save
            == prior_environment_hashes_after_save),
          "prior_environment_preserved_on_reload": (
            prior_environment_hashes_before_save
            == reload_prior_environment_hashes),
        }
        movement_integrity_valid = all((
          movement_integrity["saved_frame_count"] == expected_total_frames,
          movement_integrity["reload_frame_count"] == expected_total_frames,
          movement_integrity["all_frames_distinct"],
          movement_integrity["unchanged_by_save"],
          movement_integrity["unchanged_by_reload"],
          movement_integrity["reload_copy_matches"],
          movement_integrity["prior_frames_match_source"],
          movement_integrity["prior_frames_unchanged_by_save"],
          movement_integrity["prior_frames_preserved_on_reload"],
          movement_integrity["prior_environment_matches_source"],
          movement_integrity["prior_environment_unchanged_by_save"],
          movement_integrity["prior_environment_preserved_on_reload"],
        ))
        memory_preserved = all(
          _memory_counts(server.personas[name])
          == _memory_counts(reloaded.personas[name])
          for name in config.cognitive_actors)
        state_preserved = all(
          all(actor_state_after[name][field]
              == actor_state_reload[name][field]
              for field in (
                "memory_node_count", "embedding_count", "node_ids_valid",
                "node_ids_unique", "embedding_references_valid",
                "orphan_embedding_count", "daily_plan_present",
                "schedule_present", "current_action_present",
                "current_action_actor_aligned", "daily_plan_hash",
                "schedule_hash", "action_hash"))
          for name in config.cognitive_actors)
        embeddings_valid = all(
          audit.classification == controlled_replay.MODERN_COMPATIBLE
          and audit.manifest_present
          and audit.model == TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.model
          and audit.dimensions == TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.dimensions
          and not audit.internal_mismatch
          and audit.orphan_embedding_count == 0
          for audit in (*saved_embedding_audits.values(),
                        *reload_embedding_audits.values()))
        isolation_valid = all(
          all(checks.values()) for checks in (
            isolation_before, isolation_after, isolation_reload))
        reload_passed = all((
          calls_before_reload == calls_after_reload,
          reloaded.step == server.step,
          reloaded.curr_time == server.curr_time,
          len(reloaded.personas) == len(VISIBLE_ACTORS),
          memory_preserved, state_preserved, embeddings_valid, isolation_valid,
          all(continuity.values()), movement_integrity_valid,
          expected_total_frames
          == len(list((saved_root / "movement").glob("*.json"))),
        ))
        reloaded_summary = {
          "step": reloaded.step, "curr_time": reloaded.curr_time,
          "persona_count": len(reloaded.personas),
          "actors": actor_state_reload,
          "provider_calls": calls_after_reload - calls_before_reload,
        }
        if not reload_passed:
          raise ModernRuntimeInvariantError("offline reload verification failed")
        final_step, final_time = server.step, server.curr_time
  except Exception as caught:
    error = caught
    if server is not None:
      final_step, final_time = server.step, server.curr_time
  finally:
    reverie_module.fs_storage = previous_storage
    reverie_module.fs_temp_storage = previous_temp
    run_gpt_prompt.debug = previous_debug
    os.chdir(previous_cwd)

  events = tuple(get_telemetry())
  logical_calls = len({event.logical_call_id for event in events})
  physical_attempts = len(events)
  passive_provider_calls = sum(
    event.actor_id in config.passive_actors for event in events)
  requested_models = {event.model_or_engine for event in events}
  canonical_models = {
    providers.chat_config.chat_model, providers.completion_config.model,
    providers.embedding_config.embedding_model,
  }
  legacy_count = sum(model not in canonical_models for model in requested_models)
  snapshot = cost_guard.snapshot() if cost_guard else None
  cost_records = cost_guard.records() if cost_guard else ()
  total_cost = snapshot.accumulated_cost if snapshot else Decimal("0")
  retry_count = physical_attempts - logical_calls
  telemetry_by_actor = {}
  planning_callers = (
    "wake_up_hour", "daily_plan", "generate_hourly_schedule", "task_decomp")
  for name in config.cognitive_actors:
    actor_events = tuple(event for event in events if event.actor_id == name)
    actor_records = tuple(
      record for record in cost_records if record.actor_id == name)
    caller_counts = Counter(event.caller_id for event in actor_events)
    telemetry_by_actor[name] = {
      "logical_calls": len({event.logical_call_id for event in actor_events}),
      "physical_attempts": len(actor_events),
      "retries": len(actor_events) - len({
        event.logical_call_id for event in actor_events}),
      "input_tokens": sum(event.input_tokens or 0 for event in actor_events),
      "output_tokens": sum(event.output_tokens or 0 for event in actor_events),
      "cost_usd": sum(
        (record.estimated_total_cost_usd or Decimal("0")
         for record in actor_records), Decimal("0")),
      "operations": dict(Counter(event.operation for event in actor_events)),
      "callers": dict(Counter(
        event.caller_id for event in actor_events if event.caller_id)),
      "planning_callers": {
        caller: caller_counts[caller] for caller in planning_callers},
    }
  telemetry_by_tick = []
  for progression in tick_progression:
    step = progression["step_before"]
    tick_events = tuple(
      event for event in events if event.simulation_step == step)
    tick_records = tuple(
      record for record in cost_records if record.simulation_step == step)
    by_actor = {}
    for name in config.cognitive_actors:
      actor_events = tuple(
        event for event in tick_events if event.actor_id == name)
      actor_records = tuple(
        record for record in tick_records if record.actor_id == name)
      previous = (actor_state_before[name]
                  if progression["tick"] == 0
                  else tick_progression[progression["tick"] - 1][
                    "actors"][name])
      current = progression["actors"][name]
      actor_logical = len({
        event.logical_call_id for event in actor_events})
      by_actor[name] = {
        "logical_calls": actor_logical,
        "physical_attempts": len(actor_events),
        "retries": len(actor_events) - actor_logical,
        "input_tokens": sum(
          event.input_tokens or 0 for event in actor_events),
        "output_tokens": sum(
          event.output_tokens or 0 for event in actor_events),
        "cost_usd": sum(
          (record.estimated_total_cost_usd or Decimal("0")
           for record in actor_records), Decimal("0")),
        "operations": dict(Counter(
          event.operation for event in actor_events)),
        "callers": dict(Counter(
          event.caller_id for event in actor_events if event.caller_id)),
        "memory_delta": (
          current["memory_node_count"] - previous["memory_node_count"]),
        "embedding_delta": (
          current["embedding_count"] - previous["embedding_count"]),
      }
    tick_logical = len({event.logical_call_id for event in tick_events})
    telemetry_by_tick.append({
      "tick": progression["tick"], "simulation_step": step,
      "logical_calls": tick_logical,
      "physical_attempts": len(tick_events),
      "retries": len(tick_events) - tick_logical,
      "input_tokens": sum(event.input_tokens or 0 for event in tick_events),
      "output_tokens": sum(event.output_tokens or 0 for event in tick_events),
      "cost_usd": sum(
        (record.estimated_total_cost_usd or Decimal("0")
         for record in tick_records), Decimal("0")),
      "operations": dict(Counter(event.operation for event in tick_events)),
      "by_actor": by_actor,
    })
  global_operations = dict(Counter(event.operation for event in events))
  cognitive_moves_valid = all(
    actor_move_counts[name] == config.ticks
    for name in config.cognitive_actors)
  passive_moves_valid = all(
    actor_move_counts[name] == 0 for name in config.passive_actors)
  actor_structures_valid = bool(actor_state_after) and all(
    state["identity_aligned"] and state["node_ids_valid"]
    and state["node_ids_unique"] and state["embedding_references_valid"]
    and state["orphan_embedding_count"] == 0
    and state["daily_plan_present"] and state["schedule_present"]
    and state["current_action_present"]
    and state["current_action_actor_aligned"]
    for state in actor_state_after.values())
  isolation_valid = bool(isolation_before) and all(
    all(checks.values()) for checks in (
      isolation_before, *tick_isolation, isolation_after, isolation_reload))
  continuity_valid = bool(tick_progression) and all(continuity.values())
  tick_progression_valid = (
    len(tick_progression) == config.ticks
    and all(item["step_after"] == item["step_before"] + 1
            and item["time_after"] == item["time_before"] + dt.timedelta(
              seconds=config.tick_seconds)
            for item in tick_progression))
  telemetry_attribution_valid = all(
    event.actor_id in config.cognitive_actors
    and type(event.simulation_step) is int
    and initial_step <= event.simulation_step < initial_step + config.ticks
    for event in events)
  encounters = interaction_observer.encounters if interaction_observer else []
  reactions = interaction_observer.reactions if interaction_observer else []
  conversations = (
    interaction_observer.conversations if interaction_observer else [])
  retrieval_events = (
    interaction_observer.retrieval_events if interaction_observer else [])
  relationship_events = (
    interaction_observer.relationship_events if interaction_observer else [])
  last_chat_events = (
    interaction_observer.last_chat_events if interaction_observer else [])
  causal_observation_enabled = bool(
    resume is not None and resume.request.observe_causal_social_memory)
  retrieved_derived_refs = {
    node_ref for event in retrieval_events
    for node_ref in event[
      "retrieved_pre_resume_derived_social_node_refs"]}
  consumed_derived_refs = {
    node_ref for event in relationship_events
    for node_ref in event[
      "pre_resume_derived_social_node_refs_present"]}
  causal_classification = _classify_causal_social_memory(
    raw_social_memory_persisted=(
      causal_hydration.get("raw_social_memory_persisted", False)),
    derived_social_memory_persisted=(
      causal_hydration.get("derived_social_memory_persisted", False)),
    lineage_preserved_after_hydration=(
      causal_hydration.get("lineage_preserved", False)),
    pre_resume_derived_node_ids=(
      causal_pre_resume.get("derived_node_refs", ())),
    retrieved_node_ids=retrieved_derived_refs,
    consumed_node_ids=consumed_derived_refs)
  causal_report = {
    "observation_enabled": causal_observation_enabled,
    "pre_resume": (causal_pre_resume if resume is not None
                   else causal_run_snapshot),
    "hydration": ({
      **causal_hydration,
      "hydrated_snapshot": causal_hydrated,
    } if causal_observation_enabled else None),
    "retrieval": {
      "events": retrieval_events,
      "pre_resume_derived_nodes_retrieved":
        causal_classification[
          "retrieved_pre_resume_derived_social_node_ids"],
    },
    "relationship": {
      "events": relationship_events,
      "pre_resume_derived_nodes_consumed":
        causal_classification[
          "consumed_pre_resume_derived_social_node_ids"],
    },
    "last_chat": {
      "events": last_chat_events,
      "last_chat_social_context_observed": any(
        event["pre_resume_chat_node_observed"]
        for event in last_chat_events),
    },
    **causal_classification,
    "last_accessed_persistence_gap_detected": (
      causal_pre_resume.get(
        "last_accessed_persistence_gap_detected", False)
      if resume is not None else causal_run_snapshot.get(
        "last_accessed_persistence_gap_detected", False)),
    "candidate_material_preserved": (
      causal_hydration.get("derived_social_memory_persisted", False)
      if causal_observation_enabled else None),
    "ranking_state_fidelity": (
      "NOT_FULLY_GUARANTEED" if causal_observation_enabled else None),
  }
  bilateral_encounter = all(any(
    row["observer"] == actor and row["target"] == target
    for row in encounters)
    for actor, target in (R1M3C_ACTORS, tuple(reversed(R1M3C_ACTORS))))
  conversation_started = bool(conversations)
  def conversation_has_integrity(row):
    return (
      row["turn_count"] > 0
      and set(row["participants"]) == set(R1M3C_ACTORS)
      and all(actor in row["speaker_sequence"] for actor in R1M3C_ACTORS)
      and row["distinct_chat_objects"])

  conversation_committed = any(
    conversation_has_integrity(row) for row in conversations)
  model_end_observed = any(
    conversation_has_integrity(row) and row["termination"] == "MODEL_END"
    for row in conversations)
  safety_ceiling_reached = any(
    conversation_has_integrity(row)
    and row["termination"] == "SAFETY_CEILING"
    for row in conversations)
  valid_conversation = conversation_committed and model_end_observed
  bilateral_memory = bool(actor_state_after) and all(
    actor_state_after.get(name, {}).get("chat_count", 0)
    > actor_state_before.get(name, {}).get("chat_count", 0)
    for name in R1M3C_ACTORS)
  bilateral_memory_reloaded = bilateral_memory and all(
    actor_state_reload.get(name, {}).get("chat_node_ids")
    == actor_state_after.get(name, {}).get("chat_node_ids")
    for name in R1M3C_ACTORS)
  success = all((error is None, completed == config.ticks,
                 movement_count == config.ticks,
                 final_step == initial_step + config.ticks,
                 final_time == initial_time + dt.timedelta(
                 seconds=config.tick_seconds * config.ticks),
                 save_passed, reload_passed, passive_provider_calls == 0,
                 passive_mutations == 0, legacy_count == 0,
                 cognitive_moves_valid, passive_moves_valid,
                 isolation_valid, actor_structures_valid,
                 continuity_valid, tick_progression_valid,
                 movement_integrity_valid, telemetry_attribution_valid,
                 resume is None or (
                   resume_hydration.get("all_checks_passed", False)
                   and source_hash_before == _tree_sha256(source))))
  r1m3b_policy = (
    config.ticks == 5 and config.cognitive_actors == VISIBLE_ACTORS
    and not config.passive_actors)
  r1m3a_policy = (
    config.ticks == 1 and config.cognitive_actors == VISIBLE_ACTORS
    and not config.passive_actors)
  if causal_observation_enabled:
    if not success:
      verdict = R1CLI_A2_B_BLOCKED_VERDICT
    elif not relationship_events:
      verdict = R1CLI_A2_B_PATH_NOT_REACHED_VERDICT
    elif causal_classification["causal_link_verified"]:
      verdict = R1CLI_A2_B_READY_VERDICT
    else:
      verdict = R1CLI_A2_B_BLOCKED_VERDICT
  elif resume is not None:
    verdict = (R1CLI_A2_A_READY_VERDICT if success
               else R1CLI_A2_A_BLOCKED_VERDICT)
  elif success and r1m3b_policy:
    verdict = "R1M3_B_THREE_COGNITIVE_ACTORS_FIVE_TICKS_PASSED"
  elif success and r1m3a_policy:
    verdict = "R1M3_A_THREE_COGNITIVE_ACTORS_ONE_TICK_PASSED"
  elif success:
    verdict = "MODERN_SMALLVILLE_HEADLESS_RUN_PASSED"
  elif r1m3b_policy:
    verdict = "R1M3_B_BLOCKED"
  elif r1m3a_policy:
    verdict = "R1M3_A_BLOCKED"
  else:
    verdict = "MODERN_SMALLVILLE_HEADLESS_RUN_FAILED"
  if resume is None and config.controlled_proximity:
    conversation_callers = {
      "agent_chat_summarize_relationship", "iterative_chat_utterance",
      "summarize_conversation", "chat_poignancy"}
    last_caller = events[-1].caller_id if events else None
    chat_was_selected = any(
      row["decision_category"] == "CHAT" for row in reactions)
    if (isinstance(error, (ModernDeferredCallerError,
                           LLMIncompleteResponseError))
        or (error is not None and chat_was_selected
            and last_caller in conversation_callers)):
      verdict = "R1M3_C_MODERN_CALLER_BLOCKED"
    elif not encounters:
      verdict = "R1M3_C_ENCOUNTER_NOT_REACHED"
    elif not conversation_started:
      verdict = "R1M3_C_ENCOUNTER_PASSED_NO_NATURAL_CONVERSATION"
    else:
      verdict = _classify_r1m3c_conversation(
        conversation_committed=conversation_committed,
        model_end_observed=model_end_observed,
        safety_ceiling_reached=safety_ceiling_reached,
        bilateral_memory=bilateral_memory,
        bilateral_memory_reloaded=bilateral_memory_reloaded,
        memory_integrity_valid=actor_structures_valid,
        save_passed=save_passed, reload_passed=reload_passed)
  result = ModernRunResult(
    verdict=verdict, run_directory=run_dir, completed_ticks=completed,
    cognitive_actors=config.cognitive_actors,
    passive_actors=config.passive_actors,
    movement_count=movement_count, initial_step=initial_step,
    final_step=final_step, initial_time=initial_time, final_time=final_time,
    logical_calls=logical_calls, physical_attempts=physical_attempts,
    input_tokens=sum(event.input_tokens or 0 for event in events),
    output_tokens=sum(event.output_tokens or 0 for event in events),
    total_cost_usd=total_cost, cost_ceiling_usd=config.cost_ceiling_usd,
    save_passed=save_passed, reload_passed=reload_passed,
    actor_move_counts=tuple((name, actor_move_counts[name])
                            for name in VISIBLE_ACTORS),
    passive_provider_calls=passive_provider_calls,
    passive_memory_mutations=passive_mutations,
    legacy_fallback_count=legacy_count, retry_count=retry_count,
    exception_type=type(error).__name__ if error else None,
    exception_message=str(error)[:512] if error else None,
  )
  report = {
    "result": asdict(result),
    "config": asdict(config),
    "resume": (None if resume is None else {
      "source_run": resume.source_run,
      "source_simulation": resume.simulation_root,
      "persisted_initial_step": resume.source_meta["step"],
      "persisted_initial_time": resume.source_meta["curr_time"],
      "continuation_strategy": "STANFORD_FORK_COPY",
      "hydration": resume_hydration,
      "hydration_provider_calls": hydration_provider_calls,
      "history_preserved": movement_integrity_valid,
      "cognitive_state_preserved": resume_hydration.get(
        "all_checks_passed", False),
      "old_frame_hashes": resume.movement_hashes,
      "new_frame_hashes": movement_integrity.get("frame_hashes", {}),
      "old_environment_hashes": resume.environment_hashes,
    }),
    "causal_resume": causal_report,
    "actor_policy": {"cognitive": config.cognitive_actors,
                     "passive": config.passive_actors,
                     "visible": config.visible_actors},
    "fixture": {
      "source_simulation": config.source_simulation,
      "source_hash_before": source_hash_before,
      "source_hash_after": _tree_sha256(source),
      "source_unchanged": source_hash_before == _tree_sha256(source),
      "seed": fixture_seed,
      "validation": controlled_fixture,
    },
    "interaction": {
      "encounter_gate": {
        "reached": bool(encounters), "bilateral": bilateral_encounter,
        "events": encounters,
      },
      "reaction_gate": {"reached": bool(reactions), "events": reactions},
      "conversation_gate": {
        "started": conversation_started,
        "committed": conversation_committed,
        "valid": valid_conversation,
        "model_end_observed": model_end_observed,
        "safety_ceiling_reached": safety_ceiling_reached,
        "social_pipeline_functional": (
          verdict in (R1M3C_NATURAL_VERDICT, R1M3C_FUNCTIONAL_VERDICT)),
        "conversations": conversations,
      },
      "bilateral_memory": {
        "saved": bilateral_memory,
        "reloaded": bilateral_memory_reloaded,
        "actors": {
          name: {
            "before": actor_state_before.get(name, {}).get("chat_count", 0),
            "after": actor_state_after.get(name, {}).get("chat_count", 0),
            "reload": actor_state_reload.get(name, {}).get("chat_count", 0),
            "new_node_ids": sorted(set(
              actor_state_after.get(name, {}).get("chat_node_ids", ()))
              - set(actor_state_before.get(name, {}).get(
                "chat_node_ids", ()))),
          } for name in R1M3C_ACTORS},
      },
    },
    "runtime": {
      "chat_model": providers.chat_config.chat_model,
      "completion_compat_model": providers.completion_config.model,
      "embedding_model": providers.embedding_config.embedding_model,
      "embedding_dimensions": TEXT_EMBEDDING_3_SMALL_1536_MANIFEST.dimensions,
      "sdk_version": importlib.metadata.version("openai"),
      "sdk_retries": 0, "legacy_fallback_count": legacy_count,
    },
    "embedding_preflight": {
      "before": [audit.classification
                 for audit in embedding_preflight.audits_before],
      "after": [audit.classification
                for audit in embedding_preflight.audits_after],
      "bootstrapped_personas": embedding_preflight.bootstrapped_personas,
    },
    "actors": {
      name: {
        "before": actor_state_before.get(name, {}),
        "after": actor_state_after.get(name, {}),
        "reload": actor_state_reload.get(name, {}),
      }
      for name in config.cognitive_actors},
    "multi_actor_isolation": {
      "before": isolation_before,
      "after": isolation_after,
      "reload": isolation_reload,
      "by_tick": tick_isolation,
      "all_checks_passed": isolation_valid,
    },
    "continuity": {
      **continuity, "all_checks_passed": continuity_valid,
    },
    "tick_progression": tick_progression,
    "movement_integrity": {
      **movement_integrity, "all_checks_passed": movement_integrity_valid,
    },
    "embedding_stores": {
      "saved": saved_embedding_metadata,
      "reload": reload_embedding_metadata,
    },
    "telemetry": {
      "global": {
        "logical_calls": logical_calls,
        "physical_attempts": physical_attempts,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": total_cost,
        "operations": global_operations,
        "retries": retry_count,
      },
      "by_actor": telemetry_by_actor,
      "by_tick": telemetry_by_tick,
      "attribution_valid": telemetry_attribution_valid,
    },
    "failure": ({
      "stage": execution_state["stage"],
      "actor": execution_state["actor"],
      "tick": execution_state["tick"],
      "exception_type": result.exception_type,
      "exception_message": result.exception_message,
      "caller": (getattr(error, "caller", None)
                 or (events[-1].caller_id if events else None)),
      "operation": (getattr(error, "operation", None)
                    or (events[-1].operation if events else None)),
    } if error else None),
    "reload": reloaded_summary,
    "artifacts": {
      "simulation": storage_root / simulation_code,
      "movement": storage_root / simulation_code / "movement",
      "status": status_path, "report": report_path,
    },
  }
  _write_json(report_path, report)
  _write_json(status_path, {
    "run_id": config.run_name, "state": "PASSED" if success else "FAILED",
    "completed_ticks": completed, "step": final_step,
    "curr_time": final_time, "total_cost_usd": total_cost,
    "exception_type": result.exception_type,
    "exception_message": result.exception_message,
    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
  })
  return result


def _decimal_argument(value: str) -> Decimal:
  try:
    result = Decimal(value)
  except InvalidOperation as error:
    raise argparse.ArgumentTypeError("must be a decimal number") from error
  if not result.is_finite() or result <= 0:
    raise argparse.ArgumentTypeError("must be greater than zero")
  return result


def _positive_int(value: str) -> int:
  try:
    result = int(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError("must be an integer") from error
  if result <= 0:
    raise argparse.ArgumentTypeError("must be greater than zero")
  return result


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="modern_smallville")
  commands = parser.add_subparsers(dest="command", required=True)
  run = commands.add_parser("run", help="run modern Smallville headlessly")
  run.add_argument("--ticks", type=_positive_int, default=1)
  run.add_argument("--name")
  run.add_argument("--cost-ceiling", type=_decimal_argument,
                   default=DEFAULT_COST_CEILING)
  run.add_argument("--source", default=DEFAULT_SOURCE)
  run.add_argument(
    "--cognitive", choices=("isabella", "all"), default="isabella",
    help="run cognition for Isabella only or for all visible actors")
  run.add_argument(
    "--controlled-proximity", action="store_true",
    help="use the isolated daytime R1M3-C encounter fixture")
  resume = commands.add_parser(
    "resume", help="continue a validated persisted modern run")
  resume.add_argument("--from", dest="source_run", type=Path, required=True)
  resume.add_argument("--ticks", type=_positive_int, default=5)
  resume.add_argument("--name")
  resume.add_argument("--cost-ceiling", type=_decimal_argument,
                      default=DEFAULT_COST_CEILING)
  resume.add_argument(
    "--observe-causal-social-memory", action="store_true",
    help="observe persisted Chat-derived Event/Thought social cognition")
  return parser


def _render(result: ModernRunResult) -> str:
  moves = dict(result.actor_move_counts)
  remaining = max(Decimal("0"),
                  result.cost_ceiling_usd - result.total_cost_usd)
  cognitive_lines = tuple(f"  {name}" for name in result.cognitive_actors)
  passive_lines = (tuple(f"  {name}" for name in result.passive_actors)
                   or ("  (none)",))
  move_summary = ", ".join(
    f"{name.split()[0]}={moves.get(name, 0)}" for name in VISIBLE_ACTORS)
  return "\n".join((
    result.verdict,
    "",
    f"Run:               {result.run_directory.name}",
    f"Completed ticks:   {result.completed_ticks}",
    f"Movement frames:   {result.movement_count}",
    f"Step:              {result.initial_step} -> {result.final_step}",
    f"Time:              {result.initial_time} -> {result.final_time}",
    "",
    "Cognitive actors:", *cognitive_lines,
    "Passive actors:", *passive_lines,
    "",
    f"Persona.move:       {move_summary}",
    f"Logical calls:      {result.logical_calls}",
    f"Physical attempts:  {result.physical_attempts}",
    f"Input tokens:       {result.input_tokens}",
    f"Output tokens:      {result.output_tokens}",
    f"Total cost:         ${result.total_cost_usd}",
    f"Cost ceiling:       ${result.cost_ceiling_usd}",
    f"Remaining ceiling:  ${remaining}",
    f"Save:               {'PASS' if result.save_passed else 'FAIL'}",
    f"Offline reload:     {'PASS' if result.reload_passed else 'FAIL'}",
    "", "Artifacts:", str(result.run_directory),
  ))


def main(argv: Optional[list[str]] = None) -> int:
  parser = build_parser()
  try:
    args = parser.parse_args(argv)
    if args.command == "resume":
      config = ModernResumeConfig(
        source_run=args.source_run,
        run_name=args.name or generate_run_name(), ticks=args.ticks,
        cost_ceiling_usd=args.cost_ceiling,
        observe_causal_social_memory=args.observe_causal_social_memory)
      result = run_modern_smallville_resume(config)
    else:
      cognitive_actors = (
        VISIBLE_ACTORS if args.cognitive == "all" else (COGNITIVE_ACTOR,))
      passive_actors = tuple(
        name for name in VISIBLE_ACTORS if name not in cognitive_actors)
      config = ModernRunConfig(
        source_simulation=args.source,
        run_name=args.name or generate_run_name(),
        ticks=args.ticks,
        cost_ceiling_usd=args.cost_ceiling,
        cognitive_actors=cognitive_actors,
        passive_actors=passive_actors,
        controlled_proximity=args.controlled_proximity,
      )
      result = run_modern_smallville(config)
  except ModernRunConfigurationError as error:
    print(f"configuration error: {error}", file=sys.stderr)
    return EXIT_CONFIGURATION
  except (ReplayCostCeilingExceededError,
          ReplayCostGuardAlreadyTrippedError) as error:
    print(f"cost ceiling exceeded: {error}", file=sys.stderr)
    return EXIT_COST_CEILING
  except (ModernRuntimeInvariantError,
          controlled_replay.ControlledReplayLegacyConfigurationError) as error:
    print(f"modern runtime invariant failed: {error}", file=sys.stderr)
    return EXIT_MODERN_INVARIANT
  except Exception as error:
    print(f"runtime failure: {type(error).__name__}: {error}", file=sys.stderr)
    return EXIT_RUNTIME_FAILURE
  print(_render(result))
  if (result.verdict.endswith("PASSED")
      or result.verdict in (
        R1CLI_A2_A_READY_VERDICT, R1CLI_A2_B_READY_VERDICT)):
    return EXIT_SUCCESS
  if result.exception_type in (
      "ReplayCostCeilingExceededError", "ReplayCostGuardAlreadyTrippedError"):
    return EXIT_COST_CEILING
  if result.exception_type in (
      "ModernRuntimeInvariantError", "ModernDeferredCallerError",
      "ControlledReplayLegacyConfigurationError"):
    return EXIT_MODERN_INVARIANT
  return EXIT_RUNTIME_FAILURE


if __name__ == "__main__":
  raise SystemExit(main())
