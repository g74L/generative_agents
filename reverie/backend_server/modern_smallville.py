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

  def __init__(self, server, execution_state, simulation_id):
    self.server = server
    self.execution_state = execution_state
    self.simulation_id = simulation_id
    self.encounters = []
    self.reactions = []
    self.conversations = []
    self.turn_results = []
    self._original_perceive = {}
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
        return self._originals["new_retrieve"](persona, *args, **kwargs)

    def actor_relationship(init_persona, *args, **kwargs):
      with self._actor_context(init_persona.name):
        return self._originals["relationship"](
          init_persona, *args, **kwargs)

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


@contextmanager
def _observe_conversations(server, execution_state, simulation_id):
  observer = _ConversationObserver(
    server, execution_state, simulation_id).install()
  try:
    yield observer
  finally:
    observer.restore()


def run_modern_smallville(
    config: ModernRunConfig, *, adapter=None,
    runtime_root: Optional[Path] = None) -> ModernRunResult:
  """Run one isolated modern Smallville and return a compact typed result."""
  if not isinstance(config, ModernRunConfig):
    raise TypeError("config must be ModernRunConfig")
  root = Path(runtime_root or RUNTIME_ROOT).resolve()
  source = (SOURCE_ROOT / config.source_simulation).resolve()
  if not source.is_dir() or source.parent != SOURCE_ROOT.resolve():
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
  embedding_preflight = controlled_replay.prepare_isolated_embedding_stores(
    fixture)

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

        server = reverie_module.ReverieServer(source_code, simulation_code)
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
                simulation_code) as interaction_observer):
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
        execution_state.update(stage="save", actor=None, tick=server.step)
        server.save()
        save_passed = True
        frame_hashes_after_save = {
          path.name: _file_sha256(path) for path in expected_frames}
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
        movement_integrity = {
          "expected_frame_count": config.ticks,
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
        }
        movement_integrity_valid = all((
          movement_integrity["saved_frame_count"] == config.ticks,
          movement_integrity["reload_frame_count"] == config.ticks,
          movement_integrity["all_frames_distinct"],
          movement_integrity["unchanged_by_save"],
          movement_integrity["unchanged_by_reload"],
          movement_integrity["reload_copy_matches"],
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
          movement_count == len(list((saved_root / "movement").glob("*.json"))),
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
                 movement_integrity_valid, telemetry_attribution_valid))
  r1m3b_policy = (
    config.ticks == 5 and config.cognitive_actors == VISIBLE_ACTORS
    and not config.passive_actors)
  r1m3a_policy = (
    config.ticks == 1 and config.cognitive_actors == VISIBLE_ACTORS
    and not config.passive_actors)
  if success and r1m3b_policy:
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
  if config.controlled_proximity:
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
  if result.verdict.endswith("PASSED"):
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
