"""Operational headless launcher for the modern Smallville runtime."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
import datetime as dt
from decimal import Decimal, InvalidOperation
import importlib.metadata
import importlib.util
import json
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
from persona.prompt_template.modern_openai_provider import (
  ModernOpenAIClientAdapter,
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
DEFAULT_COST_CEILING = Decimal("0.03")
RUNTIME_ROOT = REPOSITORY_ROOT / ".runtime" / "live-runs"
SOURCE_ROOT = (REPOSITORY_ROOT / "environment" / "frontend_server"
               / "storage")
DATE_FORMAT = "%B %d, %Y, %H:%M:%S"
CANONICAL_START_TIME = dt.datetime(2023, 2, 13, 5, 55, 0)
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

EXIT_SUCCESS = 0
EXIT_RUNTIME_FAILURE = 1
EXIT_CONFIGURATION = 2
EXIT_COST_CEILING = 3
EXIT_MODERN_INVARIANT = 4


class ModernSmallvilleError(RuntimeError):
  """Base error for the operational launcher."""


class ModernRunConfigurationError(ModernSmallvilleError, ValueError):
  pass


class ModernRuntimeInvariantError(ModernSmallvilleError):
  pass


@dataclass(frozen=True)
class ModernRunConfig:
  source_simulation: str = DEFAULT_SOURCE
  run_name: str = "modern-smallville"
  ticks: int = 1
  tick_seconds: int = 10
  cognitive_actors: tuple[str, ...] = (COGNITIVE_ACTOR,)
  visible_actors: tuple[str, ...] = VISIBLE_ACTORS
  cost_ceiling_usd: Decimal = DEFAULT_COST_CEILING

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
    if self.cognitive_actors != (COGNITIVE_ACTOR,):
      raise ModernRunConfigurationError(
        "R1CLI-A requires Isabella Rodriguez as its only cognitive actor")
    if self.visible_actors != VISIBLE_ACTORS:
      raise ModernRunConfigurationError(
        "R1CLI-A requires Isabella, Maria and Klaus as visible actors")
    if not set(self.cognitive_actors).issubset(self.visible_actors):
      raise ModernRunConfigurationError(
        "every cognitive actor must be visible")
    if self.tick_seconds != 10:
      raise ModernRunConfigurationError("R1CLI-A requires 10-second ticks")


@dataclass(frozen=True)
class ModernRunResult:
  verdict: str
  run_directory: Path
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


def _bootstrap_isolated_temporal_source(source: Path) -> None:
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
    if name == COGNITIVE_ACTOR:
      scratch.update({
        "act_address": "the Ville:Isabella Rodriguez's apartment:main room:bed",
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
    caller = kwargs["replay_context"].caller_id
    if caller in controlled_replay.FORBIDDEN_COGNITIVE_CALLERS:
      raise controlled_replay.ControlledReplayPathForbiddenError(
        "conversation/reflection provider path is outside R1CLI-A")
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
  _bootstrap_isolated_temporal_source(isolated_source)
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
  passive_mutations = 0
  passive_provider_calls = 0
  legacy_count = 0
  observer = None
  cost_guard = None
  error = None
  reloaded_summary = {}
  passive_before = {}

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
        active = server.personas[COGNITIVE_ACTOR]
        passive_before = {
          name: passive_cognitive_fingerprint(server.personas[name])
          for name in PASSIVE_ACTORS}
        for name, persona in server.personas.items():
          original = persona.move

          def counted_move(*args, _name=name, _original=original, **kwargs):
            actor_move_counts[_name] += 1
            return _original(*args, **kwargs)

          persona.move = counted_move

        saved_root = storage_root / simulation_code
        with install_passive_visible_moves(
            server, PASSIVE_ACTORS) as passive_controller:
          for tick in range(config.ticks):
            context = LLMReplayContext(
              cognitive_category="WORLD_TICK", actor_id=COGNITIVE_ACTOR,
              simulation_id=simulation_code, simulation_step=server.step)
            ledger = CostLedgerContext(
              simulation_id=simulation_code, simulation_step=server.step,
              actor_id=COGNITIVE_ACTOR, cognitive_category="WORLD_TICK")
            with use_llm_replay_context(context), use_cost_ledger_context(ledger):
              server.start_server(1)
            movement_path = saved_root / "movement" / f"{initial_step + tick}.json"
            movement = _read_json(movement_path)
            if set(movement.get("persona", {})) != set(VISIBLE_ACTORS):
              raise ModernRuntimeInvariantError(
                "movement frame does not contain all visible actors")
            environment = _read_json(
              saved_root / "environment" / f"{initial_step + tick}.json")
            for name in VISIBLE_ACTORS:
              coordinate = movement["persona"][name].get("movement")
              if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ModernRuntimeInvariantError(
                  "movement frame contains an invalid coordinate")
              environment[name] = dict(environment[name])
              environment[name]["x"], environment[name]["y"] = coordinate
            _write_json(saved_root / "environment" / f"{server.step}.json",
                        environment)
            movement_count += 1
            completed += 1

          passive_after = {
            name: passive_cognitive_fingerprint(server.personas[name])
            for name in PASSIVE_ACTORS}
          passive_mutations = sum(
            passive_before[name] != passive_after[name]
            for name in PASSIVE_ACTORS)
          if passive_mutations:
            raise ModernRuntimeInvariantError("passive cognitive state mutated")
          if any(actor_move_counts[name] for name in PASSIVE_ACTORS):
            raise ModernRuntimeInvariantError("passive Persona.move was invoked")
          if any(passive_controller.frame_emissions[name] != config.ticks
                 for name in PASSIVE_ACTORS):
            raise ModernRuntimeInvariantError("passive frame emission mismatch")

        server.save()
        save_passed = True
        calls_before_reload = len(get_telemetry())
        reloaded = reverie_module.ReverieServer(simulation_code, reload_code)
        calls_after_reload = len(get_telemetry())
        original_memory = _memory_counts(server.personas[COGNITIVE_ACTOR])
        reload_memory = _memory_counts(reloaded.personas[COGNITIVE_ACTOR])
        reload_passed = all((
          calls_before_reload == calls_after_reload,
          reloaded.step == server.step,
          reloaded.curr_time == server.curr_time,
          len(reloaded.personas) == len(VISIBLE_ACTORS),
          original_memory == reload_memory,
          movement_count == len(list((saved_root / "movement").glob("*.json"))),
        ))
        reloaded_summary = {
          "step": reloaded.step, "curr_time": reloaded.curr_time,
          "persona_count": len(reloaded.personas),
          "memory_nodes": reload_memory["nodes"],
          "embedding_count": reload_memory["embeddings"],
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
    event.actor_id in PASSIVE_ACTORS for event in events)
  requested_models = {event.model_or_engine for event in events}
  canonical_models = {
    providers.chat_config.chat_model, providers.completion_config.model,
    providers.embedding_config.embedding_model,
  }
  legacy_count = sum(model not in canonical_models for model in requested_models)
  snapshot = cost_guard.snapshot() if cost_guard else None
  total_cost = snapshot.accumulated_cost if snapshot else Decimal("0")
  retry_count = physical_attempts - logical_calls
  success = all((error is None, completed == config.ticks,
                 movement_count == config.ticks,
                 final_step == initial_step + config.ticks,
                 final_time == initial_time + dt.timedelta(
                   seconds=config.tick_seconds * config.ticks),
                 save_passed, reload_passed, passive_provider_calls == 0,
                 passive_mutations == 0, legacy_count == 0))
  verdict = ("MODERN_SMALLVILLE_HEADLESS_RUN_PASSED" if success
             else "MODERN_SMALLVILLE_HEADLESS_RUN_FAILED")
  result = ModernRunResult(
    verdict=verdict, run_directory=run_dir, completed_ticks=completed,
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
                     "passive": PASSIVE_ACTORS,
                     "visible": config.visible_actors},
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
  return parser


def _render(result: ModernRunResult) -> str:
  moves = dict(result.actor_move_counts)
  remaining = max(Decimal("0"),
                  result.cost_ceiling_usd - result.total_cost_usd)
  return "\n".join((
    result.verdict,
    "",
    f"Run:               {result.run_directory.name}",
    f"Completed ticks:   {result.completed_ticks}",
    f"Movement frames:   {result.movement_count}",
    f"Step:              {result.initial_step} -> {result.final_step}",
    f"Time:              {result.initial_time} -> {result.final_time}",
    "",
    "Cognitive actors:", "  Isabella Rodriguez",
    "Passive actors:", "  Maria Lopez", "  Klaus Mueller",
    "",
    f"Persona.move:       Isabella={moves.get(COGNITIVE_ACTOR, 0)}, "
    f"Maria={moves.get(PASSIVE_ACTORS[0], 0)}, "
    f"Klaus={moves.get(PASSIVE_ACTORS[1], 0)}",
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
    config = ModernRunConfig(
      source_simulation=args.source,
      run_name=args.name or generate_run_name(),
      ticks=args.ticks,
      cost_ceiling_usd=args.cost_ceiling,
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
      "ModernRuntimeInvariantError", "ControlledReplayLegacyConfigurationError"):
    return EXIT_MODERN_INVARIANT
  return EXIT_RUNTIME_FAILURE


if __name__ == "__main__":
  raise SystemExit(main())
