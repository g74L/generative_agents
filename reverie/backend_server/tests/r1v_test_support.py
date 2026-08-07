"""Sanitized, versionable support for offline R1V contract tests."""
import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
  sys.path.insert(0, str(BACKEND_ROOT))

from modern_smallville import (
  PassiveVisibleMoveController,
  install_passive_visible_moves,
  passive_cognitive_fingerprint,
)


DATE_FORMAT = "%B %d, %Y, %H:%M:%S"
ACTOR = "Isabella Rodriguez"
PASSIVES = ("Maria Lopez", "Klaus Mueller")
PERSONAS = (ACTOR,) + PASSIVES
RUN_OUTPUT_NAMES = (
  "status.json", "report.json", "execution.log", "checkpoints.jsonl",
  "tick-reports",
)


def safe(value):
  if isinstance(value, Decimal):
    return str(value)
  if isinstance(value, datetime.datetime):
    return value.strftime(DATE_FORMAT)
  if isinstance(value, dict):
    return {str(key): safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [safe(item) for item in value]
  if isinstance(value, set):
    return sorted(safe(item) for item in value)
  return value


def atomic_json(path, value):
  path = Path(path)
  temporary = path.with_name(path.name + ".tmp")
  with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(safe(value), stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(temporary, path)


def read_json(path):
  return json.loads(Path(path).read_text(encoding="utf-8"))


def tree_fingerprint(root):
  root = Path(root)
  rows = [
    (path.relative_to(root).as_posix(),
     hashlib.sha256(path.read_bytes()).hexdigest())
    for path in sorted(root.rglob("*")) if path.is_file()
  ]
  serialized = json.dumps(rows, separators=(",", ":")).encode("utf-8")
  return {
    "file_count": len(rows),
    "sha256": hashlib.sha256(serialized).hexdigest(),
  }


def assert_fresh_run_outputs(run_dir, storage_root, sim_code,
                             reload_code=None):
  """Fail closed without modifying an existing run or simulation."""
  run_dir = Path(run_dir)
  storage_root = Path(storage_root)
  existing = [name for name in RUN_OUTPUT_NAMES
              if (run_dir / name).exists()]
  simulation_codes = tuple(
    code for code in (sim_code, reload_code) if code)
  simulations = [code for code in simulation_codes
                 if (storage_root / code).exists()]
  if existing or simulations:
    reasons = existing + [f"simulation:{code}" for code in simulations]
    raise RuntimeError(
      "R1V_RUN_OUTPUT_ALREADY_EXISTS: " + ", ".join(reasons))


def traceback_frames(error, repository_root=None):
  repository_root = (Path(repository_root).resolve()
                     if repository_root else None)
  result = []
  node = error.__traceback__
  while node:
    path = Path(node.tb_frame.f_code.co_filename).resolve()
    if (repository_root is not None
        and (path == repository_root or repository_root in path.parents)):
      label = path.relative_to(repository_root).as_posix()
    else:
      label = "<external>/" + path.name
    result.append({
      "file": label,
      "function": node.tb_frame.f_code.co_name,
      "line": node.tb_lineno,
    })
    node = node.tb_next
  return result


def build_failure_observability(error, current_tick, server, per_tick,
                                checkpoint_names, repository_root=None):
  """Build additive, JSON-safe evidence without prompts, responses or locals."""
  return {
    "exception_type": type(error).__name__ if error else None,
    "exception_message": str(error) if error else None,
    "exception_repr": repr(error) if error else None,
    "traceback": (traceback_frames(error, repository_root)
                  if error else []),
    "failure_context": {
      "tick": current_tick if error else None,
      "step": server.step if error and server else None,
      "curr_time": server.curr_time if error and server else None,
      "completed_ticks": len(per_tick),
      "last_completed_checkpoint": (
        checkpoint_names[-1] if checkpoint_names else None),
    },
  }


def build_synthetic_cognitive_seed(baseline_root, seed_root):
  """Create a sanitized sleeping-state seed inside a caller-owned temp root."""
  baseline_root = Path(baseline_root).resolve()
  seed_root = Path(seed_root).resolve()
  if seed_root.exists():
    raise RuntimeError("synthetic R1V seed already exists")
  if not baseline_root.is_dir():
    raise RuntimeError("versioned structural baseline is unavailable")
  shutil.copytree(baseline_root, seed_root)

  meta_path = seed_root / "reverie" / "meta.json"
  meta = read_json(meta_path)
  meta.update({
    "curr_time": "February 13, 2023, 00:00:00",
    "sec_per_step": 10,
    "step": 0,
  })
  meta["persona_names"] = list(PERSONAS)
  atomic_json(meta_path, meta)

  environment_dir = seed_root / "environment"
  for path in environment_dir.glob("*.json"):
    if path.name != "0.json":
      path.unlink()
  environment = read_json(environment_dir / "0.json")
  if set(environment) != set(PERSONAS):
    raise RuntimeError("versioned structural environment is incomplete")

  movement_dir = seed_root / "movement"
  if movement_dir.exists():
    shutil.rmtree(movement_dir)
  movement_dir.mkdir()

  for name in PERSONAS:
    scratch_path = (seed_root / "personas" / name / "bootstrap_memory"
                    / "scratch.json")
    scratch = read_json(scratch_path)
    scratch["curr_time"] = "February 13, 2023, 00:00:00"
    scratch["daily_req"] = ["synthetic day"]
    scratch["daily_plan_req"] = ["synthetic plan"]
    scratch["f_daily_schedule"] = [
      ["sleeping", 360], ["synthetic action", 1080]]
    scratch["f_daily_schedule_hourly_org"] = list(
      scratch["f_daily_schedule"])
    if name == ACTOR:
      scratch.update({
        "act_address": "synthetic world:synthetic room:bed",
        "act_description": "sleeping",
        "act_duration": 360,
        "act_event": [name, "is", "sleeping"],
        "act_path_set": False,
        "act_pronunciatio": "",
        "act_start_time": "February 13, 2023, 00:00:00",
        "planned_path": [],
      })
    else:
      scratch.update({
        "act_address": None,
        "act_description": None,
        "act_duration": None,
        "act_event": [name, None, None],
        "act_path_set": False,
        "act_pronunciatio": None,
        "act_start_time": None,
        "planned_path": [],
      })
    atomic_json(scratch_path, scratch)
  return seed_root


def bootstrap_temporal_source(seed_root, source_root, start_time):
  """Copy a synthetic cognitive seed and reset only its temporal origin."""
  seed_root = Path(seed_root).resolve()
  source_root = Path(source_root).resolve()
  if source_root.exists():
    raise RuntimeError("R1V temporal source already exists")
  if not seed_root.is_dir():
    raise RuntimeError("R1V cognitive seed is unavailable")
  if not isinstance(start_time, datetime.datetime):
    raise TypeError("start_time must be a datetime")

  meta = read_json(seed_root / "reverie" / "meta.json")
  seed_step = meta.get("step")
  if type(seed_step) is not int or seed_step < 0:
    raise RuntimeError("R1V cognitive seed step is invalid")
  seed_environment = seed_root / "environment" / f"{seed_step}.json"
  if not seed_environment.is_file():
    raise RuntimeError("R1V cognitive seed environment is unavailable")
  if meta.get("sec_per_step") != 10:
    raise RuntimeError("R1V cognitive seed must use 10 seconds per step")
  if set(meta.get("persona_names", ())) != set(PERSONAS):
    raise RuntimeError("R1V cognitive seed persona registry is incomplete")

  shutil.copytree(seed_root, source_root)
  environment = read_json(seed_environment)
  if set(environment) != set(PERSONAS):
    raise RuntimeError("R1V cognitive seed environment is incomplete")
  environment_dir = source_root / "environment"
  for path in environment_dir.glob("*.json"):
    path.unlink()
  atomic_json(environment_dir / "0.json", environment)
  movement_dir = source_root / "movement"
  if movement_dir.exists():
    shutil.rmtree(movement_dir)
  movement_dir.mkdir()

  serialized_start = start_time.strftime(DATE_FORMAT)
  meta["curr_time"] = serialized_start
  meta["step"] = 0
  atomic_json(source_root / "reverie" / "meta.json", meta)

  scratch_state = {}
  for name in PERSONAS:
    scratch_path = (source_root / "personas" / name / "bootstrap_memory"
                    / "scratch.json")
    scratch = read_json(scratch_path)
    scratch["curr_time"] = serialized_start
    atomic_json(scratch_path, scratch)
    scratch_state[name] = {
      "curr_time": scratch.get("curr_time"),
      "act_start_time": scratch.get("act_start_time"),
      "act_duration": scratch.get("act_duration"),
      "act_description": scratch.get("act_description"),
      "act_address": scratch.get("act_address"),
      "planned_path_length": len(scratch.get("planned_path") or []),
      "schedule_length": len(scratch.get("f_daily_schedule") or []),
    }

  actor = read_json(
    source_root / "personas" / ACTOR / "bootstrap_memory" / "scratch.json")
  action_start = datetime.datetime.strptime(
    actor["act_start_time"], DATE_FORMAT)
  action_end = action_start + datetime.timedelta(
    minutes=actor["act_duration"])
  if actor.get("act_description") != "sleeping":
    raise RuntimeError("R1V cognitive seed action is not sleeping")
  if not str(actor.get("act_address", "")).endswith(":bed"):
    raise RuntimeError("R1V cognitive seed sleeping address is not a bed")
  if actor.get("act_duration") != 360:
    raise RuntimeError("R1V cognitive seed sleeping duration is not 360 minutes")
  if action_end != datetime.datetime(2023, 2, 13, 6, 0):
    raise RuntimeError("R1V cognitive seed sleeping window does not end at 06:00")
  if actor.get("planned_path") not in ([], None):
    raise RuntimeError("R1V cognitive seed sleeping path is not stationary")
  if not (action_start <= start_time < action_end):
    raise RuntimeError("R1V start time is outside the sleeping window")
  return {
    "seed_step": seed_step,
    "initial_step": 0,
    "initial_time": serialized_start,
    "sleep_end_time": action_end.strftime(DATE_FORMAT),
    "environment_source": f"environment/{seed_step}.json",
    "scratch": scratch_state,
  }


def _normalize(value):
  if isinstance(value, datetime.datetime):
    return value.isoformat()
  if isinstance(value, dict):
    return {str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
  if isinstance(value, (list, tuple)):
    return [_normalize(item) for item in value]
  if isinstance(value, set):
    return sorted((_normalize(item) for item in value), key=repr)
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  if hasattr(value, "__dict__"):
    return _normalize(vars(value))
  return {"type": type(value).__name__}


def _digest(value):
  serialized = json.dumps(
    _normalize(value), sort_keys=True, separators=(",", ":"),
    ensure_ascii=False)
  return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def summarize_transitions(initial_action, initial_coordinate, observations):
  actions = [row.get("action_description") for row in observations]
  coordinates = [list(row["coordinate"]) for row in observations]
  action_series = [initial_action] + actions
  coordinate_series = [list(initial_coordinate)] + coordinates
  action_ticks = [
    tick for tick in range(len(observations))
    if action_series[tick + 1] != action_series[tick]]
  movement_ticks = [
    tick for tick in range(len(observations))
    if coordinate_series[tick + 1] != coordinate_series[tick]]
  return {
    "distinct_action_descriptions": len(set(actions)),
    "action_transitions": action_ticks,
    "first_transition_tick": action_ticks[0] if action_ticks else None,
    "initial_coordinate": list(initial_coordinate),
    "final_coordinate": (coordinates[-1] if coordinates
                         else list(initial_coordinate)),
    "distinct_isabella_coordinates": len({tuple(row) for row in coordinates}),
    "coordinate_transitions": movement_ticks,
    "first_movement_tick": movement_ticks[0] if movement_ticks else None,
  }
