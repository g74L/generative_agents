"""Privacy-safe normalization helpers for deterministic golden call traces."""
import hashlib
from typing import Iterable, List, Optional


def _hash_text(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _essential_arguments(fake_call):
  arguments = fake_call.arguments
  if fake_call.operation == "CHAT":
    messages = arguments["messages"]
    return {
      "message_count": len(messages),
      "message_roles": [message["role"] for message in messages],
    }
  if fake_call.operation == "COMPLETION":
    return {
      key: arguments[key]
      for key in (
        "temperature", "max_tokens", "top_p", "frequency_penalty",
        "presence_penalty", "stream", "stop")
    }
  if fake_call.operation == "EMBEDDING":
    return {
      "input_count": len(arguments["input"]),
      "normalized_input_hashes": [
        _hash_text(value) for value in arguments["input"]],
    }
  raise AssertionError(f"Unsupported operation {fake_call.operation}")


def build_golden_trace(events: Iterable, fake_calls: Optional[Iterable] = None) -> List[dict]:
  """Group physical events by logical ID without retaining raw input."""
  event_list = list(events)
  call_list = list(fake_calls) if fake_calls is not None else [None] * len(event_list)
  if len(event_list) != len(call_list):
    raise AssertionError("Telemetry and fake call counts differ")

  trace = []
  logical_positions = {}
  for event, fake_call in zip(event_list, call_list):
    if event.logical_call_id not in logical_positions:
      logical_positions[event.logical_call_id] = len(trace)
      item = {
        "logical_index": len(trace) + 1,
        "operation": event.operation,
        "physical_attempts": 0,
        "model": event.model_or_engine,
        "outcomes": [],
        "attempt_numbers": [],
        "input_fingerprints": [],
      }
      if fake_call is not None:
        item["essential_arguments"] = _essential_arguments(fake_call)
      trace.append(item)

    item = trace[logical_positions[event.logical_call_id]]
    if item["operation"] != event.operation:
      raise AssertionError("A logical call crossed operation types")
    item["physical_attempts"] += 1
    item["outcomes"].append(event.outcome)
    item["attempt_numbers"].append(event.physical_attempt)
    item["input_fingerprints"].append(event.input_fingerprint)

  return trace
