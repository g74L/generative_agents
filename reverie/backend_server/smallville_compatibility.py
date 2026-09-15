"""Pure representation conversions for the legacy Smallville boundary."""
from __future__ import annotations

from typing import Any


def copy_environment_actor_with_movement(
    environment_record: Any,
    coordinate: list[Any],
) -> dict[Any, Any]:
  """Return a shallow actor-record copy with its movement coordinate applied."""
  converted = dict(environment_record)
  converted["x"], converted["y"] = coordinate
  return converted
