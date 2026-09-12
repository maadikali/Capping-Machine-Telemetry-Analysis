"""
Maps the machine's raw closure-status codes to four business categories
(see config/status_codes.json for the human-readable reference table).

  success  status 0 - a valid closure.
  no_load  status 2 - head cycled with no bottle present, not a fault.
  reject   reject_signal == YES - a genuine quality failure (torque,
           turns, timing, tracking error, etc).
  fault    reject_signal == NO and code not in {0, 2} - an abnormal
           machine condition that isn't a quality reject. Kept separate
           from "reject" so it doesn't distort the reject rate.
  unknown  any code not in STATUS_TABLE (logged once, never crashes).
"""
from __future__ import annotations

import logging
from typing import Dict, List, TypedDict

logger = logging.getLogger(__name__)

CATEGORY_SUCCESS = "success"
CATEGORY_NO_LOAD = "no_load"
CATEGORY_REJECT = "reject"
CATEGORY_FAULT = "fault"
CATEGORY_UNKNOWN = "unknown"


class _StatusEntry(TypedDict):
    reject_signal: bool
    description: str


# Kept in sync with config/status_codes.json - see module docstring.
STATUS_TABLE: Dict[int, _StatusEntry] = {
    0: {"reject_signal": False, "description": "Closure OK"},
    2: {"reject_signal": False, "description": "No Load"},
    3: {"reject_signal": True, "description": "Failing to reach the first torque threshold (SlowTorque)"},
    4: {"reject_signal": False, "description": "No Closure"},
    5: {"reject_signal": True, "description": "Failing to reach the final torque (ClosureTorque)"},
    8: {"reject_signal": False, "description": "No InTorque"},
    9: {"reject_signal": True, "description": "Closure Head raises before the TimeInTorque time was elapsed"},
    16: {"reject_signal": False, "description": "No CapTurns"},
    17: {"reject_signal": True, "description": "The cap is closed but with less degrees than CapTurns"},
    32: {"reject_signal": False, "description": "Following Error"},
    33: {"reject_signal": True, "description": "Tracking error between the real position and the controlled position of the head"},
    64: {"reject_signal": False, "description": "Bad Closure"},
    65: {"reject_signal": True, "description": "ClosureTorque reached but cap is still rotating when head raises"},
}

_warned_unknown_codes: set = set()


def classify(code) -> str:
    """Returns one of: success, no_load, reject, fault, unknown."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return CATEGORY_UNKNOWN

    if code == 0:
        return CATEGORY_SUCCESS
    if code == 2:
        return CATEGORY_NO_LOAD

    entry = STATUS_TABLE.get(code)
    if entry is None:
        if code not in _warned_unknown_codes:
            logger.warning("Unrecognized status code %s - treating as 'unknown'", code)
            _warned_unknown_codes.add(code)
        return CATEGORY_UNKNOWN

    return CATEGORY_REJECT if entry["reject_signal"] else CATEGORY_FAULT


def describe(code) -> str:
    """Human-readable description of a status code, for reports and
    fault_code_breakdown. Falls back gracefully for undocumented codes."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return f"Unrecognized status value ({code!r})"

    entry = STATUS_TABLE.get(code)
    if entry is None:
        return f"Unrecognized status code ({code})"
    return entry["description"]


def is_reject_signal(code) -> bool:
    """Raw reject_signal for a code (False for unknown codes)."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return False
    entry = STATUS_TABLE.get(code)
    return bool(entry and entry["reject_signal"])


def full_table() -> List[dict]:
    """The complete status-code reference, for documentation or a
    quick lookup of what every code means."""
    return [
        {
            "code": code,
            "category": classify(code),
            "reject_signal": entry["reject_signal"],
            "description": entry["description"],
        }
        for code, entry in sorted(STATUS_TABLE.items())
    ]
