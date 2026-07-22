"""In-memory guard against unsafe consecutive tool calls."""

import hashlib
import json

from pydantic import BaseModel


class LoopGuard:
    """Block identical calls and successful effective repeats within one run."""

    def __init__(self) -> None:
        self._last_call_by_run: dict[str, tuple[str, str, bool]] = {}

    def check(self, run_id: str, name: str, args: BaseModel) -> bool:
        """Return whether the call repeats the last non-blocked call for this run."""
        dumped_args = args.model_dump(mode="json")
        full_hash = _hash_call(name, dumped_args)
        effective_args = dict(dumped_args)
        effective_args.pop("rationale", None)
        effective_hash = _hash_call(name, effective_args)
        previous = self._last_call_by_run.get(run_id)

        if previous is not None:
            previous_full, previous_effective, previous_ok = previous
            if previous_full == full_hash:
                return True
            if previous_ok and previous_effective == effective_hash:
                return True

        self._last_call_by_run[run_id] = (full_hash, effective_hash, False)
        return False

    def mark_success(self, run_id: str) -> None:
        """Mark the last non-blocked call for this run as successfully executed."""
        previous = self._last_call_by_run.get(run_id)
        if previous is None:
            return
        full_hash, effective_hash, _previous_ok = previous
        self._last_call_by_run[run_id] = (full_hash, effective_hash, True)


def _hash_call(name: str, args: dict[str, object]) -> str:
    serialized_args = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{name}\0{serialized_args}".encode()).hexdigest()
