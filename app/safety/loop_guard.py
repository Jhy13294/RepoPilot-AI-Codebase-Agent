"""In-memory guard against identical consecutive tool calls."""

import hashlib
import json

from pydantic import BaseModel


class LoopGuard:
    """Block identical consecutive validated calls within the same run."""

    def __init__(self) -> None:
        self._last_call_hash_by_run: dict[str, str] = {}

    def check(self, run_id: str, name: str, args: BaseModel) -> bool:
        """Return whether the call repeats the last non-blocked call for this run."""
        serialized_args = json.dumps(args.model_dump(mode="json"), sort_keys=True)
        call_hash = hashlib.sha256(f"{name}\0{serialized_args}".encode()).hexdigest()

        if self._last_call_hash_by_run.get(run_id) == call_hash:
            return True

        self._last_call_hash_by_run[run_id] = call_hash
        return False
