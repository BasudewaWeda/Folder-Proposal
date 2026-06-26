"""
Minimal persistence for demo-registered users (cold-start users).

The trained VAE + RSVD ensemble only knows the original 943 MovieLens users.
Users created through `POST /api/register` get IDs starting at 944 and are
stored here as plain JSON so they survive a backend restart. They have no
learned factors, so the API serves them cold-start recommendations
(highest-average-rated movies) instead of model predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Optional

from data import N_USERS

SCRIPT_DIR     = Path(__file__).resolve().parent      # Programs/App/backend
NEW_USERS_PATH = SCRIPT_DIR / "new_users.json"


class NewUserStore:
    """In-memory registry of demo users, persisted to ``new_users.json``."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._users: dict[int, dict] = {}
        if NEW_USERS_PATH.exists():
            raw = json.loads(NEW_USERS_PATH.read_text(encoding="utf-8"))
            self._users = {int(k): v for k, v in raw.items()}

    def __contains__(self, user_id: int) -> bool:
        return user_id in self._users

    def __len__(self) -> int:
        return len(self._users)

    def get(self, user_id: int) -> Optional[dict]:
        return self._users.get(user_id)

    def register(self, age: int, gender: str, occupation: str) -> dict:
        """Create a new user with the next free id (>= 944) and persist it."""
        with self._lock:
            next_id = max([N_USERS, *self._users.keys()]) + 1
            info = {
                "user_id"   : next_id,
                "age"       : int(age),
                "gender"    : gender,
                "occupation": occupation,
            }
            self._users[next_id] = info
            self._save()
            return info

    def _save(self) -> None:
        NEW_USERS_PATH.write_text(
            json.dumps({str(k): v for k, v in self._users.items()}, indent=2),
            encoding="utf-8",
        )
