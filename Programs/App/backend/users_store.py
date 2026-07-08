"""
Minimal persistence for demo-registered users (cold-start users).

The trained VAE + RSVD ensemble only knows the original 943 MovieLens users.
Users created through `POST /api/register` get IDs starting at 944 and are
stored here as plain JSON so they survive a backend restart. They have no
learned factors, so the API serves them cold-start recommendations
(highest-average-rated movies) instead of model predictions.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from threading import Lock
from typing import Optional

from data import N_USERS

SCRIPT_DIR     = Path(__file__).resolve().parent      # Programs/App/backend
NEW_USERS_PATH = SCRIPT_DIR / "new_users.json"

# Password shared by every original MovieLens user (id 1..943). Those users
# come from the dataset and never set a password, so the demo accepts this
# fixed value for them. Documented in the manual book.
DEFAULT_PASSWORD = "movielens"

_PBKDF2_ITERATIONS = 100_000


def hash_password(password: str) -> str:
    """Return a salted PBKDF2-SHA256 hash string: ``pbkdf2_sha256$iters$salt$hash``."""
    salt   = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: Optional[str]) -> bool:
    """Check ``password`` against a hash produced by :func:`hash_password`."""
    if not stored:
        return False
    try:
        algo, iters, salt, expected = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iters)
        )
    except (ValueError, AttributeError):
        return False
    return secrets.compare_digest(digest.hex(), expected)


def _public(info: dict) -> dict:
    """Strip the password hash before exposing a user record over the API."""
    return {k: v for k, v in info.items() if k != "password_hash"}


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
        """Return public user info (never the password hash)."""
        user = self._users.get(user_id)
        return _public(user) if user is not None else None

    def register(
        self, username: str, age: int, gender: str, occupation: str, password: str
    ) -> dict:
        """Create a new user with the next free id (>= 944) and persist it.

        The password is stored only as a salted hash under ``password_hash``.
        """
        with self._lock:
            next_id = max([N_USERS, *self._users.keys()]) + 1
            info = {
                "user_id"      : next_id,
                "username"     : username,
                "age"          : int(age),
                "gender"       : gender,
                "occupation"   : occupation,
                "password_hash": hash_password(password),
            }
            self._users[next_id] = info
            self._save()
            return _public(info)

    def by_username(self, username: str) -> Optional[dict]:
        """Return the public record whose username matches (case-insensitive)."""
        key = username.strip().lower()
        for user in self._users.values():
            if str(user.get("username", "")).lower() == key:
                return _public(user)
        return None

    def username_taken(self, username: str) -> bool:
        return self.by_username(username) is not None

    def check_password(self, user_id: int, password: str) -> bool:
        """Verify ``password`` for a registered user.

        Entries created before passwords existed (no ``password_hash``) fall
        back to the shared :data:`DEFAULT_PASSWORD`.
        """
        user = self._users.get(user_id)
        if user is None:
            return False
        stored = user.get("password_hash")
        if not stored:
            return password == DEFAULT_PASSWORD
        return verify_password(password, stored)

    def _save(self) -> None:
        NEW_USERS_PATH.write_text(
            json.dumps({str(k): v for k, v in self._users.items()}, indent=2),
            encoding="utf-8",
        )
