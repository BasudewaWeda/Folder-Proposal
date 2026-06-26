"""
Minimal persistence for ratings submitted *through the app* (any user).

The trained models are static. To let a user's freshly-given ratings influence
their recommendations, we store them here and fold them into the user's rating
vector at request time (see `inference.Predictor.predict_vae_foldin` +
`app._scores_excluding_rated`). Persisted as plain JSON so they survive a
restart. These are kept separate from the original MovieLens `u.data` ratings.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

SCRIPT_DIR        = Path(__file__).resolve().parent       # Programs/App/backend
USER_RATINGS_PATH = SCRIPT_DIR / "user_ratings.json"


class RatingsStore:
    """In-memory ``{user_id: {item_id: rating}}``, persisted to JSON."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._data: dict[int, dict[int, float]] = {}
        if USER_RATINGS_PATH.exists():
            raw = json.loads(USER_RATINGS_PATH.read_text(encoding="utf-8"))
            self._data = {
                int(uid): {int(iid): float(r) for iid, r in items.items()}
                for uid, items in raw.items()
            }

    def get_user(self, user_id: int) -> dict[int, float]:
        """Return this user's app-submitted ratings (item_id -> rating)."""
        return self._data.get(user_id, {})

    def total(self) -> int:
        return sum(len(v) for v in self._data.values())

    def set(self, user_id: int, item_id: int, rating: float) -> None:
        with self._lock:
            self._data.setdefault(user_id, {})[item_id] = float(rating)
            self._save()

    def _save(self) -> None:
        USER_RATINGS_PATH.write_text(
            json.dumps(
                {str(uid): {str(iid): r for iid, r in items.items()}
                 for uid, items in self._data.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
