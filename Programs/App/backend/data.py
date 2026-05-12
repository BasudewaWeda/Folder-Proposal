"""
MovieLens metadata loaders + poster cache + small helpers.

All dataset IDs are 1-indexed. We keep them 1-indexed in the API surface
and only subtract 1 when slicing NumPy arrays.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


SCRIPT_DIR    = Path(__file__).resolve().parent          # Programs/App/backend
PROGRAMS_DIR  = SCRIPT_DIR.parent.parent                 # Programs
ML_DIR        = PROGRAMS_DIR / "Data" / "ml-100k"
U_ITEM_PATH   = ML_DIR / "u.item"
U_USER_PATH   = ML_DIR / "u.user"
U_DATA_PATH   = ML_DIR / "u.data"
POSTERS_CACHE = SCRIPT_DIR / "posters_cache.json"

N_USERS = 943
N_ITEMS = 1682

# 19 genre columns in u.item, in their canonical MovieLens 100K order.
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
# Genres shown as their own shelves. "unknown" is excluded.
SHELF_GENRES = [g for g in GENRES if g != "unknown"]

_ARTICLES_AT_END = (
    "The", "A", "An",
    "Le", "La", "Les",
    "Il", "Lo", "Gli",
    "El", "Los", "Las",
    "Der", "Die", "Das",
)
_YEAR_SUFFIX_RE = re.compile(r"\s*\((\d{4})\)\s*$")


def _denormalize_article(title: str) -> str:
    """MovieLens stores 'Birdcage, The' — flip back to 'The Birdcage' for display."""
    for article in _ARTICLES_AT_END:
        suffix = f", {article}"
        if title.endswith(suffix):
            return f"{article} {title[: -len(suffix)]}"
    return title


@dataclass
class Catalog:
    items: pd.DataFrame                # index = item_id (1..1682), cols: title, year, genres
    users: pd.DataFrame                # index = user_id (1..943),  cols: age, gender, occupation
    posters: dict                      # str(item_id) -> {poster_url, overview, tmdb_id, ...}
    genre_mask: dict[str, np.ndarray]  # genre -> bool array of length N_ITEMS

    def movie_card(self, item_id: int, rating: Optional[float] = None) -> dict:
        row    = self.items.loc[item_id]
        poster = self.posters.get(str(item_id)) or {}
        return {
            "movie_id"  : int(item_id),
            "title"     : str(row["title"]),
            "year"      : int(row["year"]) if pd.notna(row["year"]) else None,
            "genres"    : list(row["genres"]),
            "poster_url": poster.get("poster_url"),
            "overview"  : poster.get("overview", ""),
            "rating"    : float(rating) if rating is not None else None,
        }


def _load_items() -> pd.DataFrame:
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    df = pd.read_csv(U_ITEM_PATH, sep="|", header=None, names=cols, encoding="latin-1")

    # Pull year from "Title (YYYY)" suffix; fall back to last 4 digits of release_date.
    year_from_title    = df["title"].str.extract(_YEAR_SUFFIX_RE.pattern, expand=False)
    year_from_release  = df["release_date"].str.extract(r"(\d{4})\s*$", expand=False)
    df["year"]         = pd.to_numeric(year_from_title.fillna(year_from_release), errors="coerce")
    df["title"]        = df["title"].str.replace(_YEAR_SUFFIX_RE, "", regex=True).str.strip()
    df["title"]        = df["title"].apply(_denormalize_article)

    # Collapse the 19 boolean genre columns into a list[str] per row (skip 'unknown').
    df["genres"] = df[GENRES].apply(
        lambda r: [g for g, v in zip(GENRES, r) if v == 1 and g != "unknown"],
        axis=1,
    )

    df = df.set_index("item_id").sort_index()
    return df[["title", "year", "genres"]]


def _load_users() -> pd.DataFrame:
    cols = ["user_id", "age", "gender", "occupation", "zip"]
    df = pd.read_csv(U_USER_PATH, sep="|", header=None, names=cols, encoding="latin-1")
    return df.set_index("user_id").sort_index()[["age", "gender", "occupation"]]


def _load_posters() -> dict:
    if not POSTERS_CACHE.exists():
        return {}
    return json.loads(POSTERS_CACHE.read_text(encoding="utf-8"))


def _build_genre_mask(items: pd.DataFrame) -> dict[str, np.ndarray]:
    """Bool mask per genre, aligned with item_id 1..N_ITEMS in order."""
    return {
        g: items["genres"].apply(lambda gs, g=g: g in gs).to_numpy()
        for g in SHELF_GENRES
    }


def load_catalog() -> Catalog:
    items = _load_items()
    return Catalog(
        items      = items,
        users      = _load_users(),
        posters    = _load_posters(),
        genre_mask = _build_genre_mask(items),
    )


def load_ratings() -> pd.DataFrame:
    """All ratings (train + val + test) on the original 1..5 scale."""
    return pd.read_csv(
        U_DATA_PATH,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )


def build_rated_mask(ratings: pd.DataFrame) -> np.ndarray:
    """(N_USERS, N_ITEMS) bool — True wherever the user has any rating in u.data."""
    mask = np.zeros((N_USERS, N_ITEMS), dtype=bool)
    mask[ratings["user_id"].to_numpy() - 1, ratings["item_id"].to_numpy() - 1] = True
    return mask


if __name__ == "__main__":
    cat = load_catalog()
    print(f"items   : {len(cat.items)}")
    print(f"users   : {len(cat.users)}")
    print(f"posters : {len(cat.posters)}")
    print(f"genres  : {list(cat.genre_mask.keys())}")
    print(f"\nsample card (id=71, 'Lion King, The (1994)'):")
    print(f"  {cat.movie_card(71, rating=5.0)}")
    print(f"\nuser 1 demographics: {cat.users.loc[1].to_dict()}")
