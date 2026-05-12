"""
FastAPI app: serves login + recommendation endpoints for the local demo.

The heavy work (loading TF + computing the 943x1682 prediction matrix) happens
once at startup. Each request is then a pure NumPy slice.

Run:  uvicorn app:app --reload --port 8000   (from Programs/App/backend/)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data import (
    Catalog,
    SHELF_GENRES,
    build_rated_mask,
    load_catalog,
    load_ratings,
)
from inference import Predictor, build_predictor


_state: dict = {}   # populated in `lifespan`; keys: predictor, catalog, ratings, rated_mask


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[app] Loading catalog and ratings ...")
    catalog    = load_catalog()
    ratings    = load_ratings()
    rated_mask = build_rated_mask(ratings)
    print("[app] Building predictor ...")
    predictor  = build_predictor()
    _state.update(
        predictor  = predictor,
        catalog    = catalog,
        ratings    = ratings,
        rated_mask = rated_mask,
    )
    print(f"[app] Ready — {len(catalog.items)} items, {len(catalog.users)} users, "
          f"{len(catalog.posters)} posters cached.")
    yield
    _state.clear()


app = FastAPI(title="VAE+RSVD Recommender Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins   = ["http://localhost:3000"],
    allow_methods   = ["*"],
    allow_headers   = ["*"],
)


# ----- helpers -----

def _catalog()    -> Catalog:   return _state["catalog"]
def _predictor()  -> Predictor: return _state["predictor"]
def _ratings():                 return _state["ratings"]
def _rated_mask() -> np.ndarray: return _state["rated_mask"]


def _ensure_user(user_id: int) -> None:
    if user_id not in _catalog().users.index:
        raise HTTPException(status_code=404, detail=f"User {user_id} tidak ditemukan")


def _scores_excluding_rated(user_id: int) -> np.ndarray:
    """Predicted ratings for one user, with already-rated items set to -inf."""
    scores = _predictor().predictions_for_user(user_id).copy()
    scores[_rated_mask()[user_id - 1]] = -np.inf
    return scores


# ----- endpoints -----

class LoginRequest(BaseModel):
    username: str
    password: Optional[str] = None   # accepted but ignored — demo only


@app.post("/api/login")
def login(req: LoginRequest):
    try:
        user_id = int(req.username)
    except ValueError:
        raise HTTPException(status_code=401, detail="Username harus berupa angka (1-943)")
    if user_id not in _catalog().users.index:
        raise HTTPException(status_code=401, detail=f"User {user_id} tidak ada di dataset")
    info = _catalog().users.loc[user_id]
    return {
        "user_id"   : user_id,
        "age"       : int(info["age"]),
        "gender"    : str(info["gender"]),
        "occupation": str(info["occupation"]),
    }


@app.get("/api/users/{user_id}/rated")
def user_rated(user_id: int):
    _ensure_user(user_id)
    rows = (
        _ratings()
        .query("user_id == @user_id")
        .sort_values(["rating", "timestamp"], ascending=[False, False])
    )
    return [
        _catalog().movie_card(int(r.item_id), rating=float(r.rating))
        for r in rows.itertuples()
    ]


@app.get("/api/users/{user_id}/recommendations")
def user_recommendations(user_id: int, limit: int = 50):
    _ensure_user(user_id)
    scores = _scores_excluding_rated(user_id)
    top    = np.argsort(scores)[::-1][:limit]
    return [
        _catalog().movie_card(int(idx + 1), rating=float(scores[idx]))
        for idx in top
        if np.isfinite(scores[idx])
    ]


@app.get("/api/users/{user_id}/recommendations/by-genre")
def user_recommendations_by_genre(user_id: int, limit: int = 20):
    _ensure_user(user_id)
    scores = _scores_excluding_rated(user_id)
    out: dict[str, list[dict]] = {}
    for genre in SHELF_GENRES:
        mask          = _catalog().genre_mask[genre]
        masked_scores = np.where(mask, scores, -np.inf)
        top           = np.argsort(masked_scores)[::-1][:limit]
        cards         = [
            _catalog().movie_card(int(idx + 1), rating=float(scores[idx]))
            for idx in top
            if np.isfinite(scores[idx])
        ]
        if cards:
            out[genre] = cards
    return out


@app.get("/api/movies/{movie_id}")
def movie_detail(movie_id: int):
    if movie_id not in _catalog().items.index:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} tidak ditemukan")
    return _catalog().movie_card(movie_id)


@app.get("/api/health")
def health():
    return {
        "ok"         : True,
        "n_items"    : len(_catalog().items),
        "n_users"    : len(_catalog().users),
        "n_posters"  : len(_catalog().posters),
        "alpha_vae"  : _predictor().alpha,
    }
