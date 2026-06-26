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
    N_ITEMS,
    SHELF_GENRES,
    build_avg_rating_scores,
    build_rated_mask,
    load_catalog,
    load_ratings,
)
from inference import MAX_RATING, Predictor, build_predictor
from ratings_store import RatingsStore
from users_store import NewUserStore


# populated in `lifespan`; keys: predictor, catalog, ratings, rated_mask,
# avg_scores (cold-start), new_users (registered demo users),
# user_ratings (in-app ratings, all users)
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[app] Loading catalog and ratings ...")
    catalog    = load_catalog()
    ratings    = load_ratings()
    rated_mask = build_rated_mask(ratings)
    avg_scores = build_avg_rating_scores(ratings)
    new_users  = NewUserStore()
    user_ratings = RatingsStore()
    print("[app] Building predictor ...")
    predictor  = build_predictor()
    _state.update(
        predictor    = predictor,
        catalog      = catalog,
        ratings      = ratings,
        rated_mask   = rated_mask,
        avg_scores   = avg_scores,
        new_users    = new_users,
        user_ratings = user_ratings,
    )
    print(f"[app] Ready — {len(catalog.items)} items, {len(catalog.users)} users "
          f"(+{len(new_users)} registered), {len(catalog.posters)} posters cached.")
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

def _catalog()      -> Catalog:      return _state["catalog"]
def _predictor()    -> Predictor:    return _state["predictor"]
def _ratings():                      return _state["ratings"]
def _rated_mask()   -> np.ndarray:   return _state["rated_mask"]
def _avg_scores()   -> np.ndarray:   return _state["avg_scores"]
def _new_users()    -> NewUserStore: return _state["new_users"]
def _user_ratings() -> RatingsStore: return _state["user_ratings"]


def _is_new_user(user_id: int) -> bool:
    """A user registered via /api/register — no learned factors, cold-start only."""
    return user_id in _new_users()


def _ensure_user(user_id: int) -> None:
    if user_id not in _catalog().users.index and not _is_new_user(user_id):
        raise HTTPException(status_code=404, detail=f"User {user_id} tidak ditemukan")


def _user_rating_vector_norm(user_id: int) -> np.ndarray:
    """The user's full rating vector (normalized [0,1]) for VAE fold-in.

    Base = their training ratings (zeros for brand-new users), with their
    app-submitted ratings overlaid on top.
    """
    if _is_new_user(user_id):
        vec = np.zeros(N_ITEMS, dtype=np.float32)
    else:
        vec = _predictor().train_data[user_id - 1].copy()
    for item_id, rating in _user_ratings().get_user(user_id).items():
        vec[item_id - 1] = rating / MAX_RATING
    return vec


def _excluded_mask(user_id: int) -> np.ndarray:
    """Items to drop from recommendations: prior ratings + app-submitted ones."""
    if _is_new_user(user_id):
        mask = np.zeros(N_ITEMS, dtype=bool)
    else:
        mask = _rated_mask()[user_id - 1].copy()
    for item_id in _user_ratings().get_user(user_id):
        mask[item_id - 1] = True
    return mask


def _scores_excluding_rated(user_id: int) -> np.ndarray:
    """Recommendation scores for one user, with already-rated items set to -inf.

    Unified hybrid fold-in: once a user has rated movies in-app, *both* halves
    respond to those ratings — `α·VAE_foldin + (1−α)·RSVD_foldin` — for new and
    original users alike (RSVD fold-in estimates the user's factors on the fixed
    item factors, no retraining). With no in-app ratings yet: registered users
    get the cold-start average-rating shelf, original users get the precomputed
    hybrid matrix.
    """
    app_ratings = _user_ratings().get_user(user_id)
    if app_ratings:
        vec       = _user_rating_vector_norm(user_id)
        vae_pred  = _predictor().predict_vae_foldin(vec)
        rsvd_pred = _predictor().predict_rsvd_foldin(vec)
        a = _predictor().alpha
        scores = np.clip(
            a * vae_pred + (1.0 - a) * rsvd_pred, 1.0, MAX_RATING
        ).astype(np.float32)
    elif _is_new_user(user_id):
        scores = _avg_scores().copy()
    else:
        scores = _predictor().predictions_for_user(user_id).copy()

    scores = np.array(scores, dtype=np.float32, copy=True)
    scores[_excluded_mask(user_id)] = -np.inf
    return scores


# ----- endpoints -----

class LoginRequest(BaseModel):
    username: str
    password: Optional[str] = None   # accepted but ignored — demo only


class RegisterRequest(BaseModel):
    age: int = 25
    gender: str = "M"
    occupation: str = "other"


@app.post("/api/login")
def login(req: LoginRequest):
    try:
        user_id = int(req.username)
    except ValueError:
        raise HTTPException(status_code=401, detail="Username harus berupa angka")
    if user_id in _catalog().users.index:
        info = _catalog().users.loc[user_id]
        return {
            "user_id"   : user_id,
            "age"       : int(info["age"]),
            "gender"    : str(info["gender"]),
            "occupation": str(info["occupation"]),
            "is_new"    : False,
        }
    new = _new_users().get(user_id)
    if new is not None:
        return {**new, "is_new": True}
    raise HTTPException(status_code=401, detail=f"User {user_id} tidak ada di dataset")


@app.post("/api/register")
def register(req: RegisterRequest):
    """Create a new demo user (id >= 944) served cold-start recommendations."""
    info = _new_users().register(
        age=req.age, gender=req.gender, occupation=req.occupation
    )
    return {**info, "is_new": True}


@app.get("/api/users/{user_id}/rated")
def user_rated(user_id: int):
    _ensure_user(user_id)
    # Original MovieLens ratings (none for registered users) ...
    ratings_map: dict[int, float] = {}
    if not _is_new_user(user_id):
        rows = _ratings().query("user_id == @user_id")
        for r in rows.itertuples():
            ratings_map[int(r.item_id)] = float(r.rating)
    # ... then overlay app-submitted ratings (they win on conflict).
    for item_id, rating in _user_ratings().get_user(user_id).items():
        ratings_map[item_id] = float(rating)
    cards = [_catalog().movie_card(i, rating=rt) for i, rt in ratings_map.items()]
    cards.sort(key=lambda c: c["rating"], reverse=True)
    return cards


class RatingRequest(BaseModel):
    item_id: int
    rating: float   # 1..5


@app.post("/api/users/{user_id}/ratings")
def add_rating(user_id: int, req: RatingRequest):
    """Record a rating the user gave in-app; it feeds the VAE fold-in."""
    _ensure_user(user_id)
    if req.item_id not in _catalog().items.index:
        raise HTTPException(status_code=404, detail=f"Movie {req.item_id} tidak ditemukan")
    if not (1.0 <= req.rating <= MAX_RATING):
        raise HTTPException(status_code=400, detail="Rating harus antara 1 dan 5")
    _user_ratings().set(user_id, req.item_id, req.rating)
    return _catalog().movie_card(req.item_id, rating=req.rating)


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
        "ok"            : True,
        "n_items"       : len(_catalog().items),
        "n_users"       : len(_catalog().users),
        "n_new_users"   : len(_new_users()),
        "n_app_ratings" : _user_ratings().total(),
        "n_posters"     : len(_catalog().posters),
        "alpha_vae"     : _predictor().alpha,
    }
