"""
Loads the trained VAE + RSVD ensemble and precomputes the full 943x1682
predicted-rating matrix so per-user recommendations are a single array slice
at request time.

Mirrors the prediction pipeline in Notebooks/Testing.ipynb sections 4 & 5.
"""
from __future__ import annotations

import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR        = Path(__file__).resolve().parent       # Programs/App/backend
PROGRAMS_DIR      = SCRIPT_DIR.parent.parent              # Programs
SCRIPTS_DIR       = PROGRAMS_DIR / "Scripts"
DATA_DIR          = PROGRAMS_DIR / "Data"

VAE_PARAMS_PATH   = DATA_DIR / "HyperParameters" / "tuning_progress_vae.json"
VAE_WEIGHTS_PATH  = DATA_DIR / "SavedModels"      / "trained_best_vae_weights.weights.h5"
RSVD_DIR          = DATA_DIR / "SavedModels"
TRAIN_DATA_PATH   = DATA_DIR / "TrainTest"        / "train_data.npy"
EVAL_RESULTS_GLOB = str(DATA_DIR / "EvaluationResults" / "final_testing_results_*.json")

MAX_RATING = 5.0


@dataclass
class Predictor:
    pred_full: np.ndarray       # (943, 1682) float32, denormalized to [1, 5]
    train_data: np.ndarray      # (943, 1682) float32, normalized [0, 1]; > 0 = rated
    alpha: float                # ensemble weight (VAE proportion)

    def is_rated(self, user_id_1based: int) -> np.ndarray:
        return self.train_data[user_id_1based - 1] > 0

    def predictions_for_user(self, user_id_1based: int) -> np.ndarray:
        return self.pred_full[user_id_1based - 1]


def _latest_eval_alpha() -> float:
    """Pick the most recent EvaluationResults JSON and return its tuned alpha."""
    files = sorted(glob.glob(EVAL_RESULTS_GLOB))
    if not files:
        raise FileNotFoundError(f"No evaluation result files matched {EVAL_RESULTS_GLOB}")
    with open(files[-1]) as f:
        return float(json.load(f)["ensemble_weights"]["best_alpha_vae"])


def _load_rsvd_components():
    mu    = np.load(RSVD_DIR / "final_mu.npy")
    b_u   = np.load(RSVD_DIR / "final_b_u.npy")
    b_i   = np.load(RSVD_DIR / "final_b_i.npy")
    U     = np.load(RSVD_DIR / "best_U.npy")
    Sigma = np.load(RSVD_DIR / "best_Sigma.npy")
    V     = np.load(RSVD_DIR / "best_V.npy")
    return mu, b_u, b_i, U, Sigma, V


def _load_vae(train_data: np.ndarray):
    # Defer TF import so this module is cheap when only paths/constants are needed.
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from model import Encoder, Decoder, VAE  # type: ignore

    with open(VAE_PARAMS_PATH) as f:
        params = json.load(f)["best_params"]

    num_items = train_data.shape[1]
    encoder = Encoder(
        hidden_dims  = params["hidden_dims"],
        latent_dim   = params["latent_dim"],
        dropout_rate = params["dropout_rate"],
    )
    decoder = Decoder(hidden_dims=params["hidden_dims"][::-1], output_dim=num_items)
    vae     = VAE(encoder, decoder, beta=params["beta"])
    # Materialize variables before load_weights — otherwise Keras has nothing to load into.
    _ = vae(train_data[:1])
    vae.load_weights(str(VAE_WEIGHTS_PATH))
    return vae


def build_predictor() -> Predictor:
    """Load both models, compute the ensemble matrix, return a Predictor."""
    print("[inference] Loading train_data ...")
    train_data = np.load(TRAIN_DATA_PATH).astype(np.float32)

    print("[inference] Loading RSVD components ...")
    mu, b_u, b_i, U, Sigma, V = _load_rsvd_components()
    pred_rsvd_norm = float(mu) + b_u[:, None] + b_i[None, :] + U @ Sigma @ V.T
    pred_rsvd      = np.clip(pred_rsvd_norm * MAX_RATING, 1.0, MAX_RATING).astype(np.float32)

    print("[inference] Loading VAE and running encoder/decoder over training matrix ...")
    import tensorflow as tf  # local import to keep cold-import latency low
    vae       = _load_vae(train_data)
    train_tf  = tf.constant(train_data, dtype=tf.float32)
    z_mean, _ = vae.encoder.predict(train_tf, verbose=0)
    pred_vae_norm = vae.decoder.predict(z_mean, verbose=0)
    pred_vae      = np.clip(pred_vae_norm * MAX_RATING, 1.0, MAX_RATING).astype(np.float32)

    alpha = _latest_eval_alpha()
    print(f"[inference] Ensemble alpha (VAE proportion) = {alpha:.2f}")

    pred_full = np.clip(alpha * pred_vae + (1.0 - alpha) * pred_rsvd, 1.0, MAX_RATING).astype(np.float32)
    print(f"[inference] pred_full ready: shape={pred_full.shape}, dtype={pred_full.dtype}")

    return Predictor(pred_full=pred_full, train_data=train_data, alpha=alpha)


if __name__ == "__main__":
    # Smoke test — print top 5 unrated movies for user 1.
    p = build_predictor()
    rated  = p.is_rated(1)
    scores = p.predictions_for_user(1).copy()
    scores[rated] = -np.inf
    top = np.argsort(scores)[::-1][:5]
    print("\n[smoke] Top 5 unrated movies for user 1 (1-indexed item_id, predicted rating):")
    for idx in top:
        print(f"  item_id = {idx + 1:4d}   pred = {scores[idx]:.3f}")
