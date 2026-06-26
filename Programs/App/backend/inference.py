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

# Ridge strength for RSVD fold-in. Shrinks a folded-in user's latent vector
# toward 0 — large enough to stay stable when a user has only a few ratings,
# small enough to recover taste once they've rated more. Tuned (sweep over a
# sample of users) so folding a known user's training ratings reproduces their
# trained RSVD prediction: at λ=4, corr ≈ 0.97, MAE ≈ 0.09 stars vs. trained.
RSVD_FOLDIN_LAMBDA = 4.0


@dataclass
class Predictor:
    pred_full: np.ndarray       # (943, 1682) float32, denormalized to [1, 5]
    pred_rsvd: np.ndarray       # (943, 1682) float32, RSVD-only, denormalized to [1, 5]
    train_data: np.ndarray      # (943, 1682) float32, normalized [0, 1]; > 0 = rated
    alpha: float                # ensemble weight (VAE proportion)
    vae: object = None          # kept alive for live fold-in (re-encode new ratings)
    # RSVD pieces needed to fold a user in (estimate U[u], b_u[u] on fixed items):
    rsvd_mu: float = 0.0                       # global mean (normalized space)
    rsvd_bi: np.ndarray = None                 # (n_items,) item biases (normalized)
    rsvd_item_factors: np.ndarray = None       # (n_items, k) = V · diag(Σ)
    rsvd_foldin_lambda: float = RSVD_FOLDIN_LAMBDA

    def is_rated(self, user_id_1based: int) -> np.ndarray:
        return self.train_data[user_id_1based - 1] > 0

    def predictions_for_user(self, user_id_1based: int) -> np.ndarray:
        return self.pred_full[user_id_1based - 1]

    def predict_vae_foldin(self, rating_vec_norm: np.ndarray) -> np.ndarray:
        """VAE prediction for an arbitrary (normalized) rating vector.

        Runs one deterministic encoder→decoder pass (uses z_mean, dropout off)
        so a user's freshly-given ratings flow straight into fresh predictions.
        Returns a (num_items,) array denormalized and clipped to [1, 5].
        """
        import tensorflow as tf  # local import — keep module cold-import cheap
        x = tf.constant(rating_vec_norm[None, :], dtype=tf.float32)
        z_mean, _ = self.vae.encoder(x, training=False)
        pred_norm = self.vae.decoder(z_mean, training=False)
        pred = np.asarray(pred_norm)[0] * MAX_RATING
        return np.clip(pred, 1.0, MAX_RATING).astype(np.float32)

    def predict_rsvd_foldin(self, rating_vec_norm: np.ndarray) -> np.ndarray:
        """RSVD prediction for an arbitrary user via fold-in (no retraining).

        Holds the trained item factors (μ, b_i, Σ, V) fixed and solves a ridge
        regression for just this user's bias b_u and latent vector U[u] from the
        items they've rated, then scores all items. Returns a (n_items,) array
        denormalized and clipped to [1, 5].
        """
        rated = np.where(rating_vec_norm > 0)[0]
        if rated.size == 0:                       # nothing to fit -> bias baseline
            base = self.rsvd_mu + self.rsvd_bi
            return np.clip(base * MAX_RATING, 1.0, MAX_RATING).astype(np.float32)

        X = self.rsvd_item_factors[rated]                         # (r, k)
        y = rating_vec_norm[rated] - self.rsvd_mu - self.rsvd_bi[rated]  # (r,)
        # Prepend a constant column so the intercept recovers b_u[u].
        Xa = np.concatenate([np.ones((rated.size, 1), dtype=X.dtype), X], axis=1)
        reg = np.eye(Xa.shape[1], dtype=X.dtype)
        reg[0, 0] = 0.0                           # don't regularize the intercept
        w = np.linalg.solve(Xa.T @ Xa + self.rsvd_foldin_lambda * reg, Xa.T @ y)
        b_u, u_vec = w[0], w[1:]
        pred_norm = self.rsvd_mu + b_u + self.rsvd_bi + self.rsvd_item_factors @ u_vec
        return np.clip(pred_norm * MAX_RATING, 1.0, MAX_RATING).astype(np.float32)


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
    # Effective item factors for fold-in: row i = Σ · V[i] (Σ is diagonal).
    item_factors   = (V * np.diag(Sigma)[None, :]).astype(np.float32)

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

    return Predictor(
        pred_full         = pred_full,
        pred_rsvd         = pred_rsvd,
        train_data        = train_data,
        alpha             = alpha,
        vae               = vae,
        rsvd_mu           = float(mu),
        rsvd_bi           = b_i.astype(np.float32),
        rsvd_item_factors = item_factors,
    )


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
