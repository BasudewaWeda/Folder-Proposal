# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hybrid recommender system thesis project on MovieLens 100K (`Data/ml-100k/u.data`). Combines a **Variational Autoencoder (VAE, Keras/TensorFlow)** with a **Regularized SVD (RSVD, hand-rolled NumPy)** via a linear ensemble. All notebooks, code comments, prints, and existing docs are in **Bahasa Indonesia** — match that language when modifying notebooks or adding new prints. Code identifiers stay in English.

## Environment

- Windows + bash shell. A venv lives at `venv/`. Activate with `source venv/Scripts/activate` before running Jupyter.
- Install deps: `pip install -r requirements.txt` (pinned: TF 2.20, NumPy 2.2, Keras 3.12, Optuna 4.7, scikit-learn, scipy, matplotlib).
- Notebooks assume CWD is `Notebooks/` and reference siblings via `../Data/...` and `../Scripts/`. They prepend `../Scripts` to `sys.path` so `from model import ...` works only when launched from `Notebooks/`.

## Pipeline (run notebooks in order)

1. `Notebooks/DataPreprocessing.ipynb` — pivot `u.data` into a 943×1682 user×item matrix, normalize by dividing by `MAX_RATING=5.0` (preserving the 0 ⇔ "unrated" sentinel), then hold-out split 70/10/20 → `Data/TrainTest/{train,val,test}_data.npy` plus the unsplit `normalized_matrix.npy` (used by CV).
2. `Notebooks/HyperParameterTuning.ipynb` (grid) **or** `HyperParameterTuningOptuna.ipynb` — writes `Data/HyperParameters/tuning_progress_{vae,rsvd}.json` (resumable; tracks `completed_indices`), `best_vae_weights.weights.h5`, and `best_{U,V,Sigma,mu,b_u,b_i}.npy`.
3. `Notebooks/Training.ipynb` — retrains both models with the best params on full `train_data`, saves `Data/SavedModels/trained_best_vae_weights.weights.h5` and `final_{mu,b_u,b_i}.npy` / `best_{U,V,Sigma}.npy`, plus loss history under `Data/LossHistory/`.
4. `Notebooks/Testing.ipynb` — loads saved weights, finds ensemble α on **validation set** (grid 0..1 step 0.01), applies that α to the test set exactly once, writes `Data/EvaluationResults/final_testing_results_<timestamp>.json`.
5. `Notebooks/CrossValidation.ipynb` — 10-fold CV with inner val-split per fold, paired t-tests in section 12. Each run dumps into `Data/CrossValidation/cv_<timestamp>/`.

There is no test/build/lint tooling — `Scripts/preprocessing.py` is empty, and the suite runs entirely through notebooks.

## Architecture (`Scripts/model.py`)

`model.py` is the single source of truth for both models; every notebook imports from it. Key classes:

- **`VAE(encoder, decoder, beta)`** — subclasses `tf.keras.Model` with a custom `train_step`. Two critical behaviors:
  - **Masked reconstruction loss**: only positions where `data > 0` contribute. `mse_loss = sum(masked_se) / sum(mask)`, then scaled by `num_items` so the magnitude is comparable to a per-row sum. Do not "fix" this by averaging differently without understanding the trade-off.
  - **`self.beta` is a `tf.Variable`** (not a Python float) so `KLAnnealingCallback` can mutate it each epoch without rebuilding the model. If you reconstruct the VAE for inference, you must still call it once on a dummy batch (`_ = vae(train_data[:1])`) before `load_weights` — this is what every notebook does.
- **`KLAnnealingCallback(beta_target, annealing_epochs)`** — linearly ramps `model.beta` from 0 → `beta_target` over the first `annealing_epochs` of training, then holds. Always place it **before** `EarlyStopping` in the `callbacks=[...]` list so β is updated before loss is read.
- **`RSVD`** — pure NumPy. Uses SGD over `(U · Σ · Vᵀ) + μ + b_u + b_i`. **Σ is diagonal and not L2-regularized** (regularizing it caused weight collapse in earlier experiments — leave that decision alone unless explicitly asked). `fit()` does its own internal val_ratio split for early-stopping on validation MSE; loss history stored is `train_mse + λ·(‖U‖² + ‖V‖²)`.

## Ensemble convention

Prediction = `α · VAE_pred + (1−α) · RSVD_pred`, both denormalized by `*MAX_RATING` and clipped to `[1.0, 5.0]`. α is **always** chosen on the validation set (`val_data.npy` for the single hold-out flow, or `val_inner` per fold in CV) — never on test. With the current saved hyperparameters α ≈ 0.24–0.25 (RSVD dominates).

## Data conventions

- The matrix uses **0 as "missing rating" sentinel**, not as a true rating value. Anywhere you compute a mean, sum, or loss over ratings, mask with `> 0` first.
- All saved `.npy` rating matrices are float32 normalized to `[0, 1]`. Denormalize by `*5.0` before reporting metrics.
- File paths in notebooks are relative to `Notebooks/` — when adding new notebooks or scripts elsewhere, recompute paths accordingly.

## Working with the notebooks

- VAE training in tuning loops uses `tf.constant(train_data)` (not raw NumPy) to avoid TensorFlow re-allocating per iteration (this addresses the question in `Questions To Ponder.txt` about why train data is wrapped in tensors).
- Hyperparameter tuning is resumable — both grid search and Optuna persist progress to JSON / SQLite (`Data/HyperParameters/optuna_vae_study.db`). When iterating, delete or rename the progress file to force a fresh run.
- After each VAE-tuning trial, the loop calls `tf.keras.backend.clear_session()` + `gc.collect()`. Preserve that cleanup pattern when adding new trials — without it RAM grows unboundedly.
- Current branch is `kl-annealing-vae`; the KL-annealing wiring is the active work-in-progress, so when touching VAE training make sure the callback ordering and `beta=0.0` init are kept.

## Demo web app (`Programs/App/`)

A localhost-only Netflix-style demo that consumes the trained artifacts. Two processes:

- **Backend** (`App/backend/`, FastAPI + uvicorn) — at startup loads the VAE weights, RSVD arrays, full `u.data` ratings, and `posters_cache.json`; precomputes the 943×1682 hybrid prediction matrix once and serves slices per request. α is read from the **latest** `Data/EvaluationResults/final_testing_results_*.json`. Run: `uvicorn app:app --port 8000` from `App/backend/` with the venv (`Programs/venv/`) activated. Endpoints under `/api/`: `login`, `users/{id}/rated`, `users/{id}/recommendations`, `users/{id}/recommendations/by-genre`, `movies/{id}`, `health`. Error messages are in Bahasa Indonesia.
- **Frontend** (`App/frontend/`, Next.js 16.2.6 + React 19 + Tailwind v4) — login page + browse view with horizontal shelves and a click-to-open modal. Run: `npm run dev` from `App/frontend/`. **Heads-up:** Next.js 16 has breaking changes — `frontend/AGENTS.md` requires consulting `frontend/node_modules/next/dist/docs/` before writing or editing frontend code. The user_id is kept in `localStorage`; there is no real auth.

Files of interest in `App/backend/`:
- `inference.py` — rebuilds the VAE via `Scripts/model.py`, **must call `vae(train_data[:1])` before `load_weights`** to materialize variables (same rule as the notebooks).
- `data.py` — denormalizes article-trailing MovieLens titles (`"Lion King, The"` → `"The Lion King"`), precomputes per-genre boolean masks on `(1682,)` so the genre-shelf endpoint stays O(N).
- `fetch_posters.py` — one-shot, resumable TMDB fetcher. The cache (`posters_cache.json`) is loaded once at backend startup, so restart uvicorn after refreshing it. API key lives in `App/backend/.env` (gitignored).

Reference docs: `App/PLAN.md` (original plan), `App/RUNNING.md` (install + run instructions), `App/FILES.md` (per-file purpose for backend and frontend).
