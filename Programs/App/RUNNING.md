# Running the MovieMatch Demo

Localhost-only demo of the hybrid VAE + RSVD recommender on MovieLens 100K.
Two processes: a **FastAPI backend** that serves the model, and a **Next.js frontend** that consumes it.

## Prerequisites

- Python 3.11 with the project venv at `Folder-Proposal/venv/` (already has TF 2.20, NumPy, etc.).
- Node.js 20+ and npm.
- TMDB API key already saved in `backend/.env` (gitignored).
- Trained model artifacts must already exist:
  - `Data/SavedModels/trained_best_vae_weights.weights.h5`
  - `Data/SavedModels/{final_mu,final_b_u,final_b_i,best_U,best_Sigma,best_V}.npy`
  - `Data/EvaluationResults/final_testing_results_*.json` (the latest is used for α)
  - `Data/TrainTest/train_data.npy`
  - `Data/HyperParameters/tuning_progress_vae.json`

If any of these are missing, re-run the notebooks in `Notebooks/` in order (see `Programs/CLAUDE.md`).

## One-time setup

### 1. Backend deps (inside the project venv)

```bash
cd Folder-Proposal
source venv/Scripts/activate
pip install "fastapi==0.115.5" "uvicorn[standard]==0.32.1"
```

### 2. Build the poster cache (slow, ~7–10 minutes for 1682 movies)

```bash
cd Programs/App/backend
python fetch_posters.py
```

The script reads `Data/ml-100k/u.item`, queries TMDB for each title, and writes
`backend/posters_cache.json`. It's resumable — re-run it any time to fill in
movies that previously failed. Cache hit rate is typically ~98%.

### 3. Frontend deps

```bash
cd Programs/App/frontend
npm install
```

## Running the demo

Open two terminals.

**Terminal 1 — backend** (from `Programs/App/backend/`):

```bash
source ../../../venv/Scripts/activate
uvicorn app:app --reload --port 8000
```

Wait for `[app] Ready — 1682 items, 943 users, N posters cached.`
Swagger docs live at http://localhost:8000/docs.

**Terminal 2 — frontend** (from `Programs/App/frontend/`):

```bash
npm run dev
```

Wait for `Ready in …ms` at http://localhost:3000.

## Using the app

1. Open http://localhost:3000 — you'll be redirected to `/login`.
2. Enter a **User ID between 1 and 943** as the username. Password is ignored.
   Or click **"Daftar di sini"** to register a new user (id ≥ 944). New users have
   no rating history, so they're shown the **highest-average-rated movies** as a
   cold start. Registered users are saved in `backend/new_users.json`.
3. The browse page shows:
   - **Top Recommended for You** — the 50 highest-predicted unrated movies
   - **Already Rated** — every movie this user actually rated, sorted by rating desc
   - **Best in <Genre>** — one shelf per of the 18 displayed genres (excluding "unknown"), top-20 unrated within each

Hover over a shelf to reveal left/right scroll arrows. Click **Logout** to clear
localStorage and return to `/login`.

**Rating movies:** click any movie to open its modal and pick a 1–5 star rating.
The rating is saved (`backend/user_ratings.json`) and folded into your
recommendations live via a **unified hybrid fold-in** — `α·VAE_foldin +
(1−α)·RSVD_foldin` for every user. Both halves re-estimate from your ratings (the
RSVD half solves a ridge regression for your latent vector on the fixed item
factors — no retraining), so recommendations respond for new *and* existing
users. Rated movies move to the **Already Rated** shelf and drop out of
recommendations. (A full retrain is only needed to add brand-new *movies* to the
catalog, not to react to a user's ratings.)

## Tips

- If you grow the poster cache while the backend is already running, restart the
  backend to pick up new entries (cache is loaded once at startup).
- The backend allows CORS only from `http://localhost:3000` — change it in
  `app.py` if you serve the frontend elsewhere.
- α (VAE weight in the ensemble) is read from the **latest**
  `Data/EvaluationResults/final_testing_results_*.json`. Hit
  `GET /api/health` to confirm which value is in use.
- Logged-in user is stored client-side in `localStorage` (`user_id`,
  `user_info`). Clearing site data forces re-login.

## Common failure modes

| Symptom | Likely cause |
|---|---|
| Backend fails on `load_weights` | Mismatch between the saved VAE architecture and `tuning_progress_vae.json`. Re-run `Notebooks/Training.ipynb`. |
| `Login gagal: User N tidak ada di dataset` | Username out of `[1, 943]` or non-numeric. |
| Posters all show title placeholders | `posters_cache.json` is empty or missing — run `fetch_posters.py`. |
| `next/image` error about disallowed host | `next.config.ts` doesn't list `image.tmdb.org` in `remotePatterns`. |
| CORS error in browser console | Frontend is not on `http://localhost:3000`, or backend isn't running. |
