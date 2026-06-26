# File-by-file Guide

What every file under `Programs/App/` does, organized by directory.

## Top-level

| File | Purpose |
|---|---|
| `PLAN.md` | Original implementation plan — tech stack, endpoint table, decisions made before coding. Historical reference; the code is the source of truth now. |
| `RUNNING.md` | How to install, build the poster cache, and start the two dev servers. |
| `FILES.md` | This file — what each source file does. |

## `backend/` — FastAPI service

The backend loads the trained models once at startup and serves a few thin JSON
endpoints. Each request is a NumPy slice over the precomputed 943×1682 prediction
matrix; nothing is computed lazily per request.

| File | Purpose |
|---|---|
| `.env` | Holds `TMDB_API_KEY`. Gitignored. Read by `fetch_posters.py` via a tiny home-grown loader (no `python-dotenv` dependency). |
| `app.py` | FastAPI app. Defines the `lifespan` handler that calls `load_catalog()`, `load_ratings()`, `build_rated_mask()`, `build_avg_rating_scores()`, `build_predictor()` and instantiates `NewUserStore()` + `RatingsStore()` at startup, stashing them in `_state`. Defines all routes: `POST /api/login`, `POST /api/register`, `GET /api/users/{id}/rated`, `POST /api/users/{id}/ratings`, `GET /api/users/{id}/recommendations`, `GET /api/users/{id}/recommendations/by-genre`, `GET /api/movies/{id}`, `GET /api/health`. **Scoring (`_scores_excluding_rated`)**: a user who has rated movies in-app gets a live **unified hybrid fold-in** — `α·VAE_foldin + (1−α)·RSVD_foldin` — for new and original users alike, so *both* halves respond to their ratings (no retraining). With no in-app ratings yet: registered users (`id ≥ 944`) get the cold-start `avg_scores`, original users get the precomputed hybrid matrix. `/rated` merges `u.data` ratings with app-submitted ones (app wins on conflict). CORS is locked to `http://localhost:3000`. |
| `users_store.py` | `NewUserStore` — minimal persistence for demo users created via `POST /api/register`. New users get ids starting at 944 and are saved to `new_users.json` (so they survive a restart). Exposes `register(age, gender, occupation)`, `get(id)`, `__contains__`, `__len__`. |
| `ratings_store.py` | `RatingsStore` — persists ratings submitted in-app by **any** user (`POST /api/users/{id}/ratings`) to `user_ratings.json`, kept separate from the original `u.data`. Exposes `get_user(id) → {item_id: rating}`, `set(id, item_id, rating)`, `total()`. Read at request time to fold a user's fresh ratings into their rating vector. |
| `new_users.json` | Runtime output of `NewUserStore` — `{user_id: {age, gender, occupation}}`. Gitignored; created on first registration. |
| `user_ratings.json` | Runtime output of `RatingsStore` — `{user_id: {item_id: rating}}`. Gitignored; created on first in-app rating. |
| `inference.py` | Loads the VAE weights and RSVD arrays, then **precomputes the full hybrid prediction matrix** (`α·VAE + (1−α)·RSVD`, clipped to [1, 5]). Reads α from the latest `Data/EvaluationResults/final_testing_results_*.json`. Exposes a `Predictor` holding `pred_full`, `pred_rsvd`, `train_data`, `alpha`, the live `vae`, and the RSVD fold-in pieces (`rsvd_mu`, `rsvd_bi`, `rsvd_item_factors = V·diag(Σ)`, `rsvd_foldin_lambda`). Methods: `predictions_for_user(id)`; `predict_vae_foldin(vec)` (one deterministic encoder→decoder pass); `predict_rsvd_foldin(vec)` (ridge-regression fold-in — estimates the user's `b_u`/`U[u]` on the fixed item factors, no retraining). `RSVD_FOLDIN_LAMBDA` is tuned (λ=4) so folding a known user's training ratings reproduces their trained RSVD (corr ≈ 0.97). Calls `vae(train_data[:1])` once before `load_weights` so Keras materializes the layer variables — without that step `load_weights` errors. |
| `data.py` | Pandas loaders for the MovieLens flat files. `load_catalog()` reads `u.item` and `u.user`, denormalizes article-trailing titles (`"Lion King, The"` → `"The Lion King"`), and precomputes a per-genre boolean mask of shape `(1682,)` so the genre-shelf endpoint can mask in O(N) instead of running pandas filters per request. `load_ratings()` reads `u.data`. `build_rated_mask(ratings)` returns a `(943, 1682)` bool array used to set `-inf` on already-rated items when ranking. `build_avg_rating_scores(ratings, min_votes=50)` returns a `(1682,)` array of mean rating per movie (movies under `min_votes` set to `-inf`) — the cold-start ranking shown to freshly-registered users. `Catalog.movie_card(item_id, rating=None)` is the canonical dict shape every endpoint returns. |
| `fetch_posters.py` | One-shot CLI script. Reads `u.item`, queries TMDB `/search/movie?query=<title>&year=<year>`, falls back to a query without `year` if the first try misses, and writes `posters_cache.json` keyed by `item_id`. Resumable — re-running skips entries already in the cache. Sleeps 0.25 s between calls, flushes to disk every 25 entries. |
| `posters_cache.json` | Output of `fetch_posters.py`. Maps `item_id` → `{poster_url, tmdb_id, overview}`. Loaded by `data.py`. ~98% hit rate on MovieLens 100K. |

## `frontend/` — Next.js 16 App Router

Standard `create-next-app` scaffold with TypeScript, Tailwind v4, the `src/`
directory, and Turbopack. **Heads-up**: Next.js 16 has breaking changes from
earlier versions — when modifying frontend code, consult the bundled docs at
`frontend/node_modules/next/dist/docs/` (this is the same rule `frontend/AGENTS.md`
gives to AI assistants).

### Config and root

| File | Purpose |
|---|---|
| `package.json` | Pinned to `next@16.2.6`, `react@19.2.4`, Tailwind v4, ESLint 9. The `dev`, `build`, `start`, `lint` scripts are standard. |
| `next.config.ts` | Sets `images.remotePatterns` to allow `https://image.tmdb.org/t/p/**`. Without this entry, `next/image` rejects every poster URL. |
| `tsconfig.json` | Path alias `@/* → ./src/*` lets components import `@/lib/api`. |
| `postcss.config.mjs` | Tailwind v4 PostCSS pipeline. |
| `eslint.config.mjs` | ESLint flat config (`eslint-config-next`). |
| `AGENTS.md` | One-screen warning to AI assistants that Next.js 16 ≠ earlier versions. |
| `CLAUDE.md` | Imports `AGENTS.md`. |

### `src/lib/`

| File | Purpose |
|---|---|
| `api.ts` | Typed client for the backend. Exports `MovieCard`, `LoginResponse` (now with optional `is_new`), `RegisterRequest`, `GenreShelves` types. Functions: `login()`, `register()`, `fetchRated()`, `fetchRecommendations()`, `fetchByGenre()`, `rateMovie(userId, itemId, rating)`. Every fetch uses `cache: "no-store"` because the data is per-user and ephemeral. `API_BASE` is hard-coded to `http://localhost:8000`. |

### `src/app/` — App Router pages

| File | Purpose |
|---|---|
| `layout.tsx` | Root layout. Loads Geist + Geist Mono fonts, sets `<html lang="en">`, dark body background. Defines page `<title>` metadata. |
| `globals.css` | Tailwind import + CSS custom properties for the Netflix-style theme (`--background: #000`, `--accent: #e50914`). Defines `.shelf-scroll` utility that hides the scrollbar while leaving scroll behavior intact. |
| `page.tsx` | Main browse view, client component. Reads `user_id` from `localStorage` on mount (redirects to `/login` if missing), then fetches `/rated`, `/recommendations?limit=50`, and `/recommendations/by-genre?limit=20` **in parallel** via `Promise.all`. Renders `Navbar`, a `Hero` block with the user's demographics, then a stack of `<Shelf>`s. For **new users** (`user_info.is_new`) the Hero copy and the first shelf title switch to a cold-start framing ("Film dengan Rating Tertinggi"). Passes `userId` and an `onRated` callback (a silent `refresh()` that refetches all shelves without the loading flash) down to each `<Shelf>` so a rating immediately updates the page. |
| `favicon.ico` | Default scaffold icon — replace if you want a real brand mark. |

### `src/app/login/`

| File | Purpose |
|---|---|
| `page.tsx` | Login page, client component. Controlled `<form>` with username (User ID) and ignored-password fields. On submit calls `login(username, password)`, stores `user_id` and full `user_info` in `localStorage`, then `router.replace("/")`. Surfaces backend error messages (in Bahasa Indonesia, since `app.py` raises them that way) in a red banner above the submit button. Includes a `<Link>` to `/register` for new users. |

### `src/app/register/`

| File | Purpose |
|---|---|
| `page.tsx` | Registration page, client component. Mirrors the login page styling. Controlled `<form>` with age (number), gender (M/F), and occupation (MovieLens occupation list) selects. On submit calls `register({age, gender, occupation})`, stores the returned `user_id` + `user_info` in `localStorage`, then `router.replace("/")`. The new user lands on the browse page seeing the highest-average-rated movies. Links back to `/login`. |

### `src/app/components/`

| File | Purpose |
|---|---|
| `Navbar.tsx` | Sticky top bar. Shows the MovieMatch wordmark, the current user ID and occupation, and a Logout button that clears `localStorage` and routes back to `/login`. |
| `Shelf.tsx` | Horizontal scrollable row of `<MovieCard>`s. Title above the row, optional left/right arrow buttons that appear on hover (`group-hover/shelf:opacity-100`) and call `scrollBy()`. CSS scroll-snap (`snap-x snap-mandatory`) for the touch-friendly feel. Renders nothing (or an empty-state message) when `movies` is empty. Forwards `userId` and `onRated` to each card. |
| `MovieCard.tsx` | Single poster card. Uses `next/image` with `fill` + `aspect-[2/3]` so the parent controls width. Falls back to a centered-title plate when `poster_url` is null or the image errors out. Optional rating badge in the top-right corner — shown as `★ N.N` for "Already Rated" and as the raw predicted score for recommendation shelves. Opens `<MovieModal>` on click, passing `userId`, `onRated`, and `userRating` (the user's actual rating, only for "Already Rated" cards). |
| `MovieModal.tsx` | Click-to-open detail modal (poster, year, genres, overview). Includes an interactive **5-star rating** widget: clicking a star `POST`s to `/api/users/{id}/ratings` via `rateMovie()`, shows a saved/saving/error state, and pre-fills from `userRating`. On close, if a rating was submitted, fires `onRated()` so the page silently refreshes (the movie moves to "Already Rated" and recommendations update). |

## Sanity check

After a fresh clone + setup, the smoke test is:

1. `python backend/fetch_posters.py` (one-shot)
2. `uvicorn app:app --reload --port 8000` → `http://localhost:8000/api/health` returns
   `{"ok": true, "n_items": 1682, "n_users": 943, "n_new_users": 0, "n_app_ratings": 0, "n_posters": >0, "alpha_vae": ~0.25}`
3. `npm run dev` in `frontend/` → http://localhost:3000/login
4. Sign in as `42`. Top recommendation should be **A Close Shave** (predicted ~4.76).
5. Open any movie and click a star to rate it — it should move to **Already Rated** and the
   recommendation shelves refresh. For a brand-new user (register first), rating a few films
   visibly re-themes the recommendations (VAE-only fold-in).
