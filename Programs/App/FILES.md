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
| `app.py` | FastAPI app. Defines the `lifespan` handler that calls `load_catalog()`, `load_ratings()`, `build_rated_mask()`, `build_predictor()` at startup and stashes them in `_state`. Defines all routes: `POST /api/login`, `GET /api/users/{id}/rated`, `GET /api/users/{id}/recommendations`, `GET /api/users/{id}/recommendations/by-genre`, `GET /api/movies/{id}`, `GET /api/health`. CORS is locked to `http://localhost:3000`. |
| `inference.py` | Loads the VAE weights and RSVD arrays, then **precomputes the full hybrid prediction matrix** (`α·VAE + (1−α)·RSVD`, clipped to [1, 5]). Reads α from the latest `Data/EvaluationResults/final_testing_results_*.json`. Exposes a `Predictor` with `predictions_for_user(user_id_1based) → np.ndarray[1682]`. Calls `vae(train_data[:1])` once before `load_weights` so Keras materializes the layer variables — without that step `load_weights` errors. |
| `data.py` | Pandas loaders for the MovieLens flat files. `load_catalog()` reads `u.item` and `u.user`, denormalizes article-trailing titles (`"Lion King, The"` → `"The Lion King"`), and precomputes a per-genre boolean mask of shape `(1682,)` so the genre-shelf endpoint can mask in O(N) instead of running pandas filters per request. `load_ratings()` reads `u.data`. `build_rated_mask(ratings)` returns a `(943, 1682)` bool array used to set `-inf` on already-rated items when ranking. `Catalog.movie_card(item_id, rating=None)` is the canonical dict shape every endpoint returns. |
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
| `api.ts` | Typed client for the backend. Exports `MovieCard`, `LoginResponse`, `GenreShelves` types. Functions: `login()`, `fetchRated()`, `fetchRecommendations()`, `fetchByGenre()`. Every fetch uses `cache: "no-store"` because the data is per-user and ephemeral. `API_BASE` is hard-coded to `http://localhost:8000`. |

### `src/app/` — App Router pages

| File | Purpose |
|---|---|
| `layout.tsx` | Root layout. Loads Geist + Geist Mono fonts, sets `<html lang="en">`, dark body background. Defines page `<title>` metadata. |
| `globals.css` | Tailwind import + CSS custom properties for the Netflix-style theme (`--background: #000`, `--accent: #e50914`). Defines `.shelf-scroll` utility that hides the scrollbar while leaving scroll behavior intact. |
| `page.tsx` | Main browse view, client component. Reads `user_id` from `localStorage` on mount (redirects to `/login` if missing), then fetches `/rated`, `/recommendations?limit=50`, and `/recommendations/by-genre?limit=20` **in parallel** via `Promise.all`. Renders `Navbar`, a `Hero` block with the user's demographics, then a stack of `<Shelf>`s. |
| `favicon.ico` | Default scaffold icon — replace if you want a real brand mark. |

### `src/app/login/`

| File | Purpose |
|---|---|
| `page.tsx` | Login page, client component. Controlled `<form>` with username (User ID) and ignored-password fields. On submit calls `login(username, password)`, stores `user_id` and full `user_info` in `localStorage`, then `router.replace("/")`. Surfaces backend error messages (in Bahasa Indonesia, since `app.py` raises them that way) in a red banner above the submit button. |

### `src/app/components/`

| File | Purpose |
|---|---|
| `Navbar.tsx` | Sticky top bar. Shows the MovieMatch wordmark, the current user ID and occupation, and a Logout button that clears `localStorage` and routes back to `/login`. |
| `Shelf.tsx` | Horizontal scrollable row of `<MovieCard>`s. Title above the row, optional left/right arrow buttons that appear on hover (`group-hover/shelf:opacity-100`) and call `scrollBy()`. CSS scroll-snap (`snap-x snap-mandatory`) for the touch-friendly feel. Renders nothing (or an empty-state message) when `movies` is empty. |
| `MovieCard.tsx` | Single poster card. Uses `next/image` with `fill` + `aspect-[2/3]` so the parent controls width. Falls back to a centered-title plate when `poster_url` is null or the image errors out. Optional rating badge in the top-right corner — shown as `★ N.N` for "Already Rated" and as the raw predicted score for recommendation shelves. |

## Sanity check

After a fresh clone + setup, the smoke test is:

1. `python backend/fetch_posters.py` (one-shot)
2. `uvicorn app:app --reload --port 8000` → `http://localhost:8000/api/health` returns
   `{"ok": true, "n_items": 1682, "n_users": 943, "n_posters": >0, "alpha_vae": ~0.25}`
3. `npm run dev` in `frontend/` → http://localhost:3000/login
4. Sign in as `42`. Top recommendation should be **A Close Shave** (predicted ~4.76).
