"""
Fetch poster URLs from TMDB for every movie in MovieLens 100K.

Reads  : Data/ml-100k/u.item
Writes : App/backend/posters_cache.json

Resumable — if posters_cache.json already exists, only movies missing from it
are fetched. Safe to interrupt with Ctrl-C; the cache is flushed every 25
successful lookups and again on shutdown.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import requests


SCRIPT_DIR   = Path(__file__).resolve().parent           # Programs/App/backend
PROGRAMS_DIR = SCRIPT_DIR.parent.parent                  # Programs
U_ITEM_PATH  = PROGRAMS_DIR / "Data" / "ml-100k" / "u.item"
CACHE_PATH   = SCRIPT_DIR / "posters_cache.json"
ENV_PATH     = SCRIPT_DIR / ".env"

TMDB_SEARCH_URL      = "https://api.themoviedb.org/3/search/movie"
POSTER_BASE          = "https://image.tmdb.org/t/p/w500"
SLEEP_BETWEEN_CALLS  = 0.25   # seconds — TMDB free tier allows ~40 req/sec, this is comfortably under
SAVE_EVERY           = 25     # flush cache to disk every N successful lookups


def load_env(path: Path) -> dict:
    """Minimal .env reader; only handles KEY=VALUE lines, ignores quotes and comments."""
    env = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def parse_u_item(path: Path):
    """
    Yield (item_id, title, year) for every movie in u.item.

    u.item format (pipe-separated):
        item_id | title | release_date | video_release_date | IMDb_URL | <19 genre flags>

    title typically looks like "Toy Story (1995)" — extract year from the
    parenthesized suffix when present, otherwise fall back to release_date.
    The file uses latin-1 encoding (a few non-ASCII titles).
    """
    year_re = re.compile(r"\s*\((\d{4})\)\s*$")
    with path.open(encoding="latin-1") as f:
        for raw in f:
            parts = raw.rstrip("\n").split("|")
            if len(parts) < 5:
                continue
            item_id   = int(parts[0])
            title_raw = parts[1]
            release   = parts[2]

            m = year_re.search(title_raw)
            if m:
                year  = int(m.group(1))
                title = title_raw[: m.start()].strip()
            else:
                title = title_raw.strip()
                year  = int(release[-4:]) if release and release[-4:].isdigit() else None
            yield item_id, title, year


def normalize_movielens_title(title: str) -> str:
    """
    MovieLens moves leading articles to the end: "Postman, The" instead of
    "The Postman". TMDB indexes the natural form, so move the article back.
    """
    for article in ("The", "A", "An", "Le", "La", "Les", "Il", "Lo", "Gli", "El", "Los", "Las", "Der", "Die", "Das"):
        suffix = f", {article}"
        if title.endswith(suffix):
            return f"{article} {title[: -len(suffix)]}"
    return title


def tmdb_search(title: str, year, api_key: str) -> dict | None:
    """
    Search TMDB and return the top match's metadata dict (or None).
    If a year-filtered search returns nothing, retry without the year — TMDB's
    year filter is strict and MovieLens release dates aren't always exact.
    """
    params = {"api_key": api_key, "query": title, "include_adult": "false"}
    if year:
        params["year"] = year
    try:
        r = requests.get(TMDB_SEARCH_URL, params=params, timeout=10)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    results = r.json().get("results") or []

    if not results and year:
        params.pop("year")
        try:
            r = requests.get(TMDB_SEARCH_URL, params=params, timeout=10)
        except requests.RequestException:
            return None
        if r.status_code == 200:
            results = r.json().get("results") or []

    return results[0] if results else None


def write_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    env     = load_env(ENV_PATH)
    api_key = env.get("TMDB_API_KEY") or os.environ.get("TMDB_API_KEY")
    if not api_key:
        print(f"[ERROR] TMDB_API_KEY not found in {ENV_PATH} or environment.", file=sys.stderr)
        return 1

    if not U_ITEM_PATH.exists():
        print(f"[ERROR] u.item not found at {U_ITEM_PATH}", file=sys.stderr)
        return 1

    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        print(f"[INFO] Resuming with {len(cache)} cached entries.")
    else:
        cache = {}

    movies = list(parse_u_item(U_ITEM_PATH))
    print(f"[INFO] Total movies in u.item : {len(movies)}")

    pending = [(mid, t, y) for mid, t, y in movies if str(mid) not in cache]
    print(f"[INFO] To fetch this run      : {len(pending)}")
    if not pending:
        print("[INFO] Nothing to do — cache is already complete.")
        return 0

    fetched = 0
    try:
        for i, (item_id, title, year) in enumerate(pending, 1):
            query  = normalize_movielens_title(title)
            result = tmdb_search(query, year, api_key)

            if result and result.get("poster_path"):
                cache[str(item_id)] = {
                    "tmdb_id"   : result.get("id"),
                    "poster_url": POSTER_BASE + result["poster_path"],
                    "overview"  : result.get("overview") or "",
                    "title"     : title,
                    "year"      : year,
                }
                tag = "OK  "
            else:
                cache[str(item_id)] = {
                    "tmdb_id"   : None,
                    "poster_url": None,
                    "overview"  : "",
                    "title"     : title,
                    "year"      : year,
                }
                tag = "MISS"

            fetched += 1
            print(f"  [{i:4d}/{len(pending)}] {tag}  id={item_id:4d}  {title} ({year})")

            if fetched % SAVE_EVERY == 0:
                write_cache(cache)

            time.sleep(SLEEP_BETWEEN_CALLS)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user — flushing cache before exit.")
    finally:
        write_cache(cache)
        with_poster = sum(1 for v in cache.values() if v.get("poster_url"))
        print(f"\n[SUCCESS] Cache saved to: {CACHE_PATH}")
        print(f"          {with_poster}/{len(cache)} movies have posters "
              f"({100 * with_poster / len(cache):.1f}%).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
