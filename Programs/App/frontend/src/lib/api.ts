export const API_BASE = "http://localhost:8000";

export type MovieCard = {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  poster_url: string | null;
  overview: string | null;
  rating: number | null;
};

export type LoginResponse = {
  user_id: number;
  age: number;
  gender: string;
  occupation: string;
  is_new?: boolean;
};

export type RegisterRequest = {
  age: number;
  gender: string;
  occupation: string;
};

export type GenreShelves = Record<string, MovieCard[]>;

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  const res = await fetch(`${API_BASE}/api/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? `Login gagal: ${res.status}`);
  }
  return res.json();
}

export async function register(body: RegisterRequest): Promise<LoginResponse> {
  const res = await fetch(`${API_BASE}/api/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? `Registrasi gagal: ${res.status}`);
  }
  return res.json();
}

export function fetchRated(userId: number) {
  return getJson<MovieCard[]>(`/api/users/${userId}/rated`);
}

export function fetchRecommendations(userId: number, limit = 50) {
  return getJson<MovieCard[]>(`/api/users/${userId}/recommendations?limit=${limit}`);
}

export function fetchByGenre(userId: number, limit = 20) {
  return getJson<GenreShelves>(
    `/api/users/${userId}/recommendations/by-genre?limit=${limit}`
  );
}

export async function rateMovie(
  userId: number,
  itemId: number,
  rating: number
): Promise<MovieCard> {
  const res = await fetch(`${API_BASE}/api/users/${userId}/ratings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ item_id: itemId, rating }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? `Gagal menyimpan rating: ${res.status}`);
  }
  return res.json();
}
