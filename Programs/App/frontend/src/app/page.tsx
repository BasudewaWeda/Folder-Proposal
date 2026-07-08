"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  fetchByGenre,
  fetchRated,
  fetchRecommendations,
  type GenreShelves,
  type LoginResponse,
  type MovieCard,
} from "@/lib/api";
import Navbar from "./components/Navbar";
import Shelf from "./components/Shelf";

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; rated: MovieCard[]; recs: MovieCard[]; byGenre: GenreShelves }
  | { kind: "error"; message: string };

export default function HomePage() {
  const router = useRouter();
  const [userId, setUserId] = useState<number | null>(null);
  const [userInfo, setUserInfo] = useState<LoginResponse | null>(null);
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  useEffect(() => {
    const stored = window.localStorage.getItem("user_id");
    if (!stored) {
      router.replace("/login");
      return;
    }
    const id = Number(stored);
    if (!Number.isFinite(id)) {
      window.localStorage.removeItem("user_id");
      router.replace("/login");
      return;
    }
    setUserId(id);
    const infoRaw = window.localStorage.getItem("user_info");
    if (infoRaw) {
      try {
        setUserInfo(JSON.parse(infoRaw) as LoginResponse);
      } catch {
        // ignore stale value
      }
    }
  }, [router]);

  useEffect(() => {
    if (userId === null) return;
    let cancelled = false;
    setState({ kind: "loading" });
    Promise.all([
      fetchRated(userId),
      fetchRecommendations(userId, 50),
      fetchByGenre(userId, 20),
    ])
      .then(([rated, recs, byGenre]) => {
        if (cancelled) return;
        setState({ kind: "ready", rated, recs, byGenre });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setState({
          kind: "error",
          message: err instanceof Error ? err.message : "Gagal memuat data",
        });
      });
    return () => {
      cancelled = true;
    };
  }, [userId]);

  // Silent refetch (no loading flash) after the user rates a movie.
  const refresh = useCallback(() => {
    if (userId === null) return;
    Promise.all([
      fetchRated(userId),
      fetchRecommendations(userId, 50),
      fetchByGenre(userId, 20),
    ])
      .then(([rated, recs, byGenre]) =>
        setState({ kind: "ready", rated, recs, byGenre })
      )
      .catch((err: unknown) =>
        setState({
          kind: "error",
          message: err instanceof Error ? err.message : "Gagal memuat data",
        })
      );
  }, [userId]);

  return (
    <div className="flex flex-1 flex-col bg-black">
      <Navbar username={userInfo?.username} occupation={userInfo?.occupation} />

      <main className="flex-1 pb-16">
        {state.kind === "loading" && (
          <div className="flex h-64 items-center justify-center text-zinc-400">
            Loading rekomendasi…
          </div>
        )}

        {state.kind === "error" && (
          <div className="mx-6 mt-6 rounded-md bg-red-950/50 px-4 py-3 text-sm text-red-300 ring-1 ring-red-800">
            {state.message}
          </div>
        )}

        {state.kind === "ready" && (
          <>
            <Hero userInfo={userInfo} ratedCount={state.rated.length} />

            <Shelf
              title={
                userInfo?.is_new
                  ? "Film dengan Rating Tertinggi"
                  : "Top Recommended for You"
              }
              movies={state.recs}
              userId={userId}
              onRated={refresh}
              emptyMessage="Tidak ada rekomendasi yang tersedia."
            />

            <Shelf
              title="Already Rated"
              movies={state.rated}
              showRating
              userId={userId}
              onRated={refresh}
              emptyMessage="User ini belum memiliki rating."
            />

            {Object.entries(state.byGenre).map(([genre, movies]) => (
              <Shelf
                key={genre}
                title={`Best in ${capitalize(genre)}`}
                movies={movies}
                userId={userId}
                onRated={refresh}
              />
            ))}
          </>
        )}
      </main>
    </div>
  );
}

function capitalize(s: string): string {
  if (!s) return s;
  return s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, " ");
}

function Hero({
  userInfo,
  ratedCount,
}: {
  userInfo: LoginResponse | null;
  ratedCount: number;
}) {
  const isNew = userInfo?.is_new ?? false;
  const name = userInfo?.username ?? "";
  return (
    <section className="px-6 pb-6 pt-2">
      <h1 className="text-3xl font-bold text-white sm:text-4xl">
        {isNew ? `Selamat datang, ${name}` : `Welcome back, ${name}`}
      </h1>
      <p className="mt-2 text-sm text-zinc-400">
        {userInfo
          ? `${userInfo.age} yrs · ${userInfo.gender} · ${userInfo.occupation}. `
          : ""}
        {isNew
          ? "Akun baru — belum ada histori rating, jadi kami tampilkan film dengan rating tertinggi untuk memulai."
          : `${ratedCount} film sudah kamu rating. Berikut hasil prediksi dari model hybrid VAE + RSVD.`}
      </p>
    </section>
  );
}
