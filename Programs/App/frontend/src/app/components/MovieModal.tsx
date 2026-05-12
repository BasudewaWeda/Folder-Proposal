"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import type { MovieCard as MovieCardType } from "@/lib/api";

type Props = {
  movie: MovieCardType;
  onClose: () => void;
};

const ANIM_MS = 250;

export default function MovieModal({ movie, onClose }: Props) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const raf = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") handleClose();
    }
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleClose() {
    setVisible(false);
    window.setTimeout(onClose, ANIM_MS);
  }

  return (
    <div
      onClick={handleClose}
      className={`fixed inset-0 z-50 flex items-center justify-center px-4 py-8 transition-all ease-out ${
        visible
          ? "bg-black/80 opacity-100 backdrop-blur-md"
          : "bg-black/0 opacity-0 backdrop-blur-0"
      }`}
      style={{ transitionDuration: `${ANIM_MS}ms` }}
      role="dialog"
      aria-modal="true"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className={`relative flex w-full max-w-3xl flex-col overflow-hidden rounded-lg bg-zinc-950 shadow-2xl ring-1 ring-white/10 transition-all ease-out sm:flex-row ${
          visible
            ? "scale-100 opacity-100 blur-0"
            : "scale-95 opacity-0 blur-md"
        }`}
        style={{ transitionDuration: `${ANIM_MS}ms` }}
      >
        <button
          type="button"
          aria-label="Close"
          onClick={handleClose}
          className="absolute right-3 top-3 z-10 flex h-9 w-9 items-center justify-center rounded-full bg-black/70 text-xl text-white transition hover:bg-black"
        >
          ×
        </button>

        <div className="relative aspect-[2/3] w-full shrink-0 bg-zinc-900 sm:w-64">
          {movie.poster_url ? (
            <Image
              src={movie.poster_url}
              alt={movie.title}
              fill
              sizes="(min-width: 640px) 256px, 100vw"
              className="object-cover"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center p-4 text-center text-sm text-zinc-400">
              {movie.title}
            </div>
          )}
        </div>

        <div className="flex flex-1 flex-col gap-3 p-6">
          <div>
            <h2 className="text-2xl font-bold text-white">{movie.title}</h2>
            <p className="mt-1 text-sm text-zinc-400">
              {movie.year ?? "Unknown year"}
              {movie.genres.length > 0 && (
                <span className="ml-2 text-zinc-500">
                  · {movie.genres.join(", ")}
                </span>
              )}
            </p>
          </div>

          <p className="text-sm leading-relaxed text-zinc-300">
            {movie.overview && movie.overview.trim().length > 0
              ? movie.overview
              : "Tidak ada deskripsi yang tersedia untuk film ini."}
          </p>

          {movie.rating != null && (
            <p className="mt-auto text-xs text-zinc-500">
              Predicted / actual rating:{" "}
              <span className="text-amber-300">★ {movie.rating.toFixed(2)}</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
