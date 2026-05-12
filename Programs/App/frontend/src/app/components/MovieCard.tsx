"use client";

import Image from "next/image";
import { useState } from "react";
import type { MovieCard as MovieCardType } from "@/lib/api";
import MovieModal from "./MovieModal";

type Props = {
  movie: MovieCardType;
  showRating?: boolean;
};

export default function MovieCard({ movie, showRating = false }: Props) {
  const [imgError, setImgError] = useState(false);
  const [open, setOpen] = useState(false);
  const hasPoster = !!movie.poster_url && !imgError;
  const ratingLabel = movie.rating == null
    ? null
    : showRating
      ? `★ ${movie.rating.toFixed(1)}`
      : `${movie.rating.toFixed(2)}`;

  return (
    <div className="group relative w-40 shrink-0 snap-start">
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="relative aspect-[2/3] w-full overflow-hidden rounded-md bg-zinc-900 ring-1 ring-white/5 transition-transform duration-200 group-hover:scale-105 group-hover:ring-white/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/60"
        aria-label={`Show details for ${movie.title}`}
      >
        {hasPoster ? (
          <Image
            src={movie.poster_url!}
            alt={movie.title}
            fill
            sizes="160px"
            className="object-cover"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center p-2 text-center text-xs text-zinc-400">
            {movie.title}
          </div>
        )}
        {ratingLabel && (
          <div className="absolute right-1 top-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-semibold text-amber-300">
            {ratingLabel}
          </div>
        )}
      </button>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="mt-2 block w-full text-left"
      >
        <p className="line-clamp-1 text-sm font-medium text-zinc-100" title={movie.title}>
          {movie.title}
        </p>
        <p className="text-xs text-zinc-500">
          {movie.year ?? ""}
          {movie.genres.length > 0 && (
            <span className="ml-1 text-zinc-600">· {movie.genres.slice(0, 2).join(", ")}</span>
          )}
        </p>
      </button>
      {open && <MovieModal movie={movie} onClose={() => setOpen(false)} />}
    </div>
  );
}
