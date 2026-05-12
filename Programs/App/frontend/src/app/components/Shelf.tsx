"use client";

import { useRef } from "react";
import type { MovieCard as MovieCardType } from "@/lib/api";
import MovieCard from "./MovieCard";

type Props = {
  title: string;
  movies: MovieCardType[];
  showRating?: boolean;
  emptyMessage?: string;
};

export default function Shelf({ title, movies, showRating, emptyMessage }: Props) {
  const scrollerRef = useRef<HTMLDivElement>(null);

  function scrollBy(direction: 1 | -1) {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollBy({ left: direction * el.clientWidth * 0.9, behavior: "smooth" });
  }

  if (movies.length === 0) {
    if (!emptyMessage) return null;
    return (
      <section className="py-4">
        <h2 className="mb-3 px-6 text-xl font-semibold text-white">{title}</h2>
        <p className="px-6 text-sm text-zinc-500">{emptyMessage}</p>
      </section>
    );
  }

  return (
    <section className="group/shelf relative py-4">
      <h2 className="mb-3 px-6 text-xl font-semibold text-white">{title}</h2>

      <button
        type="button"
        aria-label="Scroll left"
        onClick={() => scrollBy(-1)}
        className="absolute left-0 top-1/2 z-10 hidden h-32 w-10 -translate-y-1/4 items-center justify-center bg-gradient-to-r from-black/80 to-transparent text-2xl text-white opacity-0 transition-opacity group-hover/shelf:opacity-100 md:flex"
      >
        ‹
      </button>
      <button
        type="button"
        aria-label="Scroll right"
        onClick={() => scrollBy(1)}
        className="absolute right-0 top-1/2 z-10 hidden h-32 w-10 -translate-y-1/4 items-center justify-center bg-gradient-to-l from-black/80 to-transparent text-2xl text-white opacity-0 transition-opacity group-hover/shelf:opacity-100 md:flex"
      >
        ›
      </button>

      <div
        ref={scrollerRef}
        className="shelf-scroll flex snap-x snap-mandatory gap-3 overflow-x-auto scroll-smooth px-6 pb-2"
      >
        {movies.map((m) => (
          <MovieCard key={m.movie_id} movie={m} showRating={showRating} />
        ))}
      </div>
    </section>
  );
}
