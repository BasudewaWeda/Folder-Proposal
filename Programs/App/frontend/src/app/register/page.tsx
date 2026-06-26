"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { register } from "@/lib/api";

// MovieLens 100K occupation vocabulary (u.occupation).
const OCCUPATIONS = [
  "administrator", "artist", "doctor", "educator", "engineer",
  "entertainment", "executive", "healthcare", "homemaker", "lawyer",
  "librarian", "marketing", "none", "other", "programmer", "retired",
  "salesman", "scientist", "student", "technician", "writer",
];

export default function RegisterPage() {
  const router = useRouter();
  const [age, setAge] = useState("25");
  const [gender, setGender] = useState("M");
  const [occupation, setOccupation] = useState("other");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const info = await register({
        age: Number(age),
        gender,
        occupation,
      });
      if (typeof window !== "undefined") {
        window.localStorage.setItem("user_id", String(info.user_id));
        window.localStorage.setItem("user_info", JSON.stringify(info));
      }
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registrasi gagal");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex flex-1 items-center justify-center bg-gradient-to-br from-black via-zinc-950 to-[#1a0205] px-4 py-12">
      <div className="absolute left-6 top-6 text-2xl font-black tracking-tight text-[color:var(--accent)]">
        MovieMatch
      </div>

      <form
        onSubmit={onSubmit}
        className="w-full max-w-md rounded-lg bg-black/75 p-10 shadow-2xl ring-1 ring-white/5"
      >
        <h1 className="mb-6 text-3xl font-bold text-white">Buat Akun Baru</h1>
        <p className="mb-6 text-sm text-zinc-400">
          User baru belum punya histori rating, jadi kami mulai dengan film
          berrating tertinggi. ID akan dibuat otomatis.
        </p>

        <label className="mb-4 block">
          <span className="mb-1 block text-sm text-zinc-300">Umur</span>
          <input
            type="number"
            min={1}
            max={120}
            value={age}
            onChange={(e) => setAge(e.target.value)}
            required
            className="w-full rounded-md bg-zinc-800 px-4 py-3 text-white placeholder-zinc-500 outline-none ring-1 ring-transparent transition focus:bg-zinc-700 focus:ring-white/30"
          />
        </label>

        <label className="mb-4 block">
          <span className="mb-1 block text-sm text-zinc-300">Gender</span>
          <select
            value={gender}
            onChange={(e) => setGender(e.target.value)}
            className="w-full rounded-md bg-zinc-800 px-4 py-3 text-white outline-none ring-1 ring-transparent transition focus:bg-zinc-700 focus:ring-white/30"
          >
            <option value="M">Laki-laki (M)</option>
            <option value="F">Perempuan (F)</option>
          </select>
        </label>

        <label className="mb-6 block">
          <span className="mb-1 block text-sm text-zinc-300">Pekerjaan</span>
          <select
            value={occupation}
            onChange={(e) => setOccupation(e.target.value)}
            className="w-full rounded-md bg-zinc-800 px-4 py-3 text-white outline-none ring-1 ring-transparent transition focus:bg-zinc-700 focus:ring-white/30"
          >
            {OCCUPATIONS.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        </label>

        {error && (
          <div className="mb-4 rounded-md bg-red-950/60 px-3 py-2 text-sm text-red-300 ring-1 ring-red-800">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded-md bg-[color:var(--accent)] py-3 text-base font-semibold text-white transition hover:bg-[#f6121d] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {submitting ? "Membuat akun…" : "Daftar"}
        </button>

        <p className="mt-6 text-sm text-zinc-400">
          Sudah punya akun?{" "}
          <Link
            href="/login"
            className="font-semibold text-[color:var(--accent)] hover:underline"
          >
            Masuk di sini
          </Link>
        </p>
      </form>
    </div>
  );
}
