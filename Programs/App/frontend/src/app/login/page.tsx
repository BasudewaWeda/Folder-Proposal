"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { login } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const info = await login(username.trim(), password);
      if (typeof window !== "undefined") {
        window.localStorage.setItem("user_id", String(info.user_id));
        window.localStorage.setItem("user_info", JSON.stringify(info));
      }
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login gagal");
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
        <h1 className="mb-6 text-3xl font-bold text-white">Sign In</h1>
        <p className="mb-6 text-sm text-zinc-400">
          Masukkan ID user MovieLens 100K (1–943). Password boleh diisi apa saja.
        </p>

        <label className="mb-4 block">
          <span className="sr-only">Username</span>
          <input
            type="text"
            inputMode="numeric"
            placeholder="User ID (e.g. 42)"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            className="w-full rounded-md bg-zinc-800 px-4 py-3 text-white placeholder-zinc-500 outline-none ring-1 ring-transparent transition focus:bg-zinc-700 focus:ring-white/30"
          />
        </label>

        <label className="mb-6 block">
          <span className="sr-only">Password</span>
          <input
            type="password"
            placeholder="Password (ignored)"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-md bg-zinc-800 px-4 py-3 text-white placeholder-zinc-500 outline-none ring-1 ring-transparent transition focus:bg-zinc-700 focus:ring-white/30"
          />
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
          {submitting ? "Signing in…" : "Sign In"}
        </button>

        <p className="mt-6 text-sm text-zinc-400">
          Belum punya akun?{" "}
          <Link
            href="/register"
            className="font-semibold text-[color:var(--accent)] hover:underline"
          >
            Daftar di sini
          </Link>
        </p>

        <p className="mt-3 text-xs text-zinc-500">
          Demo lokal — tidak ada otentikasi sungguhan. Username harus berupa angka.
        </p>
      </form>
    </div>
  );
}
