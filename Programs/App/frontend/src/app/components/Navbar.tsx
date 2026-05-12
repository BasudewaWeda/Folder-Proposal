"use client";

import { useRouter } from "next/navigation";

type Props = {
  userId: number | null;
  occupation?: string;
};

export default function Navbar({ userId, occupation }: Props) {
  const router = useRouter();

  function logout() {
    if (typeof window !== "undefined") {
      window.localStorage.removeItem("user_id");
      window.localStorage.removeItem("user_info");
    }
    router.replace("/login");
  }

  return (
    <header className="sticky top-0 z-20 flex items-center justify-between bg-gradient-to-b from-black via-black/85 to-transparent px-6 py-4 backdrop-blur-sm">
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-black tracking-tight text-[color:var(--accent)]">
          Movie Recommender System
        </span>
      </div>

      <div className="flex items-center gap-3 text-sm">
        {userId !== null && (
          <span className="text-zinc-300">
            User <span className="font-semibold text-white">#{userId}</span>
            {occupation && <span className="ml-1 text-zinc-500">({occupation})</span>}
          </span>
        )}
        <button
          type="button"
          onClick={logout}
          className="rounded-md border border-white/15 px-3 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:bg-white/10"
        >
          Logout
        </button>
      </div>
    </header>
  );
}
