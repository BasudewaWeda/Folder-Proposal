# III. POTONGAN PROGRAM

Bagian ini menampilkan potongan-potongan kode inti dari aplikasi MovieMatch
beserta penjelasan singkatnya. Kode dikelompokkan sesuai alur kerja aplikasi,
mulai dari penyiapan lingkungan, pemuatan model, penyediaan layanan (API) di
sisi *backend*, hingga antarmuka di sisi *frontend*.

---

## 1. Minimum Library Requirements

Aplikasi membutuhkan sejumlah pustaka (*library*) Python. Berikut pustaka utama
beserta versinya (daftar lengkap tersedia pada berkas `requirements.txt`).

*Berkas: `Programs/requirements.txt`*

```text
# Machine learning & komputasi numerik
tensorflow==2.20.0
keras==3.12.1
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.7.2
scipy==1.15.3
optuna==4.7.0

# Web service (backend API)
fastapi==0.115.5
uvicorn==0.32.1
pydantic==2.13.4
requests==2.32.5

# Notebook & visualisasi
jupyter_client==8.8.0
ipykernel==7.2.0
matplotlib==3.10.8
tqdm==4.67.3
```

Seluruh pustaka dipasang sekaligus dengan perintah:

```powershell
pip install -r requirements.txt
```

---

## 2. Aktivasi Virtual Environment

Sebelum menjalankan program, dibuat lingkungan virtual (*virtual environment*)
agar seluruh pustaka terisolasi dari sistem. Perintah dijalankan dari folder
`Programs`.

```powershell
# Membuat virtual environment bernama "venv"
python -m venv venv

# Mengaktifkan virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Bila PowerShell menolak karena execution policy, jalankan sekali:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Memasang seluruh dependency
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika berhasil, akan muncul penanda `(venv)` di depan baris terminal.

---

## 3. Memuat Model Hybrid VAE + RSVD

Saat *backend* dinyalakan, kedua model dimuat dari folder `Data`. Komponen RSVD
berupa berkas NumPy (`.npy`), sedangkan VAE dimuat dari bobot terlatih
(`.weights.h5`) menggunakan arsitektur yang sama seperti saat pelatihan.

*Berkas: `App/backend/inference.py`*

```python
def _load_rsvd_components():
    mu    = np.load(RSVD_DIR / "final_mu.npy")
    b_u   = np.load(RSVD_DIR / "final_b_u.npy")
    b_i   = np.load(RSVD_DIR / "final_b_i.npy")
    U     = np.load(RSVD_DIR / "best_U.npy")
    Sigma = np.load(RSVD_DIR / "best_Sigma.npy")
    V     = np.load(RSVD_DIR / "best_V.npy")
    return mu, b_u, b_i, U, Sigma, V


def _load_vae(train_data: np.ndarray):
    from model import Encoder, Decoder, VAE

    with open(VAE_PARAMS_PATH) as f:
        params = json.load(f)["best_params"]

    num_items = train_data.shape[1]
    encoder = Encoder(
        hidden_dims  = params["hidden_dims"],
        latent_dim   = params["latent_dim"],
        dropout_rate = params["dropout_rate"],
    )
    decoder = Decoder(hidden_dims=params["hidden_dims"][::-1], output_dim=num_items)
    vae     = VAE(encoder, decoder, beta=params["beta"])
    # Bangun variabel model dulu sebelum memuat bobot
    _ = vae(train_data[:1])
    vae.load_weights(str(VAE_WEIGHTS_PATH))
    return vae
```

---

## 4. Perhitungan Matriks Prediksi Awal

Setelah kedua model dimuat, seluruh matriks prediksi rating berukuran
`943 × 1682` dihitung **satu kali** di awal. Prediksi akhir merupakan gabungan
berbobot (*ensemble*) dari VAE dan RSVD: `α · VAE + (1 − α) · RSVD`. Dengan cara
ini, setiap permintaan rekomendasi cukup membaca satu baris matriks (cepat).

*Berkas: `App/backend/inference.py`*

```python
def build_predictor() -> Predictor:
    # 1) Muat data latih dan komponen RSVD, lalu susun prediksi RSVD
    train_data = np.load(TRAIN_DATA_PATH).astype(np.float32)
    mu, b_u, b_i, U, Sigma, V = _load_rsvd_components()
    pred_rsvd_norm = float(mu) + b_u[:, None] + b_i[None, :] + U @ Sigma @ V.T
    pred_rsvd = np.clip(pred_rsvd_norm * MAX_RATING, 1.0, MAX_RATING).astype(np.float32)

    # 2) Jalankan encoder -> decoder VAE atas seluruh matriks rating
    vae       = _load_vae(train_data)
    z_mean, _ = vae.encoder.predict(train_data, verbose=0)
    pred_vae_norm = vae.decoder.predict(z_mean, verbose=0)
    pred_vae = np.clip(pred_vae_norm * MAX_RATING, 1.0, MAX_RATING).astype(np.float32)

    # 3) Gabungkan kedua prediksi dengan bobot alpha (hasil penyetelan)
    alpha = _latest_eval_alpha()
    pred_full = np.clip(
        alpha * pred_vae + (1.0 - alpha) * pred_rsvd, 1.0, MAX_RATING
    ).astype(np.float32)

    return Predictor(pred_full=pred_full, pred_rsvd=pred_rsvd,
                     train_data=train_data, alpha=alpha, vae=vae, ...)
```

---

## 5. Endpoint Login dan Verifikasi Password

*Endpoint* `POST /api/login` menerima **username** dan **password**. Pengguna
hasil registrasi diverifikasi terhadap *hash* password tersimpan, sedangkan
pengguna bawaan dataset (`User_1`–`User_943`) memakai password default
`movielens`.

*Berkas: `App/backend/app.py`*

```python
@app.post("/api/login")
def login(req: LoginRequest):
    username = req.username.strip()

    # 1) Pengguna hasil registrasi memiliki username bebas
    rec = _new_users().by_username(username)
    if rec is not None:
        if not _new_users().check_password(rec["user_id"], req.password):
            raise HTTPException(status_code=401, detail="Password salah")
        return {**rec, "is_new": True}

    # 2) Pengguna dataset login sebagai "User_<id>" + password default
    user_id = _dataset_username_id(username)
    if user_id is not None and user_id in _catalog().users.index:
        if req.password != DEFAULT_PASSWORD:
            raise HTTPException(status_code=401, detail="Password salah")
        info = _catalog().users.loc[user_id]
        return {
            "user_id": user_id, "username": f"User_{user_id}",
            "age": int(info["age"]), "gender": str(info["gender"]),
            "occupation": str(info["occupation"]), "is_new": False,
        }

    raise HTTPException(status_code=401, detail="Username tidak ditemukan")
```

Verifikasi password memakai algoritma **PBKDF2-SHA256** dengan *salt*. Password
tidak pernah disimpan sebagai teks polos.

*Berkas: `App/backend/users_store.py`*

```python
def verify_password(password: str, stored: Optional[str]) -> bool:
    """Mencocokkan password dengan hash yang tersimpan."""
    if not stored:
        return False
    try:
        algo, iters, salt, expected = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iters)
        )
    except (ValueError, AttributeError):
        return False
    return secrets.compare_digest(digest.hex(), expected)
```

---

## 6. Endpoint Registrasi Pengguna Baru

*Endpoint* `POST /api/register` membuat pengguna baru. Backend memvalidasi
username (unik, minimal 3 karakter, bukan format `User_<angka>`) dan password
(minimal 4 karakter), lalu menyimpan datanya.

*Berkas: `App/backend/app.py`*

```python
@app.post("/api/register")
def register(req: RegisterRequest):
    username = req.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username minimal 3 karakter")
    if _dataset_username_id(username) is not None:
        raise HTTPException(status_code=400,
            detail='Username dengan format "User_<angka>" dipakai user dataset')
    if _new_users().username_taken(username):
        raise HTTPException(status_code=409, detail="Username sudah dipakai")
    if not req.password or len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Password minimal 4 karakter")
    info = _new_users().register(
        username=username, age=req.age, gender=req.gender,
        occupation=req.occupation, password=req.password,
    )
    return {**info, "is_new": True}
```

Password diubah menjadi *hash* bersalt sebelum disimpan ke berkas
`new_users.json`.

*Berkas: `App/backend/users_store.py`*

```python
def hash_password(password: str) -> str:
    """Menghasilkan hash PBKDF2-SHA256 bersalt: 'pbkdf2_sha256$iters$salt$hash'."""
    salt   = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"
```

---

## 7. Menghasilkan Rekomendasi Personal

*Endpoint* `GET /api/users/{id}/recommendations` mengambil skor prediksi
pengguna, mengurutkannya dari tertinggi, lalu mengembalikan film-film teratas
(film yang sudah dirating dikecualikan).

*Berkas: `App/backend/app.py`*

```python
@app.get("/api/users/{user_id}/recommendations")
def user_recommendations(user_id: int, limit: int = 50):
    _ensure_user(user_id)
    scores = _scores_excluding_rated(user_id)
    top    = np.argsort(scores)[::-1][:limit]
    return [
        _catalog().movie_card(int(idx + 1), rating=float(scores[idx]))
        for idx in top
        if np.isfinite(scores[idx])
    ]
```

Fungsi `_scores_excluding_rated` menentukan sumber skor: matriks *hybrid* untuk
pengguna dataset, skor rata-rata (*cold-start*) untuk pengguna baru, atau hasil
*fold-in* bila pengguna sudah memberi rating lewat aplikasi. Film yang sudah
dirating diberi nilai `-inf` agar tidak muncul kembali.

```python
def _scores_excluding_rated(user_id: int) -> np.ndarray:
    app_ratings = _user_ratings().get_user(user_id)
    if app_ratings:                                   # ada rating dari aplikasi
        vec       = _user_rating_vector_norm(user_id)
        vae_pred  = _predictor().predict_vae_foldin(vec)
        rsvd_pred = _predictor().predict_rsvd_foldin(vec)
        a = _predictor().alpha
        scores = np.clip(a * vae_pred + (1.0 - a) * rsvd_pred, 1.0, MAX_RATING)
    elif _is_new_user(user_id):                       # pengguna baru: cold-start
        scores = _avg_scores().copy()
    else:                                             # pengguna dataset: matriks hybrid
        scores = _predictor().predictions_for_user(user_id).copy()

    scores = np.array(scores, dtype=np.float32, copy=True)
    scores[_excluded_mask(user_id)] = -np.inf         # buang film yang sudah dirating
    return scores
```

---

## 8. Rekomendasi Berdasarkan Genre

*Endpoint* `GET /api/users/{id}/recommendations/by-genre` menyaring skor
pengguna per genre, lalu mengembalikan film terbaik untuk masing-masing genre.

*Berkas: `App/backend/app.py`*

```python
@app.get("/api/users/{user_id}/recommendations/by-genre")
def user_recommendations_by_genre(user_id: int, limit: int = 20):
    _ensure_user(user_id)
    scores = _scores_excluding_rated(user_id)
    out: dict[str, list[dict]] = {}
    for genre in SHELF_GENRES:
        mask          = _catalog().genre_mask[genre]          # film bergenre ini
        masked_scores = np.where(mask, scores, -np.inf)        # sisanya diabaikan
        top           = np.argsort(masked_scores)[::-1][:limit]
        cards = [
            _catalog().movie_card(int(idx + 1), rating=float(scores[idx]))
            for idx in top
            if np.isfinite(scores[idx])
        ]
        if cards:
            out[genre] = cards
    return out
```

---

## 9. Menyimpan Rating dan Proses Fold-in

*Endpoint* `POST /api/users/{id}/ratings` menyimpan rating yang diberikan
pengguna lewat aplikasi. Rating ini kemudian ikut diperhitungkan saat menghitung
rekomendasi melalui proses *fold-in*.

*Berkas: `App/backend/app.py`*

```python
@app.post("/api/users/{user_id}/ratings")
def add_rating(user_id: int, req: RatingRequest):
    _ensure_user(user_id)
    if req.item_id not in _catalog().items.index:
        raise HTTPException(status_code=404, detail=f"Movie {req.item_id} tidak ditemukan")
    if not (1.0 <= req.rating <= MAX_RATING):
        raise HTTPException(status_code=400, detail="Rating harus antara 1 dan 5")
    _user_ratings().set(user_id, req.item_id, req.rating)
    return _catalog().movie_card(req.item_id, rating=req.rating)
```

**Fold-in** memungkinkan rating baru langsung memengaruhi prediksi **tanpa
melatih ulang model**. Pada VAE, vektor rating pengguna cukup dilewatkan sekali
melalui *encoder* dan *decoder*. Pada RSVD, vektor laten pengguna diestimasi
ulang lewat regresi *ridge* atas faktor film yang sudah tetap.

*Berkas: `App/backend/inference.py`*

```python
def predict_vae_foldin(self, rating_vec_norm: np.ndarray) -> np.ndarray:
    """Prediksi VAE untuk vektor rating apa pun (satu lintasan encoder->decoder)."""
    import tensorflow as tf
    x = tf.constant(rating_vec_norm[None, :], dtype=tf.float32)
    z_mean, _ = self.vae.encoder(x, training=False)
    pred_norm = self.vae.decoder(z_mean, training=False)
    pred = np.asarray(pred_norm)[0] * MAX_RATING
    return np.clip(pred, 1.0, MAX_RATING).astype(np.float32)


def predict_rsvd_foldin(self, rating_vec_norm: np.ndarray) -> np.ndarray:
    """Prediksi RSVD via fold-in: estimasi bias b_u dan vektor laten U[u]."""
    rated = np.where(rating_vec_norm > 0)[0]
    if rated.size == 0:                        # belum ada rating -> baseline bias
        base = self.rsvd_mu + self.rsvd_bi
        return np.clip(base * MAX_RATING, 1.0, MAX_RATING).astype(np.float32)

    X  = self.rsvd_item_factors[rated]
    y  = rating_vec_norm[rated] - self.rsvd_mu - self.rsvd_bi[rated]
    Xa = np.concatenate([np.ones((rated.size, 1), dtype=X.dtype), X], axis=1)
    reg = np.eye(Xa.shape[1], dtype=X.dtype)
    reg[0, 0] = 0.0                            # intercept tidak diregularisasi
    w = np.linalg.solve(Xa.T @ Xa + self.rsvd_foldin_lambda * reg, Xa.T @ y)
    b_u, u_vec = w[0], w[1:]
    pred_norm = self.rsvd_mu + b_u + self.rsvd_bi + self.rsvd_item_factors @ u_vec
    return np.clip(pred_norm * MAX_RATING, 1.0, MAX_RATING).astype(np.float32)
```

---

## 10. Pemanggilan API dari Frontend

Di sisi *frontend*, seluruh komunikasi dengan *backend* dikumpulkan dalam satu
berkas. Setiap fungsi mengirim permintaan HTTP dan mengembalikan data JSON.

*Berkas: `App/frontend/src/lib/api.ts`*

```typescript
export const API_BASE = "http://localhost:8000";

// Login dengan username + password
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

// Mengambil rekomendasi personal
export function fetchRecommendations(userId: number, limit = 50) {
  return getJson<MovieCard[]>(`/api/users/${userId}/recommendations?limit=${limit}`);
}

// Mengirim rating baru
export async function rateMovie(userId: number, itemId: number, rating: number): Promise<MovieCard> {
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
```

---

## 11. Komponen Shelf Rekomendasi

Komponen `Shelf` menampilkan satu deret film (poster) yang dapat digeser
horizontal, lengkap dengan tombol panah navigasi. Setiap film dirender melalui
komponen `MovieCard`.

*Berkas: `App/frontend/src/app/components/Shelf.tsx`*

```tsx
export default function Shelf({ title, movies, showRating, userId, onRated }: Props) {
  const scrollerRef = useRef<HTMLDivElement>(null);

  // Geser satu layar penuh ke kiri/kanan
  function scrollBy(direction: 1 | -1) {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollBy({ left: direction * el.clientWidth * 0.9, behavior: "smooth" });
  }

  return (
    <section className="group/shelf relative py-4">
      <h2 className="mb-3 px-6 text-xl font-semibold text-white">{title}</h2>

      <button type="button" aria-label="Scroll left" onClick={() => scrollBy(-1)}>‹</button>
      <button type="button" aria-label="Scroll right" onClick={() => scrollBy(1)}>›</button>

      <div ref={scrollerRef} className="shelf-scroll flex snap-x gap-3 overflow-x-auto px-6">
        {movies.map((m) => (
          <MovieCard
            key={m.movie_id}
            movie={m}
            showRating={showRating}
            userId={userId}
            onRated={onRated}
          />
        ))}
      </div>
    </section>
  );
}
```

---

## 12. Modal Detail Film dan Fitur Rating

Komponen `MovieModal` menampilkan detail film (poster, sinopsis, genre) dan
menyediakan pilihan bintang 1–5. Ketika bintang dipilih, rating dikirim ke
*backend* melalui fungsi `rateMovie`, lalu daftar rekomendasi diperbarui saat
modal ditutup.

*Berkas: `App/frontend/src/app/components/MovieModal.tsx`*

```tsx
// Mengirim rating ke backend saat bintang dipilih
async function submitRating(star: number) {
  if (userId == null || saving) return;
  setSaving(true);
  setRateError(null);
  try {
    await rateMovie(userId, movie.movie_id, star);
    setRating(star);
    setSaved(true);
    ratedDuringSession.current = true;
  } catch (err) {
    setRateError(err instanceof Error ? err.message : "Gagal menyimpan rating");
  } finally {
    setSaving(false);
  }
}
```

```tsx
{/* Pilihan bintang 1-5 di dalam modal */}
<div className="flex items-center gap-1">
  {[1, 2, 3, 4, 5].map((star) => {
    const active = (hover ?? rating ?? 0) >= star;
    return (
      <button
        key={star}
        type="button"
        disabled={saving}
        onMouseEnter={() => setHover(star)}
        onMouseLeave={() => setHover(null)}
        onClick={() => submitRating(star)}
        className={active ? "text-amber-400" : "text-zinc-600"}
      >
        ★
      </button>
    );
  })}
  {rating != null && <span className="ml-2 text-sm text-zinc-300">{rating}/5</span>}
</div>
```

Setelah modal ditutup, aplikasi memanggil ulang data sehingga film yang baru
dirating berpindah ke daftar "Already Rated" dan rekomendasi ikut menyesuaikan
tanpa memuat ulang halaman.

```tsx
function handleClose() {
  setVisible(false);
  window.setTimeout(() => {
    onClose();
    // Perbarui shelf hanya bila ada rating baru selama modal terbuka
    if (ratedDuringSession.current) onRated?.();
  }, ANIM_MS);
}
```
