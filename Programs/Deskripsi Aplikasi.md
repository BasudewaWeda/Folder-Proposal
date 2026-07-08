# MovieMatch — Deskripsi Aplikasi & Panduan Fitur

> Dokumen ini merangkum **cara instalasi** dan **seluruh fitur** aplikasi
> MovieMatch. Dipakai sebagai bahan sumber untuk menyusun *manual book* /
> buku panduan pengguna pada dokumen tugas akhir.

---

## 1. Gambaran Umum

**MovieMatch** adalah aplikasi web sistem rekomendasi film yang berjalan secara
lokal. Aplikasi menampilkan rekomendasi film bergaya *streaming platform*
(mirip Netflix): deretan poster film yang bisa digeser, dikelompokkan per
kategori, lengkap dengan halaman detail dan fitur memberi rating.

Rekomendasi dihasilkan oleh model *machine learning* **hybrid VAE + RSVD**
yang dilatih pada dataset **MovieLens 100K** (943 pengguna, 1.682 film,
100.000 rating). Model memprediksi rating yang kemungkinan diberikan seorang
pengguna untuk film-film yang belum ia tonton, lalu film dengan prediksi
tertinggi ditampilkan sebagai rekomendasi.

Aplikasi terdiri dari dua bagian:

| Bagian    | Teknologi                        | Peran                                          |
|-----------|----------------------------------|------------------------------------------------|
| Backend   | Python, FastAPI, TensorFlow, NumPy, pandas | Memuat model, menghitung prediksi, menyediakan API |
| Frontend  | Next.js (React), TypeScript, Tailwind CSS | Antarmuka pengguna di browser                  |

Backend berjalan di `http://localhost:8000`, frontend di
`http://localhost:3000`. Frontend mengambil data dari backend, jadi backend
harus dinyalakan lebih dulu.

---

## 2. Cara Instalasi & Menjalankan

Semua perintah dijalankan lewat PowerShell / Terminal dari dalam folder
`Programs`, kecuali disebutkan lain.

### 2.1 Prasyarat

- **Python 3.11** (yang dipakai: 3.11.8)
- **Node.js 18+** (yang dipakai: versi 22)

### 2.2 Setup Python (virtual environment)

```powershell
# Buat dan aktifkan virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # PowerShell
# atau: venv\Scripts\activate.bat  # CMD

# Jika PowerShell menolak karena execution policy, jalankan sekali:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Install seluruh dependency (agak lama karena ada TensorFlow)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kalau venv aktif, akan ada tulisan `(venv)` di depan baris terminal.

### 2.3 (Opsional) Melatih Ulang Model

Model hasil training sudah tersimpan di folder `Data`, jadi langkah ini
**tidak wajib** untuk menjalankan aplikasi. Urutan notebook jika ingin melatih
dari awal (folder `Notebooks`):

1. `DataPreprocessing.ipynb` — menyiapkan & membagi data train/test
2. `HyperParameterTuningOptuna.ipynb` — mencari parameter terbaik (opsional, lama)
3. `Training.ipynb` — melatih model, menyimpan bobot
4. `CrossValidation.ipynb` — validasi (opsional)
5. `Testing.ipynb` — evaluasi akhir

### 2.4 Menjalankan Backend (FastAPI)

```powershell
cd App\backend
# pastikan venv aktif
uvicorn app:app --reload --port 8000
```

- API berjalan di `http://localhost:8000`
- Dokumentasi API interaktif: `http://localhost:8000/docs`
- Loading pertama agak lama karena model dihitung terlebih dahulu. Biarkan
  terminal ini tetap terbuka selama aplikasi dipakai.

### 2.5 Menjalankan Frontend (Next.js)

Buka terminal **baru** (jangan yang dipakai backend):

```powershell
cd App\frontend
npm install        # cukup sekali di awal
npm run dev
```

Buka `http://localhost:3000` di browser.

### 2.6 Ringkasan Urutan Menyalakan

```
Terminal 1  ->  cd App\backend   ->  aktifkan venv  ->  uvicorn app:app --reload --port 8000
Terminal 2  ->  cd App\frontend  ->  npm run dev
```

Lalu buka `http://localhost:3000`.

---

## 3. Akun & Cara Login

Aplikasi login menggunakan **username + password**. Ada dua jenis akun:

### 3.1 Akun Pengguna Dataset (bawaan MovieLens 100K)

- **Username:** `User_1` sampai `User_943` (contoh: `User_42`)
- **Password:** `movielens` (sama untuk semua pengguna dataset)

Pengguna ini sudah memiliki histori rating dari dataset, sehingga rekomendasinya
dihasilkan langsung oleh model hybrid VAE + RSVD.

### 3.2 Akun Baru (didaftarkan sendiri)

- Dibuat lewat halaman **Daftar**. Pengguna mengisi username, umur, gender,
  pekerjaan, dan password.
- **Username** minimal 3 karakter, tidak boleh sama dengan yang sudah ada, dan
  tidak boleh berformat `User_<angka>` (dicadangkan untuk pengguna dataset).
- **Password** minimal 4 karakter.
- Sistem memberi ID pengguna otomatis (mulai dari 944) di belakang layar —
  pengguna cukup mengingat username & password.
- Karena belum punya histori rating, awalnya ditampilkan film dengan rating
  rata-rata tertinggi (*cold-start*).

> **Keamanan:** password disimpan sebagai *hash* (PBKDF2-SHA256 + salt), bukan
> teks polos. Ini autentikasi tingkat demo untuk keperluan tugas akhir (belum
> menggunakan token/sesi).

---

## 4. Daftar Fitur

### 4.1 Registrasi Akun (`/register`)

- Formulir pendaftaran berisi: **Username, Umur, Gender, Pekerjaan, Password,
  Konfirmasi Password**.
- Pilihan **Pekerjaan** mengikuti kosakata MovieLens (administrator, artist,
  doctor, educator, engineer, programmer, student, dst. — 21 pilihan).
- Validasi di sisi frontend: username ≥ 3 karakter, password ≥ 4 karakter, dan
  konfirmasi password harus cocok.
- Validasi di sisi backend: username unik, bukan format `User_<angka>`, password
  ≥ 4 karakter.
- Setelah berhasil, pengguna langsung masuk (login otomatis) dan diarahkan ke
  beranda.

### 4.2 Login & Logout (`/login`)

- Login memakai **username + password** (lihat Bagian 3).
- Sesi disimpan di `localStorage` browser (`user_id` dan `user_info`), sehingga
  pengguna tetap login saat halaman dimuat ulang.
- Pesan error yang jelas: "Password salah", "Username tidak ditemukan", dll.
- Tombol **Logout** di kanan atas menghapus sesi dan kembali ke halaman login.
- Halaman yang butuh login otomatis mengalihkan ke `/login` bila belum masuk.

### 4.3 Beranda / Halaman Rekomendasi (`/`)

Tampilan utama bergaya platform streaming, terdiri dari:

- **Navbar** — nama aplikasi, username + pekerjaan pengguna, dan tombol Logout.
- **Hero / sapaan** — "Welcome back, `<username>`" (pengguna dataset) atau
  "Selamat datang, `<username>`" (pengguna baru), disertai ringkasan
  demografi (umur, gender, pekerjaan) dan jumlah film yang sudah dirating.
- Beberapa **shelf** (deretan poster horizontal), lihat 4.4–4.6.

### 4.4 Rekomendasi Personal ("Top Recommended for You")

- Shelf berisi film dengan **prediksi rating tertinggi** untuk pengguna,
  dihasilkan model hybrid VAE + RSVD.
- Film yang **sudah pernah dirating** pengguna dikecualikan dari rekomendasi.
- Untuk pengguna baru (cold-start), judul shelf berubah menjadi
  **"Film dengan Rating Tertinggi"**, berisi film dengan rata-rata rating
  tertinggi (minimal 50 vote agar tidak bias film obscure).

### 4.5 Rekomendasi per Genre ("Best in ...")

- Untuk setiap genre, ditampilkan shelf tersendiri berisi film terbaik menurut
  prediksi model pada genre tersebut.
- Genre mengikuti 18 kategori MovieLens: Action, Adventure, Animation,
  Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror,
  Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western.

### 4.6 Film yang Sudah Dirating ("Already Rated")

- Shelf berisi seluruh film yang sudah dirating pengguna, diurutkan dari rating
  tertinggi.
- Menggabungkan rating asli dari dataset (untuk pengguna dataset) dengan rating
  yang diberikan lewat aplikasi (rating dari aplikasi menang bila ada konflik).

### 4.7 Kartu Film & Poster

- Tiap film ditampilkan sebagai **kartu poster** (rasio 2:3) dengan judul,
  tahun, dan hingga 2 genre.
- Poster diambil dari cache lokal (`posters_cache.json`, sumber TMDB). Bila
  poster tidak tersedia, ditampilkan judul film sebagai pengganti.
- Badge angka di pojok poster menampilkan **prediksi rating** (mis. `4.37`)
  atau **rating bintang** (mis. `★ 5.0`) pada shelf "Already Rated".
- Efek *hover*: kartu sedikit membesar dan ada highlight.

### 4.8 Navigasi Shelf

- Setiap shelf bisa **digeser horizontal** (scroll).
- Pada layar lebar muncul tombol panah **‹ / ›** saat kursor diarahkan ke shelf,
  untuk menggeser satu layar penuh.

### 4.9 Detail Film (Modal)

- Klik kartu film membuka **modal detail** dengan animasi *fade + zoom*.
- Menampilkan: poster besar, judul, tahun, seluruh genre, **sinopsis/overview**,
  dan prediksi/rating aktual.
- Bisa ditutup lewat tombol **×**, klik area gelap di luar modal, atau tekan
  **Esc**. Latar belakang halaman dikunci agar tidak ikut ter-scroll.

### 4.10 Memberi Rating (1–5 Bintang)

- Di dalam modal detail terdapat pilihan **bintang 1–5** untuk memberi rating.
- Rating tersimpan ke backend dan langsung memberi umpan balik ("Menyimpan…",
  lalu "✓ Rating tersimpan").
- Rating memakai interaksi *hover* untuk pratinjau jumlah bintang.

### 4.11 Rekomendasi Menyesuaikan Rating Baru (Real-time Fold-in)

Ini fitur inti yang membedakan aplikasi dari sekadar tabel statis:

- Saat pengguna memberi rating baru, backend melakukan **fold-in**: rating baru
  langsung dimasukkan ke vektor rating pengguna dan model **menghitung ulang
  prediksi** tanpa perlu melatih ulang model.
- Setelah modal ditutup, shelf otomatis diperbarui: film yang baru dirating
  pindah ke "Already Rated" dan daftar rekomendasi menyesuaikan — tanpa reload
  halaman.
- Fold-in berlaku baik untuk pengguna dataset maupun pengguna baru; semakin
  banyak film yang dirating, rekomendasi makin sesuai selera.

---

## 5. Cara Kerja Mesin Rekomendasi (Ringkas)

Untuk keperluan manual book, cukup dijelaskan pada tingkat konsep:

- **RSVD (Regularized SVD)** — model faktorisasi matriks yang mempelajari
  vektor laten pengguna & film beserta bias, memprediksi rating berdasarkan pola
  kesamaan selera antar pengguna.
- **VAE (Variational Autoencoder)** — jaringan saraf yang meng-*encode* seluruh
  vektor rating seorang pengguna menjadi representasi laten, lalu me-*decode*-nya
  kembali menjadi prediksi rating untuk semua film.
- **Ensemble (Hybrid)** — prediksi akhir adalah gabungan berbobot kedua model:
  `prediksi = α · VAE + (1 − α) · RSVD`, dengan bobot `α` hasil penyetelan pada
  data evaluasi.
- Saat backend menyala, seluruh matriks prediksi `943 × 1.682` dihitung sekali
  di awal, sehingga tiap permintaan rekomendasi hanya membaca satu baris array
  (cepat).
- **Fold-in** memungkinkan pengguna di luar 943 pengguna asli (atau pengguna
  asli yang menambah rating baru) tetap memperoleh prediksi tanpa melatih ulang
  model: vektor laten pengguna diestimasi ulang dari rating yang tersedia.

---

## 6. Penyimpanan Data

Model bersifat statis, tetapi aktivitas pengguna disimpan agar bertahan setelah
backend di-restart:

| File                              | Isi                                                        |
|-----------------------------------|------------------------------------------------------------|
| `App/backend/new_users.json`      | Data pengguna baru hasil registrasi (termasuk hash password) |
| `App/backend/user_ratings.json`   | Rating yang diberikan lewat aplikasi (semua pengguna)      |
| `App/backend/posters_cache.json`  | Cache URL poster + sinopsis film (sumber TMDB)             |
| `Data/ml-100k/`                   | Dataset MovieLens 100K asli (`u.data`, `u.item`, `u.user`) |
| `Data/SavedModels/`               | Bobot model VAE & komponen RSVD hasil training             |

---

## 7. Ringkasan API Backend

Semua endpoint berbasis JSON. Dokumentasi interaktif tersedia di
`http://localhost:8000/docs`.

| Method & Endpoint                                   | Fungsi                                             |
|-----------------------------------------------------|----------------------------------------------------|
| `POST /api/login`                                   | Login dengan username + password                   |
| `POST /api/register`                                | Mendaftarkan pengguna baru                         |
| `GET  /api/users/{id}/recommendations`              | Rekomendasi personal (prediksi tertinggi)          |
| `GET  /api/users/{id}/recommendations/by-genre`     | Rekomendasi dikelompokkan per genre                |
| `GET  /api/users/{id}/rated`                        | Daftar film yang sudah dirating pengguna           |
| `POST /api/users/{id}/ratings`                      | Menyimpan rating baru (memicu fold-in)             |

---

## 8. Struktur Folder (Ringkas)

```
Programs/
├─ App/
│  ├─ backend/          # FastAPI: app.py, inference.py, data.py, *_store.py
│  └─ frontend/         # Next.js: src/app (halaman & komponen), src/lib/api.ts
├─ Data/                # dataset MovieLens, model tersimpan, hasil evaluasi
├─ Notebooks/           # notebook preprocessing, training, evaluasi
├─ Scripts/             # definisi model (VAE) & utilitas
└─ requirements.txt     # dependency Python
```

---

## 9. Catatan untuk Penyusunan Manual Book

- Aplikasi ini adalah **demo lokal untuk tugas akhir**, bukan produk publik.
- Bagian yang layak dijadikan tangkapan layar di manual book: halaman Login,
  halaman Daftar, beranda dengan shelf rekomendasi, modal detail film, dan
  interaksi memberi rating bintang.
- Alur pengguna yang disarankan untuk didemokan di manual book:
  1. Login sebagai pengguna dataset (mis. `User_42` / `movielens`) untuk melihat
     rekomendasi personal yang kaya.
  2. Mendaftar akun baru untuk memperlihatkan kondisi *cold-start*.
  3. Memberi beberapa rating dan menunjukkan rekomendasi berubah secara langsung.
