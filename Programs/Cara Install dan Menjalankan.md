# Cara Install dan Menjalankan Program

> Catatan singkat buat nyiapin environment, jalanin notebook/script Python,
> sama nyalain aplikasi (backend + frontend). Semua perintah di bawah
> dijalankan lewat PowerShell / Terminal dari dalam folder `Programs` kecuali
> disebutkan lain.

**Yang perlu sudah terpasang di laptop:**

- **Python 3.11** (yang dipakai di sini 3.11.8)
- **Node.js 18** ke atas (dipakai versi 22)

---

## 1. Setup Python (venv)

Buka terminal di folder `Programs` ini, terus buat virtual environment:

```powershell
python -m venv venv
```

Aktifkan venv-nya:

```powershell
.\venv\Scripts\Activate.ps1        # PowerShell
venv\Scripts\activate.bat          # CMD
```

Kalau di PowerShell muncul error soal execution policy, jalankan ini dulu
sekali aja lalu ulangi perintah activate di atas:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Kalau venv sudah aktif, di depan baris terminal bakal ada tulisan `(venv)`.
Sekarang install semua library yang dibutuhin:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Install-nya lumayan lama karena ada TensorFlow. Sabar aja.

---

## 2. Jalanin Notebook / Script Python

Pastikan venv masih aktif (lihat tulisan `(venv)` di terminal). Buka Jupyter:

```powershell
jupyter notebook
```

atau kalau pakai VS Code, tinggal buka file `.ipynb` di folder `Notebooks` dan
pilih kernel-nya yang dari venv tadi.

Urutan notebook kalau mau jalanin dari awal:

1. `DataPreprocessing.ipynb` — nyiapin & bagi data train/test
2. `HyperParameterTuningOptuna.ipynb` — cari parameter terbaik (opsional, lama)
3. `Training.ipynb` — latih model, simpan bobotnya
4. `CrossValidation.ipynb` — validasi (opsional)
5. `Testing.ipynb` — evaluasi hasil akhir

Model hasil training dan file pendukung lain sudah tersimpan di folder `Data`,
jadi kalau cuma mau nyalain aplikasi, langkah training ini nggak wajib diulang.

Buat script biasa (`.py`) tinggal dijalanin langsung, contoh:

```powershell
python Notebooks\GenerateTabelRekomendasi.py
```

---

## 3. Jalanin Backend (FastAPI)

Backend baca model dari folder `Data`, jadi struktur folder jangan diubah.
Masuk ke foldernya:

```powershell
cd App\backend
```

Pastikan venv-nya aktif, lalu jalanin server:

```powershell
uvicorn app:app --reload --port 8000
```

- Backend jalan di `http://localhost:8000`
- Dokumentasi API-nya bisa dibuka di `http://localhost:8000/docs`

> Biarin terminal ini tetap kebuka selama aplikasi dipakai. Loading pertama
> agak lama karena model-nya dihitung dulu.

---

## 4. Jalanin Frontend (Next.js)

Buka terminal **BARU** (jangan yang dipakai backend), masuk ke folder frontend:

```powershell
cd App\frontend
```

Install dependency-nya (cukup sekali di awal):

```powershell
npm install
```

Jalanin:

```powershell
npm run dev
```

Frontend jalan di `http://localhost:3000`. Buka alamat itu di browser.

---

## 5. Ringkasan Urutan Nyalain Aplikasi

Kalau environment sudah pernah di-setup, tiap mau pakai aplikasi tinggal:

```
Terminal 1  ->  cd App\backend   ->  aktifkan venv  ->  uvicorn app:app --reload --port 8000
Terminal 2  ->  cd App\frontend  ->  npm run dev
```

Terus buka `http://localhost:3000` di browser.

> Frontend nyambung ke backend lewat `http://localhost:8000`, jadi backend harus
> dinyalain duluan biar datanya kebaca.

---

## 6. Cara Login / Akun

Aplikasi sekarang login pakai **username + password** (bukan ID lagi). Ada dua
jenis akun:

### a) User bawaan dataset MovieLens 100K (943 user)

- **Username:** `User_1` sampai `User_943` (contoh: `User_42`)
- **Password:** `movielens` (sama untuk semua user dataset)

User ini sudah punya histori rating, jadi rekomendasinya keluar dari model
hybrid VAE + RSVD.

### b) User baru (daftar sendiri lewat menu "Daftar di sini")

- Isi **username** (min. 3 karakter), umur, gender, pekerjaan, dan **password**
  (min. 4 karakter). Username tidak boleh sama dengan yang sudah dipakai, dan
  tidak boleh berformat `User_<angka>` karena itu dicadangkan buat user dataset.
- Setelah daftar, login pakai username & password yang tadi dibuat.

User baru belum punya histori rating, jadi awalnya ditampilkan film dengan
rating rata-rata tertinggi (*cold-start*).

> **Catatan:** password disimpan dalam bentuk hash (PBKDF2-SHA256 + salt) di file
> `App\backend\new_users.json`, bukan teks polos. Ini autentikasi tingkat demo
> untuk keperluan tugas akhir, belum pakai token/sesi.
