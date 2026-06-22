# =============================================================================
# Skrip untuk menghasilkan dua tabel contoh hasil rekomendasi (Sub-bab 4.7):
#   Tabel 1 : Verifikasi akurasi pada film yang sudah dinilai pengguna contoh
#             (sebaran beberapa film per level rating 5..1)
#   Tabel 2 : Top-N rekomendasi film yang belum ditonton (judul + genre)
#
# Mengikuti pipeline prediksi pada Testing.ipynb. Jalankan dari folder Notebooks
# (sama seperti Testing.ipynb), atau sesuaikan path pada bagian KONFIGURASI.
# =============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# ============================ KONFIGURASI ====================================
scripts_path = os.path.abspath(os.path.join('..', 'Programs', 'Scripts'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

data_dir            = os.path.abspath(os.path.join('..', 'Programs', 'Data', 'TrainTest'))
save_dir            = os.path.abspath(os.path.join('..', 'Programs', 'Data', 'SavedModels'))
hyperparameters_dir = os.path.abspath(os.path.join('..', 'Programs', 'Data', 'HyperParameters'))
uitem_path          = os.path.abspath(os.path.join('..', 'Programs', 'Data', 'ml-100k', 'u.item'))

MAX_RATING      = 5.0
example_user_id = 655   # user_id (1-based) pengguna contoh; ganti sesuai pilihan
k_per_rating    = 2     # jumlah film per level rating asli pada Tabel 1
n_rekomendasi   = 10    # jumlah rekomendasi Top-N pada Tabel 2

# ============================ 1. LOAD DATA ===================================
train_data = np.load(os.path.join(data_dir, 'train_data.npy')).astype(np.float32)
val_data   = np.load(os.path.join(data_dir, 'val_data.npy')).astype(np.float32)
test_data  = np.load(os.path.join(data_dir, 'test_data.npy')).astype(np.float32)
num_users, num_items = train_data.shape

# (Opsional) Kandidat pengguna dengan rating test terbanyak, untuk membantu memilih
jumlah_rating_test = (test_data > 0).sum(axis=1)
kandidat = np.argsort(jumlah_rating_test)[::-1][:10]
print("[INFO] Kandidat pengguna contoh (user_id : jumlah rating test):")
for i in kandidat:
    print(f"        user_id {i + 1:>3} : {int(jumlah_rating_test[i])}")

# ============================ 2. LOAD & PREDIKSI VAE =========================
from model import Encoder, Decoder, VAE

with open(os.path.join(hyperparameters_dir, 'tuning_progress_vae.json'), 'r') as f:
    best_vae_params = json.load(f)['best_params']

encoder = Encoder(hidden_dims=best_vae_params['hidden_dims'],
                  latent_dim=best_vae_params['latent_dim'],
                  dropout_rate=best_vae_params['dropout_rate'])
decoder = Decoder(hidden_dims=best_vae_params['hidden_dims'][::-1], output_dim=num_items)
eval_vae = VAE(encoder, decoder, beta=best_vae_params['beta'])
_ = eval_vae(train_data[:1])  # membangun graph sebelum memuat bobot
eval_vae.load_weights(os.path.join(save_dir, 'trained_best_vae_weights.weights.h5'))

# Prediksi VAE lewat z_mean (deterministik), sama seperti Testing.ipynb (skala 0-1)
z_mean, _     = eval_vae.encoder.predict(tf.constant(train_data, dtype=tf.float32), verbose=0)
pred_vae_norm = eval_vae.decoder.predict(z_mean, verbose=0)

# ============================ 3. LOAD & PREDIKSI RSVD ========================
mu    = np.load(os.path.join(save_dir, 'final_mu.npy'))
b_u   = np.load(os.path.join(save_dir, 'final_b_u.npy'))
b_i   = np.load(os.path.join(save_dir, 'final_b_i.npy'))
U     = np.load(os.path.join(save_dir, 'best_U.npy'))
Sigma = np.load(os.path.join(save_dir, 'best_Sigma.npy'))
V     = np.load(os.path.join(save_dir, 'best_V.npy'))

# Prediksi penuh RSVD (skala 0-1)
full_rsvd_pred = (float(mu) + b_u[:, np.newaxis] + b_i[np.newaxis, :]) + np.dot(np.dot(U, Sigma), V.T)

# ============================ 4. CARI BOBOT ENSEMBLE (ALPHA) =================
# Grid search pada validation set, identik dengan Testing.ipynb agar konsisten
val_indices       = np.where(val_data > 0)
actual_val_values = val_data[val_indices] * MAX_RATING
pred_vae_val      = np.clip(pred_vae_norm[val_indices]  * MAX_RATING, 1.0, MAX_RATING)
pred_rsvd_val     = np.clip(full_rsvd_pred[val_indices] * MAX_RATING, 1.0, MAX_RATING)

best_val_rmse, best_alpha = float('inf'), 0.0
for alpha in np.linspace(0, 1, 101):
    beta          = 1.0 - alpha
    pred_ensemble = np.clip(alpha * pred_vae_val + beta * pred_rsvd_val, 1.0, MAX_RATING)
    rmse          = np.sqrt(np.mean(np.square(actual_val_values - pred_ensemble)))
    if rmse < best_val_rmse:
        best_val_rmse, best_alpha = rmse, alpha
best_beta = 1.0 - best_alpha
print(f"\n[INFO] Bobot ensemble: alpha (VAE) = {best_alpha:.2f}, beta (RSVD) = {best_beta:.2f}")

# ============================ 5. MATRIKS PREDIKSI HYBRID (skala 1-5) =========
vae_clip    = np.clip(pred_vae_norm  * MAX_RATING, 1.0, MAX_RATING)
rsvd_clip   = np.clip(full_rsvd_pred * MAX_RATING, 1.0, MAX_RATING)
hybrid_full = np.clip(best_alpha * vae_clip + best_beta * rsvd_clip, 1.0, MAX_RATING)

# ============================ 6. METADATA FILM (u.item) ======================
nama_genre = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

item_df = pd.read_csv(uitem_path, sep='|', encoding='latin-1', header=None)
assert len(item_df) == num_items, "Jumlah film di u.item tidak sama dengan jumlah kolom matriks!"

# Pemetaan indeks kolom matriks (0-based) -> judul & genre.
# item_id MovieLens 100K kontigu 1..1682, sehingga kolom j berkorespondensi dengan item_id (j+1).
judul_per_kolom, genre_per_kolom = {}, {}
for _, row in item_df.iterrows():
    kolom = int(row[0]) - 1
    judul_per_kolom[kolom] = row[1]
    genre_aktif = [nama_genre[k] for k in range(19)
                   if int(row[5 + k]) == 1 and nama_genre[k] != 'unknown']
    genre_per_kolom[kolom] = ", ".join(genre_aktif) if genre_aktif else "-"

# ============================ 7. TABEL 1: VERIFIKASI AKURASI =================
# Sebaran beberapa film per level rating (5..1) agar mewakili seluruh rentang
baris_user = example_user_id - 1   # user_id (1-based) -> indeks baris (0-based)
film_test  = np.where(test_data[baris_user] > 0)[0]

# Rata-rata selisih absolut atas SELURUH rating test pengguna (konteks caption)
asli_semua   = test_data[baris_user, film_test] * MAX_RATING
pred_semua   = hybrid_full[baris_user, film_test]
mae_pengguna = float(np.mean(np.abs(asli_semua - pred_semua)))

rows_akurasi = []
for nilai in [5, 4, 3, 2, 1]:
    film_level = [j for j in film_test
                  if round(float(test_data[baris_user, j] * MAX_RATING)) == nilai]
    for j in film_level[:k_per_rating]:
        asli     = round(float(test_data[baris_user, j] * MAX_RATING), 1)
        prediksi = round(float(hybrid_full[baris_user, j]), 2)
        rows_akurasi.append([judul_per_kolom[j], asli, prediksi, round(abs(asli - prediksi), 2)])

df_akurasi = pd.DataFrame(
    rows_akurasi, columns=['Judul Film', 'Rating Asli', 'Prediksi Hybrid', 'Selisih'])

# ============================ 8. TABEL 2: TOP-N REKOMENDASI ==================
# Film "sudah dilihat" (pernah dinilai di train/val/test) dikecualikan dari rekomendasi
sudah_dilihat = (train_data[baris_user] > 0) | (val_data[baris_user] > 0) | (test_data[baris_user] > 0)
skor          = hybrid_full[baris_user].copy()
skor[sudah_dilihat] = -np.inf

idx_top = np.argsort(skor)[::-1][:n_rekomendasi]
rows_rekomendasi = []
for peringkat, j in enumerate(idx_top, start=1):
    rows_rekomendasi.append([
        peringkat,
        judul_per_kolom[j],
        genre_per_kolom[j],
        round(float(hybrid_full[baris_user, j]), 2),
    ])

df_rekomendasi = pd.DataFrame(
    rows_rekomendasi, columns=['Peringkat', 'Judul Film', 'Genre', 'Prediksi Rating'])

# ============================ 9. TAMPILKAN HASIL (untuk skripsi) =============
print(f"\n=== Pengguna contoh: user_id {example_user_id} "
      f"({int(np.sum(test_data[baris_user] > 0))} rating di test set) ===")

print("\n--- Tabel 1: Verifikasi Akurasi pada Film yang Sudah Dinilai ---")
print(df_akurasi.to_string(index=False))
print(f"Rata-rata selisih absolut (seluruh {len(film_test)} rating test pengguna ini): {mae_pengguna:.2f}")

print(f"\n--- Tabel 2: Top-{n_rekomendasi} Rekomendasi Film yang Belum Ditonton ---")
print(df_rekomendasi.to_string(index=False))

# ============================ 10. DIAGNOSTIK (opsional, tidak wajib di skripsi)
# Rata-rata selisih absolut per level rating: memperlihatkan kompresi prediksi
# (error besar di rating ekstrem 1 & 5, kecil di rating tengah).
df_semua = pd.DataFrame({
    'Rating Asli': np.round(asli_semua, 1),
    'Selisih': np.abs(asli_semua - pred_semua),
})
print("\n[DIAGNOSTIK] Rata-rata selisih absolut per level rating:")
print(df_semua.groupby('Rating Asli')['Selisih'].mean().round(2).to_string())

print(item_df)