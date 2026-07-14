# =============================================================================
# profil_kompleksitas.py
# -----------------------------------------------------------------------------
# Mengukur data empiris untuk Tabel 8 artikel: waktu pelatihan, waktu inferensi,
# dan penggunaan memori puncak untuk VAE standalone, RSVD standalone, dan
# hybrid ensemble.
#
# CATATAN PENTING:
# - Angka waktu & memori BERGANTUNG PADA PERANGKAT KERAS. Jalankan di mesin yang
#   sama dengan yang dipakai untuk eksperimen, lalu catat spesifikasinya
#   (CPU/GPU, RAM) pada kalimat pengantar Tabel 8.
# - Jalankan dari root project (folder yang berisi model.py dan Data/).
# =============================================================================

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from model import Encoder, Decoder, VAE, RSVD, KLAnnealingCallback

# ---------------------------------------------------------------------------
# Konfigurasi terbaik (hasil hyperparameter tuning) — samakan dengan Training
# ---------------------------------------------------------------------------
VAE_PARAMS = {
    "hidden_dims": [1024, 512],
    "latent_dim": 100,
    "learning_rate": 0.001,
    "batch_size": 64,
    "dropout_rate": 0.1,
    "beta": 0.01,
}
RSVD_PARAMS = {"n_factors": 10, "learning_rate": 0.005, "lambda_reg": 0.001}

VAE_EPOCHS = 500          # sama dengan Training.ipynb
VAE_PATIENCE = 10
KL_ANNEALING_EPOCHS = 30
RSVD_EPOCHS = 100         # sama dengan Training.ipynb
RSVD_PATIENCE = 5
BEST_ALPHA = 0.25         # bobot VAE pada hybrid (dari final_testing_results)


def peak_memory_mb():
    """Kembalikan penggunaan memori puncak proses dalam MB.
    Lintas-platform: pakai resource (Linux/Unix) bila tersedia, jika tidak
    (mis. Windows) pakai psutil. Jalankan: pip install psutil"""
    try:
        import resource  # hanya ada di Linux/Unix
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        import os
        import psutil
        mem_info = psutil.Process(os.getpid()).memory_info()
        # Windows menyimpan puncak working set pada peak_wset (satuan byte)
        peak_bytes = getattr(mem_info, "peak_wset", None) or mem_info.rss
        return peak_bytes / (1024.0 * 1024.0)


def clip_scale(pred_norm):
    """Kembalikan prediksi ternormalisasi [0,1] ke skala rating 1-5 lalu klip."""
    return np.clip(pred_norm * 5.0, 1.0, 5.0)


# ---------------------------------------------------------------------------
# 1. Muat data (matriks pengguna x item; 0 = belum dirating)
# ---------------------------------------------------------------------------
train_data = np.load("Data/TrainTest/train_data.npy")
val_data = np.load("Data/TrainTest/val_data.npy")
test_data = np.load("Data/TrainTest/test_data.npy")
num_items = train_data.shape[1]

# Masukan encoder saat inferensi = riwayat rating yang diketahui model.
# Sesuaikan baris ini bila Testing.ipynb Anda memakai train+val sebagai input.
inference_input = train_data

# ---------------------------------------------------------------------------
# 2. VAE — waktu pelatihan
# ---------------------------------------------------------------------------
encoder = Encoder(hidden_dims=VAE_PARAMS["hidden_dims"],
                  latent_dim=VAE_PARAMS["latent_dim"],
                  dropout_rate=VAE_PARAMS["dropout_rate"])
decoder = Decoder(hidden_dims=VAE_PARAMS["hidden_dims"][::-1], output_dim=num_items)
vae = VAE(encoder, decoder, beta=0.0)
vae.compile(optimizer=Adam(learning_rate=VAE_PARAMS["learning_rate"]))

kl_annealing = KLAnnealingCallback(beta_target=VAE_PARAMS["beta"],
                                   annealing_epochs=KL_ANNEALING_EPOCHS)
early_stop = EarlyStopping(monitor="loss", patience=VAE_PATIENCE,
                           restore_best_weights=True)

t0 = time.perf_counter()
vae.fit(train_data, epochs=VAE_EPOCHS, batch_size=VAE_PARAMS["batch_size"],
        callbacks=[kl_annealing, early_stop], verbose=0)
vae_train_time = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# 3. VAE — waktu inferensi (seluruh pengguna, jalur deterministik z_mean)
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
z_mean, _ = encoder.predict(inference_input, verbose=0)
vae_recon = decoder.predict(z_mean, verbose=0)
vae_infer_time = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# 4. RSVD — waktu pelatihan & inferensi
# ---------------------------------------------------------------------------
rsvd = RSVD(n_factors=RSVD_PARAMS["n_factors"],
            learning_rate=RSVD_PARAMS["learning_rate"],
            lambda_reg=RSVD_PARAMS["lambda_reg"],
            epochs=RSVD_EPOCHS, patience=RSVD_PATIENCE)

t0 = time.perf_counter()
rsvd.fit(train_data)
rsvd_train_time = time.perf_counter() - t0

t0 = time.perf_counter()
rsvd_recon = (rsvd.mu + rsvd.b_u[:, None] + rsvd.b_i[None, :]
              + rsvd.U @ rsvd.Sigma @ rsvd.V.T)
rsvd_infer_time = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# 5. Hybrid — waktu inferensi (penjumlahan berbobot; overhead saja)
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
hybrid_recon = np.clip(
    BEST_ALPHA * clip_scale(vae_recon) + (1 - BEST_ALPHA) * clip_scale(rsvd_recon),
    1.0, 5.0)
hybrid_combine_time = time.perf_counter() - t0
# Waktu inferensi hybrid = inferensi kedua model + overhead penggabungan
hybrid_infer_time = vae_infer_time + rsvd_infer_time + hybrid_combine_time

# ---------------------------------------------------------------------------
# 6. Ringkasan (isi ke Tabel 8)
# ---------------------------------------------------------------------------
print("\n================ HASIL PENGUKURAN (untuk Tabel 8) ================")
print(f"Memori puncak proses         : {peak_memory_mb():.1f} MB")
print("-" * 66)
print(f"VAE   | latih {vae_train_time:8.2f} s | inferensi {vae_infer_time*1000:8.2f} ms")
print(f"RSVD  | latih {rsvd_train_time:8.2f} s | inferensi {rsvd_infer_time*1000:8.2f} ms")
print(f"Hybrid| latih {vae_train_time + rsvd_train_time:8.2f} s "
      f"| inferensi {hybrid_infer_time*1000:8.2f} ms "
      f"(overhead gabung {hybrid_combine_time*1000:.3f} ms)")
print("=" * 66)
print("Catatan: waktu pelatihan hybrid = VAE + RSVD (dua model dilatih).")
