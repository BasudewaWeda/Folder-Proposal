# --- CELL 0: DEFINISI MODEL (PERBAIKAN FINAL LOSS FUNCTION) ---
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.models import Model

class Sampling(Layer):
    """
    Kelas ini mengimplementasikan 'Reparameterization Trick'.
    Tujuannya agar operasi pengambilan sampel acak (random sampling) dari 
    ruang laten tetap bisa diturunkan secara matematis (differentiable), 
    sehingga proses Backpropagation pada jaringan saraf tiruan bisa berjalan.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        # Mengambil noise acak (epsilon) dari distribusi normal standar (mean=0, std=1)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        # Menghitung standar deviasi (sigma) dari log_var
        sigma = tf.exp(0.5 * z_log_var)

        # Mengembalikan titik sampel laten: z = mean + (sigma * epsilon)
        return z_mean + (sigma * epsilon)

class Encoder(Model):
    """
    Encoder bertugas memetakan data rating asli dari pengguna (dimensi besar)
    menjadi representasi selera laten (dimensi kecil/kompak).
    """
    def __init__(self, hidden_dims=[512, 256], latent_dim=50, dropout_rate=0.5, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        # Dropout digunakan untuk mencegah overfitting dengan mematikan sebagian neuron secara acak
        self.dropout = Dropout(dropout_rate)

        # Membangun lapisan tersembunyi (Hidden Layers) secara dinamis
        self.hidden_layers = [Dense(dim, activation='relu') for dim in hidden_dims]

        # Output layer dari Encoder: Nilai Rata-rata (Mean) dan Log Varians dari distribusi ruang laten
        self.z_mean = Dense(latent_dim, name='z_mean')
        self.z_log_var = Dense(latent_dim, name='z_log_var')

    def call(self, inputs):
        x = self.dropout(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.z_mean(x), self.z_log_var(x)

class Decoder(Model):
    """
    Decoder bertugas membangun ulang (rekonstruksi) matriks laten
    kembali menjadi bentuk tebakan rating asli (dengan dimensi awal).

    Menggunakan aktivasi Sigmoid agar output berada di rentang (0, 1)
    sehingga bisa didenormalisasi ke skala rating 1-5 saat prediksi.
    Loss function di train_step VAE menggunakan Multinomial Log-Likelihood
    yang dihitung dari distribusi output sigmoid yang dinormalisasi per pengguna.
    """
    def __init__(self, hidden_dims=[256, 512], output_dim=1682, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.hidden_layers = [Dense(dim, activation='relu') for dim in hidden_dims]

        # Layer output menggunakan Sigmoid agar output berada di rentang 0-1
        self.reconstruction = Dense(output_dim, activation='sigmoid', name='decoder_output')

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.reconstruction(x)

class VAE(Model):
    """
    Model utama yang menggabungkan Encoder, Sampling, dan Decoder.

    Menggunakan Masked Multinomial Log-Likelihood sebagai reconstruction loss
    mengikuti pendekatan Liang et al. (2018) "Variational Autoencoders for
    Collaborative Filtering", namun dengan output Sigmoid (bukan Softmax) agar
    prediksi tetap dalam skala 0-1 yang bisa didenormalisasi ke rating 1-5.

    Formula reconstruction loss:
        loss = -sum( x_norm * log(sigmoid_out + eps) ) untuk sel yang ada rating
      di mana x_norm adalah distribusi rating yang sudah dinormalisasi per pengguna.
    - KL Annealing tetap digunakan via KLAnnealingCallback.
    """
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # Simpan beta sebagai tf.Variable agar nilainya bisa diperbarui secara
        # dinamis oleh KLAnnealingCallback di setiap epoch tanpa perlu rebuild model
        self.beta = tf.Variable(float(beta), trainable=False, dtype=tf.float32, name='beta')

        self.sampling = Sampling()

        # Pelacak nilai metrik selama proses training
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        # Forward pass biasa untuk tahap evaluasi/testing
        # Output Decoder adalah sigmoid (0-1), langsung bisa didenormalisasi ke skala rating
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        return self.decoder(z)

    def train_step(self, data):
        """
        Modifikasi proses training: menggunakan Masked Multinomial Log-Likelihood
        sebagai reconstruction loss, hanya menghitung loss pada sel yang ada rating.

        Langkah:
        1. Encoder menghasilkan z_mean dan z_log_var
        2. Sampling menghasilkan z via reparameterization trick
        3. Decoder menghasilkan output Sigmoid (0-1)
        4. Output Sigmoid dinormalisasi per pengguna menjadi distribusi
        5. Reconstruction loss = -sum( x_norm * log(sigmoid_out + eps) ) per pengguna,
           hanya pada item yang ada rating (masking)
        6. KL loss dihitung seperti biasa
        7. Total loss = reconstruction_loss + beta * kl_loss
        """
        if isinstance(data, tuple):
            data = data[0]  # ambil x saja, y dibuang

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decoder(z)

            # A. Reconstruction Loss (Masked Multinomial Log-Likelihood)
            # Membuat mask bernilai 1 jika rating ada, dan 0 jika kosong
            mask = tf.cast(data > 0, tf.float32)

            # Normalisasi rating per pengguna menjadi distribusi (jumlah per baris = 1)
            # Hanya item yang dirating yang dihitung - item kosong diabaikan
            masked_data = data * mask
            row_sums    = tf.reduce_sum(masked_data, axis=1, keepdims=True) + 1e-8
            x_norm      = masked_data / row_sums

            # Normalisasi output Sigmoid per pengguna menjadi distribusi
            # Ditambah eps kecil untuk mencegah log(0)
            recon_masked = reconstruction * mask
            recon_sums   = tf.reduce_sum(recon_masked, axis=1, keepdims=True) + 1e-8
            recon_norm   = recon_masked / recon_sums

            # Hitung negative log-likelihood hanya pada item yang ada rating
            # x_norm * log(recon_norm): nol otomatis untuk item yang tidak dirating
            nll_per_item = -tf.reduce_sum(x_norm * tf.math.log(recon_norm + 1e-8) * mask, axis=1)

            # Rata-rata loss di semua pengguna
            reconstruction_loss = tf.reduce_mean(nll_per_item)

            # B. KL Divergence Loss
            # Berfungsi meregulasi agar distribusi ruang laten mendekati normal Gaussian
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # C. Total Loss dengan Hyperparameter Beta
            total_loss = reconstruction_loss + (self.beta * kl_loss)

        # Backpropagation: Menghitung gradien dan memperbarui bobot (weights)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Memperbarui history loss untuk ditampilkan di progress bar
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class KLAnnealingCallback(tf.keras.callbacks.Callback):
    """
    Callback untuk menaikkan nilai beta VAE secara bertahap (linear annealing)
    dari 0.0 menuju beta_target selama fase annealing.

    Tujuan:
    - Di epoch-epoch awal, beta = 0 sehingga model fokus meminimalkan
      reconstruction loss terlebih dahulu tanpa tekanan dari KL divergence.
    - Secara bertahap, beta dinaikkan agar model mulai memperhatikan
      struktur ruang laten dan mencegah posterior collapse.
    - Setelah annealing_epochs tercapai, beta tetap di nilai beta_target.

    Jadwal annealing (linear):
        epoch 0                    → beta = 0.0
        epoch annealing_epochs - 1 → beta = beta_target
        epoch >= annealing_epochs  → beta = beta_target (konstan)
    """
    def __init__(self, beta_target, annealing_epochs):
        """
        Args:
            beta_target      : nilai beta akhir yang ingin dicapai,
                               diambil dari hasil hyperparameter tuning
            annealing_epochs : jumlah epoch untuk menaikkan beta dari 0 ke beta_target
        """
        super(KLAnnealingCallback, self).__init__()
        self.beta_target      = beta_target
        self.annealing_epochs = annealing_epochs

    def on_epoch_begin(self, epoch, logs=None):
        """
        Dipanggil otomatis oleh Keras di awal setiap epoch.
        Menghitung dan menetapkan nilai beta baru secara linear.
        """
        # Hitung proporsi kemajuan annealing (0.0 hingga 1.0)
        if self.annealing_epochs > 0:
            progress = min(epoch / self.annealing_epochs, 1.0)
        else:
            progress = 1.0

        # Hitung nilai beta saat ini secara linear
        beta_now = progress * self.beta_target

        # Perbarui tf.Variable beta di model tanpa perlu rebuild
        self.model.beta.assign(beta_now)

    def on_epoch_end(self, epoch, logs=None):
        """
        Tampilkan nilai beta aktif di akhir setiap epoch untuk monitoring.
        """
        current_beta = float(self.model.beta.numpy())
        print(f"  [KL Annealing] Epoch {epoch + 1}: beta = {current_beta:.6f} / {self.beta_target}")


class RSVD:
    """
    Implementasi algoritma faktorisasi matriks tradisional yang diperkaya dengan:
    1. Perhitungan Bias (Global, User, Item)
    2. Identitas Inisialisasi pada matriks Sigma untuk menghindari Vanishing Gradient

    Catatan Desain:
    - Regularisasi L2 diterapkan pada U, V, b_u, dan b_i.
    - Sigma TIDAK diregularisasi karena eksperimen awal menunjukkan weight collapse
      (nilai Sigma mendekati 0 secara kaskade, menghambat gradien U dan V).
    - loss_history menyimpan Total Loss = MSE + L2 Penalty (sesuai rumus 8 proposal).
    - Early stopping memantau validation MSE (bukan training MSE) untuk mendeteksi
      overfitting secara akurat — training MSE selalu turun sehingga tidak informatif.
    """
    def __init__(self, n_factors=50, learning_rate=0.001, lambda_reg=0.001, epochs=100, patience=5):
        self.k        = n_factors      # Jumlah dimensi laten
        self.eta      = learning_rate  # Kecepatan belajar (Learning rate)
        self.lam      = lambda_reg     # Penalti regularisasi untuk mencegah overfitting
        self.epochs   = epochs
        self.patience = patience       # Jumlah epoch tanpa perbaikan val MSE sebelum dihentikan
        self.loss_history = []         # Menyimpan nilai Total Loss (MSE + L2 Penalty) setiap epoch

    def fit(self, Z, val_ratio=0.1, random_seed=42):
        """
        Melatih model RSVD dengan validation split untuk early stopping.

        Args:
            Z           : np.ndarray matriks rating (users x items), nilai 0 = belum dirating
            val_ratio   : float proporsi rating yang dipakai sebagai validation set (default 10%)
            random_seed : int seed untuk reproduksibilitas pembagian validation split
        """
        m, n = Z.shape  # m = Jumlah pengguna, n = Jumlah film

        # -------------------------------------------------------
        # Validation Split
        # Pisahkan sebagian rating sebagai validation set
        # agar early stopping bisa memantau performa di data
        # yang tidak dilihat selama training
        # -------------------------------------------------------
        np.random.seed(random_seed)

        # Ambil koordinat semua rating yang ada nilainya
        rated_indices = np.argwhere(Z > 0)
        n_val         = int(len(rated_indices) * val_ratio)

        # Pilih indeks validation secara acak tanpa penggantian
        val_chosen  = np.random.choice(len(rated_indices), n_val, replace=False)
        val_idx     = rated_indices[val_chosen]
        val_rows    = val_idx[:, 0]
        val_cols    = val_idx[:, 1]

        # Simpan nilai rating asli untuk evaluasi validation
        val_true = Z[val_rows, val_cols].copy()

        # Buat training matrix: salin Z lalu hapus rating validation
        Z_train = Z.copy()
        Z_train[val_rows, val_cols] = 0.0

        print(f"[INFO] Total rating     : {len(rated_indices)}")
        print(f"[INFO] Training ratings : {len(rated_indices) - n_val}")
        print(f"[INFO] Validation ratings: {n_val:,}")

        # -------------------------------------------------------
        # Inisialisasi Parameter
        # -------------------------------------------------------

        # Hitung rata-rata global HANYA dari rating training (val sudah dihapus)
        nonzero_ratings = Z_train[Z_train > 0]
        self.mu = np.mean(nonzero_ratings) if len(nonzero_ratings) > 0 else 0

        # Bias user (pemberi rating) dan item (film)
        self.b_u = np.zeros(m)
        self.b_i = np.zeros(n)

        # Inisialisasi U dan V dengan nilai kecil
        self.U     = np.random.normal(scale=0.1, size=(m, self.k))
        self.V     = np.random.normal(scale=0.1, size=(n, self.k))

        # Inisialisasi Sigma dengan Matriks Identitas
        # Mencegah error gradient terperangkap mendekati 0 di epoch awal
        self.Sigma = np.eye(self.k)

        # Variabel untuk early stopping — memantau validation MSE
        best_val_mse  = float('inf')
        no_improve    = 0

        # Snapshot bobot terbaik (disimpan saat val MSE mencapai nilai terendah)
        best_U, best_V, best_Sigma = None, None, None
        best_b_u, best_b_i         = None, None

        # -------------------------------------------------------
        # Training Loop
        # -------------------------------------------------------
        for epoch in range(self.epochs):
            for u in range(m):
                for i in range(n):
                    # KUNCI UTAMA: Hanya perbarui bobot jika user benar-benar memberi rating
                    # Gunakan Z_train (bukan Z) agar rating validation tidak ikut dilatih
                    if Z_train[u, i] > 0:
                        # Prediksi rating: Bias Global + Bias User + Bias Item + (U * Sigma * V^T)
                        dot_product = np.dot(np.dot(self.U[u, :], self.Sigma), self.V[i, :].T)
                        pred        = self.mu + self.b_u[u] + self.b_i[i] + dot_product

                        # Hitung jarak error (Asli dikurangi Prediksi)
                        e_ui = Z_train[u, i] - pred

                        # Update Bias menggunakan Gradient Descent & L2 Regularization
                        self.b_u[u] += self.eta * (e_ui - self.lam * self.b_u[u])
                        self.b_i[i] += self.eta * (e_ui - self.lam * self.b_i[i])

                        # Update Matriks Laten
                        for k in range(self.k):
                            U_uk     = self.U[u, k]
                            V_ik     = self.V[i, k]
                            Sigma_kk = self.Sigma[k, k]

                            # Memperbarui bobot U, V, dan Sigma berdasarkan gradien error
                            self.U[u, k]     += self.eta * (e_ui * Sigma_kk * V_ik - self.lam * U_uk)
                            self.V[i, k]     += self.eta * (e_ui * Sigma_kk * U_uk - self.lam * V_ik)
                            self.Sigma[k, k] += self.eta * (e_ui * U_uk * V_ik)

            # -------------------------------------------------------
            # Evaluasi akhir epoch
            # -------------------------------------------------------

            # Rekonstruksi matriks penuh
            bias_matrix     = self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :]
            latent_matrix   = np.dot(np.dot(self.U, self.Sigma), self.V.T)
            reconstructed_Z = bias_matrix + latent_matrix

            # Hitung training MSE (hanya pada rating training, sel kosong dan val diabaikan)
            train_mask  = (Z_train > 0)
            current_mse = np.sum(np.square(Z_train[train_mask] - reconstructed_Z[train_mask])) / np.sum(train_mask)

            # Hitung validation MSE (pada rating yang disisihkan di awal)
            val_pred    = reconstructed_Z[val_rows, val_cols]
            val_mse     = np.mean(np.square(val_true - val_pred))

            # Hitung penalti regularisasi L2 sesuai rumus (8) proposal:
            # lambda * (||U||^2_F + ||V||^2_F)
            # Catatan: Sigma tidak diregularisasi (lihat Catatan Desain di docstring)
            l2_penalty   = self.lam * (np.sum(np.square(self.U)) + np.sum(np.square(self.V)))

            # Total Loss = Training MSE + L2 Penalty (rumus 8 proposal)
            current_loss = current_mse + l2_penalty
            self.loss_history.append(current_loss)

            # Print progress setiap epoch
            print(f"Epoch {epoch+1:03d}/{self.epochs} | Train MSE: {current_mse:.6f} | Val MSE: {val_mse:.6f} | L2: {l2_penalty:.6f} | Total Loss: {current_loss:.6f}")

            # -------------------------------------------------------
            # Early Stopping — pantau validation MSE
            # Validation MSE dipakai karena training MSE selalu turun
            # dan tidak mencerminkan performa di data yang belum dilihat
            # -------------------------------------------------------
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                no_improve   = 0

                # Simpan snapshot bobot terbaik saat ini
                best_U     = self.U.copy()
                best_V     = self.V.copy()
                best_Sigma = self.Sigma.copy()
                best_b_u   = self.b_u.copy()
                best_b_i   = self.b_i.copy()
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"\n[Early Stopping] Tidak ada perbaikan Val MSE selama {self.patience} epoch.")
                    print(f"[Early Stopping] Berhenti di epoch {epoch+1}. Best Val MSE: {best_val_mse:.6f}")

                    # Kembalikan bobot ke snapshot terbaik
                    self.U     = best_U
                    self.V     = best_V
                    self.Sigma = best_Sigma
                    self.b_u   = best_b_u
                    self.b_i   = best_b_i
                    break

class RSVDWithFeatures:
    """
    Varian RSVD yang diperkaya dengan fitur laten dari VAE (Feature Stacking).

    Ide utama:
    - VAE mengekstrak representasi laten (z_mean) dari matriks rating.
    - z_mean setiap pengguna digabungkan sebagai kolom tambahan ke matriks rating asli,
      sehingga RSVD bisa memanfaatkan informasi non-linear dari VAE secara langsung.
    - RSVD tetap bekerja di ruang rating (bukan ruang laten), sehingga konteks
      user-item tetap terjaga — berbeda dari arsitektur awal di mana RSVD
      memfaktorisasi ruang laten VAE secara langsung.

    Perbedaan dari RSVD biasa:
    - Input fit() adalah (Z, z_mean) bukan hanya Z.
    - Matriks Z diperluas: Z_aug = [Z | z_mean] dengan dimensi (m x (n + latent_dim)).
    - Prediksi dilakukan di Z_aug, lalu kolom laten dipotong — hanya n kolom rating
      yang diambil sebagai output prediksi akhir.

    Catatan desain:
    - Kolom laten dari z_mean tidak dikenai mask (karena selalu terisi, bukan sparse).
    - Regularisasi, early stopping, dan struktur training identik dengan RSVD biasa.
    - Sigma tidak diregularisasi (alasan sama dengan RSVD biasa — lihat docstring RSVD).
    """
    def __init__(self, n_factors=50, learning_rate=0.001, lambda_reg=0.001, epochs=100, patience=5):
        self.k        = n_factors      # Jumlah dimensi laten
        self.eta      = learning_rate  # Kecepatan belajar (Learning rate)
        self.lam      = lambda_reg     # Penalti regularisasi untuk mencegah overfitting
        self.epochs   = epochs
        self.patience = patience       # Jumlah epoch tanpa perbaikan val MSE sebelum dihentikan
        self.loss_history = []         # Menyimpan nilai Total Loss (MSE + L2 Penalty) setiap epoch
        self.n_items  = None           # Jumlah item rating asli (disimpan saat fit untuk prediksi)

    def fit(self, Z, z_mean_features, val_ratio=0.1, random_seed=42):
        """
        Melatih RSVDWithFeatures dengan matriks rating yang diperkaya fitur laten VAE.

        Args:
            Z                : np.ndarray matriks rating asli (users x items), nilai 0 = belum dirating
            z_mean_features  : np.ndarray matriks laten VAE (users x latent_dim) hasil encoder
            val_ratio        : float proporsi rating yang dipakai sebagai validation set (default 10%)
            random_seed      : int seed untuk reproduksibilitas pembagian validation split
        """
        m, n = Z.shape

        # Simpan jumlah item asli agar prediksi bisa dipotong dengan benar
        self.n_items = n

        # -------------------------------------------------------
        # Feature Stacking: Gabungkan z_mean ke matriks rating
        # Z_aug memiliki dimensi (m x (n + latent_dim))
        # Kolom 0..n-1   : rating asli (sparse)
        # Kolom n..n+d-1 : fitur laten VAE (dense, selalu terisi)
        # -------------------------------------------------------
        Z_aug = np.concatenate([Z, z_mean_features], axis=1)
        m_aug, n_aug = Z_aug.shape

        # -------------------------------------------------------
        # Validation Split — hanya dari kolom rating asli (sparse)
        # Kolom laten tidak dimasukkan ke validation karena selalu terisi
        # -------------------------------------------------------
        np.random.seed(random_seed)

        # Ambil koordinat semua rating yang ada nilainya (hanya di kolom rating asli)
        rated_indices = np.argwhere(Z > 0)
        n_val         = int(len(rated_indices) * val_ratio)

        # Pilih indeks validation secara acak tanpa penggantian
        val_chosen = np.random.choice(len(rated_indices), n_val, replace=False)
        val_idx    = rated_indices[val_chosen]
        val_rows   = val_idx[:, 0]
        val_cols   = val_idx[:, 1]

        # Simpan nilai rating asli untuk evaluasi validation
        val_true = Z[val_rows, val_cols].copy()

        # Buat training matrix: salin Z_aug lalu hapus rating validation di kolom rating asli
        Z_train = Z_aug.copy()
        Z_train[val_rows, val_cols] = 0.0

        print(f"[INFO] Total rating      : {len(rated_indices)}")
        print(f"[INFO] Training ratings  : {len(rated_indices) - n_val}")
        print(f"[INFO] Validation ratings: {n_val:,}")
        print(f"[INFO] Dimensi Z_aug     : {Z_aug.shape} (items asli: {n}, fitur laten: {z_mean_features.shape[1]})")

        # -------------------------------------------------------
        # Inisialisasi Parameter
        # -------------------------------------------------------

        # Hitung rata-rata global HANYA dari rating training (hanya kolom rating asli)
        nonzero_ratings = Z[Z > 0]
        self.mu = np.mean(nonzero_ratings) if len(nonzero_ratings) > 0 else 0

        # Bias user dan item (mencakup semua kolom Z_aug termasuk kolom laten)
        self.b_u = np.zeros(m_aug)
        self.b_i = np.zeros(n_aug)

        # Inisialisasi U dan V untuk dimensi Z_aug
        self.U     = np.random.normal(scale=0.1, size=(m_aug, self.k))
        self.V     = np.random.normal(scale=0.1, size=(n_aug, self.k))

        # Inisialisasi Sigma dengan Matriks Identitas
        self.Sigma = np.eye(self.k)

        # Variabel untuk early stopping
        best_val_mse  = float('inf')
        no_improve    = 0

        # Snapshot bobot terbaik
        best_U, best_V, best_Sigma = None, None, None
        best_b_u, best_b_i         = None, None

        # -------------------------------------------------------
        # Training Loop
        # Mask untuk kolom laten: kolom ini selalu terisi (tidak di-mask)
        # -------------------------------------------------------
        for epoch in range(self.epochs):
            for u in range(m_aug):
                for i in range(n_aug):
                    # Tentukan apakah sel ini perlu dilatih:
                    # - Kolom rating asli (i < n): hanya jika ada rating (> 0)
                    # - Kolom laten (i >= n): selalu dilatih karena selalu terisi
                    if i < n:
                        should_train = Z_train[u, i] > 0
                    else:
                        should_train = True

                    if should_train:
                        # Prediksi: Bias Global + Bias User + Bias Item + (U * Sigma * V^T)
                        dot_product = np.dot(np.dot(self.U[u, :], self.Sigma), self.V[i, :].T)
                        pred        = self.mu + self.b_u[u] + self.b_i[i] + dot_product

                        # Hitung error
                        e_ui = Z_train[u, i] - pred

                        # Update Bias menggunakan Gradient Descent & L2 Regularization
                        self.b_u[u] += self.eta * (e_ui - self.lam * self.b_u[u])
                        self.b_i[i] += self.eta * (e_ui - self.lam * self.b_i[i])

                        # Update Matriks Laten
                        for k in range(self.k):
                            U_uk     = self.U[u, k]
                            V_ik     = self.V[i, k]
                            Sigma_kk = self.Sigma[k, k]

                            self.U[u, k]     += self.eta * (e_ui * Sigma_kk * V_ik - self.lam * U_uk)
                            self.V[i, k]     += self.eta * (e_ui * Sigma_kk * U_uk - self.lam * V_ik)
                            self.Sigma[k, k] += self.eta * (e_ui * U_uk * V_ik)

            # -------------------------------------------------------
            # Evaluasi akhir epoch
            # -------------------------------------------------------

            # Rekonstruksi matriks penuh (Z_aug)
            bias_matrix     = self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :]
            latent_matrix   = np.dot(np.dot(self.U, self.Sigma), self.V.T)
            reconstructed_Z = bias_matrix + latent_matrix

            # Hitung training MSE (hanya kolom rating asli, sel tidak kosong)
            train_mask  = (Z_train[:, :n] > 0)
            current_mse = np.sum(np.square(Z_train[:, :n][train_mask] - reconstructed_Z[:, :n][train_mask])) / np.sum(train_mask)

            # Hitung validation MSE (pada kolom rating asli yang disisihkan)
            val_pred = reconstructed_Z[val_rows, val_cols]
            val_mse  = np.mean(np.square(val_true - val_pred))

            # Hitung penalti regularisasi L2
            l2_penalty   = self.lam * (np.sum(np.square(self.U)) + np.sum(np.square(self.V)))

            # Total Loss = Training MSE + L2 Penalty
            current_loss = current_mse + l2_penalty
            self.loss_history.append(current_loss)

            print(f"Epoch {epoch+1:03d}/{self.epochs} | Train MSE: {current_mse:.6f} | Val MSE: {val_mse:.6f} | L2: {l2_penalty:.6f} | Total Loss: {current_loss:.6f}")

            # -------------------------------------------------------
            # Early Stopping — pantau validation MSE
            # -------------------------------------------------------
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                no_improve   = 0

                # Simpan snapshot bobot terbaik
                best_U     = self.U.copy()
                best_V     = self.V.copy()
                best_Sigma = self.Sigma.copy()
                best_b_u   = self.b_u.copy()
                best_b_i   = self.b_i.copy()
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"\n[Early Stopping] Tidak ada perbaikan Val MSE selama {self.patience} epoch.")
                    print(f"[Early Stopping] Berhenti di epoch {epoch+1}. Best Val MSE: {best_val_mse:.6f}")

                    # Kembalikan bobot ke snapshot terbaik
                    self.U     = best_U
                    self.V     = best_V
                    self.Sigma = best_Sigma
                    self.b_u   = best_b_u
                    self.b_i   = best_b_i
                    break

    def predict(self, z_mean_features):
        """
        Menghasilkan prediksi rating untuk seluruh pengguna.

        Args:
            z_mean_features : np.ndarray matriks laten VAE (users x latent_dim)
                              yang sama digunakan saat fit() — diperlukan untuk
                              menyusun ulang Z_aug saat prediksi

        Returns:
            np.ndarray prediksi rating dengan dimensi (users x n_items)
            — kolom laten sudah dipotong, hanya rating asli yang dikembalikan
        """
        m = z_mean_features.shape[0]

        # Susun ulang bias dan rekonstruksi menggunakan bobot yang sudah dilatih
        bias_matrix     = self.mu + self.b_u[:m, np.newaxis] + self.b_i[np.newaxis, :]
        latent_matrix   = np.dot(np.dot(self.U[:m, :], self.Sigma), self.V.T)
        reconstructed   = bias_matrix + latent_matrix

        # Potong: hanya ambil kolom rating asli (0..n_items-1), buang kolom laten
        return reconstructed[:, :self.n_items]