# --- CELL 0: DEFINISI MODEL (PERBAIKAN FINAL LOSS FUNCTION) ---
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.models import Model

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        sigma = tf.exp(0.5 * z_log_var)
        return z_mean + (sigma * epsilon)

class Encoder(Model):
    def __init__(self, hidden_dims=[512, 256], latent_dim=50, dropout_rate=0.5, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dropout = Dropout(dropout_rate)
        self.hidden_layers = [Dense(dim, activation='relu') for dim in hidden_dims]
        self.z_mean = Dense(latent_dim, name='z_mean')
        self.z_log_var = Dense(latent_dim, name='z_log_var')

    def call(self, inputs):
        x = self.dropout(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.z_mean(x), self.z_log_var(x)

class Decoder(Model):
    def __init__(self, hidden_dims=[256, 512], output_dim=1682, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.hidden_layers = [Dense(dim, activation='relu') for dim in hidden_dims]
        self.reconstruction = Dense(output_dim, activation='sigmoid', name='decoder_output')

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.reconstruction(x)

class VAE(Model):
    # Tambahkan parameter beta di init (default 1.0 seperti VAE biasa)
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # Simpan nilai beta ke dalam state model
        self.sampling = Sampling()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        return self.decoder(z)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decoder(z)

            # A. Reconstruction Loss (Masked MSE)
            mask = tf.cast(data > 0, tf.float32)
            squared_diff = tf.square(data - reconstruction)
            masked_se = squared_diff * mask
            mse_loss = tf.reduce_sum(masked_se) / (tf.reduce_sum(mask) + 1e-8)
            num_items = tf.cast(tf.shape(data)[1], tf.float32)
            reconstruction_loss = mse_loss * num_items

            # B. KL Divergence Loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # C. Total Loss dengan Hyperparameter Beta
            # Gunakan self.beta yang didapat dari inisialisasi model
            total_loss = reconstruction_loss + (self.beta * kl_loss)

        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class RSVD:
    def __init__(self, n_factors=50, learning_rate=0.001, lambda_reg=0.001, epochs=100):
        self.k = n_factors
        self.eta = learning_rate
        self.lam = lambda_reg
        self.epochs = epochs
        self.loss_history = []
        
    def fit(self, Z):
        m, n = Z.shape
        
        # Hitung rata-rata global HANYA dari rating yang memiliki nilai (> 0)
        nonzero_ratings = Z[Z > 0]
        self.mu = np.mean(nonzero_ratings) if len(nonzero_ratings) > 0 else 0
        
        self.b_u = np.zeros(m)
        self.b_i = np.zeros(n)
        
        # 1. Besarkan sedikit inisialisasi awal U dan V
        self.U = np.random.normal(scale=0.1, size=(m, self.k))
        self.V = np.random.normal(scale=0.1, size=(n, self.k))
        
        # 2. Inisialisasi Sigma dengan angka 1 (Matriks Identitas)
        self.Sigma = np.eye(self.k)
        
        for epoch in range(self.epochs):
            for u in range(m):
                for i in range(n):
                    # KUNCI UTAMA: Hanya perbarui bobot jika user benar-benar memberi rating
                    if Z[u, i] > 0: 
                        dot_product = np.dot(np.dot(self.U[u, :], self.Sigma), self.V[i, :].T)
                        pred = self.mu + self.b_u[u] + self.b_i[i] + dot_product
                        
                        e_ui = Z[u, i] - pred
                        
                        # Update Bias
                        self.b_u[u] += self.eta * (e_ui - self.lam * self.b_u[u])
                        self.b_i[i] += self.eta * (e_ui - self.lam * self.b_i[i])
                        
                        # Update Laten
                        for k in range(self.k):
                            U_uk = self.U[u, k]
                            V_ik = self.V[i, k]
                            Sigma_kk = self.Sigma[k, k]
                            
                            self.U[u, k] += self.eta * (e_ui * Sigma_kk * V_ik - self.lam * U_uk)
                            self.V[i, k] += self.eta * (e_ui * Sigma_kk * U_uk - self.lam * V_ik)
                            self.Sigma[k, k] += self.eta * (e_ui * U_uk * V_ik)

            # Evaluasi MSE (hanya pada nilai yang > 0)
            bias_matrix = self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :]
            latent_matrix = np.dot(np.dot(self.U, self.Sigma), self.V.T)
            reconstructed_Z = bias_matrix + latent_matrix
            
            # Masking saat evaluasi agar sel kosong tidak dihitung error
            mask = (Z > 0)
            current_mse = np.sum(np.square(Z[mask] - reconstructed_Z[mask])) / np.sum(mask)
            self.loss_history.append(current_mse)

            # Print progress dynamically
            print(f"Epoch {epoch+1:03d}/{self.epochs} | Training MSE: {current_mse:.6f}")