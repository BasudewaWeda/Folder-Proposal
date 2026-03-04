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
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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
        # Membongkar tuple dari fit()
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            # 1. Forward Pass
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decoder(z)

            # --- PERBAIKAN MATEMATIS LOSS ---
            
            # A. Reconstruction Loss (Masked MSE)
            # Mask, 1 jika rating > 0, 0 jika rating == 0
            mask = tf.cast(data > 0, tf.float32)

            # Hitung Squarred Error hanya pada elemen mask
            squared_diff = tf.square(data - reconstruction)
            masked_squared_error = squared_diff * mask

            # Hitung rata-rata MSE berdasarkan jumlah elemen yang ada ratingnya
            # Ditambah epsilon (1e-8) agar tidak terjadi pembagian dengan nol
            mse_loss = tf.reduce_sum(masked_squared_error) / (tf.reduce_sum(mask) + 1e-8)

            # Kalikan dengan dimensi item (1682) agar skalanya seimbang dengan KL Loss
            num_items = tf.cast(tf.shape(data)[1], tf.float32)
            reconstruction_loss = mse_loss * num_items

            # B. KL Divergence Loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # C. Total Loss
            total_loss = reconstruction_loss + kl_loss

        # 2. Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # 3. Update Metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class RSVD:
    def __init__(self, n_factors=50, learning_rate=0.01, lambda_reg=0.1, epochs=100):
        self.k = n_factors
        self.eta = learning_rate
        self.lam = lambda_reg
        self.epochs = epochs

        # List for storing loss history during training
        self.loss_history = []
        
    def fit(self, Z):
        m, n = Z.shape
        self.U = np.random.normal(scale=1./self.k, size=(m, self.k))
        self.V = np.random.normal(scale=1./self.k, size=(n, self.k))
        self.Sigma = np.diag(np.random.normal(scale=1./self.k, size=self.k))
        
        for epoch in range(self.epochs):
            # 1. Stochastic Gradient Descent Update
            for u in range(m):
                for i in range(n):
                    # Calculate prediction for current user-item pair
                    pred = np.dot(np.dot(self.U[u, :], self.Sigma), self.V[i, :].T)
                    e_ui = Z[u, i] - pred
                    
                    # Update each latent factor
                    for k in range(self.k):
                        U_uk = self.U[u, k]
                        V_ik = self.V[i, k]
                        Sigma_kk = self.Sigma[k, k]
                        
                        # Calculate gradients
                        grad_U = -2 * e_ui * (Sigma_kk * V_ik) + 2 * self.lam * U_uk
                        grad_V = -2 * e_ui * (Sigma_kk * U_uk) + 2 * self.lam * V_ik
                        grad_Sigma = -2 * e_ui * (U_uk * V_ik) + 2 * self.lam * Sigma_kk
                        
                        # Apply gradients
                        self.U[u, k] -= self.eta * grad_U
                        self.V[i, k] -= self.eta * grad_V
                        self.Sigma[k, k] -= self.eta * grad_Sigma

            # Reconstruct the full matrix using current parameters
            reconstructed_Z = np.dot(np.dot(self.U, self.Sigma), self.V.T)
            
            # Calculate Mean Squared Error
            current_mse = np.mean(np.square(Z - reconstructed_Z))
            
            # Save to history list
            self.loss_history.append(current_mse)
            
            # Print progress dynamically
            print(f"Epoch {epoch+1:03d}/{self.epochs} | Training MSE: {current_mse:.6f}")