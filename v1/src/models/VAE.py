"""
Variational Autoencoder (VAE) Implementation for Generative Modeling.

This module implements a Variational Autoencoder as described in
"Auto-Encoding Variational Bayes" by Kingma and Welling (2014). The VAE
learns a probabilistic latent space that enables generation of new samples.

Architecture Overview:
    The VAE consists of two neural networks and a sampling mechanism:
    
    1. ENCODER: Maps input images to a probability distribution in latent
       space, parameterized by mean (μ) and log-variance (log σ²).
       
       Input: Image → Conv layers → Flatten → μ and log(σ²) → Sampling → z
    
    2. DECODER: Generates images from latent samples.
       
       Input: z ∈ R^z_dim → Dense → Reshape → ConvTranspose → Image
    
    The VAE is trained with two loss components:
    - Reconstruction Loss: MSE between input and reconstruction
    - KL Divergence: Regularizes latent space to be close to N(0, I)
    
    Total Loss = r_loss_factor × Reconstruction + KL Divergence

Key Concepts:
    - Reparameterization Trick: z = μ + σ × ε, where ε ~ N(0, 1)
      Enables backpropagation through the sampling process.
    
    - KL Divergence: Forces the latent distribution q(z|x) to be
      close to the prior p(z) = N(0, I).
    
    - Latent Space Continuity: Unlike AE, VAE's latent space is smooth,
      allowing interpolation and generation.

Example Usage:
    >>> # Initialize VAE
    >>> vae = VariationalAutoencoder(
    ...     input_dim=(128, 128, 3),
    ...     encoder_conv_filters=[32, 64, 64, 64],
    ...     encoder_conv_kernel_size=[3, 3, 3, 3],
    ...     encoder_conv_strides=[2, 2, 2, 2],
    ...     decoder_conv_t_filters=[64, 64, 32, 3],
    ...     decoder_conv_t_kernel_size=[3, 3, 3, 3],
    ...     decoder_conv_t_strides=[2, 2, 2, 2],
    ...     z_dim=200,
    ...     use_batch_norm=True
    ... )
    >>>
    >>> # Compile and train
    >>> vae.compile(learning_rate=0.0005, r_loss_factor=10000)
    >>> vae.train(x_train, batch_size=32, epochs=200, run_folder='../run/vae')

References:
    - Kingma & Welling "Auto-Encoding Variational Bayes" (2014)
    - Chapter 3 of "Generative Deep Learning" book
    - documentation/NOTEBOOK_STANDARDIZATION.md for training workflow

Author: Generative Deep Learning Book / Refactored by Antigravity AI
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
import json
import os
import pickle
from typing import Any, List, Optional, Tuple

# Third-party imports
import numpy as np

# Keras imports
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    Layer,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Local imports
from src.utils.callbacks import CustomCallback, step_decay_schedule


# =============================================================================
# KL LOSS LAYER
# =============================================================================
class KLLossLayer(Layer):
    """
    Custom Keras layer that computes and adds KL Divergence loss.
    
    This layer implements the KL divergence term of the VAE loss function.
    It measures how much the learned latent distribution q(z|x) diverges
    from the standard normal prior p(z) = N(0, I).
    
    The KL divergence is computed as:
        KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
    
    This is analytically derived assuming both distributions are Gaussian.
    
    Usage:
        The layer takes [mu, log_var] as input and adds the KL loss
        to the model's total loss via self.add_loss().
    
    Example:
        >>> mu = Dense(z_dim, name='mu')(x)
        >>> log_var = Dense(z_dim, name='log_var')(x)
        >>> mu, log_var = KLLossLayer()([mu, log_var])
    
    Note:
        This layer passes its inputs through unchanged (identity function).
        Its only purpose is to compute and register the KL loss.
    """
    
    def call(self, inputs):
        """
        Compute KL divergence and add to model loss.
        
        Args:
            inputs: List of [mu, log_var] tensors.
                mu: Mean of the latent distribution, shape (batch, z_dim).
                log_var: Log variance of latent distribution.
        
        Returns:
            Same inputs unchanged (identity function).
        
        Side Effects:
            Adds KL divergence loss to self.losses for inclusion
            in the model's total loss.
        """
        mu, log_var = inputs
        
        # KL divergence formula for Gaussian distributions
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * K.sum(
            1 + log_var - K.square(mu) - K.exp(log_var),
            axis=1
        )
        
        # Add mean KL loss to model's losses
        self.add_loss(K.mean(kl_loss))
        
        return inputs


# =============================================================================
# VARIATIONAL AUTOENCODER CLASS
# =============================================================================
class VariationalAutoencoder:
    """
    Variational Autoencoder for generative modeling.
    
    This class implements a configurable VAE with support for:
    - Probabilistic latent space with μ and σ parameterization
    - Reparameterization trick for gradient flow
    - KL divergence regularization
    - Custom encoder and decoder architectures
    - Batch normalization and dropout
    - Training with arrays or data generators
    
    Attributes:
        name (str): Model identifier ('variational_autoencoder').
        input_dim (tuple): Input image dimensions (H, W, C).
        z_dim (int): Dimension of the latent space.
        encoder (Model): Maps images to latent samples.
        encoder_mu_log_var (Model): Maps images to (μ, log σ²).
        decoder (Model): Generates images from latent vectors.
        model (Model): The full VAE for training.
        mu (Tensor): Mean output layer.
        log_var (Tensor): Log variance output layer.
    
    Loss Function:
        Total Loss = r_loss_factor × MSE + KL Divergence
        
        Where:
        - MSE: Mean squared error between input and reconstruction
        - KL: Regularization term pushing q(z|x) toward N(0, I)
        - r_loss_factor: Weight to balance reconstruction vs regularization
    """
    
    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        encoder_conv_filters: List[int],
        encoder_conv_kernel_size: List[int],
        encoder_conv_strides: List[int],
        decoder_conv_t_filters: List[int],
        decoder_conv_t_kernel_size: List[int],
        decoder_conv_t_strides: List[int],
        z_dim: int,
        use_batch_norm: bool = False,
        use_dropout: bool = False
    ):
        """
        Initialize the VAE with architecture parameters.
        
        Args:
            input_dim: Image dimensions as (height, width, channels).
                Example: (128, 128, 3) for RGB face images.
            
            encoder_conv_filters: Number of filters for each encoder
                convolutional layer.
                Example: [32, 64, 64, 64] for 4 conv layers.
            
            encoder_conv_kernel_size: Kernel sizes for encoder layers.
                Example: [3, 3, 3, 3] for 3x3 kernels.
            
            encoder_conv_strides: Stride for each encoder layer.
                Stride > 1 reduces spatial dimensions.
                Example: [2, 2, 2, 2] to downsample 4 times.
            
            decoder_conv_t_filters: Number of filters for decoder layers.
                Last value should match input channels.
            
            decoder_conv_t_kernel_size: Kernel sizes for decoder layers.
            
            decoder_conv_t_strides: Stride for decoder layers.
                Should mirror encoder for symmetric architecture.
            
            z_dim: Dimension of the latent space.
                Higher values allow more expressive representations.
                Example: 200 for face generation.
            
            use_batch_norm: If True, add BatchNormalization.
                Default is False.
            
            use_dropout: If True, add Dropout(0.25).
                Default is False.
        """
        # =====================================================================
        # MODEL IDENTIFICATION
        # =====================================================================
        self.name = 'variational_autoencoder'
        
        # =====================================================================
        # ARCHITECTURE PARAMETERS - ENCODER
        # =====================================================================
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        
        # =====================================================================
        # ARCHITECTURE PARAMETERS - DECODER
        # =====================================================================
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        
        # =====================================================================
        # SHARED PARAMETERS
        # =====================================================================
        self.z_dim = z_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        # Calculate number of layers for each network
        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)
        
        # =====================================================================
        # BUILD THE NETWORKS
        # =====================================================================
        self._build()
    
    # =========================================================================
    # NETWORK BUILDING METHODS
    # =========================================================================
    
    def _build(self) -> None:
        """
        Build the encoder, decoder, and full VAE models.
        
        This method constructs:
        1. Encoder: Maps images to latent distribution (μ, log σ²)
        2. Sampling layer: Uses reparameterization trick
        3. Decoder: Generates images from latent samples
        4. Full Model: End-to-end VAE for training
        
        Architecture Details:
            Encoder:
                Input → [Conv2D → (BatchNorm) → LeakyReLU → (Dropout)] × N
                      → Flatten → Dense(μ) and Dense(log σ²)
                      → KLLossLayer → Sampling → z
            
            Decoder:
                z → Dense → Reshape → [ConvTranspose → (BatchNorm)
                  → LeakyReLU → (Dropout)] × (N-1)
                → ConvTranspose → Sigmoid → Output
        
        Models created:
            - self.encoder_mu_log_var: Returns (μ, log σ²)
            - self.encoder: Returns sampled z
            - self.decoder: Generates from z
            - self.model: Full VAE for training
        """
        # =====================================================================
        # BUILD ENCODER
        # =====================================================================
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input
        
        # Apply convolutional layers
        for i in range(self.n_layers_encoder):
            # -----------------------------------------------------------------
            # Convolutional layer
            # -----------------------------------------------------------------
            conv_layer = Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding='same',
                name=f'encoder_conv_{i}'
            )
            x = conv_layer(x)
            
            # -----------------------------------------------------------------
            # BatchNorm → Activation → Dropout
            # Note: BatchNorm before activation is a valid choice
            # -----------------------------------------------------------------
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            
            x = LeakyReLU()(x)
            
            if self.use_dropout:
                x = Dropout(rate=0.25)(x)
        
        # Store shape before flattening for decoder reshape
        shape_before_flattening = K.int_shape(x)[1:]
        
        # Flatten and project to latent parameters
        x = Flatten()(x)
        
        # =====================================================================
        # LATENT DISTRIBUTION PARAMETERS
        # =====================================================================
        # Mean (μ) of the latent distribution
        self.mu = Dense(self.z_dim, name='mu')(x)
        
        # Log variance (log σ²) of the latent distribution
        # Using log variance ensures σ² > 0 and stabilizes training
        self.log_var = Dense(self.z_dim, name='log_var')(x)
        
        # =====================================================================
        # KL DIVERGENCE LOSS LAYER
        # =====================================================================
        # This layer adds KL loss to the model and passes through μ, log_var
        self.mu, self.log_var = KLLossLayer()([self.mu, self.log_var])
        
        # Create model that outputs distribution parameters
        self.encoder_mu_log_var = Model(
            inputs=encoder_input,
            outputs=(self.mu, self.log_var),
            name='encoder_mu_log_var'
        )
        
        # =====================================================================
        # REPARAMETERIZATION TRICK
        # =====================================================================
        def sampling(args):
            """
            Sample from the latent distribution using reparameterization.
            
            The reparameterization trick:
                z = μ + σ × ε, where ε ~ N(0, 1)
            
            This allows gradients to flow through the sampling operation.
            
            Args:
                args: Tuple of (mu, log_var) tensors.
            
            Returns:
                Sampled latent vector z.
            """
            mu, log_var = args
            # Sample ε from standard normal
            epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
            # Compute z = μ + σ × ε (note: σ = exp(log_var / 2))
            return mu + K.exp(log_var / 2) * epsilon
        
        # Create sampling layer
        encoder_output = Lambda(sampling, name='encoder_output')(
            [self.mu, self.log_var]
        )
        
        # Create encoder model (outputs sampled z)
        self.encoder = Model(
            inputs=encoder_input,
            outputs=encoder_output,
            name='encoder'
        )
        
        # =====================================================================
        # BUILD DECODER
        # =====================================================================
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        
        # Dense layer to expand latent to spatial dimensions
        x = Dense(units=int(np.prod(shape_before_flattening)))(decoder_input)
        x = Reshape(target_shape=shape_before_flattening)(x)
        
        # Apply transposed convolutional layers
        for i in range(self.n_layers_decoder):
            # -----------------------------------------------------------------
            # Transposed convolution for upsampling
            # -----------------------------------------------------------------
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding='same',
                name=f'decoder_conv_t_{i}'
            )
            x = conv_t_layer(x)
            
            # -----------------------------------------------------------------
            # Intermediate layers: BatchNorm → LeakyReLU → Dropout
            # Final layer: Sigmoid activation
            # -----------------------------------------------------------------
            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                # Final layer: output pixel values in [0, 1]
                x = Activation('sigmoid')(x)
        
        decoder_output = x
        
        # Create decoder model
        self.decoder = Model(
            inputs=decoder_input,
            outputs=decoder_output,
            name='decoder'
        )
        
        # =====================================================================
        # BUILD FULL VAE
        # =====================================================================
        model_input = encoder_input
        model_output = self.decoder(encoder_output)
        
        self.model = Model(
            inputs=model_input,
            outputs=model_output,
            name='variational_autoencoder'
        )
    
    # =========================================================================
    # TRAINING METHODS
    # =========================================================================
    
    def compile(self, learning_rate: float, r_loss_factor: float) -> None:
        """
        Compile the VAE with optimizer and loss functions.
        
        The VAE loss consists of:
        1. Reconstruction loss: Weighted MSE between input and output
        2. KL divergence: Added automatically by KLLossLayer
        
        Args:
            learning_rate: Learning rate for Adam optimizer.
            
            r_loss_factor: Weight for reconstruction loss.
                Higher values emphasize reconstruction quality.
                Lower values emphasize latent space regularization.
                Typical values: 1000-10000 for face images.
        
        Note:
            Must be called before train() or train_with_generator().
        """
        self.learning_rate = learning_rate
        
        # =====================================================================
        # DEFINE LOSS FUNCTIONS
        # =====================================================================
        def vae_r_loss(y_true, y_pred):
            """
            Weighted reconstruction loss (MSE).
            
            Computes mean squared error scaled by r_loss_factor.
            
            Args:
                y_true: Original input images.
                y_pred: Reconstructed images.
            
            Returns:
                Weighted MSE loss per sample.
            """
            r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss_factor * r_loss
        
        def vae_loss(y_true, y_pred):
            """
            Total VAE loss (reconstruction only, KL added by layer).
            
            Note: KL divergence is added via KLLossLayer.add_loss()
            and is automatically included in the total loss.
            """
            return vae_r_loss(y_true, y_pred)
        
        # =====================================================================
        # COMPILE MODEL
        # =====================================================================
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=vae_loss,
            metrics=[vae_r_loss]
        )
    
    def train(
        self,
        x_train: np.ndarray,
        batch_size: int,
        epochs: int,
        run_folder: str,
        print_every_n_batches: int = 100,
        initial_epoch: int = 0,
        lr_decay: float = 1.0,
        extra_callbacks: Optional[List] = None
    ) -> None:
        """
        Train the VAE on numpy array data.
        
        Args:
            x_train: Training images as numpy array.
                Shape: (num_samples, height, width, channels).
                Values should be normalized to [0, 1].
            
            batch_size: Number of samples per batch.
            
            epochs: Total training epochs.
            
            run_folder: Directory to save weights and visualizations.
            
            print_every_n_batches: Frequency of sample generation.
                Default is 100.
            
            initial_epoch: Starting epoch (for resuming).
                Default is 0.
            
            lr_decay: Learning rate decay per epoch.
                Default is 1.0 (no decay).
            
            extra_callbacks: Additional Keras callbacks.
                Example: [WandbMetricsLogger(), EarlyStopping()]
        """
        # Custom callback for sample generation
        custom_callback = CustomCallback(
            run_folder,
            print_every_n_batches,
            initial_epoch,
            self
        )
        
        # Checkpoints for saving weights
        checkpoint_filepath = os.path.join(
            run_folder,
            "weights/weights-{epoch:03d}-{loss:.2f}.weights.h5"
        )
        checkpoint1 = ModelCheckpoint(
            checkpoint_filepath,
            save_weights_only=True,
            verbose=1
        )
        checkpoint2 = ModelCheckpoint(
            os.path.join(run_folder, 'weights/weights.weights.h5'),
            save_weights_only=True,
            verbose=1
        )
        
        callbacks_list = [checkpoint1, checkpoint2, custom_callback]
        
        # Add LR decay if specified
        if lr_decay != 1.0:
            lr_sched = step_decay_schedule(
                initial_lr=self.learning_rate,
                decay_factor=lr_decay,
                step_size=1
            )
            callbacks_list.append(lr_sched)
        
        # Add extra callbacks
        if extra_callbacks:
            callbacks_list.extend(extra_callbacks)
        
        # Train (input = output for VAE)
        self.model.fit(
            x=x_train,
            y=x_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_list
        )
    
    def train_with_generator(
        self,
        data_flow: Any,
        epochs: int,
        steps_per_epoch: int,
        run_folder: str,
        print_every_n_batches: int = 100,
        initial_epoch: int = 0,
        lr_decay: float = 1.0,
        extra_callbacks: Optional[List] = None
    ) -> None:
        """
        Train the VAE using a data generator.
        
        Use this method for large datasets that don't fit in memory,
        or when using data augmentation.
        
        Args:
            data_flow: Keras data generator yielding (x, y) batches.
                For VAE, should yield (images, images).
                Example: ImageDataGenerator().flow(x_train, x_train)
            
            epochs: Total training epochs.
            
            steps_per_epoch: Number of batches per epoch.
                For full dataset: len(x_train) // batch_size
            
            run_folder: Directory to save weights.
            
            print_every_n_batches: Sample generation frequency.
                Default is 100.
            
            initial_epoch: Starting epoch (for resuming).
                Default is 0.
            
            lr_decay: Learning rate decay per epoch.
                Default is 1.0 (no decay).
                Note: Set to 1.0 when using external LR schedulers
                like ReduceLROnPlateau.
            
            extra_callbacks: Additional Keras callbacks.
                Example: [WandbMetricsLogger(), ReduceLROnPlateau()]
        
        Note:
            - lr_decay creates an internal step decay scheduler
            - If using external schedulers, set lr_decay=1.0 to avoid conflicts
        """
        # Custom callback for sample generation
        custom_callback = CustomCallback(
            run_folder,
            print_every_n_batches,
            initial_epoch,
            self
        )
        
        # Checkpoints
        checkpoint_filepath = os.path.join(
            run_folder,
            "weights/weights-{epoch:03d}-{loss:.2f}.weights.h5"
        )
        checkpoint1 = ModelCheckpoint(
            checkpoint_filepath,
            save_weights_only=True,
            verbose=1
        )
        checkpoint2 = ModelCheckpoint(
            os.path.join(run_folder, 'weights/weights.weights.h5'),
            save_weights_only=True,
            verbose=1
        )
        
        callbacks_list = [checkpoint1, checkpoint2, custom_callback]
        
        # Add LR decay only if specified (avoids conflict with external schedulers)
        if lr_decay != 1.0:
            lr_sched = step_decay_schedule(
                initial_lr=self.learning_rate,
                decay_factor=lr_decay,
                step_size=1
            )
            callbacks_list.append(lr_sched)
        
        # Add extra callbacks (W&B, LR schedulers, early stopping, etc.)
        if extra_callbacks:
            callbacks_list.extend(extra_callbacks)
        
        # Save initial weights
        self.model.save_weights(
            os.path.join(run_folder, 'weights/weights.weights.h5')
        )
        
        # Train using generator
        self.model.fit(
            data_flow,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_list,
            steps_per_epoch=steps_per_epoch
        )
    
    # =========================================================================
    # PERSISTENCE METHODS
    # =========================================================================
    
    def save(self, folder: str) -> None:
        """
        Save model architecture parameters and visualizations.
        
        Args:
            folder: Directory to save model files.
                Creates subdirectories: viz/, weights/, images/
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))
        
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim,
                self.encoder_conv_filters,
                self.encoder_conv_kernel_size,
                self.encoder_conv_strides,
                self.decoder_conv_t_filters,
                self.decoder_conv_t_kernel_size,
                self.decoder_conv_t_strides,
                self.z_dim,
                self.use_batch_norm,
                self.use_dropout,
            ], f)
        
        self.plot_model(folder)
    
    def load_weights(self, filepath: str) -> None:
        """
        Load model weights from a file.
        
        Args:
            filepath: Path to the weights file (.weights.h5).
        """
        self.model.load_weights(filepath)
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def plot_model(self, run_folder: str) -> None:
        """
        Generate and save architecture diagrams.
        
        Saves diagrams to {run_folder}/viz/ for:
        - model.png: Full VAE
        - encoder.png: Encoder network
        - decoder.png: Decoder network
        
        Args:
            run_folder: Directory containing 'viz/' subdirectory.
        """
        plot_model(
            self.model,
            to_file=os.path.join(run_folder, 'viz/model.png'),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.encoder,
            to_file=os.path.join(run_folder, 'viz/encoder.png'),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.decoder,
            to_file=os.path.join(run_folder, 'viz/decoder.png'),
            show_shapes=True,
            show_layer_names=True
        )
    
    # =========================================================================
    # ACCESSOR METHODS
    # =========================================================================
    
    def get_model(self) -> Model:
        """
        Get the full VAE model.
        
        Returns:
            The compiled VAE model for training or inference.
        
        Note:
            Renamed from getModel() to follow PEP 8 naming conventions.
            getModel() is deprecated but kept for compatibility.
        """
        return self.model
    
    def getModel(self) -> Model:
        """
        Get the full VAE model (deprecated).
        
        Use get_model() instead for PEP 8 compliance.
        
        Returns:
            The compiled VAE model.
        """
        return self.get_model()
