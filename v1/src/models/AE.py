"""
Autoencoder Implementation for Image Compression and Reconstruction.

This module implements a standard convolutional Autoencoder architecture
for unsupervised representation learning. The autoencoder learns to compress
images into a low-dimensional latent space and reconstruct them with
minimal loss.

Architecture Overview:
    The Autoencoder consists of two neural networks:
    
    1. ENCODER: A convolutional network that compresses input images into
       a compact latent representation (encoding).
       
       Input: Image (H, W, C) → Conv layers → Flatten → Dense(z_dim) → z
    
    2. DECODER: A transposed convolutional network that reconstructs images
       from the latent representation.
       
       Input: z ∈ R^z_dim → Dense → Reshape → ConvTranspose layers → Image
    
    The networks are trained jointly to minimize reconstruction error:
        Loss = MSE(input_image, reconstructed_image)

Key Concepts:
    - Latent Space: The compressed representation in R^z_dim
    - Bottleneck: Forces the model to learn efficient representations
    - Reconstruction: The decoder's output attempts to match the input

Example Usage:
    >>> # Initialize Autoencoder
    >>> ae = Autoencoder(
    ...     input_dim=(28, 28, 1),
    ...     encoder_conv_filters=[32, 64, 64, 64],
    ...     encoder_conv_kernel_size=[3, 3, 3, 3],
    ...     encoder_conv_strides=[1, 2, 2, 1],
    ...     decoder_conv_t_filters=[64, 64, 32, 1],
    ...     decoder_conv_t_kernel_size=[3, 3, 3, 3],
    ...     decoder_conv_t_strides=[1, 2, 2, 1],
    ...     z_dim=2,
    ...     use_batch_norm=True,
    ...     use_dropout=True
    ... )
    >>>
    >>> # Compile and train
    >>> ae.compile(learning_rate=0.0005)
    >>> ae.train(x_train, batch_size=32, epochs=200, run_folder='../run/ae')

References:
    - Hinton & Salakhutdinov "Reducing the Dimensionality of Data" (2006)
    - Chapter 3 of "Generative Deep Learning" book

Author: Generative Deep Learning Book / Refactored by Antigravity AI
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
import json
import os
import pickle
from typing import List, Optional, Tuple

# Third-party imports
import numpy as np
import tensorflow as tf

# Keras imports
import keras.ops as ops
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

# Local imports
from src.utils.callbacks import CustomCallback, step_decay_schedule


# =============================================================================
# AUTOENCODER CLASS
# =============================================================================
class Autoencoder:
    """
    Convolutional Autoencoder for image compression and reconstruction.
    
    This class implements a configurable autoencoder with support for:
    - Custom encoder and decoder architectures
    - Batch normalization for training stability
    - Dropout regularization to prevent overfitting
    - Learning rate decay during training
    - Model checkpointing and visualization
    
    Attributes:
        name (str): Model identifier ('autoencoder').
        input_dim (tuple): Input image dimensions (H, W, C).
        z_dim (int): Dimension of the latent space (bottleneck).
        encoder (Model): The encoder network (image → latent).
        decoder (Model): The decoder network (latent → image).
        model (Model): The full autoencoder (image → image).
        use_batch_norm (bool): Whether batch normalization is enabled.
        use_dropout (bool): Whether dropout is enabled.
        learning_rate (float): Current learning rate (set after compile).
    
    Architecture Parameters:
        The encoder and decoder are configured via constructor parameters.
        Each convolutional layer can have different:
        - Number of filters
        - Kernel size
        - Stride (controls downsampling/upsampling)
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
        Initialize the Autoencoder with architecture parameters.
        
        Args:
            input_dim: Image dimensions as (height, width, channels).
                Example: (28, 28, 1) for grayscale MNIST images.
            
            encoder_conv_filters: Number of filters for each encoder
                convolutional layer. Length determines number of layers.
                Example: [32, 64, 64, 64] for 4 conv layers.
            
            encoder_conv_kernel_size: Kernel sizes for each encoder layer.
                Example: [3, 3, 3, 3] for 3x3 kernels.
            
            encoder_conv_strides: Stride for each encoder layer.
                Stride > 1 reduces spatial dimensions.
                Example: [1, 2, 2, 1] to downsample twice.
            
            decoder_conv_t_filters: Number of filters for each decoder
                transposed convolutional layer.
            
            decoder_conv_t_kernel_size: Kernel sizes for decoder layers.
            
            decoder_conv_t_strides: Stride for each decoder layer.
                Stride > 1 increases spatial dimensions.
            
            z_dim: Dimension of the latent space (encoding).
                Controls the compression ratio.
                Example: 2 for 2D visualization, 200 for complex data.
            
            use_batch_norm: If True, add BatchNormalization after each
                conv layer. Helps with training stability.
                Default is False.
            
            use_dropout: If True, add Dropout(0.25) after each conv layer.
                Helps prevent overfitting.
                Default is False.
        
        Note:
            - Encoder and decoder should be symmetric for best results
            - Final decoder layer always uses sigmoid activation
            - LeakyReLU is used for all intermediate activations
        """
        # =====================================================================
        # MODEL IDENTIFICATION
        # =====================================================================
        self.name = 'autoencoder'
        
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
        Build the encoder, decoder, and full autoencoder models.
        
        This method constructs:
        1. Encoder: Compresses input images to latent vectors
        2. Decoder: Reconstructs images from latent vectors
        3. Full Model: End-to-end autoencoder (encoder + decoder)
        
        Architecture Details:
            Encoder:
                Input → [Conv2D → LeakyReLU → (BatchNorm) → (Dropout)] × N
                      → Flatten → Dense(z_dim) → Latent Vector
            
            Decoder:
                Latent → Dense → Reshape → [ConvTranspose → LeakyReLU
                        → (BatchNorm) → (Dropout)] × (N-1)
                      → ConvTranspose → Sigmoid → Output Image
        
        The models are stored in:
            - self.encoder
            - self.decoder
            - self.model
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
            # Uses strided convolutions to downsample spatial dimensions
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
            # Activation: LeakyReLU allows small negative gradients
            # -----------------------------------------------------------------
            x = LeakyReLU()(x)
            
            # -----------------------------------------------------------------
            # Optional: Batch Normalization
            # Normalizes activations to stabilize training
            # -----------------------------------------------------------------
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            
            # -----------------------------------------------------------------
            # Optional: Dropout regularization
            # Randomly zeros 25% of activations to prevent overfitting
            # -----------------------------------------------------------------
            if self.use_dropout:
                x = Dropout(rate=0.25)(x)
        
        # Store shape before flattening (needed for decoder's reshape)
        shape_before_flattening = x.shape[1:]
        
        # Flatten spatial dimensions to 1D
        x = Flatten()(x)
        
        # Dense layer to produce latent vector
        encoder_output = Dense(self.z_dim, name='encoder_output')(x)
        
        # Create encoder model
        self.encoder = Model(
            inputs=encoder_input,
            outputs=encoder_output,
            name='encoder'
        )
        
        # =====================================================================
        # BUILD DECODER
        # =====================================================================
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        
        # Dense layer to expand latent vector back to spatial dimensions
        x = Dense(units=int(np.prod(shape_before_flattening)))(decoder_input)
        x = Reshape(target_shape=shape_before_flattening)(x)
        
        # Apply transposed convolutional layers
        for i in range(self.n_layers_decoder):
            # -----------------------------------------------------------------
            # Transposed convolution (deconvolution)
            # Upsamples spatial dimensions when stride > 1
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
            # Activation: LeakyReLU for intermediate layers, Sigmoid for output
            # -----------------------------------------------------------------
            if i < self.n_layers_decoder - 1:
                # Intermediate layers
                x = LeakyReLU()(x)
                
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                # Final layer: Sigmoid to output pixel values in [0, 1]
                x = Activation('sigmoid')(x)
        
        decoder_output = x
        
        # Create decoder model
        self.decoder = Model(
            inputs=decoder_input,
            outputs=decoder_output,
            name='decoder'
        )
        
        # =====================================================================
        # BUILD FULL AUTOENCODER
        # =====================================================================
        # Connect encoder output to decoder input
        model_input = encoder_input
        model_output = self.decoder(encoder_output)
        
        self.model = Model(
            inputs=model_input,
            outputs=model_output,
            name='autoencoder'
        )
    
    # =========================================================================
    # TRAINING METHODS
    # =========================================================================
    
    def compile(self, learning_rate: float) -> None:
        """
        Compile the autoencoder with optimizer and loss function.
        
        Uses Adam optimizer and mean squared error (MSE) loss for
        reconstruction. The loss is averaged over all pixels.
        
        Args:
            learning_rate: Learning rate for the Adam optimizer.
        
        Note:
            Must be called before train() method.
        """
        self.learning_rate = learning_rate
        
        optimizer = Adam(learning_rate=learning_rate)
        
        # Define reconstruction loss (MSE per image)
        def r_loss(y_true, y_pred):
            """
            Reconstruction loss: Mean Squared Error.
            
            Computes the average squared difference between input
            and reconstruction across all pixels.
            
            Args:
                y_true: Original input images.
                y_pred: Reconstructed images.
            
            Returns:
                Tensor of shape (batch_size,) with loss per sample.
            """
            return ops.mean(ops.square(y_true - y_pred), axis=[1, 2, 3])
        
        self.model.compile(optimizer=optimizer, loss=r_loss)
    
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
        Train the autoencoder on the provided data.
        
        The model learns to reconstruct its input by minimizing the
        mean squared error between input and output images.
        
        Args:
            x_train: Training images as numpy array.
                Shape: (num_samples, height, width, channels).
                Values should be normalized to [0, 1].
            
            batch_size: Number of samples per training batch.
            
            epochs: Total number of training epochs.
            
            run_folder: Directory to save weights, images, and logs.
                Must contain 'weights/' subdirectory.
            
            print_every_n_batches: Frequency of custom callback output.
                Default is 100.
            
            initial_epoch: Starting epoch number (for resuming training).
                Default is 0.
            
            lr_decay: Learning rate decay factor per epoch.
                Default is 1.0 (no decay).
                Example: 0.95 for 5% decay per epoch.
            
            extra_callbacks: Optional list of additional Keras callbacks.
                Example: [WandbMetricsLogger(), EarlyStopping()]
        
        Note:
            - compile() must be called before train()
            - Weights are saved to {run_folder}/weights/weights.weights.h5
        """
        # Create custom callback for sample generation
        custom_callback = CustomCallback(
            run_folder,
            print_every_n_batches,
            initial_epoch,
            self
        )
        
        # Checkpoint to save weights after each epoch
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(run_folder, 'weights/weights.weights.h5'),
            save_weights_only=True,
            verbose=1
        )
        
        callbacks_list = [checkpoint, custom_callback]
        
        # Add learning rate decay schedule if specified
        if lr_decay != 1.0:
            lr_sched = step_decay_schedule(
                initial_lr=self.learning_rate,
                decay_factor=lr_decay,
                step_size=1
            )
            callbacks_list.append(lr_sched)
        
        # Add any extra callbacks (e.g., W&B, early stopping)
        if extra_callbacks:
            callbacks_list.extend(extra_callbacks)
        
        # Train the model (input = output for autoencoders)
        self.model.fit(
            x=x_train,
            y=x_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_list
        )
    
    # =========================================================================
    # PERSISTENCE METHODS
    # =========================================================================
    
    def save(self, folder: str) -> None:
        """
        Save model architecture parameters and visualizations.
        
        Creates the folder structure and saves:
        - params.pkl: Constructor parameters for model reconstruction
        - viz/: Architecture diagrams
        
        Args:
            folder: Directory to save model files.
                Creates subdirectories: viz/, weights/, images/
        """
        # Create directory structure if needed
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))
        
        # Save constructor parameters
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
        
        # Save architecture visualizations
        self.plot_model(folder)
    
    def load_weights(self, filepath: str) -> None:
        """
        Load model weights from a file.
        
        Args:
            filepath: Path to the weights file (.weights.h5).
        
        Example:
            >>> ae.load_weights('../run/ae/weights/weights.weights.h5')
        """
        self.model.load_weights(filepath)
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def plot_model(self, run_folder: str) -> None:
        """
        Generate and save architecture diagrams for all models.
        
        Creates visual representations of the encoder, decoder,
        and full autoencoder architectures.
        
        Args:
            run_folder: Directory containing 'viz/' subdirectory.
        
        Output:
            Saves to:
                - {run_folder}/viz/model.png
                - {run_folder}/viz/encoder.png
                - {run_folder}/viz/decoder.png
        
        Note:
            Requires graphviz to be installed. Silently fails if
            graphviz is not available.
        """
        try:
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
        except Exception as e:
            print(
                f"Skipping plot_model due to error "
                f"(likely missing graphviz): {e}"
            )
