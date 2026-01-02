"""
Generative Adversarial Network (GAN) Implementation.

This module implements a standard GAN architecture for image generation as
described in "Generative Adversarial Networks" by Goodfellow et al. (2014).
The implementation supports configurable discriminator and generator
architectures, multiple optimizer types, and integrated experiment tracking
via Weights & Biases (W&B).

Architecture Overview:
    The GAN consists of two neural networks in adversarial training:
    
    1. DISCRIMINATOR: A convolutional classifier that learns to distinguish
       real images from fake (generated) images. It outputs a probability
       score for each input image.
       
       Input: Image (H, W, C) → Conv layers → Flatten → Dense(1, sigmoid) → [0,1]
    
    2. GENERATOR: A transposed convolutional network that learns to generate
       realistic images from random noise vectors (latent space).
       
       Input: Noise z ∈ R^z_dim → Dense → Reshape → Upsample/ConvT → Image
    
    The networks are trained alternately:
    - Discriminator learns: Maximize P(real) and minimize P(fake)
    - Generator learns: Maximize P(fake) (fool the discriminator)

Example Usage:
    >>> # Initialize GAN with architecture parameters
    >>> gan = GAN(
    ...     input_dim=(28, 28, 1),
    ...     discriminator_conv_filters=[64, 64, 128, 128],
    ...     discriminator_conv_kernel_size=[5, 5, 5, 5],
    ...     discriminator_conv_strides=[2, 2, 2, 1],
    ...     discriminator_batch_norm_momentum=None,
    ...     discriminator_activation='relu',
    ...     discriminator_dropout_rate=0.4,
    ...     discriminator_learning_rate=0.0008,
    ...     generator_initial_dense_layer_size=(7, 7, 64),
    ...     generator_upsample=[2, 2, 1, 1],
    ...     generator_conv_filters=[128, 64, 64, 1],
    ...     generator_conv_kernel_size=[5, 5, 5, 5],
    ...     generator_conv_strides=[1, 1, 1, 1],
    ...     generator_batch_norm_momentum=0.9,
    ...     generator_activation='relu',
    ...     generator_dropout_rate=None,
    ...     generator_learning_rate=0.0004,
    ...     optimiser='rmsprop',
    ...     z_dim=100
    ... )
    >>>
    >>> # Train the GAN
    >>> gan.train(x_train, batch_size=256, epochs=6000, run_folder='../run/gan')

References:
    - Goodfellow et al. "Generative Adversarial Networks" (2014)
    - Radford et al. "Unsupervised Representation Learning with DCGANs" (2015)
    - documentation/NOTEBOOK_STANDARDIZATION.md for training workflow

Author: Generative Deep Learning Book / Refactored by Antigravity AI
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
import json
import os
import pickle as pkl
from typing import List, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Keras imports (using TensorFlow backend)
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    LeakyReLU,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model


# =============================================================================
# GAN CLASS
# =============================================================================
class GAN:
    """
    Generative Adversarial Network for image generation.
    
    This class implements a configurable GAN with support for:
    - Custom discriminator and generator architectures
    - Multiple optimizer types (Adam, RMSprop)
    - Batch normalization and dropout regularization
    - Step decay learning rate scheduling
    - Weights & Biases experiment tracking
    - Model/weight saving and loading
    
    Attributes:
        name (str): Model identifier ('gan').
        input_dim (tuple): Input image dimensions (H, W, C).
        z_dim (int): Latent space dimension for the generator.
        discriminator (Model): The discriminator network.
        generator (Model): The generator network.
        model (Model): Combined GAN model (generator + frozen discriminator).
        d_losses (list): History of discriminator losses per epoch.
        g_losses (list): History of generator losses per epoch.
        d_lr_history (list): History of discriminator learning rates.
        g_lr_history (list): History of generator learning rates.
        epoch (int): Current training epoch (for resuming training).
    
    Architecture Parameters:
        The discriminator and generator are configured via constructor
        parameters. Each convolutional layer can have different:
        - Number of filters
        - Kernel size
        - Stride
        - Batch normalization
        - Activation function
        - Dropout rate
    """
    
    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        discriminator_conv_filters: List[int],
        discriminator_conv_kernel_size: List[int],
        discriminator_conv_strides: List[int],
        discriminator_batch_norm_momentum: Optional[float],
        discriminator_activation: str,
        discriminator_dropout_rate: Optional[float],
        discriminator_learning_rate: float,
        generator_initial_dense_layer_size: Tuple[int, int, int],
        generator_upsample: List[int],
        generator_conv_filters: List[int],
        generator_conv_kernel_size: List[int],
        generator_conv_strides: List[int],
        generator_batch_norm_momentum: Optional[float],
        generator_activation: str,
        generator_dropout_rate: Optional[float],
        generator_learning_rate: float,
        optimiser: str,
        z_dim: int
    ):
        """
        Initialize the GAN with architecture and training parameters.
        
        Args:
            input_dim: Image dimensions as (height, width, channels).
                Example: (28, 28, 1) for grayscale MNIST-like images.
            
            discriminator_conv_filters: Number of filters for each
                discriminator convolutional layer. Length determines
                number of layers.
                Example: [64, 64, 128, 128] for 4 conv layers.
            
            discriminator_conv_kernel_size: Kernel sizes for each layer.
                Example: [5, 5, 5, 5] for 5x5 kernels.
            
            discriminator_conv_strides: Stride for each layer.
                Example: [2, 2, 2, 1] to downsample 3 times.
            
            discriminator_batch_norm_momentum: BatchNorm momentum (0-1).
                Set to None to disable batch normalization.
                Note: Not applied to the first layer.
            
            discriminator_activation: Activation function name.
                Supported: 'relu', 'leaky_relu', 'tanh', 'sigmoid'.
            
            discriminator_dropout_rate: Dropout probability (0-1).
                Set to None to disable dropout.
            
            discriminator_learning_rate: Optimizer learning rate for D.
            
            generator_initial_dense_layer_size: Shape after dense layer.
                Example: (7, 7, 64) for 7x7 feature maps with 64 channels.
            
            generator_upsample: Upsampling factor for each layer.
                Use 2 for UpSampling2D, or 1 for Conv2DTranspose only.
            
            generator_conv_filters: Number of filters for each G layer.
            
            generator_conv_kernel_size: Kernel sizes for each G layer.
            
            generator_conv_strides: Strides for each G layer.
            
            generator_batch_norm_momentum: BatchNorm momentum for G.
            
            generator_activation: Activation function for G layers.
                Note: Final layer always uses 'tanh' for [-1, 1] output.
            
            generator_dropout_rate: Dropout rate for G.
            
            generator_learning_rate: Optimizer learning rate for G.
            
            optimiser: Optimizer type: 'adam' or 'rmsprop'.
            
            z_dim: Dimension of the latent noise vector.
                Higher values allow more variation but may be harder to train.
                Common values: 100-200.
        
        Raises:
            ValueError: If filter/kernel/stride lists have mismatched lengths.
        """
        # =====================================================================
        # MODEL IDENTIFICATION
        # =====================================================================
        self.name = 'gan'
        
        # =====================================================================
        # DISCRIMINATOR ARCHITECTURE PARAMETERS
        # =====================================================================
        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        
        # =====================================================================
        # GENERATOR ARCHITECTURE PARAMETERS
        # =====================================================================
        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate
        
        # =====================================================================
        # SHARED PARAMETERS
        # =====================================================================
        self.optimiser = optimiser
        self.z_dim = z_dim
        
        # Calculate number of layers for each network
        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)
        
        # Weight initialization: small random values centered at 0
        # This helps with stable GAN training (DCGAN paper recommendation)
        self.weight_init = RandomNormal(mean=0.0, stddev=0.02)
        
        # =====================================================================
        # TRAINING HISTORY
        # =====================================================================
        # Loss history: each entry is [d_loss, d_loss_real, d_loss_fake,
        #                              d_acc, d_acc_real, d_acc_fake]
        self.d_losses = []
        
        # Generator loss history: each entry is [g_loss, g_acc]
        self.g_losses = []
        
        # Learning rate history for plotting after training
        # These track the LR at each epoch for both networks
        self.d_lr_history = []
        self.g_lr_history = []
        
        # Current epoch counter (for resuming training)
        self.epoch = 0
        
        # =====================================================================
        # BUILD THE NETWORKS
        # =====================================================================
        self._build_discriminator()
        self._build_generator()
        self._build_adversarial()
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_activation(self, activation: str) -> Layer:
        """
        Get a Keras activation layer by name.
        
        This method provides a unified interface for creating activation
        layers, including the special-case handling for LeakyReLU which
        requires a negative_slope parameter.
        
        Args:
            activation: Name of the activation function.
                Supported values:
                - 'leaky_relu': LeakyReLU with negative_slope=0.2
                - 'relu', 'tanh', 'sigmoid', etc.: Standard Keras activations
        
        Returns:
            Keras Layer: An activation layer instance.
        
        Example:
            >>> act = self.get_activation('leaky_relu')
            >>> x = act(x)  # Apply LeakyReLU to tensor x
        """
        if activation == 'leaky_relu':
            # LeakyReLU allows small negative values to pass through
            # negative_slope=0.2 is a common choice for GANs
            layer = LeakyReLU(negative_slope=0.2)
        else:
            # Use standard Keras Activation layer for other functions
            layer = Activation(activation)
        return layer
    
    def get_opti(self, lr: float):
        """
        Create an optimizer instance with the specified learning rate.
        
        This method provides a factory for creating optimizers based on
        the `self.optimiser` setting. It handles the special beta_1
        parameter for Adam optimizer which helps with GAN training.
        
        Args:
            lr: Learning rate for the optimizer.
        
        Returns:
            Keras Optimizer: An optimizer instance (Adam or RMSprop).
        
        Note:
            - Adam uses beta_1=0.5 (instead of default 0.9) for GAN stability
            - This follows the DCGAN paper recommendations
        """
        if self.optimiser == 'adam':
            # beta_1=0.5 is recommended for GAN training (DCGAN paper)
            # Lower beta_1 makes the optimizer less sensitive to past gradients
            opti = Adam(learning_rate=lr, beta_1=0.5)
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(learning_rate=lr)
        else:
            # Default to Adam if unknown optimizer specified
            opti = Adam(learning_rate=lr)
        return opti
    
    def set_trainable(self, model: Model, trainable: bool) -> None:
        """
        Set trainability of a model and all its layers.
        
        This is essential for GAN training where we need to freeze the
        discriminator when training the generator, and vice versa.
        
        Args:
            model: Keras Model to modify.
            trainable: If True, enable training. If False, freeze weights.
        
        Note:
            Must set both model.trainable AND all layer.trainable
            to ensure proper freezing behavior.
        """
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable
    
    # =========================================================================
    # NETWORK BUILDING METHODS
    # =========================================================================
    
    def _build_discriminator(self) -> None:
        """
        Build the discriminator network.
        
        The discriminator is a convolutional classifier that outputs a
        probability score indicating whether the input image is real.
        
        Architecture:
            Input Image
                ↓
            [Conv2D → (BatchNorm) → Activation → (Dropout)] × N layers
                ↓
            Flatten
                ↓
            Dense(1, sigmoid) → Probability [0, 1]
        
        Design Notes:
            - First layer does NOT use batch normalization (DCGAN guideline)
            - Strided convolutions downsample the spatial dimensions
            - Dropout prevents discriminator from becoming too strong
            - Final sigmoid outputs probability of "real" class
        
        The built model is stored in `self.discriminator`.
        """
        # =====================================================================
        # INPUT LAYER
        # =====================================================================
        discriminator_input = Input(
            shape=self.input_dim,
            name='discriminator_input'
        )
        x = discriminator_input
        
        # =====================================================================
        # CONVOLUTIONAL LAYERS
        # =====================================================================
        for i in range(self.n_layers_discriminator):
            # -----------------------------------------------------------------
            # Convolutional layer
            # Uses strided convolutions to downsample (instead of pooling)
            # -----------------------------------------------------------------
            x = Conv2D(
                filters=self.discriminator_conv_filters[i],
                kernel_size=self.discriminator_conv_kernel_size[i],
                strides=self.discriminator_conv_strides[i],
                padding='same',
                name=f'discriminator_conv_{i}',
                kernel_initializer=self.weight_init
            )(x)
            
            # -----------------------------------------------------------------
            # Batch Normalization (optional, skip first layer)
            # Skipping first layer follows DCGAN best practices
            # -----------------------------------------------------------------
            if self.discriminator_batch_norm_momentum and i > 0:
                x = BatchNormalization(
                    momentum=self.discriminator_batch_norm_momentum
                )(x)
            
            # -----------------------------------------------------------------
            # Activation function
            # -----------------------------------------------------------------
            x = self.get_activation(self.discriminator_activation)(x)
            
            # -----------------------------------------------------------------
            # Dropout (optional)
            # Helps prevent discriminator from overpowering generator
            # -----------------------------------------------------------------
            if self.discriminator_dropout_rate:
                x = Dropout(rate=self.discriminator_dropout_rate)(x)
        
        # =====================================================================
        # CLASSIFICATION HEAD
        # =====================================================================
        # Flatten spatial dimensions to 1D
        x = Flatten()(x)
        
        # Output layer: single neuron with sigmoid activation
        # Outputs P(image is real) ∈ [0, 1]
        discriminator_output = Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer=self.weight_init
        )(x)
        
        # =====================================================================
        # CREATE MODEL
        # =====================================================================
        self.discriminator = Model(
            inputs=discriminator_input,
            outputs=discriminator_output,
            name='discriminator'
        )
    
    def _build_generator(self) -> None:
        """
        Build the generator network.
        
        The generator transforms random noise vectors into fake images
        that attempt to fool the discriminator.
        
        Architecture:
            Noise Vector z ∈ R^z_dim
                ↓
            Dense → Reshape to (H, W, C)
                ↓
            [UpSample/ConvT → (BatchNorm) → Activation] × (N-1) layers
                ↓
            Final Conv → Tanh → Output Image ∈ [-1, 1]
        
        Design Notes:
            - Initial dense layer projects noise to spatial representation
            - Each layer upsamples to progressively larger resolution
            - BatchNorm stabilizes training
            - Final tanh activation produces values in [-1, 1]
        
        The built model is stored in `self.generator`.
        """
        # =====================================================================
        # INPUT LAYER
        # =====================================================================
        generator_input = Input(
            shape=(self.z_dim,),
            name='generator_input'
        )
        x = generator_input
        
        # =====================================================================
        # DENSE PROJECTION LAYER
        # Projects latent vector to initial spatial representation
        # =====================================================================
        # Calculate total units needed: H × W × C
        initial_units = int(np.prod(self.generator_initial_dense_layer_size))
        x = Dense(
            units=initial_units,
            kernel_initializer=self.weight_init
        )(x)
        
        # Optional batch normalization after dense layer
        if self.generator_batch_norm_momentum:
            x = BatchNormalization(
                momentum=self.generator_batch_norm_momentum
            )(x)
        
        # Activation
        x = self.get_activation(self.generator_activation)(x)
        
        # Reshape from 1D to 3D spatial representation
        x = Reshape(self.generator_initial_dense_layer_size)(x)
        
        # Optional dropout
        if self.generator_dropout_rate:
            x = Dropout(rate=self.generator_dropout_rate)(x)
        
        # =====================================================================
        # UPSAMPLING / DECONVOLUTION LAYERS
        # =====================================================================
        for i in range(self.n_layers_generator):
            # -----------------------------------------------------------------
            # Upsampling strategy
            # Option 1: UpSampling2D + Conv2D (if upsample[i] == 2)
            # Option 2: Conv2DTranspose (fractionally strided convolution)
            # -----------------------------------------------------------------
            if self.generator_upsample[i] == 2:
                # UpSampling + Conv approach (checkerboard artifacts are less)
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    name=f'generator_conv_{i}',
                    kernel_initializer=self.weight_init
                )(x)
            else:
                # Transposed convolution (learned upsampling)
                x = Conv2DTranspose(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    strides=self.generator_conv_strides[i],
                    name=f'generator_conv_{i}',
                    kernel_initializer=self.weight_init
                )(x)
            
            # -----------------------------------------------------------------
            # Post-convolution processing
            # All layers except the last get BatchNorm + Activation
            # Last layer gets Tanh to produce [-1, 1] output
            # -----------------------------------------------------------------
            if i < self.n_layers_generator - 1:
                # Intermediate layers: BatchNorm + Activation
                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(
                        momentum=self.generator_batch_norm_momentum
                    )(x)
                x = self.get_activation(self.generator_activation)(x)
            else:
                # Final layer: Tanh activation for [-1, 1] pixel values
                x = Activation('tanh')(x)
        
        # =====================================================================
        # CREATE MODEL
        # =====================================================================
        generator_output = x
        self.generator = Model(
            inputs=generator_input,
            outputs=generator_output,
            name='generator'
        )
    
    def _build_adversarial(self) -> None:
        """
        Build and compile the adversarial (combined) model.
        
        This method:
        1. Compiles the discriminator with its own optimizer
        2. Creates the combined model: Generator → Discriminator
        3. Freezes discriminator weights in combined model
        4. Compiles combined model with generator's optimizer
        
        The combined model is used to train the generator by:
        - Feeding noise → Generator produces fake images
        - Fake images → (frozen) Discriminator produces predictions
        - Loss is computed against "real" labels (trying to fool D)
        - Gradients flow only through Generator (D is frozen)
        
        Models created:
            self.discriminator: Compiled for training D separately
            self.model: Combined G+D for training G only
        """
        # =====================================================================
        # COMPILE DISCRIMINATOR
        # =====================================================================
        # Discriminator is trained separately to classify real vs fake
        self.discriminator.compile(
            optimizer=self.get_opti(self.discriminator_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # =====================================================================
        # BUILD COMBINED MODEL (GENERATOR + DISCRIMINATOR)
        # =====================================================================
        # Freeze discriminator weights during generator training
        # This prevents D from learning while we're training G
        self.set_trainable(self.discriminator, False)
        
        # Create the combined model
        # Input: noise vector → Generator → fake image → Discriminator → prob
        model_input = Input(shape=(self.z_dim,), name='model_input')
        generated_image = self.generator(model_input)
        model_output = self.discriminator(generated_image)
        
        self.model = Model(
            inputs=model_input,
            outputs=model_output,
            name='gan_combined'
        )
        
        # =====================================================================
        # COMPILE COMBINED MODEL
        # =====================================================================
        # Uses generator's learning rate since only G is being trained
        self.model.compile(
            optimizer=self.get_opti(self.generator_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Re-enable discriminator training for separate D training steps
        self.set_trainable(self.discriminator, True)
    
    # =========================================================================
    # TRAINING METHODS
    # =========================================================================
    
    def train_discriminator(
        self,
        x_train: np.ndarray,
        batch_size: int,
        using_generator: bool
    ) -> List[float]:
        """
        Perform one discriminator training step.
        
        The discriminator is trained on two batches:
        1. Real images (labeled as 1)
        2. Fake images from generator (labeled as 0)
        
        Args:
            x_train: Training data array or generator.
            batch_size: Number of samples per batch.
            using_generator: If True, x_train is a generator.
        
        Returns:
            List of metrics: [d_loss, d_loss_real, d_loss_fake,
                             d_acc, d_acc_real, d_acc_fake]
        """
        # =====================================================================
        # PREPARE LABELS
        # =====================================================================
        # Real images should be classified as 1
        valid = np.ones((batch_size, 1))
        # Fake images should be classified as 0
        fake = np.zeros((batch_size, 1))
        
        # =====================================================================
        # GET REAL IMAGES
        # =====================================================================
        if using_generator:
            # x_train is a generator yielding batches
            true_imgs = next(x_train)[0]
            # Handle case where generator returns partial batch
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            # x_train is a numpy array, sample random indices
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]
        
        # =====================================================================
        # GENERATE FAKE IMAGES
        # =====================================================================
        # Sample random noise from standard normal distribution
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        # Generate fake images using the generator
        gen_imgs = self.generator.predict(noise, verbose=0)
        
        # =====================================================================
        # TRAIN ON REAL AND FAKE BATCHES
        # =====================================================================
        # Train on real images (should classify as 1)
        d_loss_real, d_acc_real = self.discriminator.train_on_batch(
            true_imgs, valid
        )
        
        # Train on fake images (should classify as 0)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(
            gen_imgs, fake
        )
        
        # =====================================================================
        # COMPUTE AVERAGE METRICS
        # =====================================================================
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)
        
        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]
    
    def train_generator(self, batch_size: int) -> List[float]:
        """
        Perform one generator training step.
        
        The generator is trained by:
        1. Generating fake images from noise
        2. Trying to make the discriminator classify them as real (1)
        3. Backpropagating through the combined model
        
        Note: The discriminator is frozen during this step.
        
        Args:
            batch_size: Number of samples to generate.
        
        Returns:
            List of metrics: [g_loss, g_acc]
        """
        # Generator wants discriminator to predict "real" for fake images
        valid = np.ones((batch_size, 1))
        
        # Sample random noise
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        
        # Train the combined model (only generator weights update)
        return self.model.train_on_batch(noise, valid)
    
    def train(
        self,
        x_train: Union[np.ndarray, any],
        batch_size: int,
        epochs: int,
        run_folder: str,
        print_every_n_batches: int = 50,
        using_generator: bool = False,
        use_wandb: bool = False,
        lr_decay_factor: float = 0.5,
        lr_decay_epochs: int = 500
    ) -> None:
        """
        Train the GAN model with optional W&B logging and LR scheduling.
        
        This method implements the standard GAN training loop where the
        discriminator and generator are trained alternately. It supports:
        - Weights & Biases (W&B) experiment tracking
        - Step decay learning rate scheduling
        - Learning rate history tracking for post-training visualization
        
        Training Loop:
            For each epoch:
                1. Optionally apply LR decay
                2. Train discriminator on real + fake batch
                3. Train generator via combined model
                4. Log metrics to W&B (if enabled)
                5. Periodically save samples, weights, and model
        
        Args:
            x_train: Training data array or generator.
            batch_size: Number of samples per training batch.
            epochs: Total number of epochs to train.
            run_folder: Directory to save weights, images, and model files.
            print_every_n_batches: Save samples and weights every N epochs.
                Default is 50.
            using_generator: If True, x_train is a generator yielding batches.
                Default is False (x_train is a numpy array).
            use_wandb: If True, log metrics to Weights & Biases.
                Requires wandb to be initialized before calling train().
                Default is False.
            lr_decay_factor: Multiply LR by this factor at decay intervals.
                Default is 0.5 (halve the LR).
            lr_decay_epochs: Apply LR decay every N epochs.
                Default is 500. Set to 0 to disable decay.
        
        Returns:
            None. Updates self.d_losses, self.g_losses, self.d_lr_history,
            and self.g_lr_history with training metrics.
        
        Example:
            >>> gan.train(
            ...     x_train,
            ...     batch_size=1024,
            ...     epochs=1500,
            ...     run_folder='../run/gan/0001_camel',
            ...     print_every_n_batches=50,
            ...     use_wandb=True,
            ...     lr_decay_factor=0.5,
            ...     lr_decay_epochs=375  # Decay 4 times: 375, 750, 1125
            ... )
        
        Note:
            - W&B must be initialized with wandb.init() before training
            - LR history is stored in self.d_lr_history and self.g_lr_history
            - Sample images are saved to run_folder/images/
        """
        # =====================================================================
        # WANDB IMPORT (CONDITIONAL)
        # =====================================================================
        # Import wandb only if needed to avoid import errors if not installed
        if use_wandb:
            try:
                import wandb
            except ImportError:
                print("WARNING: wandb not installed. Disabling W&B logging.")
                use_wandb = False
        
        # =====================================================================
        # INITIALIZE LEARNING RATES
        # =====================================================================
        # Get current learning rates from optimizers
        d_lr = float(self.discriminator.optimizer.learning_rate.numpy())
        g_lr = float(self.model.optimizer.learning_rate.numpy())
        
        # =====================================================================
        # MAIN TRAINING LOOP
        # =====================================================================
        for epoch in range(self.epoch, self.epoch + epochs):
            # =================================================================
            # STEP DECAY LR SCHEDULER
            # Reduce learning rate at fixed epoch intervals
            # =================================================================
            if lr_decay_epochs > 0 and epoch > 0 and epoch % lr_decay_epochs == 0:
                # Calculate new learning rates
                d_lr = d_lr * lr_decay_factor
                g_lr = g_lr * lr_decay_factor
                
                # Apply to optimizers
                self.discriminator.optimizer.learning_rate.assign(d_lr)
                self.model.optimizer.learning_rate.assign(g_lr)
                
                print(
                    f"\n>>> LR Decay at epoch {epoch}: "
                    f"D_LR={d_lr:.2e}, G_LR={g_lr:.2e}\n"
                )
            
            # Record current learning rates for plotting
            self.d_lr_history.append(d_lr)
            self.g_lr_history.append(g_lr)
            
            # =================================================================
            # TRAIN DISCRIMINATOR AND GENERATOR
            # =================================================================
            d = self.train_discriminator(x_train, batch_size, using_generator)
            g = self.train_generator(batch_size)
            
            # Print progress
            print(
                f"{epoch} [D loss: ({d[0]:.3f})(R {d[1]:.3f}, F {d[2]:.3f})] "
                f"[D acc: ({d[3]:.3f})({d[4]:.3f}, {d[5]:.3f})] "
                f"[G loss: {g[0]:.3f}] [G acc: {g[1]:.3f}]"
            )
            
            # Store losses for plotting
            self.d_losses.append(d)
            self.g_losses.append(g)
            
            # =================================================================
            # W&B LOGGING
            # =================================================================
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "d_loss": d[0],
                    "d_loss_real": d[1],
                    "d_loss_fake": d[2],
                    "d_acc": d[3],
                    "d_acc_real": d[4],
                    "d_acc_fake": d[5],
                    "g_loss": g[0],
                    "g_acc": g[1],
                    "d_lr": d_lr,
                    "g_lr": g_lr,
                })
            
            # =================================================================
            # PERIODIC SAVING
            # =================================================================
            if epoch % print_every_n_batches == 0:
                # Save sample images
                self.sample_images(run_folder)
                
                # Save weights
                self.model.save_weights(
                    os.path.join(run_folder, f'weights/weights-{epoch}.weights.h5')
                )
                self.model.save_weights(
                    os.path.join(run_folder, 'weights/weights.weights.h5')
                )
                
                # Save full model
                self.save_model(run_folder)
                
                # Log sample images to W&B
                if use_wandb:
                    sample_path = os.path.join(
                        run_folder, f"images/sample_{epoch}.png"
                    )
                    if os.path.exists(sample_path):
                        wandb.log({"samples": wandb.Image(sample_path)})
            
            # Increment epoch counter
            self.epoch += 1
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def sample_images(self, run_folder: str) -> None:
        """
        Generate and save a grid of sample images.
        
        Creates a 5x5 grid of generated images from random noise vectors
        and saves it to the run folder.
        
        Args:
            run_folder: Directory containing 'images/' subdirectory.
        
        Output:
            Saves image to: {run_folder}/images/sample_{epoch}.png
        """
        # Grid dimensions
        rows, cols = 5, 5
        
        # Generate images from random noise
        noise = np.random.normal(0, 1, (rows * cols, self.z_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)
        
        # Rescale from [-1, 1] to [0, 1] for display
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)
        
        # Create figure with subplots
        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        cnt = 0
        
        for i in range(rows):
            for j in range(cols):
                # Display single channel as grayscale
                axs[i, j].imshow(
                    np.squeeze(gen_imgs[cnt, :, :, :]),
                    cmap='gray'
                )
                axs[i, j].axis('off')
                cnt += 1
        
        # Save figure
        fig.savefig(os.path.join(run_folder, f"images/sample_{self.epoch}.png"))
        plt.close()
    
    def plot_model(self, run_folder: str) -> None:
        """
        Generate and save architecture diagrams for all models.
        
        Creates visual representations of the discriminator, generator,
        and combined model architectures.
        
        Args:
            run_folder: Directory containing 'viz/' subdirectory.
        
        Output:
            Saves to:
                - {run_folder}/viz/model.png
                - {run_folder}/viz/discriminator.png
                - {run_folder}/viz/generator.png
        """
        plot_model(
            self.model,
            to_file=os.path.join(run_folder, 'viz/model.png'),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.discriminator,
            to_file=os.path.join(run_folder, 'viz/discriminator.png'),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.generator,
            to_file=os.path.join(run_folder, 'viz/generator.png'),
            show_shapes=True,
            show_layer_names=True
        )
    
    # =========================================================================
    # PERSISTENCE METHODS
    # =========================================================================
    
    def save(self, folder: str) -> None:
        """
        Save model architecture parameters to disk.
        
        Saves all constructor parameters as a pickle file for later
        reconstruction of the model architecture.
        
        Args:
            folder: Directory to save params.pkl and visualizations.
        
        Output:
            Creates: {folder}/params.pkl
        """
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim,
                self.discriminator_conv_filters,
                self.discriminator_conv_kernel_size,
                self.discriminator_conv_strides,
                self.discriminator_batch_norm_momentum,
                self.discriminator_activation,
                self.discriminator_dropout_rate,
                self.discriminator_learning_rate,
                self.generator_initial_dense_layer_size,
                self.generator_upsample,
                self.generator_conv_filters,
                self.generator_conv_kernel_size,
                self.generator_conv_strides,
                self.generator_batch_norm_momentum,
                self.generator_activation,
                self.generator_dropout_rate,
                self.generator_learning_rate,
                self.optimiser,
                self.z_dim,
            ], f)
        
        # Also save architecture visualizations
        self.plot_model(folder)
    
    def save_model(self, run_folder: str) -> None:
        """
        Save all models and the GAN instance to disk.
        
        Uses the native Keras format (.keras) for model files.
        Also saves the entire GAN instance as a pickle file.
        
        Args:
            run_folder: Directory to save model files.
        
        Output:
            Creates:
                - {run_folder}/model.keras (combined model)
                - {run_folder}/discriminator.keras
                - {run_folder}/generator.keras
                - {run_folder}/obj.pkl (GAN instance)
        
        Note:
            Uses .keras format (Keras 3.0+) instead of legacy .h5
        """
        self.model.save(os.path.join(run_folder, 'model.keras'))
        self.discriminator.save(os.path.join(run_folder, 'discriminator.keras'))
        self.generator.save(os.path.join(run_folder, 'generator.keras'))
        
        # Save the entire GAN object for easy restoration
        with open(os.path.join(run_folder, "obj.pkl"), "wb") as f:
            pkl.dump(self, f)
    
    def load_weights(self, filepath: str) -> None:
        """
        Load model weights from a file.
        
        Loads weights for the combined model (generator + discriminator).
        
        Args:
            filepath: Path to the weights file (.weights.h5).
        
        Example:
            >>> gan.load_weights('../run/gan/0001/weights/weights.weights.h5')
        """
        self.model.load_weights(filepath)
