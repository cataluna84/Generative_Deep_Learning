"""
Wasserstein GAN (WGAN) Implementation.

This module provides a complete implementation of the Wasserstein GAN
architecture for image generation, as described in:
    Arjovsky, M., Chintala, S., & Bottou, L. (2017).
    "Wasserstein Generative Adversarial Networks"

Key Features:
    - Wasserstein loss for stable training
    - Weight clipping for Lipschitz constraint enforcement
    - Configurable critic and generator architectures
    - Verbose training with comprehensive metrics
    - Quality metrics (FID, Inception Score) support

Module Location:
    v1/src/models/WGAN.py

Usage:
    from models.WGAN import WGAN

    # Initialize WGAN
    gan = WGAN(
        input_dim=(32, 32, 3),
        critic_conv_filters=[64, 128, 256, 512],
        ...
    )

    # Train
    gan.train(x_train, batch_size=512, epochs=6000, run_folder=RUN_FOLDER)

Author:
    Based on "Generative Deep Learning" book implementation.
    Enhanced with PEP-8 documentation and metrics logging.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Flatten, Dense, Conv2DTranspose,
    Reshape, Activation, BatchNormalization, LeakyReLU,
    Dropout, UpSampling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomNormal

# Standard library
import numpy as np
import os
import pickle
import time

# Visualization
import matplotlib.pyplot as plt


# =============================================================================
# METRICS IMPORTS (OPTIONAL)
# =============================================================================
# These imports are optional for verbose training output.
# Training will work without them using simple output.

try:
    from utils.gan.metrics import (
        collect_epoch_metrics,
        format_verbose_output,
        EpochTimer,
    )
    from utils.gan.quality_metrics import (
        collect_quality_metrics,
        format_quality_output,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


# =============================================================================
# WEIGHTS & BIASES INTEGRATION (OPTIONAL)
# =============================================================================
# W&B is used for real-time experiment tracking and visualization.
# Training will work without it - metrics are still logged to console.

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# WGAN CLASS
# =============================================================================

class WGAN:
    """
    Wasserstein Generative Adversarial Network (WGAN).

    WGAN uses the Wasserstein distance (Earth Mover's distance) as the loss
    function, providing more stable training and meaningful loss values compared
    to traditional GANs.

    Key differences from vanilla GAN:
        - Critic outputs unbounded scores (no sigmoid)
        - Wasserstein loss instead of binary cross-entropy
        - Weight clipping to enforce Lipschitz constraint
        - Multiple critic updates per generator update

    Attributes:
        name (str): Model name identifier.
        input_dim (tuple): Input image dimensions (H, W, C).
        z_dim (int): Latent space dimensionality.
        critic (Model): The critic (discriminator) network.
        generator (Model): The generator network.
        model (Model): Combined adversarial model.
        d_losses (list): History of discriminator losses.
        g_losses (list): History of generator losses.
        metrics_history (dict): Detailed metrics history for analysis.
        epoch (int): Current training epoch counter.

    Example:
        >>> gan = WGAN(
        ...     input_dim=(32, 32, 3),
        ...     critic_conv_filters=[64, 128, 256, 512],
        ...     critic_conv_kernel_size=[5, 5, 5, 5],
        ...     critic_conv_strides=[2, 2, 2, 2],
        ...     critic_batch_norm_momentum=None,
        ...     critic_activation='leaky_relu',
        ...     critic_dropout_rate=None,
        ...     critic_learning_rate=0.00005,
        ...     generator_initial_dense_layer_size=(4, 4, 512),
        ...     generator_upsample=[2, 2, 2, 1],
        ...     generator_conv_filters=[256, 128, 64, 3],
        ...     generator_conv_kernel_size=[5, 5, 5, 5],
        ...     generator_conv_strides=[1, 1, 1, 1],
        ...     generator_batch_norm_momentum=0.9,
        ...     generator_activation='leaky_relu',
        ...     generator_dropout_rate=None,
        ...     generator_learning_rate=0.00005,
        ...     optimiser='rmsprop',
        ...     z_dim=100
        ... )
        >>> gan.train(x_train, batch_size=512, epochs=6000, run_folder='run/')
    """

    def __init__(
        self,
        input_dim,
        critic_conv_filters,
        critic_conv_kernel_size,
        critic_conv_strides,
        critic_batch_norm_momentum,
        critic_activation,
        critic_dropout_rate,
        critic_learning_rate,
        generator_initial_dense_layer_size,
        generator_upsample,
        generator_conv_filters,
        generator_conv_kernel_size,
        generator_conv_strides,
        generator_batch_norm_momentum,
        generator_activation,
        generator_dropout_rate,
        generator_learning_rate,
        optimiser,
        z_dim
    ):
        """
        Initialize the WGAN model.

        Args:
            input_dim (tuple): Shape of input images (height, width, channels).
            critic_conv_filters (list): Number of filters for each critic layer.
            critic_conv_kernel_size (list): Kernel sizes for each critic layer.
            critic_conv_strides (list): Stride values for each critic layer.
            critic_batch_norm_momentum (float or None): BatchNorm momentum.
                Use None to disable BatchNorm (recommended for WGAN).
            critic_activation (str): Activation function ('leaky_relu', 'relu').
            critic_dropout_rate (float or None): Dropout rate for critic.
            critic_learning_rate (float): Learning rate for critic optimizer.
            generator_initial_dense_layer_size (tuple): Initial reshape dimensions
                after the dense layer (height, width, channels).
            generator_upsample (list): Upsample factor per layer (2 for 2x, 1 for none).
            generator_conv_filters (list): Number of filters for each generator layer.
            generator_conv_kernel_size (list): Kernel sizes for each generator layer.
            generator_conv_strides (list): Stride values for each generator layer.
            generator_batch_norm_momentum (float or None): BatchNorm momentum.
            generator_activation (str): Activation function for generator.
            generator_dropout_rate (float or None): Dropout rate for generator.
            generator_learning_rate (float): Learning rate for generator optimizer.
            optimiser (str): Optimizer type ('adam', 'rmsprop').
            z_dim (int): Dimensionality of the latent space.
        """
        # =====================================================================
        # MODEL IDENTIFICATION
        # =====================================================================
        self.name = 'wgan'

        # =====================================================================
        # CRITIC CONFIGURATION
        # =====================================================================
        self.input_dim = input_dim
        self.critic_conv_filters = critic_conv_filters
        self.critic_conv_kernel_size = critic_conv_kernel_size
        self.critic_conv_strides = critic_conv_strides
        self.critic_batch_norm_momentum = critic_batch_norm_momentum
        self.critic_activation = critic_activation
        self.critic_dropout_rate = critic_dropout_rate
        self.critic_learning_rate = critic_learning_rate

        # =====================================================================
        # GENERATOR CONFIGURATION
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
        # TRAINING CONFIGURATION
        # =====================================================================
        self.optimiser = optimiser
        self.z_dim = z_dim

        # =====================================================================
        # DERIVED ATTRIBUTES
        # =====================================================================
        self.n_layers_critic = len(critic_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        # Weight initialization following DCGAN guidelines
        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        # =====================================================================
        # TRAINING STATE
        # =====================================================================
        # Loss history for plotting and analysis
        self.d_losses = []
        self.g_losses = []

        # Current epoch counter (allows resuming training)
        self.epoch = 0

        # Detailed metrics history for verbose logging
        self.metrics_history = {
            'wasserstein_dist': [],
            'dg_ratio': [],
            'clip_ratio': [],
            'epoch_time': [],
            'critic_weight_mean': [],
            'critic_weight_std': [],
            'generator_weight_mean': [],
            'generator_weight_std': [],
        }

        # =====================================================================
        # BUILD MODELS
        # =====================================================================
        self._build_critic()
        self._build_generator()
        self._build_adversarial()

    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================

    def wasserstein(self, y_true, y_pred):
        """
        Wasserstein loss function.

        In WGAN, the loss is the negative product of labels and predictions.
        The critic tries to maximize this (output high for real, low for fake),
        while the generator tries to minimize it.

        Args:
            y_true: Ground truth labels (+1 for real, -1 for fake).
            y_pred: Critic predictions (unbounded scores).

        Returns:
            Negative mean of element-wise product.
        """
        return -K.mean(y_true * y_pred)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_activation(self, activation):
        """
        Get the appropriate activation layer.

        Args:
            activation (str): Activation name ('leaky_relu' or standard name).

        Returns:
            Keras activation layer.
        """
        if activation == 'leaky_relu':
            # LeakyReLU with slope 0.2 (DCGAN recommendation)
            return LeakyReLU(negative_slope=0.2)
        else:
            return Activation(activation)

    def get_opti(self, lr):
        """
        Get the appropriate optimizer.

        Args:
            lr (float): Learning rate.

        Returns:
            Keras optimizer instance.

        Note:
            RMSprop is recommended for WGAN as per the original paper.
            Adam with beta_1=0.5 is used for GAN stability.
        """
        if self.optimiser == 'adam':
            return Adam(learning_rate=lr, beta_1=0.5)
        elif self.optimiser == 'rmsprop':
            return RMSprop(learning_rate=lr)
        else:
            return Adam(learning_rate=lr)

    def set_trainable(self, model, trainable):
        """
        Set the trainable status of a model and all its layers.

        Args:
            model: Keras model to modify.
            trainable (bool): Whether layers should be trainable.
        """
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

    # =========================================================================
    # MODEL BUILDING
    # =========================================================================

    def _build_critic(self):
        """
        Build the critic (discriminator) network.

        The critic outputs an unbounded score (no sigmoid activation).
        Higher scores indicate "more real" images, lower scores indicate
        "more fake" images.

        Architecture:
            - Input: Image tensor
            - N convolutional layers with configurable filters/strides
            - Optional BatchNorm (not recommended for WGAN critic)
            - Activation (typically LeakyReLU)
            - Optional Dropout
            - Flatten
            - Dense(1) with no activation

        Sets:
            self.critic: The critic Keras Model.
        """
        # Input layer
        critic_input = Input(shape=self.input_dim, name='critic_input')
        x = critic_input

        # ---------------------------------------------------------------------
        # Convolutional layers
        # ---------------------------------------------------------------------
        for i in range(self.n_layers_critic):
            # Convolution
            x = Conv2D(
                filters=self.critic_conv_filters[i],
                kernel_size=self.critic_conv_kernel_size[i],
                strides=self.critic_conv_strides[i],
                padding='same',
                name=f'critic_conv_{i}',
                kernel_initializer=self.weight_init
            )(x)

            # BatchNorm (skip first layer, optional for WGAN)
            if self.critic_batch_norm_momentum and i > 0:
                x = BatchNormalization(
                    momentum=self.critic_batch_norm_momentum
                )(x)

            # Activation
            x = self.get_activation(self.critic_activation)(x)

            # Dropout (if specified)
            if self.critic_dropout_rate:
                x = Dropout(rate=self.critic_dropout_rate)(x)

        # ---------------------------------------------------------------------
        # Output layers
        # ---------------------------------------------------------------------
        x = Flatten()(x)

        # Output: Single unbounded score (no activation)
        critic_output = Dense(
            1,
            activation=None,
            kernel_initializer=self.weight_init
        )(x)

        # Build model
        self.critic = Model(critic_input, critic_output, name='critic')

    def _build_generator(self):
        """
        Build the generator network.

        The generator transforms random noise vectors into images.
        Uses transposed convolutions or upsampling + convolution
        to progressively increase spatial dimensions.

        Architecture:
            - Input: Noise vector (z_dim,)
            - Dense to initial feature map size
            - BatchNorm (optional) + Activation
            - Reshape to (H, W, C)
            - N upsample/conv layers
            - Final tanh activation (output in [-1, 1])

        Sets:
            self.generator: The generator Keras Model.
        """
        # Input layer (latent vector)
        generator_input = Input(shape=(self.z_dim,), name='generator_input')
        x = generator_input

        # ---------------------------------------------------------------------
        # Initial dense layer
        # ---------------------------------------------------------------------
        # Calculate total units needed for initial reshape
        initial_units = int(np.prod(self.generator_initial_dense_layer_size))
        x = Dense(initial_units, kernel_initializer=self.weight_init)(x)

        # BatchNorm and activation
        if self.generator_batch_norm_momentum:
            x = BatchNormalization(
                momentum=self.generator_batch_norm_momentum
            )(x)
        x = self.get_activation(self.generator_activation)(x)

        # Reshape to initial feature map
        x = Reshape(self.generator_initial_dense_layer_size)(x)

        # Dropout (if specified)
        if self.generator_dropout_rate:
            x = Dropout(rate=self.generator_dropout_rate)(x)

        # ---------------------------------------------------------------------
        # Upsampling / Transposed convolution layers
        # ---------------------------------------------------------------------
        for i in range(self.n_layers_generator):
            if self.generator_upsample[i] == 2:
                # Upsample then convolve (often produces better results)
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    name=f'generator_conv_{i}',
                    kernel_initializer=self.weight_init
                )(x)
            else:
                # Transposed convolution
                x = Conv2DTranspose(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    strides=self.generator_conv_strides[i],
                    name=f'generator_conv_{i}',
                    kernel_initializer=self.weight_init
                )(x)

            # All layers except last: BatchNorm + activation
            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(
                        momentum=self.generator_batch_norm_momentum
                    )(x)
                x = self.get_activation(self.generator_activation)(x)
            else:
                # Last layer: tanh activation (output in [-1, 1])
                x = Activation('tanh')(x)

        generator_output = x

        # Build model
        self.generator = Model(generator_input, generator_output, name='generator')

    def _build_adversarial(self):
        """
        Build and compile the adversarial (combined) model.

        Creates two compiled models:
            1. Critic: Standalone critic for training on real/fake images
            2. Model: Generator + frozen critic for training generator

        The critic is frozen when training the generator to prevent
        the critic from updating during generator training steps.

        Sets:
            self.model: The combined adversarial model.
        """
        # ---------------------------------------------------------------------
        # Compile the critic
        # ---------------------------------------------------------------------
        self.critic.compile(
            optimizer=self.get_opti(self.critic_learning_rate),
            loss=self.wasserstein
        )

        # ---------------------------------------------------------------------
        # Build combined model (generator -> critic)
        # ---------------------------------------------------------------------
        # Freeze critic for combined model
        self.set_trainable(self.critic, False)

        # Connect generator output to critic input
        model_input = Input(shape=(self.z_dim,), name='model_input')
        generated_image = self.generator(model_input)
        model_output = self.critic(generated_image)

        self.model = Model(model_input, model_output, name='adversarial')

        # Compile combined model
        self.model.compile(
            optimizer=self.get_opti(self.generator_learning_rate),
            loss=self.wasserstein
        )

        # Unfreeze critic (will be frozen during generator training steps)
        self.set_trainable(self.critic, True)

    # =========================================================================
    # TRAINING METHODS
    # =========================================================================

    def train_critic(self, x_train, batch_size, clip_threshold, using_generator):
        """
        Perform one critic training step.

        The critic is trained to maximize the Wasserstein distance:
            maximize E[critic(real)] - E[critic(fake)]

        After training, weights are clipped to [-c, c] to enforce
        the Lipschitz constraint.

        Args:
            x_train: Training data (numpy array or generator).
            batch_size (int): Number of samples per batch.
            clip_threshold (float): Weight clipping value.
            using_generator (bool): If True, x_train is a data generator.

        Returns:
            list: [d_loss_total, d_loss_real, d_loss_fake]
        """
        # Labels: +1 for real, -1 for fake
        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))

        # ---------------------------------------------------------------------
        # Get real images
        # ---------------------------------------------------------------------
        if using_generator:
            true_imgs = next(x_train)[0]
            # Handle last batch potentially being smaller
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        # ---------------------------------------------------------------------
        # Generate fake images
        # ---------------------------------------------------------------------
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)

        # ---------------------------------------------------------------------
        # Train critic on real and fake images
        # ---------------------------------------------------------------------
        d_loss_real = self.critic.train_on_batch(true_imgs, valid)
        d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # ---------------------------------------------------------------------
        # Weight clipping (Lipschitz constraint)
        # ---------------------------------------------------------------------
        for layer in self.critic.layers:
            weights = layer.get_weights()
            clipped_weights = [
                np.clip(w, -clip_threshold, clip_threshold)
                for w in weights
            ]
            layer.set_weights(clipped_weights)

        return [d_loss, d_loss_real, d_loss_fake]

    def train_generator(self, batch_size):
        """
        Perform one generator training step.

        The generator is trained to minimize the Wasserstein distance
        by fooling the critic into outputting high scores for fake images.

        Args:
            batch_size (int): Number of samples to generate.

        Returns:
            float: Generator loss value.
        """
        # Target: we want the critic to output +1 (real) for generated images
        valid = np.ones((batch_size, 1))

        # Generate noise vectors
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))

        # Train generator via the combined model
        return self.model.train_on_batch(noise, valid)

    def train(
        self,
        x_train,
        batch_size,
        epochs,
        run_folder,
        print_every_n_batches=10,
        n_critic=5,
        clip_threshold=0.01,
        using_generator=False,
        verbose=True,
        quality_metrics_every=100,
        wandb_log=True
    ):
        """
        Train the WGAN model.

        Implements the WGAN training algorithm:
            1. For each epoch:
                a. Train critic n_critic times
                b. Train generator once
                c. Log metrics and save samples periodically

        Args:
            x_train: Training data (numpy array or generator).
            batch_size (int): Number of samples per batch.
            epochs (int): Number of epochs to train.
            run_folder (str): Path to save outputs (images, weights, models).
            print_every_n_batches (int): Save samples every N epochs.
                Default: 10.
            n_critic (int): Number of critic updates per generator update.
                Default: 5 (as recommended in WGAN paper).
            clip_threshold (float): Weight clipping threshold.
                Default: 0.01.
            using_generator (bool): If True, x_train is a data generator.
                Default: False.
            verbose (bool): If True, show detailed metrics output.
                Default: True.
            quality_metrics_every (int): Compute FID/IS every N epochs.
                Set to 0 to disable. Default: 100.
            wandb_log (bool): If True, log metrics to W&B in real-time.
                Requires W&B to be initialized. Default: True.

        Note:
            Training can be resumed by calling train() again.
            The epoch counter persists between calls.
        """
        # =====================================================================
        # TRAINING LOOP
        # =====================================================================
        # Calculate total epochs once at the start (before loop increments self.epoch)
        start_epoch = self.epoch
        total_epochs = start_epoch + epochs

        for epoch in range(start_epoch, total_epochs):
            epoch_start = time.time()

            # -----------------------------------------------------------------
            # Train critic n_critic times
            # -----------------------------------------------------------------
            for _ in range(n_critic):
                d_loss = self.train_critic(
                    x_train, batch_size, clip_threshold, using_generator
                )

            # -----------------------------------------------------------------
            # Train generator once
            # -----------------------------------------------------------------
            g_loss = self.train_generator(batch_size)

            epoch_time = time.time() - epoch_start

            # -----------------------------------------------------------------
            # Store losses
            # -----------------------------------------------------------------
            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            # -----------------------------------------------------------------
            # Output metrics
            # -----------------------------------------------------------------
            if verbose and METRICS_AVAILABLE:
                # Collect comprehensive metrics
                metrics = collect_epoch_metrics(
                    critic=self.critic,
                    generator=self.generator,
                    d_loss=d_loss,
                    g_loss=g_loss,
                    clip_threshold=clip_threshold,
                    epoch_time=epoch_time,
                    loss_history=self.d_losses if len(self.d_losses) > 1 else None
                )

                # Store metrics history
                self.metrics_history['wasserstein_dist'].append(
                    metrics.get('wasserstein_distance', 0)
                )
                for key in ['dg_ratio', 'clip_ratio', 'epoch_time']:
                    self.metrics_history[key].append(metrics.get(key, 0))
                for key in ['critic_weight_mean', 'critic_weight_std',
                            'generator_weight_mean', 'generator_weight_std']:
                    self.metrics_history[key].append(metrics.get(key, 0))

                # Print verbose output
                print(format_verbose_output(epoch, total_epochs, metrics))

                # ---------------------------------------------------------
                # Real-time W&B Logging
                # ---------------------------------------------------------
                # Log all per-epoch metrics to W&B for live dashboard.
                # Uses step=epoch for proper x-axis alignment in charts.
                if wandb_log and WANDB_AVAILABLE:
                    try:
                        wandb.log({
                            # Loss metrics
                            'd_loss': metrics['d_loss'],
                            'd_loss_real': metrics['d_loss_real'],
                            'd_loss_fake': metrics['d_loss_fake'],
                            'g_loss': metrics['g_loss'],
                            'wasserstein_distance': metrics['wasserstein_distance'],
                            # Weight statistics
                            'critic_weight_mean': metrics['critic_weight_mean'],
                            'critic_weight_std': metrics['critic_weight_std'],
                            'generator_weight_mean': metrics['generator_weight_mean'],
                            'generator_weight_std': metrics['generator_weight_std'],
                            # Stability metrics
                            'dg_ratio': metrics['dg_ratio'],
                            'clip_ratio': metrics['clip_ratio'],
                            'epoch_time': metrics['epoch_time'],
                            # Loss variance (if available)
                            'loss_variance': metrics.get('loss_variance', 0),
                        }, step=epoch)
                    except Exception as e:
                        # Silently ignore W&B errors to not interrupt training
                        pass

                # Quality metrics every N epochs OR on the final epoch
                # This ensures we always get a final quality assessment.
                is_quality_epoch = (epoch % quality_metrics_every == 0 and epoch > 0)
                is_final_epoch = (epoch == total_epochs - 1)

                if quality_metrics_every > 0 and (is_quality_epoch or is_final_epoch):
                    print("\n  Computing quality metrics...")
                    try:
                        # Generate sample images
                        noise = np.random.normal(0, 1, (100, self.z_dim))
                        gen_imgs = self.generator.predict(noise, verbose=0)

                        # Get sample real images
                        if using_generator:
                            real_imgs = next(x_train)[0]
                        else:
                            idx = np.random.randint(0, x_train.shape[0], 100)
                            real_imgs = x_train[idx]

                        quality = collect_quality_metrics(
                            real_images=real_imgs,
                            fake_images=gen_imgs,
                            compute_fid=True,
                            compute_is=True
                        )
                        print(format_quality_output(quality))

                        # -------------------------------------------------
                        # Log quality metrics to W&B
                        # -------------------------------------------------
                        # Quality metrics are logged as sparse points
                        # (every 100 epochs) since they're expensive to compute.
                        if wandb_log and WANDB_AVAILABLE:
                            try:
                                wandb.log({
                                    'fid_score': quality.get('fid_score', 0),
                                    'inception_score_mean': quality.get(
                                        'inception_score_mean', 0
                                    ),
                                    'inception_score_std': quality.get(
                                        'inception_score_std', 0
                                    ),
                                    'pixel_variance': quality.get(
                                        'pixel_variance', 0
                                    ),
                                }, step=epoch)
                            except Exception:
                                pass  # Silently ignore W&B errors

                    except Exception as e:
                        print(f"  ⚠ Quality metrics failed: {e}")
            else:
                # Simple output (original behavior)
                print(
                    f"{epoch} [D loss: ({d_loss[0]:.3f})"
                    f"(R {d_loss[1]:.3f}, F {d_loss[2]:.3f})]  "
                    f"[G loss: {g_loss:.3f}]"
                )

            # -----------------------------------------------------------------
            # Save checkpoints
            # -----------------------------------------------------------------
            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.model.save_weights(
                    os.path.join(run_folder, f'weights/weights-{epoch}.weights.h5')
                )
                self.model.save_weights(
                    os.path.join(run_folder, 'weights/weights.weights.h5')
                )
                self.save_model(run_folder)

            self.epoch += 1

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================

    def sample_images(self, run_folder):
        """
        Generate and save a grid of sample images.

        Creates a 5x5 grid of generated images and saves to the
        images/ subdirectory of run_folder.

        Args:
            run_folder (str): Path to run folder.
        """
        rows, cols = 5, 5
        num_samples = rows * cols

        # Generate images from random noise
        noise = np.random.normal(0, 1, (num_samples, self.z_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)

        # Rescale from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        # Create figure
        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        cnt = 0

        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(
                    np.squeeze(gen_imgs[cnt, :, :, :]),
                    cmap='gray_r'
                )
                axs[i, j].axis('off')
                cnt += 1

        # Save figure
        fig.savefig(
            os.path.join(run_folder, f"images/sample_{self.epoch}.png")
        )
        plt.close()

    def plot_model(self, run_folder):
        """
        Generate and save model architecture diagrams.

        Saves three diagrams to the viz/ subdirectory:
            - model.png: Combined adversarial model
            - critic.png: Critic architecture
            - generator.png: Generator architecture

        Args:
            run_folder (str): Path to run folder.
        """
        plot_model(
            self.model,
            to_file=os.path.join(run_folder, 'viz/model.png'),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.critic,
            to_file=os.path.join(run_folder, 'viz/critic.png'),
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

    def save(self, folder):
        """
        Save model parameters and architecture diagrams.

        Saves a pickle file containing all hyperparameters and
        generates architecture visualization diagrams.

        Args:
            folder (str): Path to save folder.
        """
        params = [
            self.input_dim,
            self.critic_conv_filters,
            self.critic_conv_kernel_size,
            self.critic_conv_strides,
            self.critic_batch_norm_momentum,
            self.critic_activation,
            self.critic_dropout_rate,
            self.critic_learning_rate,
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
        ]

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)

        self.plot_model(folder)

    def save_model(self, run_folder):
        """
        Save model weights and architecture.

        Saves the complete models in native Keras format (.keras)
        and a pickle of the WGAN object for easy reloading.

        Args:
            run_folder (str): Path to run folder.
        """
        # Save models in native Keras format
        self.model.save(os.path.join(run_folder, 'model.keras'))
        self.critic.save(os.path.join(run_folder, 'critic.keras'))
        self.generator.save(os.path.join(run_folder, 'generator.keras'))

        # Save WGAN object
        with open(os.path.join(run_folder, 'obj.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def load_weights(self, filepath):
        """
        Load model weights from a file.

        Args:
            filepath (str): Path to weights file (.h5 or .weights.h5).
        """
        self.model.load_weights(filepath)
