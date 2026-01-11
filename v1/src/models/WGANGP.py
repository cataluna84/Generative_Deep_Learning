"""
Wasserstein GAN with Gradient Penalty (WGAN-GP) Implementation.

This module provides a complete implementation of the WGAN-GP architecture
for image generation, as described in:
    Gulrajani et al. "Improved Training of Wasserstein GANs" (2017)
    https://arxiv.org/abs/1704.00028

WGAN-GP improves upon the original WGAN by replacing weight clipping with
a gradient penalty term, which provides more stable training and better
convergence properties.

Key Concepts:
    - **Wasserstein Distance**: Also known as Earth Mover's Distance, measures
      the minimum "cost" to transform one distribution into another.
    - **Gradient Penalty**: Enforces the Lipschitz constraint by penalizing
      gradients that deviate from unit norm, computed on interpolated samples.
    - **Critic**: In WGAN nomenclature, the discriminator is called a "critic"
      because it outputs unbounded scores (not probabilities).

Architecture Overview:
    - Critic: CNN that outputs a scalar "realness" score (no sigmoid)
    - Generator: Deconvolutional network that generates images from noise
    - Gradient Penalty: Computed on interpolated samples between real and fake

Typical Usage:
    >>> from WGANGP import WGANGP
    >>> gan = WGANGP(
    ...     input_dim=(64, 64, 3),
    ...     critic_conv_filters=[64, 128, 256],
    ...     critic_conv_kernel_size=[5, 5, 5],
    ...     critic_conv_strides=[2, 2, 2],
    ...     # ... other parameters
    ... )
    >>> gan.train(x_train, batch_size=64, epochs=10000, run_folder='./output')

References:
    - Original WGAN: Arjovsky et al. (2017) "Wasserstein GAN"
    - WGAN-GP: Gulrajani et al. (2017) "Improved Training of Wasserstein GANs"
    - Chapter 4 of "Generative Deep Learning" book

Author: Adapted from David Foster's "Generative Deep Learning" code
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================

# -----------------------------------------------------------------------------
# TensorFlow and Keras
# -----------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Flatten, Dense, Conv2DTranspose,
    Reshape, Activation, BatchNormalization, LeakyReLU,
    Dropout, UpSampling2D, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomNormal

# -----------------------------------------------------------------------------
# Standard Library
# -----------------------------------------------------------------------------
from functools import partial
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


# =============================================================================
# CUSTOM LAYERS
# =============================================================================

class RandomWeightedAverage(Layer):
    """
    Custom Keras layer that computes a random weighted average of two inputs.

    This layer is essential for computing the gradient penalty in WGAN-GP.
    It creates interpolated samples between real and generated images using
    random weights sampled uniformly from [0, 1].

    The interpolation formula is:
        x_interpolated = α * x_real + (1 - α) * x_fake

    where α is a random tensor with the same batch size as the inputs.

    Attributes:
        batch_size (int): The batch size for generating random weights.

    Example:
        >>> real_images = Input(shape=(64, 64, 3))
        >>> fake_images = generator(noise)
        >>> interpolated = RandomWeightedAverage(batch_size=64)(
        ...     [real_images, fake_images]
        ... )

    Note:
        The random weights α are sampled independently for each sample in
        the batch, and broadcast across spatial dimensions (H, W, C).
    """

    def __init__(self, batch_size, **kwargs):
        """
        Initialize the RandomWeightedAverage layer.

        Args:
            batch_size (int): Number of samples in each batch.
            **kwargs: Additional keyword arguments passed to the parent Layer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        """
        Compute the random weighted average of two input tensors.

        Args:
            inputs (list): A list of two tensors [real_samples, fake_samples]
                           with shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: Interpolated samples with the same shape as inputs.
        """
        # Generate random weights α with shape (batch_size, 1, 1, 1)
        # The 1s allow broadcasting across H, W, C dimensions
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))

        # Compute: α * real + (1 - α) * fake
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def get_config(self):
        """
        Return the layer configuration for serialization.

        Returns:
            dict: Configuration dictionary including batch_size.
        """
        config = super().get_config()
        config.update({"batch_size": self.batch_size})
        return config


# =============================================================================
# WGAN-GP CLASS
# =============================================================================

class WGANGP:
    """
    Wasserstein Generative Adversarial Network with Gradient Penalty.

    WGAN-GP replaces weight clipping (used in standard WGAN) with a gradient
    penalty term that encourages the critic to have unit gradient norm on
    interpolated samples. This provides:

    1. **More stable training**: No vanishing/exploding gradients from clipping
    2. **Better convergence**: Smoother loss landscape
    3. **Higher quality outputs**: Less mode collapse

    The total critic loss is:
        L_critic = E[f(fake)] - E[f(real)] + λ * E[(||∇f(x_interp)||₂ - 1)²]

    Where:
        - f is the critic function
        - x_interp is a random interpolation between real and fake samples
        - λ (grad_weight) controls the penalty strength (typically 10)

    Attributes:
        name (str): Model identifier ('gan').
        input_dim (tuple): Shape of input images (H, W, C).
        z_dim (int): Dimensionality of the latent noise vector.
        grad_weight (float): Weight of the gradient penalty term (λ).
        batch_size (int): Number of samples per training batch.
        critic (Model): The critic (discriminator) network.
        generator (Model): The generator network.
        critic_model (Model): Training model for the critic (includes GP).
        model (Model): Training model for the generator.
        d_losses (list): History of critic losses per epoch.
        g_losses (list): History of generator losses per epoch.
        epoch (int): Current training epoch counter.

    Example:
        >>> gan = WGANGP(
        ...     input_dim=(64, 64, 3),
        ...     critic_conv_filters=[64, 128, 256, 512],
        ...     critic_conv_kernel_size=[5, 5, 5, 5],
        ...     critic_conv_strides=[2, 2, 2, 2],
        ...     critic_batch_norm_momentum=None,  # No BN for WGAN-GP critic
        ...     critic_activation='leaky_relu',
        ...     critic_dropout_rate=None,
        ...     critic_learning_rate=0.0001,
        ...     generator_initial_dense_layer_size=(4, 4, 512),
        ...     generator_upsample=[2, 2, 2, 2],
        ...     generator_conv_filters=[256, 128, 64, 3],
        ...     generator_conv_kernel_size=[5, 5, 5, 5],
        ...     generator_conv_strides=[1, 1, 1, 1],
        ...     generator_batch_norm_momentum=0.9,
        ...     generator_activation='leaky_relu',
        ...     generator_dropout_rate=None,
        ...     generator_learning_rate=0.0001,
        ...     optimiser='adam',
        ...     grad_weight=10,
        ...     z_dim=100,
        ...     batch_size=64
        ... )
        >>> gan.train(x_train, batch_size=64, epochs=10000, run_folder='./run')
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

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
        grad_weight,
        z_dim,
        batch_size
    ):
        """
        Initialize the WGAN-GP model.

        Args:
            input_dim (tuple): Shape of input images (height, width, channels).
                Example: (64, 64, 3) for 64x64 RGB images.

            critic_conv_filters (list): Number of filters for each critic
                convolutional layer. Example: [64, 128, 256, 512].

            critic_conv_kernel_size (list): Kernel sizes for each critic layer.
                Example: [5, 5, 5, 5] for 5x5 kernels.

            critic_conv_strides (list): Stride values for each critic layer.
                Example: [2, 2, 2, 2] for 2x downsampling at each layer.

            critic_batch_norm_momentum (float or None): BatchNorm momentum.
                NOTE: BatchNorm is typically NOT used in WGAN-GP critic.
                Set to None to disable.

            critic_activation (str): Activation function ('leaky_relu' or other).

            critic_dropout_rate (float or None): Dropout rate (None to disable).

            critic_learning_rate (float): Learning rate for critic optimizer.
                Typical value: 0.0001 for Adam.

            generator_initial_dense_layer_size (tuple): Shape of the reshaped
                tensor after the initial dense layer (H, W, C).
                Example: (4, 4, 512) for 4x4 feature maps with 512 channels.

            generator_upsample (list): Upsampling factors per layer.
                Use 2 for 2x upsampling, 1 for no upsampling.

            generator_conv_filters (list): Filters for each generator layer.

            generator_conv_kernel_size (list): Kernel sizes for generator.

            generator_conv_strides (list): Strides for generator layers.

            generator_batch_norm_momentum (float or None): BatchNorm momentum
                for generator. Typically 0.9.

            generator_activation (str): Activation for generator layers.

            generator_dropout_rate (float or None): Dropout for generator.

            generator_learning_rate (float): Learning rate for generator.

            optimiser (str): Optimizer type ('adam' or 'rmsprop').

            grad_weight (float): Weight of gradient penalty term (λ).
                Typical value: 10 as per the original paper.

            z_dim (int): Dimensionality of the latent noise vector.
                Typical value: 100.

            batch_size (int): Number of samples per training batch.
                Required for RandomWeightedAverage layer.
        """
        # =====================================================================
        # MODEL IDENTIFICATION
        # =====================================================================
        self.name = 'gan'

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
        self.grad_weight = grad_weight
        self.batch_size = batch_size

        # =====================================================================
        # DERIVED ATTRIBUTES
        # =====================================================================
        self.n_layers_critic = len(critic_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        # Weight initialization following DCGAN guidelines
        # Small std (0.02) helps stable training
        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        # =====================================================================
        # TRAINING STATE
        # =====================================================================
        self.d_losses = []  # Critic loss history
        self.g_losses = []  # Generator loss history
        self.epoch = 0      # Current epoch counter

        # =====================================================================
        # BUILD MODELS
        # =====================================================================
        self._build_critic()
        self._build_generator()
        self._build_adversarial()

    # =========================================================================
    # LOSS FUNCTIONS
    # =========================================================================

    def gradient_penalty_loss(self, y_true, y_pred, interpolated_samples):
        """
        Compute the gradient penalty loss.

        The gradient penalty enforces the Lipschitz constraint by penalizing
        gradients that deviate from unit norm. This is computed on interpolated
        samples between real and fake images.

        The penalty is: λ * E[(||∇f(x_interp)||₂ - 1)²]

        Args:
            y_true: Ground truth labels (unused, required by Keras).
            y_pred: Critic predictions on interpolated samples.
            interpolated_samples: Interpolated images (mix of real and fake).

        Returns:
            tf.Tensor: Mean gradient penalty across the batch.

        Mathematical Details:
            1. Compute gradients: ∇f(x_interp) w.r.t. interpolated samples
            2. Compute L2 norm: ||∇f||₂ = sqrt(sum(∇f²))
            3. Compute penalty: (||∇f||₂ - 1)²
            4. Return mean across batch
        """
        # Compute gradients of critic output w.r.t. interpolated samples
        gradients = K.gradients(y_pred, interpolated_samples)[0]

        # Compute the L2 norm (Euclidean) of gradients
        # Step 1: Square each gradient element
        gradients_sqr = K.square(gradients)

        # Step 2: Sum over all dimensions except batch (axis 0)
        # This gives ||∇f||² for each sample
        gradients_sqr_sum = K.sum(
            gradients_sqr,
            axis=np.arange(1, len(gradients_sqr.shape))
        )

        # Step 3: Take square root to get ||∇f||₂
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)

        # Step 4: Compute penalty (||∇f||₂ - 1)²
        # We want gradients to have norm 1 (unit norm)
        gradient_penalty = K.square(1 - gradient_l2_norm)

        # Return mean penalty across the batch
        return K.mean(gradient_penalty)

    def wasserstein(self, y_true, y_pred):
        """
        Wasserstein loss function.

        The Wasserstein loss is simply the negative mean of element-wise
        product between labels and predictions. This encourages:
        - High output for real images (y_true = 1)
        - Low output for fake images (y_true = -1)

        Args:
            y_true: Ground truth labels (+1 for real, -1 for fake).
            y_pred: Critic predictions (unbounded scores).

        Returns:
            tf.Tensor: Negative mean of y_true * y_pred.
        """
        return -K.mean(y_true * y_pred)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_activation(self, activation):
        """
        Get the appropriate Keras activation layer.

        Args:
            activation (str): Activation name. Use 'leaky_relu' for
                LeakyReLU with slope 0.2, or any standard Keras activation.

        Returns:
            Layer: Keras activation layer instance.
        """
        if activation == 'leaky_relu':
            # LeakyReLU with slope 0.2 (DCGAN recommendation)
            layer = LeakyReLU(negative_slope=0.2)
        else:
            layer = Activation(activation)
        return layer

    def get_opti(self, lr):
        """
        Get the optimizer instance.

        Args:
            lr (float): Learning rate.

        Returns:
            Optimizer: Keras optimizer instance.

        Note:
            For GAN training, Adam with beta_1=0.5 is commonly used
            to reduce momentum and improve stability.
        """
        if self.optimiser == 'adam':
            # Lower beta_1 (0.5) helps GAN training stability
            opti = Adam(learning_rate=lr, beta_1=0.5)
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(learning_rate=lr)
        else:
            opti = Adam(learning_rate=lr)
        return opti

    def set_trainable(self, m, val):
        """
        Set the trainable status of a model and all its layers.

        This is used to freeze/unfreeze networks during adversarial training.
        When training the critic, the generator is frozen, and vice versa.

        Args:
            m (Model): Keras model to modify.
            val (bool): Whether layers should be trainable.
        """
        m.trainable = val
        for layer in m.layers:
            layer.trainable = val

    # =========================================================================
    # MODEL BUILDING
    # =========================================================================

    def _build_critic(self):
        """
        Build the critic (discriminator) network.

        The critic outputs an unbounded score indicating how "real" an input
        image appears. Unlike a traditional discriminator, there is no sigmoid
        activation at the output.

        Architecture:
            Input (H, W, C)
            → [Conv2D → (BatchNorm) → LeakyReLU → (Dropout)] × N layers
            → Flatten
            → Dense(1) with no activation

        Note:
            - BatchNorm is typically NOT used in WGAN-GP critic
            - The output is an unbounded scalar (no sigmoid)
        """
        # ---------------------------------------------------------------------
        # Input layer
        # ---------------------------------------------------------------------
        critic_input = Input(shape=self.input_dim, name='critic_input')
        x = critic_input

        # ---------------------------------------------------------------------
        # Convolutional layers
        # ---------------------------------------------------------------------
        for i in range(self.n_layers_critic):
            # Convolutional layer with strided downsampling
            x = Conv2D(
                filters=self.critic_conv_filters[i],
                kernel_size=self.critic_conv_kernel_size[i],
                strides=self.critic_conv_strides[i],
                padding='same',
                name=f'critic_conv_{i}',
                kernel_initializer=self.weight_init
            )(x)

            # Optional BatchNorm (skip first layer, not recommended for WGAN-GP)
            if self.critic_batch_norm_momentum and i > 0:
                x = BatchNormalization(
                    momentum=self.critic_batch_norm_momentum
                )(x)

            # Activation
            x = self.get_activation(self.critic_activation)(x)

            # Optional Dropout
            if self.critic_dropout_rate:
                x = Dropout(rate=self.critic_dropout_rate)(x)

        # ---------------------------------------------------------------------
        # Output layer
        # ---------------------------------------------------------------------
        x = Flatten()(x)

        # Single output with NO activation (unbounded score)
        critic_output = Dense(
            1,
            activation=None,
            kernel_initializer=self.weight_init
        )(x)

        self.critic = Model(critic_input, critic_output)

    def _build_generator(self):
        """
        Build the generator network.

        The generator transforms random noise vectors (z) into images.
        It uses a dense layer to expand the noise, then progressively
        upsamples through convolutional layers.

        Architecture:
            Input (z_dim,)
            → Dense → (BatchNorm) → LeakyReLU → Reshape to (H, W, C)
            → [UpSampling/Conv2D → (BatchNorm) → LeakyReLU] × (N-1) layers
            → Final Conv2D → tanh activation

        Note:
            - Final activation is tanh to output values in [-1, 1]
            - This matches input normalization to [-1, 1] range
        """
        # ---------------------------------------------------------------------
        # Input layer (latent noise vector)
        # ---------------------------------------------------------------------
        generator_input = Input(shape=(self.z_dim,), name='generator_input')
        x = generator_input

        # ---------------------------------------------------------------------
        # Initial dense layer to expand latent vector
        # ---------------------------------------------------------------------
        # Compute total units needed for initial feature map
        # Cast to int() for Keras 3.0+ compatibility (np.prod returns numpy.int64)
        initial_units = int(np.prod(self.generator_initial_dense_layer_size))
        x = Dense(
            initial_units,
            kernel_initializer=self.weight_init
        )(x)

        # Optional BatchNorm
        if self.generator_batch_norm_momentum:
            x = BatchNormalization(
                momentum=self.generator_batch_norm_momentum
            )(x)

        x = self.get_activation(self.generator_activation)(x)

        # Reshape to initial feature map (e.g., 4x4x512)
        x = Reshape(self.generator_initial_dense_layer_size)(x)

        # Optional Dropout
        if self.generator_dropout_rate:
            x = Dropout(rate=self.generator_dropout_rate)(x)

        # ---------------------------------------------------------------------
        # Upsampling / Convolutional layers
        # ---------------------------------------------------------------------
        for i in range(self.n_layers_generator):
            if self.generator_upsample[i] == 2:
                # UpSampling followed by Conv2D (often produces better results)
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    name=f'generator_conv_{i}',
                    kernel_initializer=self.weight_init
                )(x)
            else:
                # Transposed convolution (learnable upsampling)
                x = Conv2DTranspose(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    strides=self.generator_conv_strides[i],
                    name=f'generator_conv_{i}',
                    kernel_initializer=self.weight_init
                )(x)

            # All layers except the last: BatchNorm + activation
            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(
                        momentum=self.generator_batch_norm_momentum
                    )(x)
                x = self.get_activation(self.generator_activation)(x)
            else:
                # Last layer: tanh activation for output in [-1, 1]
                x = Activation('tanh')(x)

        generator_output = x
        self.generator = Model(generator_input, generator_output)

    def _build_adversarial(self):
        """
        Build and compile the adversarial training models.

        Creates two compiled models:
            1. critic_model: Trains the critic with gradient penalty
            2. model: Trains the generator to fool the critic

        The critic_model has three outputs:
            - Validity of real images
            - Validity of fake images
            - Validity of interpolated images (for gradient penalty)

        Loss function for critic:
            L = E[fake] - E[real] + λ * GP

        where GP is the gradient penalty computed on interpolated samples.
        """
        # =====================================================================
        # CRITIC TRAINING MODEL
        # =====================================================================

        # Freeze generator while training critic
        self.set_trainable(self.generator, False)

        # Input: Real images
        real_img = Input(shape=self.input_dim)

        # Input: Noise vector for generating fake images
        z_disc = Input(shape=(self.z_dim,))
        fake_img = self.generator(z_disc)

        # Critic evaluates real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Create interpolated samples for gradient penalty
        # x_interp = α * real + (1 - α) * fake
        interpolated_img = RandomWeightedAverage(self.batch_size)(
            [real_img, fake_img]
        )

        # Critic evaluates interpolated samples
        validity_interpolated = self.critic(interpolated_img)

        # Create gradient penalty loss function with interpolated samples
        # Using partial to inject the interpolated_samples argument
        partial_gp_loss = partial(
            self.gradient_penalty_loss,
            interpolated_samples=interpolated_img
        )
        partial_gp_loss.__name__ = 'gradient_penalty'

        # Build and compile critic model
        self.critic_model = Model(
            inputs=[real_img, z_disc],
            outputs=[valid, fake, validity_interpolated]
        )

        self.critic_model.compile(
            loss=[self.wasserstein, self.wasserstein, partial_gp_loss],
            optimizer=self.get_opti(self.critic_learning_rate),
            loss_weights=[1, 1, self.grad_weight]  # λ for GP term
        )

        # =====================================================================
        # GENERATOR TRAINING MODEL
        # =====================================================================

        # Freeze critic while training generator
        self.set_trainable(self.critic, False)
        self.set_trainable(self.generator, True)

        # Input: Noise vector
        model_input = Input(shape=(self.z_dim,))

        # Generate fake images
        img = self.generator(model_input)

        # Critic evaluates generated images
        model_output = self.critic(img)

        # Build and compile generator model
        self.model = Model(model_input, model_output)
        self.model.compile(
            optimizer=self.get_opti(self.generator_learning_rate),
            loss=self.wasserstein
        )

        # Unfreeze critic for next training iteration
        self.set_trainable(self.critic, True)

        # =====================================================================
        # OPTIMIZERS FOR GRADIENTTAPE-BASED TRAINING (Keras 3.0+)
        # =====================================================================
        # These are used by train_critic() and train_generator() methods
        # which use tf.GradientTape for gradient computation
        self.critic_optimizer = self.get_opti(self.critic_learning_rate)
        self.generator_optimizer = self.get_opti(self.generator_learning_rate)

    # =========================================================================
    # TRAINING METHODS
    # =========================================================================

    def train_critic(self, x_train, batch_size, using_generator):
        """
        Perform one critic training step using tf.GradientTape.

        This implementation is compatible with Keras 3.0+ which doesn't
        support K.gradients() on KerasTensors. Instead, we compute the
        gradient penalty directly using tf.GradientTape.

        The critic is trained to maximize the Wasserstein distance while
        keeping gradients near unit norm (enforced by gradient penalty).

        Args:
            x_train: Training data array or generator.
            batch_size (int): Number of samples per batch.
            using_generator (bool): If True, x_train is a data generator.

        Returns:
            list: Loss values [total, real_loss, fake_loss, gradient_penalty].
        """
        # Get batch of real images
        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        # Convert to tensor
        true_imgs = tf.cast(true_imgs, tf.float32)

        # Generate noise for fake images
        noise = tf.random.normal((batch_size, self.z_dim))

        with tf.GradientTape() as tape:
            # Generate fake images
            fake_imgs = self.generator(noise, training=True)

            # Critic evaluates real and fake images
            real_validity = self.critic(true_imgs, training=True)
            fake_validity = self.critic(fake_imgs, training=True)

            # Compute Wasserstein loss
            # Critic wants: high scores for real, low scores for fake
            # Loss = E[fake] - E[real] (minimizing this maximizes distance)
            real_loss = -tf.reduce_mean(real_validity)
            fake_loss = tf.reduce_mean(fake_validity)

            # Gradient penalty
            # Interpolate between real and fake images
            alpha = tf.random.uniform((batch_size, 1, 1, 1), 0., 1.)
            interpolated = alpha * true_imgs + (1 - alpha) * fake_imgs

            # Compute gradients of critic output w.r.t. interpolated samples
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                interp_validity = self.critic(interpolated, training=True)

            grads = gp_tape.gradient(interp_validity, interpolated)

            # Compute gradient norm
            grad_norm = tf.sqrt(
                tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-8
            )
            gradient_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))

            # Total critic loss
            d_loss = real_loss + fake_loss + self.grad_weight * gradient_penalty

        # Get critic trainable weights
        critic_grads = tape.gradient(d_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_weights)
        )

        return [
            float(d_loss),
            float(real_loss),
            float(fake_loss),
            float(gradient_penalty)
        ]

    def train_generator(self, batch_size):
        """
        Perform one generator training step using tf.GradientTape.

        The generator is trained to minimize the Wasserstein distance,
        i.e., fool the critic into outputting high scores for fake images.

        Args:
            batch_size (int): Number of samples to generate.

        Returns:
            float: Generator loss value.
        """
        # Generate noise
        noise = tf.random.normal((batch_size, self.z_dim))

        with tf.GradientTape() as tape:
            # Generate fake images
            fake_imgs = self.generator(noise, training=True)

            # Critic evaluates generated images
            fake_validity = self.critic(fake_imgs, training=True)

            # Generator wants critic to output high scores (real)
            # Loss = -E[critic(fake)] (minimizing this makes fake look real)
            g_loss = -tf.reduce_mean(fake_validity)

        # Get generator trainable weights
        gen_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_weights)
        )

        return float(g_loss)

    def train(
        self,
        x_train,
        batch_size,
        epochs,
        run_folder,
        print_every_n_batches=10,
        n_critic=5,
        using_generator=False,
        wandb_log=False
    ):
        """
        Train the WGAN-GP model.

        Implements the WGAN-GP training algorithm:
            1. For each epoch:
                a. Train critic n_critic times
                b. Train generator once
                c. Log metrics and save samples periodically

        Args:
            x_train: Training data (numpy array or generator).
            batch_size (int): Number of samples per batch.
            epochs (int): Number of epochs to train.
            run_folder (str): Path to save outputs (images, weights).
            print_every_n_batches (int): Save samples every N epochs.
                Default: 10.
            n_critic (int): Number of critic updates per generator update.
                Default: 5 (as per WGAN-GP paper).
            using_generator (bool): If True, x_train is a data generator.
                Default: False.
            wandb_log (bool): If True, log metrics to Weights & Biases.
                Default: False.

        Note:
            Every 100 epochs, the critic is trained 5 times regardless
            of n_critic to ensure stability.
        """
        # Import wandb inside method to avoid import errors if not installed
        if wandb_log:
            try:
                import wandb
            except ImportError:
                print("Warning: wandb not installed. Disabling W&B logging.")
                wandb_log = False

        for epoch in range(self.epoch, self.epoch + epochs):

            # Every 100 epochs, enforce 5 critic updates for stability
            if epoch % 100 == 0:
                critic_loops = 5
            else:
                critic_loops = n_critic

            # -----------------------------------------------------------------
            # Train critic multiple times
            # -----------------------------------------------------------------
            for _ in range(critic_loops):
                d_loss = self.train_critic(x_train, batch_size, using_generator)

            # -----------------------------------------------------------------
            # Train generator once
            # -----------------------------------------------------------------
            g_loss = self.train_generator(batch_size)

            # -----------------------------------------------------------------
            # Log progress to console
            # -----------------------------------------------------------------
            print(
                f"{epoch} ({critic_loops}, 1) "
                f"[D loss: ({d_loss[0]:.1f})"
                f"(R {d_loss[1]:.1f}, F {d_loss[2]:.1f}, GP {d_loss[3]:.1f})] "
                f"[G loss: {g_loss:.1f}]"
            )

            # Store losses
            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            # -----------------------------------------------------------------
            # Log to W&B (per-epoch logging)
            # -----------------------------------------------------------------
            if wandb_log:
                # Compute Wasserstein distance from critic losses
                wasserstein_dist = abs(d_loss[1] - d_loss[2])

                wandb.log({
                    # Epoch counter
                    "epoch": epoch,

                    # Critic losses
                    "d_loss/total": d_loss[0],
                    "d_loss/real": d_loss[1],
                    "d_loss/fake": d_loss[2],
                    "d_loss/gradient_penalty": d_loss[3],

                    # Generator loss
                    "g_loss": g_loss,

                    # Wasserstein distance (key training metric)
                    "wasserstein_distance": wasserstein_dist,

                    # Training dynamics
                    "d_g_ratio": abs(d_loss[0] / g_loss) if g_loss != 0 else 0,
                    "critic_updates": critic_loops,
                })

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

                # Log sample images to W&B
                if wandb_log:
                    # Generate sample images for W&B
                    sample_noise = np.random.normal(0, 1, (16, self.z_dim))
                    sample_imgs = self.generator.predict(sample_noise, verbose=0)
                    # Convert from [-1, 1] to [0, 1]
                    sample_imgs = (sample_imgs + 1) / 2.0
                    wandb.log({
                        "generated_images": [
                            wandb.Image(img, caption=f"Epoch {epoch}")
                            for img in sample_imgs[:8]
                        ]
                    })

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
        noise = np.random.normal(0, 1, (rows * cols, self.z_dim))
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

        fig.savefig(os.path.join(run_folder, f"images/sample_{self.epoch}.png"))
        plt.close()

    def plot_model(self, run_folder):
        """
        Generate and save model architecture diagrams.

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
            self.grad_weight,
            self.z_dim,
            self.batch_size
        ]

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)

        self.plot_model(folder)

    def save_model(self, run_folder):
        """
        Save model weights and architecture.

        Saves the complete models in native Keras format (.keras)
        and a pickle of the WGANGP object for easy reloading.

        Args:
            run_folder (str): Path to run folder.
        """
        # Save models in native Keras format
        self.model.save(os.path.join(run_folder, 'model.keras'))
        self.critic.save(os.path.join(run_folder, 'critic.keras'))
        self.generator.save(os.path.join(run_folder, 'generator.keras'))

        # Save WGANGP object
        with open(os.path.join(run_folder, 'obj.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def load_weights(self, filepath):
        """
        Load model weights from a file.

        Args:
            filepath (str): Path to weights file (.h5 or .weights.h5).
        """
        self.model.load_weights(filepath)