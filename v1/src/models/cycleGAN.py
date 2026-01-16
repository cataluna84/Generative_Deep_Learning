"""CycleGAN implementation for unpaired image-to-image translation.

This module implements the CycleGAN architecture as described in:
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
by Zhu et al. (2017).

Supports W&B logging for experiment tracking.
"""
from __future__ import print_function, division

import datetime
import gc
import os
import pickle as pkl
import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
    add,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Use our custom InstanceNormalization instead of keras_contrib
from src.models.layers.layers import InstanceNormalization, ReflectionPadding2D

# Optional W&B import (graceful fallback if not installed)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CycleGAN:
    """CycleGAN model for unpaired image-to-image translation.

    This class implements the CycleGAN architecture with support for both
    U-Net and ResNet generator architectures, PatchGAN discriminators,
    and optional Weights & Biases logging.

    Attributes:
        input_dim (tuple): Input image dimensions (height, width, channels).
        learning_rate (float): Learning rate for Adam optimizer.
        lambda_validation (float): Weight for adversarial loss.
        lambda_reconstr (float): Weight for cycle-consistency loss.
        lambda_id (float): Weight for identity loss.
        generator_type (str): Generator architecture type ('unet' or 'resnet').
        gen_n_filters (int): Base filter count for generators.
        disc_n_filters (int): Base filter count for discriminators.
        buffer_max_length (int): Maximum length of image buffer for training.
    """

    def __init__(
        self,
        input_dim,
        learning_rate,
        lambda_validation,
        lambda_reconstr,
        lambda_id,
        generator_type,
        gen_n_filters,
        disc_n_filters,
        buffer_max_length=50,
    ):
        """Initialize CycleGAN model.

        Args:
            input_dim (tuple): Input image dimensions (height, width, channels).
            learning_rate (float): Learning rate for Adam optimizer.
            lambda_validation (float): Weight for adversarial loss.
            lambda_reconstr (float): Weight for cycle-consistency loss.
            lambda_id (float): Weight for identity loss.
            generator_type (str): 'unet' or 'resnet'.
            gen_n_filters (int): Base filter count for generators.
            disc_n_filters (int): Base filter count for discriminators.
            buffer_max_length (int): Maximum buffer size for fake images.
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.buffer_max_length = buffer_max_length
        self.lambda_validation = lambda_validation
        self.lambda_reconstr = lambda_reconstr
        self.lambda_id = lambda_id
        self.generator_type = generator_type
        self.gen_n_filters = gen_n_filters
        self.disc_n_filters = disc_n_filters

        # Input shape
        self.img_rows = input_dim[0]
        self.img_cols = input_dim[1]
        self.channels = input_dim[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Training history
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        # Image buffers for training stability (stored as GPU tensors)
        # Using list instead of deque for GPU tensor compatibility
        self.buffer_A = []
        self.buffer_B = []

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 3)
        self.disc_patch = (patch, patch, 1)

        # Weight initialization
        self.weight_init = RandomNormal(mean=0.0, stddev=0.02)

        # Build and compile models
        self.compile_models()

    def compile_models(self):
        """Build and compile all CycleGAN submodels.

        Creates discriminators (d_A, d_B), generators (g_AB, g_BA),
        and the combined model for training generators.
        """
        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

        self.d_A.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate, beta_1=0.5),
            metrics=['accuracy']
        )
        self.d_B.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate, beta_1=0.5),
            metrics=['accuracy']
        )

        # Build the generators
        if self.generator_type == 'unet':
            self.g_AB = self.build_generator_unet()
            self.g_BA = self.build_generator_unet()
        else:
            self.g_AB = self.build_generator_resnet()
            self.g_BA = self.build_generator_resnet()

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # Discriminators determine validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(
            inputs=[img_A, img_B],
            outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id]
        )
        self.combined.compile(
            loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
            loss_weights=[
                self.lambda_validation, self.lambda_validation,
                self.lambda_reconstr, self.lambda_reconstr,
                self.lambda_id, self.lambda_id
            ],
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
        )

        # Re-enable discriminator training
        self.d_A.trainable = True
        self.d_B.trainable = True

    def build_generator_unet(self):
        """Build U-Net style generator.

        Returns:
            Model: Keras model for U-Net generator.
        """
        def downsample(layer_input, filters, f_size=4):
            """Downsample block with Conv2D, InstanceNorm, and ReLU."""
            d = Conv2D(
                filters,
                kernel_size=f_size,
                strides=2,
                padding='same'
            )(layer_input)
            d = InstanceNormalization(axis=-1, center=False, scale=False)(d)
            d = Activation('relu')(d)
            return d

        def upsample(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Upsample block with skip connection."""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(
                filters,
                kernel_size=f_size,
                strides=1,
                padding='same'
            )(u)
            u = InstanceNormalization(axis=-1, center=False, scale=False)(u)
            u = Activation('relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        img = Input(shape=self.img_shape)

        # Downsampling path
        d1 = downsample(img, self.gen_n_filters)
        d2 = downsample(d1, self.gen_n_filters * 2)
        d3 = downsample(d2, self.gen_n_filters * 4)
        d4 = downsample(d3, self.gen_n_filters * 8)

        # Upsampling path with skip connections
        u1 = upsample(d4, d3, self.gen_n_filters * 4)
        u2 = upsample(u1, d2, self.gen_n_filters * 2)
        u3 = upsample(u2, d1, self.gen_n_filters)

        # Final upsampling and output
        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(
            self.channels,
            kernel_size=4,
            strides=1,
            padding='same',
            activation='tanh'
        )(u4)

        return Model(img, output_img)

    def build_generator_resnet(self):
        """Build ResNet style generator with 9 residual blocks.

        Returns:
            Model: Keras model for ResNet generator.
        """
        def conv7s1(layer_input, filters, final):
            """7x7 convolution with stride 1 and reflection padding."""
            y = ReflectionPadding2D(padding=(3, 3))(layer_input)
            y = Conv2D(
                filters,
                kernel_size=(7, 7),
                strides=1,
                padding='valid',
                kernel_initializer=self.weight_init
            )(y)
            if final:
                y = Activation('tanh')(y)
            else:
                y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
                y = Activation('relu')(y)
            return y

        def downsample(layer_input, filters):
            """Downsample with 3x3 convolution, stride 2."""
            y = Conv2D(
                filters,
                kernel_size=(3, 3),
                strides=2,
                padding='same',
                kernel_initializer=self.weight_init
            )(layer_input)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = Activation('relu')(y)
            return y

        def residual(layer_input, filters):
            """Residual block with two 3x3 convolutions."""
            shortcut = layer_input
            y = ReflectionPadding2D(padding=(1, 1))(layer_input)
            y = Conv2D(
                filters,
                kernel_size=(3, 3),
                strides=1,
                padding='valid',
                kernel_initializer=self.weight_init
            )(y)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = Activation('relu')(y)

            y = ReflectionPadding2D(padding=(1, 1))(y)
            y = Conv2D(
                filters,
                kernel_size=(3, 3),
                strides=1,
                padding='valid',
                kernel_initializer=self.weight_init
            )(y)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)

            return add([shortcut, y])

        def upsample(layer_input, filters):
            """Upsample with transposed convolution."""
            y = Conv2DTranspose(
                filters,
                kernel_size=(3, 3),
                strides=2,
                padding='same',
                kernel_initializer=self.weight_init
            )(layer_input)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = Activation('relu')(y)
            return y

        # Image input
        img = Input(shape=self.img_shape)

        # Initial convolution
        y = conv7s1(img, self.gen_n_filters, final=False)

        # Downsampling
        y = downsample(y, self.gen_n_filters * 2)
        y = downsample(y, self.gen_n_filters * 4)

        # 9 Residual blocks
        for _ in range(9):
            y = residual(y, self.gen_n_filters * 4)

        # Upsampling
        y = upsample(y, self.gen_n_filters * 2)
        y = upsample(y, self.gen_n_filters)

        # Final convolution
        output = conv7s1(y, 3, final=True)

        return Model(img, output)

    def build_discriminator(self):
        """Build PatchGAN discriminator.

        Returns:
            Model: Keras model for PatchGAN discriminator.
        """
        def conv4(layer_input, filters, stride=2, norm=True):
            """4x4 convolution with LeakyReLU."""
            y = Conv2D(
                filters,
                kernel_size=(4, 4),
                strides=stride,
                padding='same',
                kernel_initializer=self.weight_init
            )(layer_input)
            if norm:
                y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = LeakyReLU(negative_slope=0.2)(y)
            return y

        # Image input
        img = Input(shape=self.img_shape)

        # Discriminator layers
        y = conv4(img, self.disc_n_filters, stride=2, norm=False)
        y = conv4(y, self.disc_n_filters * 2, stride=2)
        y = conv4(y, self.disc_n_filters * 4, stride=2)
        y = conv4(y, self.disc_n_filters * 8, stride=1)

        # Output layer (PatchGAN output)
        output = Conv2D(
            1,
            kernel_size=4,
            strides=1,
            padding='same',
            kernel_initializer=self.weight_init
        )(y)

        return Model(img, output)

    def train_discriminators(self, imgs_A, imgs_B, valid, fake):
        """Train discriminators on a batch of real and fake images.

        This method implements the discriminator training step of CycleGAN
        with an enhanced multi-batch buffer sampling strategy for improved
        training stability and diversity.

        Multi-Batch Buffer Concatenation Strategy:
            The image buffers (`buffer_A`, `buffer_B`) store previously
            generated fake images as FIFO queues. Unlike the original CycleGAN
            paper which samples a single historical batch, this implementation
            samples and concatenates multiple batches to maximize diversity:

            1. Generate new fake images from the current batch.
            2. Store them in the buffer (FIFO queue with max length).
            3. Sample up to `k` batches from the buffer (k = min(buffer_len, 4)).
            4. Concatenate sampled batches into a pool of candidates.
            5. Randomly select `batch_size` images from the pool.

            This approach ensures the discriminator sees images from multiple
            historical timesteps in each training step, reducing mode collapse
            and improving gradient quality compared to single-batch sampling.

        Args:
            imgs_A (np.ndarray): Batch of real images from domain A.
                Shape: (batch_size, height, width, channels).
            imgs_B (np.ndarray): Batch of real images from domain B.
                Shape: (batch_size, height, width, channels).
            valid (np.ndarray): Ground truth labels for real images (ones).
                Shape: (batch_size,) + disc_patch.
            fake (np.ndarray): Ground truth labels for fake images (zeros).
                Shape: (batch_size,) + disc_patch.

        Returns:
            tuple: A tuple containing 14 discriminator loss metrics:
                - d_loss_total[0]: Total discriminator loss.
                - dA_loss[0]: Discriminator A combined loss.
                - dA_loss_real[0]: Discriminator A loss on real images.
                - dA_loss_fake[0]: Discriminator A loss on fake images.
                - dB_loss[0]: Discriminator B combined loss.
                - dB_loss_real[0]: Discriminator B loss on real images.
                - dB_loss_fake[0]: Discriminator B loss on fake images.
                - d_loss_total[1]: Total discriminator accuracy.
                - dA_loss[1]: Discriminator A combined accuracy.
                - dA_loss_real[1]: Discriminator A accuracy on real images.
                - dA_loss_fake[1]: Discriminator A accuracy on fake images.
                - dB_loss[1]: Discriminator B combined accuracy.
                - dB_loss_real[1]: Discriminator B accuracy on real images.
                - dB_loss_fake[1]: Discriminator B accuracy on fake images.

        Note:
            The discriminator loss is computed as the average of losses on
            real and fake images: D_loss = 0.5 * (D_real + D_fake).

        References:
            Zhu et al. (2017). "Unpaired Image-to-Image Translation using
            Cycle-Consistent Adversarial Networks." ICCV.
        """
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Generate fake images by translating to the opposite domain
        # ─────────────────────────────────────────────────────────────────────
        fake_B = self.g_AB.predict(imgs_A, verbose=0)  # A -> B translation
        fake_A = self.g_BA.predict(imgs_B, verbose=0)  # B -> A translation

        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Add generated images to GPU buffers for training stability
        # Storing as TensorFlow tensors keeps data on GPU, increasing VRAM
        # utilization. FIFO management ensures buffer doesn't exceed max length.
        # ─────────────────────────────────────────────────────────────────────
        # Convert to TensorFlow tensors and store on GPU
        self.buffer_B.append(tf.constant(fake_B))
        self.buffer_A.append(tf.constant(fake_A))
        
        # FIFO management: remove oldest if buffer exceeds max length
        if len(self.buffer_A) > self.buffer_max_length:
            self.buffer_A.pop(0)
        if len(self.buffer_B) > self.buffer_max_length:
            self.buffer_B.pop(0)

        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Multi-batch buffer sampling with concatenation (GPU-based)
        # Sample up to k=4 batches, concatenate into pool, then randomly
        # select batch_size images. Uses TensorFlow ops to keep data on GPU.
        # ─────────────────────────────────────────────────────────────────────
        batch_size = len(imgs_A)

        # Sample fake images for discriminator A (B->A translations)
        if len(self.buffer_A) == 0:
            # Buffer empty: use current fake images
            fake_A_rnd = fake_A
        else:
            # Sample up to 4 batches from buffer
            k = min(len(self.buffer_A), 4)
            sampled_batches = random.sample(self.buffer_A, k)
            # Concatenate GPU tensors into a pool of candidate images
            pool = tf.concat(sampled_batches, axis=0)
            # Randomly select batch_size images (without replacement)
            indices = np.random.choice(pool.shape[0], size=batch_size, replace=False)
            fake_A_rnd = tf.gather(pool, indices).numpy()

        # Sample fake images for discriminator B (A->B translations)
        if len(self.buffer_B) == 0:
            # Buffer empty: use current fake images
            fake_B_rnd = fake_B
        else:
            # Sample up to 4 batches from buffer
            k = min(len(self.buffer_B), 4)
            sampled_batches = random.sample(self.buffer_B, k)
            # Concatenate GPU tensors into a pool of candidate images
            pool = tf.concat(sampled_batches, axis=0)
            # Randomly select batch_size images (without replacement)
            indices = np.random.choice(pool.shape[0], size=batch_size, replace=False)
            fake_B_rnd = tf.gather(pool, indices).numpy()

        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Train discriminators on real and fake images
        # ─────────────────────────────────────────────────────────────────────

        # Train discriminator A: distinguishes real A from fake A (B->A)
        dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
        dA_loss_fake = self.d_A.train_on_batch(fake_A_rnd, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        # Train discriminator B: distinguishes real B from fake B (A->B)
        dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = self.d_B.train_on_batch(fake_B_rnd, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total discriminator loss (average of both discriminators)
        d_loss_total = 0.5 * np.add(dA_loss, dB_loss)

        return (
            d_loss_total[0],
            dA_loss[0], dA_loss_real[0], dA_loss_fake[0],
            dB_loss[0], dB_loss_real[0], dB_loss_fake[0],
            d_loss_total[1],
            dA_loss[1], dA_loss_real[1], dA_loss_fake[1],
            dB_loss[1], dB_loss_real[1], dB_loss_fake[1],
        )

    def train_generators(self, imgs_A, imgs_B, valid):
        """Train generators on a batch of images.

        Args:
            imgs_A: Batch of images from domain A.
            imgs_B: Batch of images from domain B.
            valid: Ground truth labels for valid images.

        Returns:
            list: Generator loss values.
        """
        return self.combined.train_on_batch(
            [imgs_A, imgs_B],
            [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B]
        )

    def train(
        self,
        data_loader,
        run_folder,
        epochs,
        test_A_file,
        test_B_file,
        batch_size=1,
        sample_interval=50,
        wandb_log=False,
        n_disc_updates=5,
    ):
        """Train the CycleGAN model.

        Args:
            data_loader: DataLoader instance for loading image batches.
            run_folder (str): Path to save weights, images, and visualizations.
            epochs (int): Number of training epochs.
            test_A_file (str): Filename for test image from domain A.
            test_B_file (str): Filename for test image from domain B.
            batch_size (int): Images per batch (default: 1).
            sample_interval (int): Batches between sample generation (default: 50).
            wandb_log (bool): Enable W&B logging (default: False).
            n_disc_updates (int): Number of discriminator updates per generator
                update (default: 5). Higher values can improve training stability
                and increase GPU memory utilization.
        """
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(self.epoch, epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size=batch_size)):

                # ─────────────────────────────────────────────────────────────
                # Multiple discriminator updates per generator update
                # This improves training stability and increases GPU utilization
                # ─────────────────────────────────────────────────────────────
                for _ in range(n_disc_updates):
                    d_loss = self.train_discriminators(imgs_A, imgs_B, valid, fake)
                
                # Single generator update
                g_loss = self.train_generators(imgs_A, imgs_B, valid)

                elapsed_time = datetime.datetime.now() - start_time

                # Print progress
                print(
                    f"[Epoch {self.epoch}/{epochs}] "
                    f"[Batch {batch_i}/{data_loader.n_batches}] "
                    f"[D loss: {d_loss[0]:.6f}, acc: {100 * d_loss[7]:.0f}%] "
                    f"[G loss: {g_loss[0]:.5f}, adv: {np.sum(g_loss[1:3]):.5f}, "
                    f"recon: {np.sum(g_loss[3:5]):.5f}, id: {np.sum(g_loss[5:7]):.5f}] "
                    f"time: {elapsed_time}"
                )

                # Store losses
                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

                # Log metrics to W&B
                if wandb_log and WANDB_AVAILABLE:
                    wandb.log({
                        "epoch": self.epoch,
                        "batch": batch_i,
                        "d_loss": d_loss[0],
                        "d_acc": d_loss[7] * 100,
                        "g_loss": g_loss[0],
                        "g_adv": np.sum(g_loss[1:3]),
                        "g_recon": np.sum(g_loss[3:5]),
                        "g_id": np.sum(g_loss[5:7]),
                    })

                # Save generated image samples at intervals
                if batch_i % sample_interval == 0:
                    self.sample_images(
                        data_loader, batch_i, run_folder,
                        test_A_file, test_B_file, wandb_log=wandb_log
                    )
                    self.combined.save_weights(
                        os.path.join(run_folder, f'weights/weights-{self.epoch}.weights.h5')
                    )
                    self.combined.save_weights(
                        os.path.join(run_folder, 'weights/weights.weights.h5')
                    )
                    self.save_model(run_folder)

            self.epoch += 1

    def sample_images(
        self,
        data_loader,
        batch_i,
        run_folder,
        test_A_file,
        test_B_file,
        wandb_log=False,
    ):
        """Generate and save sample images for visualization.

        Generates translated, reconstructed, and identity-mapped images
        for test samples and saves them to files. Optionally logs to W&B.

        Args:
            data_loader: DataLoader instance.
            batch_i (int): Current batch index.
            run_folder (str): Path to save images.
            test_A_file (str): Test image filename from domain A.
            test_B_file (str): Test image filename from domain B.
            wandb_log (bool): Enable W&B image logging (default: False).
        """
        r, c = 2, 4

        for p in range(2):
            # Load test images
            if p == 1:
                imgs_A = data_loader.load_data(
                    domain="A", batch_size=1, is_testing=True
                )
                imgs_B = data_loader.load_data(
                    domain="B", batch_size=1, is_testing=True
                )
            else:
                imgs_A = data_loader.load_img(
                    f'{data_loader.data_root}/{data_loader.dataset_name}/testA/{test_A_file}'
                )
                imgs_B = data_loader.load_img(
                    f'{data_loader.data_root}/{data_loader.dataset_name}/testB/{test_B_file}'
                )

            # Generate translations
            fake_B = self.g_AB.predict(imgs_A, verbose=0)
            fake_A = self.g_BA.predict(imgs_B, verbose=0)

            # Generate reconstructions
            reconstr_A = self.g_BA.predict(fake_B, verbose=0)
            reconstr_B = self.g_AB.predict(fake_A, verbose=0)

            # Generate identity mappings
            id_A = self.g_BA.predict(imgs_A, verbose=0)
            id_B = self.g_AB.predict(imgs_B, verbose=0)

            # Concatenate all generated images
            gen_imgs = np.concatenate([
                imgs_A, fake_B, reconstr_A, id_A,
                imgs_B, fake_A, reconstr_B, id_B
            ])

            # Rescale images to [0, 1]
            gen_imgs = 0.5 * gen_imgs + 0.5
            gen_imgs = np.clip(gen_imgs, 0, 1)

            # Create visualization grid
            titles = ['Original', 'Translated', 'Reconstructed', 'ID']
            fig, axs = plt.subplots(r, c, figsize=(25, 12.5))
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[j])
                    axs[i, j].axis('off')
                    cnt += 1

            # Save figure
            fig.savefig(
                os.path.join(run_folder, f"images/{p}_{self.epoch}_{batch_i}.png")
            )

            # Log to W&B
            if wandb_log and WANDB_AVAILABLE:
                wandb.log({
                    f"generated_images_{p}": wandb.Image(
                        fig, caption=f"Epoch {self.epoch} Batch {batch_i}"
                    )
                })

            # Cleanup
            fig.clf()
            plt.close()

            del gen_imgs, imgs_A, imgs_B, fake_A, fake_B
            del reconstr_A, reconstr_B, id_A, id_B
            gc.collect()

    def plot_model(self, run_folder):
        """Save model architecture diagrams.

        Args:
            run_folder (str): Path to save visualization files.
        """
        plot_model(
            self.combined,
            to_file=os.path.join(run_folder, 'viz/combined.png'),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.d_A,
            to_file=os.path.join(run_folder, 'viz/d_A.png'),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.d_B,
            to_file=os.path.join(run_folder, 'viz/d_B.png'),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.g_BA,
            to_file=os.path.join(run_folder, 'viz/g_BA.png'),
            show_shapes=True,
            show_layer_names=True
        )
        plot_model(
            self.g_AB,
            to_file=os.path.join(run_folder, 'viz/g_AB.png'),
            show_shapes=True,
            show_layer_names=True
        )

    def save(self, folder):
        """Save model parameters and architecture diagrams.

        Args:
            folder (str): Path to save files.
        """
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim,
                self.learning_rate,
                self.buffer_max_length,
                self.lambda_validation,
                self.lambda_reconstr,
                self.lambda_id,
                self.generator_type,
                self.gen_n_filters,
                self.disc_n_filters,
            ], f)

        self.plot_model(folder)

    def save_model(self, run_folder):
        """Save all model weights and pickled object.

        Uses native Keras format (.keras) instead of legacy HDF5 (.h5).

        Args:
            run_folder (str): Path to save model files.
        """
        self.combined.save(os.path.join(run_folder, 'model.keras'))
        self.d_A.save(os.path.join(run_folder, 'd_A.keras'))
        self.d_B.save(os.path.join(run_folder, 'd_B.keras'))
        self.g_BA.save(os.path.join(run_folder, 'g_BA.keras'))
        self.g_AB.save(os.path.join(run_folder, 'g_AB.keras'))

        with open(os.path.join(run_folder, "obj.pkl"), "wb") as f:
            pkl.dump(self, f)

    def load_weights(self, filepath):
        """Load model weights from file.

        Args:
            filepath (str): Path to weights file.
        """
        self.combined.load_weights(filepath)
