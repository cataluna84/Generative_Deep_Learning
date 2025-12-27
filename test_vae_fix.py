
import numpy as np
import tensorflow as tf
import os
import sys

# Add current directory to path so we can import models
sys.path.append(os.getcwd())

from models.VAE import VariationalAutoencoder

# params
input_dim = (28,28,1)
encoder_conv_filters = [32,64]
encoder_conv_kernel_size = [3,3]
encoder_conv_strides = [1,2]
decoder_conv_t_filters = [64,1]
decoder_conv_t_kernel_size = [3,3]
decoder_conv_t_strides = [1,2]
z_dim = 2

print("Instantiating VAE...")
try:
    vae = VariationalAutoencoder(
        input_dim = input_dim
        , encoder_conv_filters = encoder_conv_filters
        , encoder_conv_kernel_size = encoder_conv_kernel_size
        , encoder_conv_strides = encoder_conv_strides
        , decoder_conv_t_filters = decoder_conv_t_filters
        , decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        , decoder_conv_t_strides = decoder_conv_t_strides
        , z_dim = z_dim
    )
    print("VAE Instantiation Successful")
except Exception as e:
    print(f"VAE Instantiation Failed: {e}")
    sys.exit(1)

print("Compiling VAE...")
try:
    vae.compile(learning_rate=0.0005, r_loss_factor=1000)
    print("VAE Compilation Successful")
except Exception as e:
    print(f"VAE Compilation Failed: {e}")
    sys.exit(1)

print("Testing training step...")
try:
    # Dummy data
    x_train = np.random.normal(0, 1, (32, 28, 28, 1)).astype('float32')
    loss = vae.model.train_on_batch(x_train, x_train)
    print(f"Training step successful. Loss: {loss}")
except Exception as e:
    print(f"Training step Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
