import os
import pickle
import numpy as np
from keras.callbacks import ModelCheckpoint

from keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, Activation, \
    BatchNormalization, Dropout
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from tensorflow_core.python.tpu.tensor_tracer import keras_layer_tracepoint
from utils.callbacks import CustomCallback, step_decay_schedule


class Autoencoder():
    def __init__(self, input_dim, encoder_conv_filters,
                 encoder_conv_kernel_size, encoder_conv_strides,
                 decoder_conv_t_filters, decoder_conv_t_kernel_size,
                 decoder_conv_t_strides, z_dim,
                 use_batch_norm = False,
                 use_dropout = False):
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layer_encoder = len(encoder_conv_filters)
        self.n_layer_decoder = len(decoder_conv_t_filters)

        self._build(self)


    def _build(self):

        ### THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(filters=self.encoder_conv_filters[i],
                                kernel_size=self.encoder_conv_kernel_size[i],
                                strides=self.encoder_conv_strides[i],
                                padding='same',
                                name='encoder_conv_'+str(i))
            x = conv_layer(x)   # Stack on top
            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)

        encoder_output = Dense(self.z_dim, name='encoder_output')(x)

        self.encoder = Model(encoder_input, encoder_output)


        ### THE DECODER
        decoder_input = Input(shape=(self.z_dim, ), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)

        for i in range(self.n_layer_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding='same',
                name='decoder_conv_t_'+str(i)
            )

            x = conv_t_layer(x)

            if i < self.n_layer_decoder - 1:
                x = LeakyReLU()(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### The Full Autoencoder
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)

    def compile(self, learning_rate):
        ### COMPILATION
        self.learning_rate = learning_rate

        optimizer = Adam(learning_rate=learning_rate)

        ## Using the Root mean squared error loss function as opposed to Binary cross-entropy loss
        def r_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

        self.model.compile(optimizer=optimizer, loss=r_loss)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0, lr_decay=1):
        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint2=ModelCheckpoint(filepath=os.path.join(run_folder, 'weight'))
