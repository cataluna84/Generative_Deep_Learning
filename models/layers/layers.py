import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config


class InstanceNormalization(Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)
    
    Replaces keras_contrib.layers.InstanceNormalization for TensorFlow 2.x compatibility.
    """
    def __init__(self, axis=-1, epsilon=1e-5, center=True, scale=True, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis < 0:
            axis = ndim + self.axis
        else:
            axis = self.axis
        
        dim = input_shape[axis]
        if dim is None:
            raise ValueError(f'Axis {self.axis} has undefined dimension')
        
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=(dim,),
                initializer='ones',
                trainable=True
            )
        else:
            self.gamma = None
            
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=(dim,),
                initializer='zeros',
                trainable=True
            )
        else:
            self.beta = None
            
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        # Calculate mean and variance per instance (per sample, per channel)
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        
        # Normalize
        x = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        
        # Apply scale and shift
        if self.scale and self.gamma is not None:
            x = x * self.gamma
        if self.center and self.beta is not None:
            x = x + self.beta
            
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale
        })
        return config
