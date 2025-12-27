"""Custom Keras layers for CycleGAN and other models.

These layers replace keras_contrib which is no longer maintained.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class InstanceNormalization(layers.Layer):
    """Instance Normalization Layer.
    
    Normalizes the activations of each instance separately.
    Used extensively in style transfer and image-to-image translation.
    """
    
    def __init__(self, axis=-1, epsilon=1e-5, center=True, scale=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        
    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(f'Axis {self.axis} of input tensor should have a defined dimension')
        
        if self.scale:
            self.gamma = self.add_weight(
                shape=(dim,),
                name='gamma',
                initializer='ones'
            )
        else:
            self.gamma = None
            
        if self.center:
            self.beta = self.add_weight(
                shape=(dim,),
                name='beta',
                initializer='zeros'
            )
        else:
            self.beta = None
            
        super().build(input_shape)
        
    def call(self, inputs):
        # Calculate mean and variance for each instance
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        
        # Normalize
        x = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        
        # Apply scale and shift
        if self.scale:
            x = x * self.gamma
        if self.center:
            x = x + self.beta
            
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        })
        return config


class ReflectionPadding2D(layers.Layer):
    """2D Reflection Padding Layer.
    
    Pads the input tensor by reflecting the values at the border.
    Used in CycleGAN to avoid boundary artifacts.
    """
    
    def __init__(self, padding=(1, 1), **kwargs):
        super().__init__(**kwargs)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        else:
            self.padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        
    def call(self, inputs):
        return tf.pad(inputs, 
                     [[0, 0], [self.padding[0][0], self.padding[0][1]], 
                      [self.padding[1][0], self.padding[1][1]], [0, 0]], 
                     mode='REFLECT')
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + self.padding[0][0] + self.padding[0][1],
                input_shape[2] + self.padding[1][0] + self.padding[1][1],
                input_shape[3])
    
    def get_config(self):
        config = super().get_config()
        config.update({'padding': self.padding})
        return config
