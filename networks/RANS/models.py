# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:52:54 2020

@author: aabdulaal
................................................................................................................................
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.constraints import NonNeg
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow.python.keras.models import Model
from typing import List, Optional

# ==============================================================================================================================
# SINCODER
# ==============================================================================================================================
class freqcoder(Layer):
    """ 
    Encode multivariate to a latent space of size 1 for extracting common oscillations in the series (similar to finding PCA).
    """
    def __init__(self, **kwargs):
        super(freqcoder, self).__init__(**kwargs)
        self.kwargs = kwargs
        
    def build(self, input_shape):
        self.latent = Dense(1, activation='linear')
        self.decoder = Dense(input_shape[-1], activation='linear')
    
    def call(self, inputs):
        z = self.latent(inputs)
        x_pred = self.decoder(z)
        return z, x_pred
    
    def get_config(self):
        base_config = super(freqcoder, self).get_config()
        return dict(list(base_config.items()))
    
class sincoder(Layer):
    """ Fit m sinusoidal waves to an input t-matrix (matrix of m epochtimes) """
    def __init__(self, freq_init: Optional[List[float]] = None, max_freqs: int = 1, trainable_freq: bool = False, **kwargs):
        super(sincoder, self).__init__(**kwargs)
        self.freq_init = freq_init
        if freq_init:
            self.max_freqs = len(freq_init)
        else:
            self.max_freqs = max_freqs
        self.trainable_freq = trainable_freq
        self.kwargs = kwargs
        
    def build(self, input_shape):
        self.amp = self.add_weight(shape=(input_shape[-1], self.max_freqs), initializer="zeros", trainable=True)
        if self.freq_init and not self.trainable_freq:
            self.freq = [self.add_weight(initializer=Constant(f), trainable=False) for f in self.freq_init]
        elif self.freq_init:
            self.freq = [self.add_weight(initializer=Constant(f), constraint=NonNeg(), trainable=True) for f in self.freq_init]
        else:
            self.freq = [
                self.add_weight(initializer="zeros", constraint=NonNeg(), trainable=True) for f in range(self.max_freqs)
            ]
        self.wb = self.add_weight(
            shape=(input_shape[-1], self.max_freqs), initializer="zeros", trainable=True
        )  # angular frequency (w) x phase shift
        self.disp = self.add_weight(shape=input_shape[-1], initializer="zeros", trainable=True)
    
    def call(self, inputs):
        th = tf.expand_dims(
            tf.expand_dims(self.freq, axis=0), axis=0
        ) * tf.expand_dims(inputs, axis=-1) + tf.expand_dims(self.wb, axis=0)
        return tf.reduce_sum(tf.expand_dims(self.amp, axis=0) * tf.sin(th), axis=-1) + self.disp
    
    def get_config(self):
        base_config = super(sincoder, self).get_config()
        config = {"freq_init": self.freq_init, "max_freqs": self.max_freqs, "trainable_freq": self.trainable_freq}
        return dict(list(base_config.items()) + list(config.items()))

# ==============================================================================================================================
# RANCODER
# ==============================================================================================================================
class Encoder(Layer):
    def __init__(self, latent_dim: int, activation: str, depth: int = 2, **kwargs,):
        super(Encoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.activation = activation
        self.depth = depth
        self.kwargs = kwargs
        
    def build(self, input_shape):
        self.hidden = {
            'hidden_{}'.format(i): Dense(
                int(input_shape[-1] / (2**(i+1))), activation=self.activation,
            ) for i in range(self.depth)
        }
        self.latent = Dense(self.latent_dim, activation=self.activation)
        
    def call(self, inputs):
        x = self.hidden['hidden_0'](inputs)
        for i in range(1, self.depth):
            x = self.hidden['hidden_{}'.format(i)](x)
        return self.latent(x)
    
    def get_config(self):
        base_config = super(Encoder, self).get_config()
        config = {"latent_dim": self.latent_dim, "activation": self.activation,"depth": self.depth,}
        return dict(list(base_config.items()) + list(config.items()))
    
class Decoder(Layer):
    def __init__(self, output_dim: int, activation: str, output_activation: str,depth: int, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        self.depth = depth
        self.kwargs = kwargs
        
    def build(self, input_shape):
        self.hidden = {
            'hidden_{}'.format(i): Dense(
                int(self.output_dim/ (2**(self.depth-i))), activation=self.activation,
            ) for i in range(self.depth)
        }
        self.restored = Dense(self.output_dim, activation=self.output_activation)
        
    def call(self, inputs):
        x = self.hidden['hidden_0'](inputs)
        for i in range(1, self.depth):
            x = self.hidden['hidden_{}'.format(i)](x)
        return self.restored(x)
    
    def get_config(self):
        base_config = super(Decoder, self).get_config()
        config = {
            "output_dim": self.output_dim, 
            "activation": self.activation, 
            "output_activation": self.output_activation, 
            "depth": self.depth,
        }
        return dict(list(base_config.items()) + list(config.items()))
    
class RANCoders(Layer):
    def __init__(
            self, 
            n_estimators: int = 100,
            max_features: int = 3,
            encoding_depth: int = 2,
            latent_dim: int = 2, 
            decoding_depth: int = 2,
            delta: float = 0.05,
            activation: str = 'linear',
            output_activation: str = 'linear',
            **kwargs,
    ):
        super(RANCoders, self).__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.encoding_depth = encoding_depth
        self.latent_dim = latent_dim
        self.decoding_depth = decoding_depth
        self.delta = delta
        self.activation = activation
        self.output_activation = output_activation
        self.kwargs = kwargs
        
    def build(self, input_shape):
        assert(input_shape[-1] > self.max_features)
        self.encoders = {
            'encoder_{}'.format(i): Encoder(
                self.latent_dim, self.activation, depth=self.encoding_depth,
            ) for i in range(self.n_estimators)
        }
        self.decoders_upper = {
            'decoder_hi_{}'.format(i): Decoder(
                input_shape[-1], self.activation, self.output_activation, self.decoding_depth
            ) for i in range(self.n_estimators)
        }
        self.decoders_lower = {
            'decoder_lo_{}'.format(i): Decoder(
                input_shape[-1], self.activation, self.output_activation, self.decoding_depth
            ) for i in range(self.n_estimators)
        }
        self.randsamples = tf.Variable(
            np.concatenate(
                [
                    np.random.choice(
                        input_shape[-1], replace=False, size=(1, self.max_features),
                    ) for i in range(self.n_estimators)
                ]
            ), trainable=False
        )  # the feature selector (bootstrapping)
        
    def call(self, inputs):
        z = {
            'z_{}'.format(i): self.encoders['encoder_{}'.format(i)](
                tf.gather(inputs, self.randsamples[i], axis=-1)
            ) for i in range(self.n_estimators)
        }
        w_hi = {
            'w_{}'.format(i): self.decoders_upper['decoder_hi_{}'.format(i)](
                z['z_{}'.format(i)]
            ) for i in range(self.n_estimators)
        }
        w_lo = {
            'w_{}'.format(i): self.decoders_lower['decoder_lo_{}'.format(i)](
                z['z_{}'.format(i)]
            ) for i in range(self.n_estimators)
        }
        o_hi = tf.concat([tf.expand_dims(i, axis=0) for i in w_hi.values()], axis=0)  
        o_lo = tf.concat([tf.expand_dims(i, axis=0) for i in w_lo.values()], axis=0)
        return o_hi, o_lo
    
    def get_config(self):
        base_config = super(RANCoders, self).get_config()
        config = {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "encoding_depth": self.encoding_depth,
            "latent_dim": self.latent_dim,
            "decoding_depth": self.decoding_depth,
            "delta": self.delta,
            "activation": self.activation,
            "output_activation": self.output_activation,
        }
        return dict(list(base_config.items()) + list(config.items()))