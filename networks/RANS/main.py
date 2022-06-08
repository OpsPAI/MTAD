# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:30:26 2020

@author: aabdulaal
................................................................................................................................
"""

import os
import logging
import numpy as np
from scipy.signal import find_peaks
from spectrum import Periodogram
from joblib import dump, load
from .models import freqcoder, sincoder, RANCoders
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, model_from_json
from typing import List, Optional


class RANSynCoders:
    """class for building, training, and testing rancoders models"""

    def __init__(
        self,
        # Rancoders inputs:
        n_estimators: int = 100,
        max_features: int = 3,
        encoding_depth: int = 2,
        latent_dim: int = 2,
        decoding_depth: int = 2,
        activation: str = "linear",
        output_activation: str = "linear",
        delta: float = 0.05,  # quantile bound for regression
        # Syncrhonization inputs
        synchronize: bool = False,
        force_synchronization: bool = True,  # if synchronization is true but no significant frequencies found
        min_periods: int = 3,  # if synchronize and forced, this is the minimum bound on cycles to look for in train set
        freq_init: Optional[
            List[float]
        ] = None,  # initial guess for the dominant angular frequency
        max_freqs: int = 1,  # the number of sinusoidal signals to fit
        min_dist: int = 60,  # maximum distance for finding local maximums in the PSD
        trainable_freq: bool = False,  # whether to make the frequency a variable during layer weight training
        bias: bool = True,  # add intercept (vertical displacement)
    ):
        # Rancoders inputs:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.encoding_depth = encoding_depth
        self.latent_dim = latent_dim
        self.decoding_depth = decoding_depth
        self.activation = activation
        self.output_activation = output_activation
        self.delta = delta

        # Syncrhonization inputs
        self.synchronize = synchronize
        self.force_synchronization = force_synchronization
        self.min_periods = min_periods
        self.freq_init = freq_init  # in radians (angular frequency)
        self.max_freqs = max_freqs
        self.min_dist = min_dist
        self.trainable_freq = trainable_freq
        self.bias = bias

        # set all variables to default to float32
        tf.keras.backend.set_floatx("float32")

    def build(self, input_shape, initial_stage: bool = False):
        x_in = Input(
            shape=(input_shape[-1],)
        )  # created for either raw signal or synchronized signal
        if initial_stage:
            freq_out = freqcoder()(x_in)
            self.freqcoder = Model(inputs=x_in, outputs=freq_out)
            self.freqcoder.compile(
                optimizer="adam", loss=lambda y, f: quantile_loss(0.5, y, f)
            )
        else:
            bounds_out = RANCoders(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                encoding_depth=self.encoding_depth,
                latent_dim=self.latent_dim,
                decoding_depth=self.decoding_depth,
                delta=self.delta,
                activation=self.activation,
                output_activation=self.output_activation,
                name="rancoders",
            )(x_in)
            self.rancoders = Model(inputs=x_in, outputs=bounds_out)
            self.rancoders.compile(
                optimizer="adam",
                loss=[
                    lambda y, f: quantile_loss(1 - self.delta, y, f),
                    lambda y, f: quantile_loss(self.delta, y, f),
                ],
            )
            if self.synchronize:
                t_in = Input(shape=(input_shape[-1],))
                sin_out = sincoder(
                    freq_init=self.freq_init, trainable_freq=self.trainable_freq
                )(t_in)
                self.sincoder = Model(inputs=t_in, outputs=sin_out)
                self.sincoder.compile(
                    optimizer="adam", loss=lambda y, f: quantile_loss(0.5, y, f)
                )

    def fit(
        self,
        x: np.ndarray,
        epochs: int = 100,
        batch_size: int = 360,
        shuffle: bool = True,
        freq_warmup: int = 10,  # number of warmup epochs to prefit the frequency
        sin_warmup: int = 10,  # number of warmup epochs to prefit the sinusoidal representation
        pos_amp: bool = True,  # whether to constraint amplitudes to be +ve only
    ):
        t = np.tile(np.array(range(x.shape[0])).reshape(-1, 1), (1, x.shape[1]))
        # Prepare the training batches.
        dataset = tf.data.Dataset.from_tensor_slices(
            (x.astype(np.float32), t.astype(np.float32))
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=x.shape[0]).batch(batch_size)

        # build and compile models (stage 1)
        if self.synchronize:
            self.build(x.shape, initial_stage=True)
            if self.freq_init:
                self.build(x.shape)
        else:
            self.build(x.shape)

        # pretraining step 1:
        if freq_warmup > 0 and self.synchronize and not self.freq_init:
            for epoch in range(freq_warmup):
                logging.info("\nStart of frequency pre-train epoch %d" % (epoch,))
                for step, (x_batch, t_batch) in enumerate(dataset):
                    # Prefit the oscillation encoder
                    with tf.GradientTape() as tape:
                        # forward pass
                        z, x_pred = self.freqcoder(x_batch)

                        # compute loss
                        x_loss = self.freqcoder.loss(x_batch, x_pred)  # median loss

                    # retrieve gradients and update weights
                    grads = tape.gradient(x_loss, self.freqcoder.trainable_weights)
                    self.freqcoder.optimizer.apply_gradients(
                        zip(grads, self.freqcoder.trainable_weights)
                    )
                logging.info("pre-reconstruction_loss: {}".format(tf.reduce_mean(x_loss).numpy()))

            # estimate dominant frequency
            z = (
                self.freqcoder(x)[0].numpy().reshape(-1)
            )  # must be done on full unshuffled series
            z = ((z - z.min()) / (z.max() - z.min())) * (
                1 - -1
            ) + -1  #  scale between -1 & 1
            p = Periodogram(z, sampling=1)
            p()
            peak_idxs = find_peaks(p.psd, distance=self.min_dist, height=(0, np.inf))[0]
            peak_order = p.psd[peak_idxs].argsort()[
                -self.min_periods - self.max_freqs :
            ][
                ::-1
            ]  # max PSDs found
            peak_idxs = peak_idxs[peak_order]
            if peak_idxs[0] < self.min_periods and not self.force_synchronization:
                self.synchronize = False
                logging.info(
                    "no common oscillations found, switching off synchronization attempts"
                )
            elif max(peak_idxs[: self.min_periods]) >= self.min_periods:
                idxs = peak_idxs[peak_idxs >= self.min_periods]
                peak_freqs = [
                    p.frequencies()[idx]
                    for idx in idxs[: min(len(idxs), self.max_freqs)]
                ]
                self.freq_init = [2 * np.pi * f for f in peak_freqs]
                logging.info(
                    "found common oscillations at period(s) = {}".format(
                        [1 / f for f in peak_freqs]
                    )
                )
            else:
                self.synchronize = False
                logging.info(
                    "no common oscillations found, switching off synchronization attempts"
                )

            # build and compile models (stage 2)
            self.build(x.shape)

        # pretraining step 2:
        if sin_warmup > 0 and self.synchronize:
            for epoch in range(sin_warmup):
                logging.info(
                    "\nStart of sine representation pre-train epoch %d" % (epoch,)
                )
                for step, (x_batch, t_batch) in enumerate(dataset):
                    # Train the sine wave encoder
                    with tf.GradientTape() as tape:
                        # forward pass
                        s = self.sincoder(t_batch)

                        # compute loss
                        s_loss = self.sincoder.loss(x_batch, s)  # median loss

                    # retrieve gradients and update weights
                    grads = tape.gradient(s_loss, self.sincoder.trainable_weights)
                    self.sincoder.optimizer.apply_gradients(
                        zip(grads, self.sincoder.trainable_weights)
                    )
                logging.info("sine_loss: {}".format(tf.reduce_mean(s_loss).numpy()))

            # invert params (all amplitudes should either be -ve or +ve). Here we make them +ve
            if pos_amp:
                a_adj = tf.where(
                    self.sincoder.layers[1].amp[:, 0] < 0,
                    self.sincoder.layers[1].amp[:, 0] * -1,
                    self.sincoder.layers[1].amp[:, 0],
                )  # invert all -ve amplitudes
                wb_adj = tf.where(
                    self.sincoder.layers[1].amp[:, 0] < 0,
                    self.sincoder.layers[1].wb[:, 0] + np.pi,
                    self.sincoder.layers[1].wb[:, 0],
                )  # shift inverted waves by half cycle
                wb_adj = tf.where(
                    wb_adj > 2 * np.pi, self.sincoder.layers[1].wb[:, 0] - np.pi, wb_adj
                )  # any cycle > freq must be reduced by half the cycle
                g_adj = tf.where(
                    self.sincoder.layers[1].amp[:, 0] < 0,
                    self.sincoder.layers[1].disp - a_adj,
                    self.sincoder.layers[1].disp,
                )  # adjust the vertical displacements after reversing amplitude signs
                K.set_value(self.sincoder.layers[1].amp[:, 0], a_adj)
                K.set_value(self.sincoder.layers[1].wb[:, 0], wb_adj)
                K.set_value(self.sincoder.layers[1].disp, g_adj)

        # train anomaly detector
        for epoch in range(epochs):
            logging.info("\nStart of epoch %d" % (epoch,))
            if self.synchronize:
                for step, (x_batch, t_batch) in enumerate(dataset):
                    # Train the sine wave encoder
                    with tf.GradientTape() as tape:
                        # forward pass
                        s = self.sincoder(t_batch)

                        # compute loss
                        s_loss = self.sincoder.loss(x_batch, s)  # median loss

                    # retrieve gradients and update weights
                    grads = tape.gradient(s_loss, self.sincoder.trainable_weights)
                    self.sincoder.optimizer.apply_gradients(
                        zip(grads, self.sincoder.trainable_weights)
                    )

                    # synchronize batch
                    b = (
                        self.sincoder.layers[1].wb / self.sincoder.layers[1].freq
                    )  # phase shift(s)
                    b_sync = b - tf.expand_dims(b[:, 0], axis=-1)
                    th_sync = tf.expand_dims(
                        tf.expand_dims(self.sincoder.layers[1].freq, axis=0), axis=0
                    ) * (
                        tf.expand_dims(t_batch, axis=-1)
                        + tf.expand_dims(b_sync, axis=0)
                    )  # synchronized angle
                    e = (x_batch - s) * tf.sin(
                        self.sincoder.layers[1].freq[0]
                        * ((np.pi / (2 * self.sincoder.layers[1].freq[0])) - b[:, 0])
                    )  # noise
                    x_batch_sync = (
                        tf.reduce_sum(
                            tf.expand_dims(self.sincoder.layers[1].amp, axis=0)
                            * tf.sin(th_sync),
                            axis=-1,
                        )
                        + self.sincoder.layers[1].disp
                        + e
                    )

                    # train the rancoders
                    with tf.GradientTape() as tape:
                        # forward pass
                        o_hi, o_lo = self.rancoders(x_batch_sync)

                        # compute losses
                        o_hi_loss = self.rancoders.loss[0](
                            tf.tile(
                                tf.expand_dims(x_batch_sync, axis=0),
                                (self.n_estimators, 1, 1),
                            ),
                            o_hi,
                        )
                        o_lo_loss = self.rancoders.loss[1](
                            tf.tile(
                                tf.expand_dims(x_batch_sync, axis=0),
                                (self.n_estimators, 1, 1),
                            ),
                            o_lo,
                        )
                        o_loss = o_hi_loss + o_lo_loss

                    # retrieve gradients and update weights
                    grads = tape.gradient(o_loss, self.rancoders.trainable_weights)
                    self.rancoders.optimizer.apply_gradients(
                        zip(grads, self.rancoders.trainable_weights)
                    )
                logging.info(
                    "sine_loss: {}, upper_bound_loss: {}, lower_bound_loss: {}".format(
                        tf.reduce_mean(s_loss).numpy(),
                        tf.reduce_mean(o_hi_loss).numpy(),
                        tf.reduce_mean(o_lo_loss).numpy(),
                    )
                )
            else:
                for step, (x_batch, t_batch) in enumerate(dataset):
                    # train the rancoders
                    with tf.GradientTape() as tape:
                        # forward pass
                        o_hi, o_lo = self.rancoders(x_batch)

                        # compute losses
                        o_hi_loss = self.rancoders.loss[0](
                            tf.tile(
                                tf.expand_dims(x_batch, axis=0),
                                (self.n_estimators, 1, 1),
                            ),
                            o_hi,
                        )
                        o_lo_loss = self.rancoders.loss[1](
                            tf.tile(
                                tf.expand_dims(x_batch, axis=0),
                                (self.n_estimators, 1, 1),
                            ),
                            o_lo,
                        )
                        o_loss = o_hi_loss + o_lo_loss

                    # retrieve gradients and update weights
                    grads = tape.gradient(o_loss, self.rancoders.trainable_weights)
                    self.rancoders.optimizer.apply_gradients(
                        zip(grads, self.rancoders.trainable_weights)
                    )
                logging.info(
                    "upper_bound_loss: {} lower_bound_loss: {}".format(
                        tf.reduce_mean(o_hi_loss).numpy(),
                        tf.reduce_mean(o_lo_loss).numpy(),
                    )
                )

    def predict_prob(
        self,
        x: np.ndarray,
        # t: np.ndarray,
        N: int,
        batch_size: int = 1000,
        desync: bool = False,
    ):
        t = np.tile(np.array(range(x.shape[0])).reshape(-1, 1), (1, x.shape[1]))
        # Prepare the training batches.
        dataset = tf.data.Dataset.from_tensor_slices(
            (x.astype(np.float32), t.astype(np.float32))
        )
        dataset = dataset.batch(batch_size)
        batches = int(np.ceil(x.shape[0] / batch_size))

        # loop through the batches of the dataset.
        if self.synchronize:
            s, x_sync, o_hi, o_lo = (
                [None] * batches,
                [None] * batches,
                [None] * batches,
                [None] * batches,
            )
            for step, (x_batch, t_batch) in enumerate(dataset):
                s_i = self.sincoder(t_batch).numpy()
                b = (
                    self.sincoder.layers[1].wb / self.sincoder.layers[1].freq
                )  # phase shift(s)
                b_sync = b - tf.expand_dims(b[:, 0], axis=-1)
                th_sync = tf.expand_dims(
                    tf.expand_dims(self.sincoder.layers[1].freq, axis=0), axis=0
                ) * (
                    tf.expand_dims(t_batch, axis=-1) + tf.expand_dims(b_sync, axis=0)
                )  # synchronized angle
                e = (x_batch - s_i) * tf.sin(
                    self.sincoder.layers[1].freq[0]
                    * ((np.pi / (2 * self.sincoder.layers[1].freq[0])) - b[:, 0])
                )  # noise
                x_sync_i = (
                    tf.reduce_sum(
                        tf.expand_dims(self.sincoder.layers[1].amp, axis=0)
                        * tf.sin(th_sync),
                        axis=-1,
                    )
                    + self.sincoder.layers[1].disp
                    + e
                ).numpy()
                o_hi_i, o_lo_i = self.rancoders(x_sync_i)
                o_hi_i, o_lo_i = (
                    tf.transpose(o_hi_i, [1, 0, 2]).numpy(),
                    tf.transpose(o_lo_i, [1, 0, 2]).numpy(),
                )
                if desync:
                    o_hi_i, o_lo_i = self.predict_desynchronize(
                        x_batch, x_sync_i, o_hi_i, o_lo_i
                    )
                s[step], x_sync[step], o_hi[step], o_lo[step] = (
                    s_i,
                    x_sync_i,
                    o_hi_i,
                    o_lo_i,
                )
            sins, synched, upper, lower = (
                np.concatenate(s, axis=0),
                np.concatenate(x_sync, axis=0),
                np.concatenate(o_hi, axis=0),
                np.concatenate(o_lo, axis=0),
            )

            synched_tiles = np.tile(
                synched.reshape(synched.shape[0], 1, synched.shape[1]), (1, N, 1)
            )
            result = np.where((synched_tiles < lower) | (synched_tiles > upper), 1, 0)
            inference = np.mean(np.mean(result, axis=1), axis=1)
            return inference
        else:
            o_hi, o_lo = [None] * batches, [None] * batches
            for step, (x_batch, t_batch) in enumerate(dataset):
                o_hi_i, o_lo_i = self.rancoders(x_batch)
                o_hi_i, o_lo_i = (
                    tf.transpose(o_hi_i, [1, 0, 2]).numpy(),
                    tf.transpose(o_lo_i, [1, 0, 2]).numpy(),
                )
                o_hi[step], o_lo[step] = o_hi_i, o_lo_i
            return np.concatenate(o_hi, axis=0), np.concatenate(o_lo, axis=0)

    def save(self, filepath: str = os.path.join(os.getcwd(), "ransyncoders.z")):
        file = {"params": self.get_config()}
        if self.synchronize:
            file["freqcoder"] = {
                "model": self.freqcoder.to_json(),
                "weights": self.freqcoder.get_weights(),
            }
            file["sincoder"] = {
                "model": self.sincoder.to_json(),
                "weights": self.sincoder.get_weights(),
            }
        file["rancoders"] = {
            "model": self.rancoders.to_json(),
            "weights": self.rancoders.get_weights(),
        }
        dump(file, filepath, compress=True)

    @classmethod
    def load(cls, filepath: str = os.path.join(os.getcwd(), "ransyncoders.z")):
        file = load(filepath)
        cls = cls()
        for param, val in file["params"].items():
            setattr(cls, param, val)
        if cls.synchronize:
            cls.freqcoder = model_from_json(
                file["freqcoder"]["model"], custom_objects={"freqcoder": freqcoder}
            )
            cls.freqcoder.set_weights(file["freqcoder"]["weights"])
            cls.sincoder = model_from_json(
                file["sincoder"]["model"], custom_objects={"sincoder": sincoder}
            )
            cls.sincoder.set_weights(file["sincoder"]["weights"])
        cls.rancoders = model_from_json(
            file["rancoders"]["model"], custom_objects={"RANCoders": RANCoders}
        )
        cls.rancoders.set_weights(file["rancoders"]["weights"])
        return cls

    def predict_desynchronize(
        self, x: np.ndarray, x_sync: np.ndarray, o_hi: np.ndarray, o_lo: np.ndarray
    ):
        if self.synchronize:
            E = (o_hi + o_lo) / 2  # expected values
            deviation = (
                tf.expand_dims(x_sync, axis=1) - E
            )  # input (synchronzied) deviation from expected
            deviation = self.desynchronize(deviation)  # desynchronize
            E = (
                tf.expand_dims(x, axis=1) - deviation
            )  # expected values in desynchronized form
            offset = (o_hi - o_lo) / 2  # this is the offet from the expected value
            offset = abs(self.desynchronize(offset))  # desynch
            o_hi, o_lo = (
                E + offset,
                E - offset,
            )  # add bound displacement to expected values
            return o_hi.numpy(), o_lo.numpy()
        else:
            raise ParameterError(
                "synchronize", "parameter not set correctly for this method"
            )

    def desynchronize(self, e: np.ndarray):
        if self.synchronize:
            b = (
                self.sincoder.layers[1].wb / self.sincoder.layers[1].freq
            )  # phase shift(s)
            return (
                e
                * tf.sin(
                    self.sincoder.layers[1].freq[0]
                    * ((np.pi / (2 * self.sincoder.layers[1].freq[0])) + b[:, 0])
                ).numpy()
            )
        else:
            raise ParameterError(
                "synchronize", "parameter not set correctly for this method"
            )

    def get_config(self):
        config = {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "encoding_depth": self.encoding_depth,
            "latent_dim": self.encoding_depth,
            "decoding_depth": self.decoding_depth,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "delta": self.delta,
            "synchronize": self.synchronize,
            "force_synchronization": self.force_synchronization,
            "min_periods": self.min_periods,
            "freq_init": self.freq_init,
            "max_freqs": self.max_freqs,
            "min_dist": self.min_dist,
            "trainable_freq": self.trainable_freq,
            "bias": self.bias,
        }
        return config


# Loss function
def quantile_loss(q, y, f):
    e = y - f
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)


class ParameterError(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
