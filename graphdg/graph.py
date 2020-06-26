from typing import Sequence

import tensorflow as tf


class GraphNet(tf.keras.layers.Layer):
    def __init__(
        self,
        encoder: tf.keras.layers.Layer,
        cores: Sequence[tf.keras.layers.Layer],
        decoder: tf.keras.layers.Layer,
        name: str = 'graph_net',
    ):
        super().__init__(name=name)

        self._encoder = encoder
        self._cores = cores
        self._decoder = decoder

    def call(self, inputs, training=False, **kwargs):
        hidden = self._encoder(inputs, training=training)

        for core in self._cores:
            hidden = core(hidden)

        return self._decoder(hidden, training=training)
