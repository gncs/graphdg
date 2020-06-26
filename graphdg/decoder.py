from typing import Callable

import tensorflow as tf


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        decoder: Callable,
        name: str = 'decoder',
    ):
        super().__init__(name=name)

        self._decoder = decoder

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        raise NotImplementedError


class EdgeDecoder(Decoder):
    def __init__(self, decoder: Callable, name: str = 'edge_decoder'):
        super().__init__(decoder=decoder, name=name)

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        return self._decoder(inputs.edges, training=training)


class NodeDecoder(Decoder):
    def __init__(self, decoder: Callable, name: str = 'node_decoder'):
        super().__init__(decoder=decoder, name=name)

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        return self._decoder(inputs.nodes, training=training)


class GlobalDecoder(Decoder):
    def __init__(self, decoder: Callable, name: str = 'global_decoder'):
        super().__init__(decoder=decoder, name=name)

    def call(self, inputs, training=False, **kwargs) -> tf.Tensor:
        return self._decoder(inputs.globals, training=training)
