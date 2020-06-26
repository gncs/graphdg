from typing import Optional, Callable

import tensorflow as tf

from graphdg import tf_tools


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dims: tf_tools.GraphFeatureDimensions,
        node_encoder: Optional[Callable] = None,
        edge_encoder: Optional[Callable] = None,
        global_encoder: Optional[Callable] = None,
        name: str = 'encoder',
    ):
        super().__init__(name=name)

        self.hidden_dims = hidden_dims

        self._node_encoder = node_encoder
        self._edge_encoder = edge_encoder
        self._global_encoder = global_encoder

    def call(self, inputs, training=False, **kwargs):
        hidden_graph = tf_tools.create_zero_graph(blue_print=inputs, feature_dims=self.hidden_dims)

        if self._node_encoder:
            hidden_graph = hidden_graph.replace(nodes=self._node_encoder(inputs.nodes, training=training))

        if self._edge_encoder:
            hidden_graph = hidden_graph.replace(edges=self._edge_encoder(inputs.edges, training=training))

        if self._global_encoder:
            hidden_graph = hidden_graph.replace(globals=self._global_encoder(inputs.globals, training=training))

        return hidden_graph
