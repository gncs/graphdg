from unittest import TestCase

import tensorflow as tf

from graphdg import tools, graph, tf_tools
from graphdg.core import Core
from graphdg.decoder import EdgeDecoder
from graphdg.encoder import Encoder

data_dicts = [{
    tools.GLOBALS: [0.0],
    tools.NODES: [[0.0], [1.0], [3.0], [4.0]],
    tools.EDGES: [[1.0], [2.0], [3.0], [4.0]],
    tools.SENDERS: [0, 3, 1, 2],
    tools.RECEIVERS: [3, 1, 2, 3],
}]


def edge_encoder(inputs, training):
    return tf.math.scalar_mul(2, inputs, name='edge_encoder')


def node_encoder(inputs, training):
    return tf.math.scalar_mul(3, inputs, name='node_encoder')


def edge_update(inputs, training):
    return tf.math.add(inputs, 1, name='edge_update')


def edge_decoder(inputs, training):
    return tf.math.reduce_max(inputs, axis=1, keepdims=True, name='edge_decoder')


class TestGraph(TestCase):
    def test_graph(self):
        feature_dims = tf_tools.GraphFeatureDimensions.from_data_dicts(data_dicts)
        graph_ph = tf_tools.create_placeholder_graph(feature_dims)

        graph_net = graph.GraphNet(
            encoder=Encoder(
                hidden_dims=tf_tools.GraphFeatureDimensions(edges=2, nodes=1, globals=1),
                edge_encoder=edge_encoder,
                node_encoder=node_encoder,
            ),
            cores=[Core(edge_update=edge_update, )],
            decoder=EdgeDecoder(decoder=edge_decoder, ),
        )

        output = graph_net(inputs=graph_ph, training=False)

        feed_dict = tf_tools.get_graph_feed_dict(graph_ph, graph=tf_tools.data_dicts_to_graphs_tuple(data_dicts))
        with tf.Session() as session:
            output_eval = session.run(output, feed_dict=feed_dict)

            # tf.summary.FileWriter('test', session.graph)

        expected = [[7.0, 8.5, 7.0, 11.5]]
        self.assertListEqual(output_eval.T.tolist(), expected)
