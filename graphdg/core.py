from typing import Optional, Callable

import graph_nets as gn
import tensorflow as tf


class Core(tf.keras.layers.Layer):
    def __init__(
        self,
        num_iterations: int = 1,
        node_update: Optional[Callable] = None,
        edge_update: Optional[Callable] = None,
        global_update: Optional[Callable] = None,
        name: str = 'core',
    ):
        super().__init__(name=name)

        self.num_iterations = num_iterations

        self._node_update = node_update
        self._edge_update = edge_update
        self._global_update = global_update

    @staticmethod
    def get_edge_features(graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
        senders = graph.replace(edges=gn.blocks.broadcast_sender_nodes_to_edges(graph, name='sn_to_e'))
        receivers = graph.replace(edges=gn.blocks.broadcast_receiver_nodes_to_edges(graph, name='rn_to_e'))
        nodes = graph.replace(edges=0.5 * (senders.edges + receivers.edges))
        globs = graph.replace(edges=gn.blocks.broadcast_globals_to_edges(graph, name='g_to_e'))

        return gn.utils_tf.concat([graph, nodes, globs], axis=1, name='concat_e_feats')

    def get_edge_to_node_aggregator(self) -> Callable:
        return gn.blocks.ReceivedEdgesToNodesAggregator(reducer=tf.unsorted_segment_sum, name='re_to_n')

    def get_node_features(self, graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
        aggregator = self.get_edge_to_node_aggregator()
        edge_to_v_agg = graph.replace(nodes=aggregator(graph))
        globs_node = graph.replace(nodes=gn.blocks.broadcast_globals_to_nodes(graph, name='g_to_n'))

        return gn.utils_tf.concat([graph, edge_to_v_agg, globs_node], axis=1, name='concat_n_feats')

    @staticmethod
    def get_edge_to_global_aggregator() -> Callable:
        return gn.blocks.EdgesToGlobalsAggregator(reducer=tf.unsorted_segment_mean, name='e_to_g')

    @staticmethod
    def get_node_to_global_aggregator() -> Callable:
        return gn.blocks.NodesToGlobalsAggregator(reducer=tf.unsorted_segment_mean, name='n_to_g')

    def get_global_features(self, graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
        edge_to_global_agg = self.get_edge_to_global_aggregator()
        edge_to_glob = graph.replace(globals=edge_to_global_agg(graph))

        node_to_global_agg = self.get_node_to_global_aggregator()
        node_to_glob = graph.replace(globals=node_to_global_agg(graph))

        return gn.utils_tf.concat([graph, edge_to_glob, node_to_glob], axis=1, name='concat_g_feats')

    def call(self, inputs, training=False, **kwargs):
        x = inputs

        for i in range(self.num_iterations):
            if self._edge_update:
                edge_features = self.get_edge_features(x)
                x = x.replace(edges=self._edge_update(edge_features.edges, training=training))

            if self._node_update:
                node_features = self.get_node_features(x)
                x = x.replace(nodes=self._node_update(node_features.nodes, training=training))

            if self._global_update:
                global_features = self.get_global_features(x)
                x = x.replace(globals=self._global_update(global_features.globals, training=training))

        return x
