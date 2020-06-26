import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import graph_nets as gn
import numpy as np
import tensorflow as tf

from graphdg.tools import NODES, GLOBALS, EDGES, SENDERS, RECEIVERS


def set_seeds(seed: int):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def get_session_config() -> tf.ConfigProto:
    return tf.ConfigProto(
        intra_op_parallelism_threads=0,
        inter_op_parallelism_threads=0,
        allow_soft_placement=True,
    )


def save_session(session: tf.Session, path: str):
    saver = tf.train.Saver()
    saver.save(sess=session, save_path=path)


def load_session(session: tf.Session, path: str):
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=path)


class SessionIO:
    def __init__(self, path: str):
        self.path = path

    def save(self, session: tf.Session):
        logging.debug(f'Saving session to {self.path}')
        saver = tf.train.Saver()
        saver.save(sess=session, save_path=self.path)

    def load(self, session: tf.Session):
        logging.debug(f'Loading session from {self.path}')
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=self.path)


def get_number_trainable_variables() -> int:
    return int(np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))


def data_dicts_to_graphs_tuple(data_dicts: Sequence[Dict]) -> gn.graphs.GraphsTuple:
    minimals = []
    for data_dict in data_dicts:
        minimals.append({k: data_dict[k] for k in [GLOBALS, NODES, EDGES, SENDERS, RECEIVERS]})
    return gn.utils_np.data_dicts_to_graphs_tuple(minimals)


def graphs_tuple_to_data_dicts(graphs_tuple: gn.graphs.GraphsTuple) -> List[Dict]:
    return gn.utils_np.graphs_tuple_to_data_dicts(graphs_tuple)


def get_graph_feed_dict(placeholders: tf.placeholder, graph: gn.graphs.GraphsTuple) -> Dict:
    return gn.utils_tf.get_feed_dict(placeholders=placeholders, graph=graph)


def print_shapes(graphs_tuple: gn.graphs.GraphsTuple) -> None:
    shapes = [
        f'globals: {graphs_tuple.globals.shape}',
        f'nodes: {graphs_tuple.nodes.shape}',
        f'edges: {graphs_tuple.edges.shape}',
    ]

    print(', '.join(shapes))


def print_graphs_tuple(graphs_tuple: gn.graphs.GraphsTuple) -> None:
    shapes = [
        f'globals:\n{graphs_tuple.globals}',
        f'nodes:\n{graphs_tuple.nodes}',
        f'edges:\n{graphs_tuple.edges}',
    ]

    print('\n'.join(shapes))


@dataclass
class GraphFeatureDimensions:
    edges: int
    nodes: int
    globals: int

    @classmethod
    def from_data_dicts(cls, data_dicts: Sequence[Dict]) -> 'GraphFeatureDimensions':
        graphs_tuple = data_dicts_to_graphs_tuple(data_dicts)
        return GraphFeatureDimensions(
            edges=graphs_tuple.edges.shape[1],
            nodes=graphs_tuple.nodes.shape[1],
            globals=graphs_tuple.globals.shape[1],
        )

    def __add__(self, other) -> 'GraphFeatureDimensions':
        return GraphFeatureDimensions(
            edges=self.edges + other.edges,
            nodes=self.nodes + other.nodes,
            globals=self.globals + other.globals,
        )


def create_placeholder_graph(feature_dims: GraphFeatureDimensions, name: str = 'graph_ph'):
    return gn.utils_tf.placeholders_from_data_dicts(
        data_dicts=[{
            NODES: np.empty([1, feature_dims.nodes], dtype=np.float64),
            GLOBALS: np.empty([feature_dims.globals], dtype=np.float64),
            EDGES: np.empty([1, feature_dims.edges], dtype=np.float64),
            SENDERS: [],
            RECEIVERS: [],
        }],
        name=name,
    )


def create_zero_graph(blue_print: gn.graphs.GraphsTuple, feature_dims: GraphFeatureDimensions):
    graph = blue_print.replace(nodes=None, edges=None, globals=None)
    graph = gn.utils_tf.set_zero_edge_features(graph, edge_size=feature_dims.edges, dtype=tf.float64)
    graph = gn.utils_tf.set_zero_node_features(graph, node_size=feature_dims.nodes, dtype=tf.float64)
    graph = gn.utils_tf.set_zero_global_features(graph, global_size=feature_dims.globals, dtype=tf.float64)
    return graph


def vae_loss(
    x: tf.Tensor,
    mean_x: tf.Tensor,
    log_var_x: tf.Tensor,
    mean_z: tf.Tensor,
    log_var_z: tf.Tensor,
    name='vae_loss',
) -> tf.Tensor:
    with tf.name_scope(name):
        # Clip log variances
        log_var_x = tf.clip_by_value(log_var_x, clip_value_min=-8, clip_value_max=8)
        log_var_z = tf.clip_by_value(log_var_z, clip_value_min=-8, clip_value_max=8)

        log_2pi = tf.fill(value=np.log(2 * np.pi), dims=tf.shape(log_var_x))
        log_likelihood = -0.5 * tf.reduce_sum(tf.square(x - mean_x) / tf.exp(log_var_x) + log_var_x + log_2pi)
        kl = 0.5 * tf.reduce_sum(tf.exp(log_var_z) + tf.square(mean_z) - 1 - log_var_z)
        elbo = log_likelihood - kl
        return -elbo
