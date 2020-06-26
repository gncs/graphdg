from typing import List

import tensorflow as tf

from graphdg import tf_tools
from graphdg.core import Core
from graphdg.decoder import EdgeDecoder, NodeDecoder
from graphdg.encoder import Encoder
from graphdg.gcvae import GraphCVAE
from graphdg.graph import GraphNet
from graphdg.mlp import MLP


def get_encoder(hidden_dims: tf_tools.GraphFeatureDimensions, node_width: int, node_depth: int, edge_width: int,
                edge_depth: int) -> tf.keras.layers.Layer:
    return Encoder(
        hidden_dims=hidden_dims,
        node_encoder=MLP([node_width] * node_depth + [hidden_dims.nodes], name='node_encoder'),
        edge_encoder=MLP([edge_width] * edge_depth + [hidden_dims.edges], name='edge_encoder'),
    )


def get_cores(hidden_dims: tf_tools.GraphFeatureDimensions, node_width: int, node_depth: int, edge_width: int,
              edge_depth: int, num_core_layers: int) -> List[tf.keras.layers.Layer]:
    return [
        Core(
            name='layer_' + str(layer),
            num_iterations=1,
            node_update=MLP([node_width] * node_depth + [hidden_dims.nodes], name='node_update'),
            edge_update=MLP([edge_width] * edge_depth + [hidden_dims.edges], name='edge_update'),
        ) for layer in range(num_core_layers)
    ]


def get_node_decoder(width: int, depth: int) -> tf.keras.layers.Layer:
    return NodeDecoder(decoder=MLP([width] * depth + [2]))


def get_edge_decoder(width: int, depth: int) -> tf.keras.layers.Layer:
    return EdgeDecoder(decoder=MLP([width] * depth + [2]))


def get_model(graph_dims: tf_tools.GraphFeatureDimensions, config: dict) -> GraphCVAE:
    q_in_dims = graph_dims + tf_tools.GraphFeatureDimensions(edges=1, nodes=0, globals=0)
    q_hidden_dims = q_in_dims + tf_tools.GraphFeatureDimensions(
        edges=config['hidden_feats_edges'],
        nodes=config['hidden_feats_nodes'],
        globals=0,
    )

    q_gnn = GraphNet(
        encoder=get_encoder(
            hidden_dims=q_in_dims,
            node_width=config['node_encoder_width'],
            node_depth=config['node_encoder_depth'],
            edge_width=config['edge_encoder_width'],
            edge_depth=config['edge_encoder_depth'],
        ),
        cores=get_cores(
            hidden_dims=q_hidden_dims,
            node_width=config['node_core_width'],
            node_depth=config['node_core_depth'],
            edge_width=config['edge_core_width'],
            edge_depth=config['edge_core_depth'],
            num_core_layers=config['num_core_models'],
        ),
        decoder=get_node_decoder(
            width=config['decoder_width'],
            depth=config['decoder_depth'],
        ),
    )

    p_in_dims = graph_dims + tf_tools.GraphFeatureDimensions(edges=0, nodes=config['latent_dims'], globals=0)
    p_hidden_dims = p_in_dims + tf_tools.GraphFeatureDimensions(
        edges=config['hidden_feats_edges'],
        nodes=config['hidden_feats_nodes'],
        globals=0,
    )

    p_gnn = GraphNet(
        encoder=get_encoder(
            hidden_dims=p_in_dims,
            node_width=config['node_encoder_width'],
            node_depth=config['node_encoder_depth'],
            edge_width=config['edge_encoder_width'],
            edge_depth=config['edge_encoder_depth'],
        ),
        cores=get_cores(
            hidden_dims=p_hidden_dims,
            node_width=config['node_core_width'],
            node_depth=config['node_core_depth'],
            edge_width=config['edge_core_width'],
            edge_depth=config['edge_core_depth'],
            num_core_layers=config['num_core_models'],
        ),
        decoder=get_edge_decoder(
            width=config['decoder_width'],
            depth=config['decoder_depth'],
        ),
    )

    return GraphCVAE(
        encoder=q_gnn,
        decoder=p_gnn,
        xs_dim=1,  # Distance
        conditions_dim=graph_dims,
        latent_dim=config['latent_dims'],
    )
