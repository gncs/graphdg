from typing import Tuple, Callable, Sequence, Optional

import numpy as np
import tensorflow as tf

from graphdg import tools, tf_tools


class GraphCVAE:
    def __init__(
        self,
        encoder: Callable,
        decoder: Callable,
        xs_dim: int,
        conditions_dim: tf_tools.GraphFeatureDimensions,
        latent_dim: int,
    ) -> None:
        self.latent_dim = latent_dim

        # Input
        self.x_ph = tf.placeholder(tf.float64, shape=[None, xs_dim], name='x')
        self.cond_ph = tf_tools.create_placeholder_graph(conditions_dim, name='graph')

        # Add targets to edges
        x_cond = self.cond_ph.replace(edges=tf.concat([self.cond_ph.edges, self.x_ph], axis=1))

        # Encode
        self.is_training_ph = tf.placeholder(tf.bool, name='is_training')
        encoded = encoder(x_cond, training=self.is_training_ph)
        self.mean_z, self.log_var_z = tf.split(encoded, 2, axis=1, name='split_encoded')

        # Reparametrization
        self.epsilon_ph = tf.placeholder(tf.float64, shape=[None, self.latent_dim], name='epsilon')
        with tf.name_scope('reparametrization'):
            z = self.mean_z + tf.sqrt(tf.exp(self.log_var_z)) * self.epsilon_ph

        # Decode
        z_cond = self.cond_ph.replace(nodes=tf.concat([self.cond_ph.nodes, z], axis=1))

        decoded = decoder(z_cond, training=self.is_training_ph)
        self.mean_x, self.log_var_x = tf.split(decoded, 2, axis=1, name='split_decoded')

        self.loss = tf_tools.vae_loss(
            x=self.x_ph,
            mean_x=self.mean_x,
            log_var_x=self.log_var_x,
            mean_z=self.mean_z,
            log_var_z=self.log_var_z,
        )

        # Sampling
        self.z_sample_ph = tf.placeholder(tf.float64, [None, self.latent_dim], name='z_sample')
        z_cond_sample = self.cond_ph.replace(nodes=tf.concat([self.cond_ph.nodes, self.z_sample_ph], axis=1))

        decoded_sample = decoder(z_cond_sample, training=False)
        self.mean_sample, self.log_var_sample = tf.split(decoded_sample, 2, axis=1, name='split_decoded_sample')

    def get_train_feed_dict(
        self,
        data: Tuple[Sequence, ...],
        indices: Optional[Sequence[int]] = None,
        is_training: bool = False,
    ) -> dict:
        xs, data_dicts = data
        assert (len(xs) == len(data_dicts))

        if indices is not None:
            xs = tools.get_items(xs, indices)
            data_dicts = tools.get_items(data_dicts, indices)

        graphs = tf_tools.data_dicts_to_graphs_tuple(data_dicts)

        return {
            self.x_ph: np.vstack(xs),
            self.cond_ph: graphs,
            self.epsilon_ph: np.random.normal(loc=0.0, scale=1.0, size=(len(graphs.nodes), self.latent_dim)),
            self.is_training_ph: is_training,
        }

    def sample(
        self,
        session: tf.Session,
        data_dicts: Sequence[dict],
        batch_size=64,
        seed=42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(seed)

        means = []
        log_vars = []

        for batch in tools.get_batches(data_dicts, batch_size=batch_size, shuffle=False):
            batch_dicts = tools.get_items(data_dicts, batch)
            graphs = tf_tools.data_dicts_to_graphs_tuple(batch_dicts)
            z_sample = np.random.normal(loc=0.0, scale=1.0, size=(len(graphs.nodes), self.latent_dim))

            feed_dict = {
                self.z_sample_ph: z_sample,
                self.cond_ph: graphs,
                self.is_training_ph: False,
            }

            batch_mean, batch_log_var = session.run([self.mean_sample, self.log_var_sample], feed_dict=feed_dict)

            means.append(batch_mean)
            log_vars.append(batch_log_var)

        return np.vstack(means), np.vstack(log_vars)
