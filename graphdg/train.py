import logging
from typing import Callable, Tuple, Sequence, Dict, Any

import sklearn.model_selection
import tensorflow as tf

from . import tools
from .tf_tools import SessionIO, get_number_trainable_variables


class Trainer:
    def __init__(self, loss_fn: Callable, feed_dict_fn: Callable):
        self.loss_fn = loss_fn
        self.minimizer = self._get_minimizer(loss_fn)
        self.feed_dict_fn = feed_dict_fn

    @staticmethod
    def _get_minimizer(loss: Callable) -> Callable:
        # Wrap minimizer for batch normalization to work
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer().minimize(loss)

    def train(
        self,
        session: tf.Session,
        data: Tuple[Sequence, ...],
        session_handler: SessionIO,
        batch_size=32,
        max_num_epochs=100,
        loss_threshold=25,
        valid_fraction=0.2,
        seed=42,
    ) -> dict:
        trainable_parameters = get_number_trainable_variables()
        info: Dict[str, Any] = {
            'trainable_parameters': trainable_parameters,
        }
        logging.info(f'Number of trainable variables: {trainable_parameters}')

        if max_num_epochs <= 0:
            return info

        assert (0 < valid_fraction < 1)
        assert (len(data) > 0)

        # Check sizes
        size = len(data[0])
        for d in data[1:]:
            assert (len(d) == size)

        valid_size = round(valid_fraction * size)
        train_size = size - valid_size

        split = sklearn.model_selection.train_test_split(
            *data,
            train_size=train_size,
            test_size=valid_size,
            random_state=seed,
        )

        train_data = split[::2]
        valid_data = split[1::2]

        best_loss = float('inf')
        loss_counter = 0

        info['valid_losses'] = []

        for epoch in range(max_num_epochs):
            for batch in tools.get_batches(range(train_size), batch_size=batch_size, seed=seed + epoch):
                batch_fd = self.feed_dict_fn(train_data, indices=batch, is_training=True)
                session.run(self.minimizer, feed_dict=batch_fd)

            valid_loss = 0
            for batch in tools.get_batches(range(valid_size), batch_size=batch_size, seed=seed):
                valid_fd = self.feed_dict_fn(valid_data, indices=batch, is_training=False)
                valid_loss += session.run(self.loss_fn, feed_dict=valid_fd)

            info['valid_losses'].append(valid_loss)
            logging.info(f'Epoch: {epoch}, loss: {valid_loss:.4f}')

            if valid_loss < best_loss:
                session_handler.save(session)
                best_loss = valid_loss
                loss_counter = 0
            else:
                loss_counter += 1

            if loss_counter > loss_threshold:
                break

        session_handler.load(session)
        logging.info(f'Lowest loss: {best_loss:.4f}')

        return info
