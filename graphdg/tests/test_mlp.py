from unittest import TestCase

import tensorflow as tf

from graphdg.mlp import MLP


class TestMLP(TestCase):
    def setUp(self) -> None:
        self.x = tf.ones(shape=(5, 3))

    def test_mlp(self):
        mlp1 = MLP(output_sizes=[7, 11, 13], with_batch_norm=False)
        mlp2 = MLP(output_sizes=[3, 11, 17], with_batch_norm=True)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            output1 = mlp1(self.x)
            output2 = mlp2(self.x)

            self.assertEqual(output1.shape, tf.TensorShape(dims=[5, 13]))
            self.assertEqual(output2.shape, tf.TensorShape(dims=[5, 17]))
