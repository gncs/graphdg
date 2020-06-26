from unittest import TestCase

import tensorflow as tf

from graphdg.mlp import LinearLayer


class TestLinear(TestCase):
    def setUp(self) -> None:
        self.x = tf.ones(shape=(5, 3))

    def test_linear(self):
        linear = LinearLayer(units=7, use_bias=True)
        linear_no_bias = LinearLayer(units=7, use_bias=False)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            output1 = linear(self.x)
            output2 = linear_no_bias(self.x)

            self.assertEqual(output1.shape, tf.TensorShape(dims=[5, 7]))
            self.assertEqual(output2.shape, tf.TensorShape(dims=[5, 7]))
