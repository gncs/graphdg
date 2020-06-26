from typing import List, Optional, Sequence

import tensorflow as tf


class LinearLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        use_bias: bool = True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        name='linear_layer',
    ):
        super().__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.kernel: Optional[tf.Tensor] = None
        self.bias: Optional[tf.Tensor] = None

    def build(self, input_shape: tf.TensorShape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1].value, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
            name='kernel',
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units, ),
                initializer=self.bias_initializer,
                trainable=True,
                name='bias',
            )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = tf.matmul(inputs, self.kernel)

        if self.use_bias:
            output += self.bias

        return output


class MLP(tf.keras.layers.Layer):
    def __init__(
        self,
        output_sizes: Sequence[int],
        activation_name: str = 'relu',
        with_batch_norm: bool = True,
        use_biases: bool = True,
        name: str = 'mlp',
    ):
        super().__init__(name=name)

        self.output_sizes = output_sizes
        self.with_batch_norm = with_batch_norm
        self.use_biases = use_biases
        self.activation_name = activation_name

        self._batch_norms: List[tf.layers.BatchNormalization] = []
        self._linear_layers: List[tf.keras.layers.Layer] = []
        self._activation_layers: List[tf.keras.layers.Activation] = []

    def build(self, input_shape: tf.TensorShape):
        for i, size in enumerate(self.output_sizes):
            if i < len(self.output_sizes) - 1:
                if self.with_batch_norm:
                    self._batch_norms.append(tf.layers.BatchNormalization())
                self._activation_layers.append(tf.keras.layers.Activation(self.activation_name))

            self._linear_layers.append(LinearLayer(units=size, use_bias=self.use_biases))

        super().build(input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = inputs

        for i in range(len(self._linear_layers)):
            x = self._linear_layers[i](x)

            # Neither batch norm nor activations in last layer
            if i < len(self._linear_layers) - 1:
                if self.with_batch_norm:
                    x = self._batch_norms[i](x, training=training)
                x = self._activation_layers[i](x)

        return x
