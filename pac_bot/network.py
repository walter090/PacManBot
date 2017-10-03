import tensorflow as tf
from . import layers


class Network(object):
    def __init__(self):
        self._encoded = None
        self._lstm_output = None

    def encoding_network(self, x, layers_config=None, activation='lrelu', name='encoding'):
        """Build the encoding network.

        Args:
            x: Input tensor.
            layers_config: Configuration for each convolution layer;
                each layer is a list with three elements: filter size, stride, and
                number of output channels.
            activation: Choose activation function, between 'lrelu' and 'elu.'
            name: Name of variable scope.

        Returns:
            fc: Output from the conv net.
        """
        with tf.variable_scope(name):
            if layers_config is None:
                layers_config = [
                    # Filter size, stride, num output channels
                    [(8, 8), (2, 2), 32],
                    [(8, 8), (4, 4), 64],
                    [(4, 4), (2, 2), 128],
                    [(4, 4), (2, 2), 256],
                ]

            conv_output = x
            for layer in layers_config:
                conv_output = layers.conv_layer(x=conv_output,
                                                conv_ksize=layer[0],
                                                conv_stride=layer[1],
                                                out_channels=layer[2],
                                                activation=activation)
            flattened = layers.flatten(conv_output)
            fc = layers.fully_conn(x=flattened, num_output=516)
            self._encoded = layers.fully_conn(x=fc, num_output=256)

            return self._encoded

    def lstm_network(self, x, action_size, cell_size=512):
        pass
