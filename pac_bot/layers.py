import numpy as np
import tensorflow as tf


def conv_layer(x,
               conv_ksize,
               conv_stride,
               out_channels,
               pool_ksize=None,
               pool_stride=None,
               alpha=0.1,
               name='conv',
               padding='SAME',
               batchnorm=False):
    """Convolution-LReLU-max pooling layers.

        This function takes the input and returns the output of the result after
        a convolution layer and an optional max pooling layer.

        Args:
            x: Input from the previous layer.
            conv_ksize: tuple, filter size.
            conv_stride: Stride for the convolution layer.
            out_channels: Out channels for the convnet.
            pool_ksize: Filter size for the max pooling layer, Defaults None.
                The max pooling layer will not be added if one of the pooling
                argument is None.
            pool_stride: Stride for the max pooling layer, Defaults None.
                The max pooling layer will not be added if one of the pooling
                argument is None.
            alpha: Alpha value for Leaky ReLU.
            name: Name of the variable scope.
            padding: Padding for the layers, default 'VALID'.
            batchnorm: Set True to use batch normalization at this layer.

        Returns:
            Output tensor.
        """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='conv_w',
                                  shape=[conv_ksize[0], conv_ksize[1],
                                         x.get_shape().as_list()[3], out_channels],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable(name='conv_b',
                               shape=[out_channels],
                               initializer=tf.zeros_initializer())

        conv_stride = [1, conv_stride[0], conv_stride[1], 1]

        convoluted = tf.nn.conv2d(x, filter=weights,
                                  strides=conv_stride, padding=padding)
        convoluted = tf.nn.bias_add(convoluted, bias)

        if batchnorm:
            convoluted = batch_normalize(convoluted)

        conv = lrelu(convoluted, alpha)

        if pool_ksize is not None and pool_stride is not None:
            pool_ksize = (1,) + pool_ksize + (1,)
            pool_stride = (1,) + pool_stride + (1,)
            conv = tf.nn.max_pool(conv, ksize=pool_ksize,
                                  strides=pool_stride, padding=padding)
        return conv


def lrelu(x, alpha=0.1):
    """Leaky ReLU activation.
    linear = 0.5 * x + 0.5 * tf.abs(x)
    leaky = 0.5 * alpha * x - 0.5 * alpha * tf.abs(x)
    output = leaky + linear

    Args:
        x(Tensor): Input from the previous layer.
        alpha(float): Parameter for if x < 0.

    Returns:
        Output tensor
    """

    linear = tf.add(
        tf.multiply(0.5, x),
        tf.multiply(0.5, tf.abs(x))
    )
    half = tf.multiply(0.5, alpha)
    leaky = tf.subtract(
        tf.multiply(half, x),
        tf.multiply(half, tf.abs(x))
    )
    output = tf.add(linear, leaky)

    return output


def batch_normalize(x, epsilon=1e-5):
    """Batch normalization for the network.

    Args:
        x: Input tensor from the previous layer.
        epsilon: Variance epsilon.

    Returns:
        Output tensor.
    """
    # Before activation
    with tf.variable_scope('batch_norm'):
        mean, variance = tf.nn.moments(x, axes=[0, 1, 2])

        scale = tf.get_variable('bn_scale',
                                shape=[x.get_shape().as_list()[-1]],
                                initializer=tf.ones_initializer())
        offset = tf.get_variable('bn_bias',
                                 shape=[x.get_shape().as_list()[-1]],
                                 initializer=tf.zeros_initializer())
        normalized = tf.nn.batch_normalization(x=x,
                                               mean=mean,
                                               variance=variance,
                                               offset=offset,
                                               scale=scale,
                                               variance_epsilon=epsilon)
        return normalized


def flatten(x):
    """Flatten a tensor for the fully connected layer.

    Args:
        x(Tensor): 4-D tensor of shape [batch, height, width, channels] to be flattened
            to the shape of [batch, height * width * channels]

    Returns:
        Flattened tensor.
    """
    return tf.reshape(x, shape=[-1, np.prod(x.get_shape().as_list()[1:])])


def fully_conn(x,
               num_output,
               name='fc',
               activation='lrelu',
               keep_prob=1.):
    """Fully connected layer, this is is last parts of convnet.
    Fully connect layer requires each image in the batch be flattened.

    Args:
        x: Input from the previous layer.
        num_output: Output size of the fully connected layer.
        name: Name for the fully connected layer variable scope.
        activation: Set to True to add a leaky relu after fully connected
            layer. Set this argument to False if this is the final layer.
        keep_prob: Keep probability for dropout layers, if keep probability is 1
            there is no dropout. Defaults 1.

    Returns:
        Output tensor.
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='fc_w',
                                  shape=[x.get_shape().as_list()[-1], num_output],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
        biases = tf.get_variable(name='fc_b',
                                 shape=[num_output],
                                 initializer=tf.zeros_initializer())

        output = tf.nn.bias_add(tf.matmul(x, weights), biases)
        output = tf.nn.dropout(output, keep_prob=keep_prob)

        if activation == 'sigmoid':
            output = tf.sigmoid(output)
        elif activation == 'lrelu':
            output = lrelu(output)
        else:
            pass

        return output


def lstm(x,
         unit_size=512,
         peepholes=True,
         initializer=tf.random_normal_initializer,
         state_is_tuple=True,
         stacked_layers=4,
         name='lstm'):
    """LSTM layer

    Args:
        x: Input to the lstm.
        unit_size: Cell size.
        peepholes: Set to True to enable peepholes.
        initializer: Initialization function for weights.
        state_is_tuple: Set to True to return state as a tuple.
        stacked_layers: Number of stacked lstm cells.
        name: Variable scope name.

    Returns:
        outputs: Output from the RNN.
        state: Final state.
    """
    with tf.variable_scope(name):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=unit_size,
                                            use_peepholes=peepholes,
                                            initializer=initializer,
                                            state_is_tuple=state_is_tuple)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(stacked_layers)])
        # outputs:
        outputs, state = tf.nn.dynamic_rnn(inputs=tf.expand_dims(x, [0]),
                                           cell=stacked_lstm,
                                           time_major=False)

        return outputs, state
