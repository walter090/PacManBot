import tensorflow as tf


def conv_layer(x,
               conv_ksize,
               conv_stride,
               out_channels,
               pool_ksize=None,
               pool_stride=None,
               alpha=0.1,
               name='conv',
               padding='VALID',
               batchnorm=False):
    """Convolution-LReLU-max pooling layers.

        This function takes the input and returns the output of the result after
        a convolution layer and an optional average pooling layer.

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
