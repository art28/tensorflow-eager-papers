from __future__ import absolute_import, division, print_function
import tensorflow as tf
import math

# https://github.com/tensorflow/models/blob/master/official/transformer/model/model_utils.py
def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.
    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.
    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position
    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


# https://github.com/tensorflow/models/blob/master/official/transformer/model/transformer.py
class LayerNormalization(tf.keras.Model):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


# most of the code is referenced by
# https://github.com/tensorflow/models/blob/master/official/transformer/model/attention_layer.py
class MultiHeadAttention(tf.keras.Model):
    def __init__(self, hidden_size, num_heads, name):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")
        super(MultiHeadAttention, self).__init__(name=name)

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.q_dense = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.mask = None

        self.output_dense = tf.layers.Dense(hidden_size, use_bias=False, name="output_transform")

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        # hidden size is last dimension size from first dense layers.
        depth = (self.hidden_size // self.num_heads)
        x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

        # make head to the front
        return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])

        return tf.reshape(x, [batch_size, length, self.hidden_size])

    def masking(self, logit):
        if self.mask is None:
            import numpy as np
            mask = np.zeros_like(logit[0][0])
            for i in range(len(mask)):
                for j in range(len(mask)):
                    if i < j:
                        mask[i][j] = -1e9
            mask = np.expand_dims(mask, 0)
            mask_heads = np.concatenate([mask for _ in range(self.num_heads)], axis=0)
            self.mask = tf.expand_dims(tf.convert_to_tensor(mask_heads), 0)
        mask_batch = tf.concat([self.mask for _ in range(tf.shape(logit)[0])], axis=0)

        return logit + mask_batch

    def call(self, Q, K, masking=False, *args, **kwargs):
        """Return attention(self or multi) Output
        Args:
          Q: Query Tensor
          K: Key & Value Tensor(if it is self-attention, Q = K)
          masking: apply decoder masking not to attend future information
        Returns:
          Tensor with shape [batch_size, length, hidden_size]
        """
        q = self.q_dense(Q)
        k = self.k_dense(K)
        v = self.v_dense(K)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        logits = tf.matmul(q, k, transpose_b=True)

        if masking:
            logits = self.masking(logits)

        weights = tf.nn.softmax(logits)

        attention_output = tf.matmul(weights, v)

        attention_output = self.combine_heads(attention_output)
        attention_output = self.output_dense(attention_output)

        return attention_output + Q


class FeedForward(tf.keras.Model):
    def __init__(self, hidden_size, filter_size, name):
        super(FeedForward, self).__init__(name=name)
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.dense1 = tf.layers.Dense(filter_size, activation=tf.nn.relu, name="ffn_filter")
        self.out = tf.layers.Dense(hidden_size, name="ffn_out")

    def call(self, X, *args, **kwargs):
        """Return Feedfoward Output
        Args:
          X: input tensor
        Returns:
          Tensor with shape [batch_size, length, hidden_size]
        """
        x = self.dense1(X)
        x = self.out(x)
        return x


# https://github.com/tensorflow/models/blob/master/official/transformer/model/transformer.py
class PrePostProcessingWrapper(tf.keras.Model):
    """Wrapper class that applies layer pre-processing and post-processing."""
    def __init__(self, layer, hidden_size, postprocess_dropout, name):
        super(PrePostProcessingWrapper, self).__init__(name=name)
        self.layer = layer
        self.postprocess_dropout = postprocess_dropout
        self.hidden_size = hidden_size

        # Create normalization layer
        self.layer_norm = LayerNormalization(self.hidden_size)

    def call(self, x, *args, **kwargs):
        """Return Layer-normalized + Drop-out + Residual Processing
        Args:
          x: main input tensor, added as residual
          train in **kwargs: if  True, dropout is  applied.
        Returns:
          Tensor with shape [batch_size, length, hidden_size]
        """

        # get "Train Attribute" explicitly.
        if "train" in kwargs:
            train = kwargs["train"]
        else:
            train = False
        # Get layer output
        y = self.layer(x, *args, **kwargs)

        if train:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)

        y = x + y

        y = self.layer_norm(y)

        return x + y



