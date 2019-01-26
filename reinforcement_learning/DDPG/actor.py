from __future__ import absolute_import, division, print_function
import tensorflow as tf


class Actor(tf.keras.Model):
    """ Actor model for DDPG
    Args:
        input_dim: shape of input(state),
        action_dim: shape of action
        action_scale: (minimum value of action, maximum value of action)
        tau: parameter for soft update
    """

    def __init__(self, input_dim, action_dim, action_scale, name):
        super(Actor, self).__init__(name=name)
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

        # active layer
        self.dense1 = tf.layers.Dense(400, activation=tf.nn.elu, name="dense1_actor")
        self.dense2 = tf.layers.Dense(300, activation=tf.nn.elu, name="dense2_actor")
        self.out = tf.layers.Dense(action_dim, activation=tf.nn.tanh, name="out_actor")

    def build(self):
        """ activate model layers.
        """
        dummy_input = tf.zeros(self.input_dim, dtype=tf.float32)
        dummy_input = tf.expand_dims(dummy_input, 0)
        self.call(dummy_input)
        self.built = True

    def call(self, S, training=True):
        """ forward pass for active layers
        Args:
            S : input(state) features batch, shape of (batch_size, input_dim)
            training : whether it is training or inference, for batch_normalization
        Returns:
            policy from active layers
          """
        x = self.dense1(S)
        # x = self.bn1(x, training=training)
        x = self.dense2(x)
        # x = self.bn2(x, training=training)
        p = self.out(x)
        p = p * self.action_scale[1]
        return p

    def get_action(self, s):
        """ get action from features
        Args:
            s : input(state) features shape of  input_dim (without batch)
        Returns:
            best action from active policy layers
        """
        x = tf.expand_dims(s, 0)
        p = self.call(x, training=False)
        return p[0]
