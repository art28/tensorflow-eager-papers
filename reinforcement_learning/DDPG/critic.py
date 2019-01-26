from __future__ import absolute_import, division, print_function
import tensorflow as tf


class Critic(tf.keras.Model):
    """ Critic model for DDPG - continuous action space case
    Args:
        input_dim: shape of input(state),
        action_dim: shape of action
        tau: parameter for soft update
    """

    def __init__(self, input_dim, action_dim, name):
        super(Critic, self).__init__(name=name)
        self.input_dim = input_dim
        self.action_dim = action_dim

        # active layer
        self.dense1 = tf.layers.Dense(400, activation=tf.nn.relu, name="dense1_critic")
        self.dense2 = tf.layers.Dense(300, activation=tf.nn.relu, name="dense2_state_critic")
        self.Q = tf.layers.Dense(1, name="Q")

    def build(self):
        """ activate model layers.
        """
        dummy_input = tf.zeros(self.input_dim, dtype=tf.float32)
        dummy_action = tf.zeros(self.action_dim, dtype=tf.float32)
        dummy_input = tf.expand_dims(dummy_input, 0)
        dummy_action = tf.expand_dims(dummy_action, 0)

        self.call(dummy_input, dummy_action)
        self.built = True

    def call(self, S, A, training=True):
        """ forward pass for active layers
        Args:
            S : input(state) features batch, shape of (batch_size, input_dim)
            A : action batch, shape of (batch_size, action_dim)
            training : whether it is training or inference, for batch_normalization
        Returns:
            q_value from active layers
          """
        x = self.dense1(S)
        # x = self.bn1(x_s, training=training)
        x = self.dense2(tf.concat([x, A], axis=1))
        # x = self.bn2(x, training=training)
        q = self.Q(x)
        return q
