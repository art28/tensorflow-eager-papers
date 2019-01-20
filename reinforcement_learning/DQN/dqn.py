from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from experience_memory import ReplayMemory


class DQN(tf.keras.Model):
    """ DQN model for atari game
    Args:
        input_dim: shape of image input, (Fx, Fy, features)
        num_action: number of actions. 4 for atari.
        memory_size : size of replay memory. 100000 needs almost 25GB memory, recommend reduce it if you need
        gamma : discount rate
        skip_frame : number of steps which automatically use past action
        device_name : name of device(normally cpu:0 or gpu:0)
    """

    def __init__(self, input_dim, num_action, memory_size, gamma, learning_rate=1e-3, device_name="cpu:0",
                 checkpoint_directory="ckpt/"):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.num_action = num_action
        self.memory_size = memory_size
        self.replay_memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.device_name = device_name

        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)

        # active layer
        self.conv1 = tf.layers.Conv2D(16, (8, 8), (4, 4), padding="same", activation=tf.nn.relu, name="conv1")
        self.conv2 = tf.layers.Conv2D(32, (4, 4), (2, 2), padding="same", activation=tf.nn.relu, name="conv2")
        self.dense = tf.layers.Dense(256, activation=tf.nn.relu, name="dense")
        self.flatten = tf.layers.Flatten(name="flatten")
        self.Q = tf.layers.Dense(num_action, name="Q")

        # target layer : non-trainable
        self.conv1_target = tf.layers.Conv2D(16, (8, 8), (4, 4), padding="same", activation=tf.nn.relu,
                                             name="conv1_target")
        self.conv2_target = tf.layers.Conv2D(32, (4, 4), (2, 2), padding="same", activation=tf.nn.relu,
                                             name="conv2_target")
        self.dense_target = tf.layers.Dense(256, activation=tf.nn.relu, name="dense_target")
        self.Q_target = tf.layers.Dense(num_action, name="Q_target")
        self.conv1_target.trainable = False
        self.conv2_target.trainable = False
        self.dense_target.trainable = False
        self.Q_target.trainable = False

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.997, epsilon=1e-09)

        # logging
        self.global_step = 0

    def copy_active2target(self):
        """ copy active layers weights to target layers.
        """
        actives = [self.conv1, self.conv2, self.dense, self.Q]
        targets = [self.conv1_target, self.conv2_target, self.dense_target, self.Q_target]
        for idx_layer in range(len(actives)):
            for idx_weight in range(len(actives[idx_layer].weights)):
                tf.assign(targets[idx_layer].weights[idx_weight], actives[idx_layer].weights[idx_weight])

    def build(self):
        """ activate model layers.
        """
        dummy_input = tf.zeros(self.input_dim, dtype=tf.float32)
        dummy_input = tf.expand_dims(dummy_input, 0)
        self.q_active(dummy_input)
        self.q_target(dummy_input)
        self.built = True

    def q_active(self, X):
        """ forward pass for active layers
        Args:
            X : input features, shape of (batch_size, Fx, Fy, features)
        Returns:
            q_value from active layers
          """
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.dense(x)
        x = self.flatten(x)
        q = self.Q(x)
        return q

    def q_target(self, X):
        """ forward pass for target layers
        Args:
            X : input features batch, shape of (batch_size, Fx, Fy, features)
        Returns:
            q_value from target layers
        """
        x = self.conv1_target(X)
        x = self.conv2_target(x)
        x = self.dense_target(x)
        x = self.flatten(x)
        q = self.Q_target(x)
        return q

    def get_action(self, x):
        """ get action from features
        Args:
            x : input features, shape of (Fx, Fy, features) (no batch)
        Returns:
            best action from active q layers
        """
        x = tf.expand_dims(x, 0)
        q = self.q_active(x)
        return tf.argmax(q, axis=1)[0]

    def loss(self, X, action, r, X_next, done):
        """ get loss of training batch
        Args:
            X : input features batch, shape of (batch_size, Fx, Fy, features)
            action : actions batch, shape of (batch_size, 1)
            r : reward batch, shape of (batch_size, 1)
            X_next : next_state features, shape of (batch_size, Fx, Fy, features)
            done : done signal batch, shape of (batch_size, 1)
        Returns:
            mean squared error of q and y value
        """
        # calculate target y-value
        done_0 = (done - 1) * -1  # toggle done(0.0, 1.0) to (1.0, 0.0)
        q_wrapped = self.q_target(X_next)
        max_q_wrapped = tf.reduce_max(q_wrapped, axis=1)
        expected_next_return = max_q_wrapped * done_0
        y = r + expected_next_return

        # calculate active q-value
        q_active = self.q_active(X)
        indices = tf.one_hot(action, self.num_action, axis=1)
        original_q = tf.reduce_sum(q_active * indices, axis=1)

        loss_val = tf.losses.mean_squared_error(labels=original_q, predictions=y)
        return loss_val

    def grad(self, X, action, r, X_next, done):
        """ get gradient of training batch
        Args:
            X : input features batch, shape of (batch_size, Fx, Fy, features)
            action : actions batch, shape of (batch_size, 1)
            r : reward batch, shape of (batch_size, 1)
            X_next : next_state features, shape of (batch_size, Fx, Fy, features)
            done : done signal batch, shape of (batch_size, 1)
        Returns:
            (gradient of layer variables, loss of batch)
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss(X, action, r, X_next, done)
        return tape.gradient(loss_val, self.variables), loss_val

    def save(self):
        """ save current weight of layers
        """
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)
        print("saved step %d in %s" % (self.global_step, self.checkpoint_directory))

    def train(self, X, action, reward, X_next, done):
        """ train mini-batch one step
        Args:
            X : input features batch, shape of (batch_size, Fx, Fy, features)
            action : actions batch, shape of (batch_size, 1)
            r : reward batch, shape of (batch_size, 1)
            X_next : next_state features, shape of (batch_size, Fx, Fy, features)
            done : done signal batch, shape of (batch_size, 1)
        """
        with tf.device(self.device_name):
            self.global_step += 1
            grads, _ = self.grad(tf.convert_to_tensor(X),
                                 tf.convert_to_tensor(action),
                                 tf.convert_to_tensor(reward),
                                 tf.convert_to_tensor(X_next),
                                 tf.convert_to_tensor(done))
            self.optimizer.apply_gradients(zip(grads, self.variables))

    def load(self, global_step="latest"):
        """ load saved weights
        Args:
            global_step : load specific step, if "latest" load latest one
        """
        self.build()
        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = global_step
