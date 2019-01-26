from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from actor import Actor
from critic import Critic
from experience_memory import ReplayMemory


def update_target_weights(var_target, var_active, tau):
    """ copy active layers weights to target layers.
    """
    for idx_weight in range(len(var_active)):
        tf.assign(var_target[idx_weight],
                  tau * var_active[idx_weight] +
                  (1 - tau) * var_target[idx_weight])


class DDPG(tf.keras.Model):
    """ DDPG model - continuous action space case
    Args:
        input_dim: shape of input
        action_dim: shape of action
        action_scale: (minimum value of action, maximum value of action)
        memory_size : size of replay memory.
        gamma : discount rate
        tau: parameter for soft update
        learning_rate_actor: learning rate for actor network
        learning_rate_critic: learning rate for critic network
        device_name : name of device(normally cpu:0 or gpu:0)
    """

    def __init__(self, input_dim, action_dim, action_scale, memory_size, gamma, tau, learning_rate_actor=1e-3,
                 learning_rate_critic=1e-3, device_name="cpu:0",
                 checkpoint_directory="ckpt/"):
        super(DDPG, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.memory_size = memory_size
        self.replay_memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.tau = tau
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.device_name = device_name

        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)

        # actor
        self.actor_active = Actor(self.input_dim, self.action_dim, self.action_scale, name="actor_active")
        self.actor_target = Actor(self.input_dim, self.action_dim, self.action_scale, name="actor_target")
        self.actor_target.trainable = False

        # critic
        self.critic_active = Critic(self.input_dim, self.action_dim, name="critic_active")
        self.critic_target = Critic(self.input_dim, self.action_dim, name="critic_target")
        self.critic_target.trainable = False

        # optimizer
        self.optimizer_actor = tf.train.AdamOptimizer(learning_rate=self.learning_rate_actor)
        self.optimizer_critic = tf.train.AdamOptimizer(learning_rate=self.learning_rate_critic)

        # logging
        self.global_step = 0

    def build(self):
        self.actor_active.build()
        self.actor_target.build()
        self.critic_active.build()
        self.critic_target.build()
        self.built = True

    def get_action(self, x):
        """ get action from features
        Args:
            x : input(state) features shape of  input_dim (without batch)
        Returns:
            best action actor network
        """
        return self.actor_active.get_action(x)

    def loss_critic(self, X, action, reward, X_next, done):
        """ get critic loss of training batch
        Args:
            X : input features batch, shape of (batch_size, input_shape)
            action : actions batch, shape of (batch_size, action_dim)
            r : reward batch, shape of (batch_size, 1)
            X_next : next_state features, shape of (batch_size, input_shape)
            done : done signal batch, shape of (batch_size, 1)
        Returns:
            mean squared error for critic q networks
        """
        # calculate target y-value
        done_0 = 1-done # toggle done(0.0, 1.0) to (1.0, 0.0)
        next_action = self.actor_target(X_next)
        q_targets_next = self.critic_target(X_next, next_action)
        expected_next_return = q_targets_next * done_0
        y = reward + (self.gamma * expected_next_return)
        # calculate active q-value
        q_active = self.critic_active(X, action)

        loss_val = tf.losses.mean_squared_error(labels=q_active, predictions=y)

        return loss_val

    def grad_critic(self, X, action, reward, X_next, done):
        """ get gradient of training batch
        Args:
            X : input features batch, shape of (batch_size, input_shape)
            action : actions batch, shape of (batch_size, action_dim)
            r : reward batch, shape of (batch_size, 1)
            X_next : next_state features, shape of (batch_size, input_shape)
            done : done signal batch, shape of (batch_size, 1)
        Returns:
            (gradient of critic variables, loss of batch)
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss_critic(X, action, reward, X_next, done)

        return tape.gradient(loss_val, self.critic_active.variables), loss_val

    def loss_actor(self, X):
        """ get actor loss of training batch
        Args:
            X : input features batch, shape of (batch_size, input_shape)
        Returns:
            -1 * mean q value of policy
        """
        q_active = self.critic_active(X, self.actor_active(X))
        loss_val = -1 * tf.reduce_mean(q_active, axis=0)

        return loss_val

    def grad_actor(self, X):
        """ get gradient of training batch
        Args:
            X : input features batch, shape of (batch_size, input_shape)
        Returns:
            (gradient of actor variables, loss of batch)
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss_actor(X)

        return tape.gradient(loss_val, self.actor_active.variables), loss_val

    def train(self, X, action, reward, X_next, done):
        """ train mini-batch one step
        Args:
            X : input features batch, shape of (batch_size, Fx, Fy, features)
            action : actions batch, shape of (batch_size, 1)
            reward : reward batch, shape of (batch_size, 1)
            X_next : next_state features, shape of (batch_size, Fx, Fy, features)
            done : done signal batch, shape of (batch_size, 1)
        """
        with tf.device(self.device_name):
            self.global_step += 1
            grads_critic, loss_critic = self.grad_critic(tf.convert_to_tensor(X),
                                               tf.convert_to_tensor(action),
                                               tf.convert_to_tensor(reward),
                                               tf.convert_to_tensor(X_next),
                                               tf.convert_to_tensor(done))
            self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic_active.variables))

            grads_actor, loss_actor = self.grad_actor(tf.convert_to_tensor(X))
            self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor_active.variables))

            update_target_weights(self.critic_target.variables, self.critic_active.variables, self.tau)
            update_target_weights(self.actor_target.variables, self.actor_active.variables, self.tau)
            return loss_critic, loss_actor

    def save(self):
        """ save current weight of layers
        """
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)
        print("saved step %d in %s" % (self.global_step, self.checkpoint_directory))

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
