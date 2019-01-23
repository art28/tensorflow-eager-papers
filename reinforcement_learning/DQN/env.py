import tensorflow as tf
import gym
import numpy as np
from collections import deque
import random
from colorama import Fore, Style
from dqn import DQN
from tqdm import tqdm, tqdm_notebook, trange
from preprocess import preprocess


class Environment:
    """ Train & simulate wrapper for Atari-DQN
    Args:
        params: dictionary of parameters
            Fx, Fy : downsampling size
            features : number of features(channles) to put in the network. total input size would be (batch_size, Fx, Fy, features)
            memory_size : size of replay memory. 100000 needs almost 25GB memory, recommend reduce it if you need
            epsilon_initial : initial number of epsilon
            epsilon_final : last number of epsilon
            exploration_step : number of steps for pure exploration
            epsilon_step : number of steps for epsilon decrease
            gamma : discount rate
            skip_frame : number of steps which automatically use past action
        device_name : name of device(normally cpu:0 or gpu:0)
    """
    def __init__(self, params, device_name):
        self.env = gym.make('Breakout-v0')
        self.dqn = DQN(input_dim=(params["Fx"], params["Fy"], params["features"]),
                       num_action=self.env.action_space.n,
                       memory_size=params["memory_size"], gamma=params["gamma"], device_name=device_name
                       )
        self.dqn.build()
        self.dqn.summary()

        self.Fx = params["Fx"]
        self.Fy = params["Fy"]

        self.epsilon = params['epsilon_initial']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_step = params['epsilon_step']
        self.epsilon_decay_rate = (self.epsilon_final - self.epsilon) / self.epsilon_step

        self.features = params['features']
        self.skip_frame = params['skip_frame']
        self.exploration_step = params['exploration_step']
        self.decay_step = params['exploration_step'] + params['epsilon_step']

        # total step operated
        self.i_step = 0

    def load(self, global_step="latest"):
        """ Load saved weights for dqn
        Args:
            global_step : load specific step, if "latest" load latest one
        """
        self.dqn.load(global_step)

    def save(self):
        """ Save current weight of dqn layers
        """
        self.dqn.save()

    def preprocess(self, observation):
        """ Downsample obervation image
        Args:
            observation: observation image
        Returns:
            downsampled observation image
        """
        return preprocess(observation, self.Fx, self.Fy)

    def train(self, episode, max_step, minibatch_size, initial_life=5, render=False, verbose=1, saving=False):
        """run the game with training network
        Args:
            episode : number of train episodes
            max_step : maximum step for each episode
            minibatch_size : minibatch size for replay memory training
            initial_life : initial life of atari game, to give penalty on death
            render : whether to show game simulating graphic
            verbose : for which step it will print the loss and accuracy (and saving)
            saving: whether to save checkpoint or not
        """
        episode_return = []
        for i_episode in trange(episode):
            return_episode = 0
            observation = self.preprocess(self.env.reset())
            inputs = deque(maxlen=self.features)
            for _ in range(self.features):
                inputs.append(observation)

            action_now = -1
            life_now = initial_life

            for t in range(max_step):
                self.i_step += 1
                if render:
                    self.env.render()

                # exploration
                if self.i_step < self.exploration_step:
                    random_action = self.env.action_space.sample()
                    observation, reward, done, info = self.env.step(random_action)
                    return_episode += reward

                    if info['ale.lives'] < life_now:
                        reward -= 1
                        life_now = info['ale.lives']

                    X = np.transpose(np.array(inputs), [1, 2, 0])
                    inputs.append(self.preprocess(observation))
                    X_next = np.transpose(np.array(inputs), [1, 2, 0])
                    self.dqn.replay_memory.append((X,
                                                   random_action,
                                                   reward,
                                                   X_next,
                                                   done
                                                   ))
                # epsilon greedy
                elif t % self.skip_frame == 0:
                    X = np.transpose(np.array(inputs), [1, 2, 0])
                    if random.random() < self.epsilon:
                        action_now = self.env.action_space.sample()
                    else:
                        action_now = self.dqn.get_action(tf.convert_to_tensor(X))
                    observation, reward, done, info = self.env.step(action_now)
                    return_episode += reward

                    if info['ale.lives'] < life_now:
                        reward -= 1
                        life_now = info['ale.lives']

                    inputs.append(self.preprocess(observation))
                    X_next = np.transpose(np.array(inputs), [1, 2, 0])
                    self.dqn.replay_memory.append((X,
                                                   action_now,
                                                   reward,
                                                   X_next,
                                                   done
                                                   ))
                    # training step
                    X_batch, action_batch, reward_batch, X_next_batch, done_batch = self.dqn.replay_memory.get_batch(
                        minibatch_size)
                    self.dqn.train(X_batch, action_batch, reward_batch, X_next_batch, done_batch)

                    # epsilon decay
                    if self.epsilon > self.epsilon_final:
                        self.epsilon += self.epsilon_decay_rate

                # skip frame
                else:
                    observation, reward, done, info = self.env.step(action_now)
                    return_episode += reward

                    if info['ale.lives'] < life_now:
                        reward -= 1
                        life_now = info['ale.lives']

                    X = np.transpose(np.array(inputs), [1, 2, 0])
                    inputs.append(self.preprocess(observation))
                    X_next = np.transpose(np.array(inputs), [1, 2, 0])
                    self.dqn.replay_memory.append((X,
                                                   action_now,
                                                   reward,
                                                   X_next,
                                                   int(done)
                                                   ))

                    # epsilon decay
                    if self.epsilon > self.epsilon_final:
                        self.epsilon += self.epsilon_decay_rate

                if done:
                    break

            episode_return.append(return_episode)
            self.dqn.copy_active2target()

            if i_episode == 0 or ((i_episode + 1) % verbose == 0):
                if self.i_step < self.exploration_step:
                    stage_tooltip = "EXPLORATION"
                elif self.epsilon > self.epsilon_final:
                    stage_tooltip = "EPSILON-GREEDY : %.4f" % self.epsilon
                else:
                    stage_tooltip = "EPSILON-GREEDY[final] : %.4f" % self.epsilon
                # print(Fore.BLACK + "=" * 50)
                print(Fore.RED + "[EPISODE %3d / STEP %5d] - %s" % (i_episode + 1, self.i_step, stage_tooltip))
                print(Fore.GREEN + "Learned Step : %4d" % (self.dqn.global_step))
                print(Fore.BLUE + "AVG   Return : %.4f" % (sum(episode_return) / len(episode_return)))
                print(Fore.BLUE + "MAX   Return : %.4f" % (max(episode_return)))
                episode_return = list()
                # print(Fore.BLACK + "=" * 50 + Style.RESET_ALL)

                if saving:
                    self.save()

    def simulate(self, episode, max_step=1000, render=False):
        """Run the game with existing dqn network
        Args:
            episode : number of train episodes
            max_step : maximum step for each episode
            render : whether to show game simulating graphic
        """
        returns = []
        for i_episode in range(episode):
            return_episode = 0
            observation = self.preprocess(self.env.reset())
            inputs = deque(maxlen=self.features)
            for _ in range(self.features):
                inputs.append(observation)

            for t in range(max_step):
                if render:
                    self.env.render()

                X = np.transpose(np.array(inputs), [1, 2, 0])

                action_now = self.dqn.get_action(tf.convert_to_tensor(X))
                observation, reward, done, info = self.env.step(action_now)
                return_episode += reward

                inputs.append(self.preprocess(observation))

                if done:
                    print(Fore.GREEN + "EPISODE %3d: REWARD: %s" % (i_episode, return_episode))
                    returns.append(return_episode)
                    break

        print(Fore.RED + "AVG REWARD : %s" % (sum(returns) / len(returns)))
        print(Fore.BLUE + "MAX REWARD : %s" % (max(returns)))
