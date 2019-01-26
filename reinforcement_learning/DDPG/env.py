import tensorflow as tf
import gym
import numpy as np
from collections import deque
import random
from colorama import Fore, Style
from ddpg import DDPG
from tqdm import tqdm, tqdm_notebook, trange
from noise import OUNoise
import time

class Environment:
    """ Train & simulate wrapper for Atari-DQN
    Args:
        params: dictionary of parameters
            memory_size : size of replay memory. 100000 needs almost 25GB memory, recommend reduce it if you need
            exploration_step : pure exploration step
            gamma : discount rate
            tau: parameter for soft update
            lr_actor: learning rate for actor network
            lr_critic: learning rate for critic network
        device_name : name of device(normally cpu:0 or gpu:0)
    """

    def __init__(self, params, device_name):
        self.env = gym.make('Pendulum-v0')
        self.ddpg = DDPG(input_dim=self.env.observation_space.shape[0],
                         action_dim=self.env.action_space.shape[0],
                         action_scale=(self.env.action_space.low[0], self.env.action_space.high[0]),
                         memory_size=params["memory_size"], gamma=params["gamma"], tau=params["tau"],
                         learning_rate_actor=params["lr_actor"],
                         learning_rate_critic=params["lr_critic"], device_name=device_name
                         )
        self.ddpg.build()
        self.ddpg.summary()

        self.random_process = OUNoise(size=self.env.action_space.shape[0])

        # total step operated
        self.i_step = 0

    def load(self, global_step="latest"):
        """ Load saved weights for ddpg
        Args:
            global_step : load specific step, if "latest" load latest one
        """
        self.ddpg.load(global_step)

    def save(self):
        """ Save current weight of ddpg layers
        """
        self.ddpg.save()

    def train(self, episode, max_step, minibatch_size, render=False, verbose=1, val_epi=5, saving=False):
        """run the game with training network
        Args:
            episode : number of train episodes
            max_step : maximum step for each episode
            minibatch_size : minibatch size for replay memory training
            render : whether to show game simulating graphic
            verbose : for which step it will print the loss and accuracy (and saving)
            val_epi : number of episode for validation
            saving: whether to save checkpoint or not
        """
        losses = []
        episode_return = []
        verbose_return = []
        episode_return_val = []

        tr = trange(episode, desc="")
        for i_episode in tr:
            return_episode = 0
            observation = self.env.reset()
            self.random_process.reset()

            for t in range(max_step):
                self.i_step += 1
                if render:
                    self.env.render()

                X = observation.astype(np.float32)
                action_policy = self.ddpg.get_action(tf.convert_to_tensor(X))
                action_policy += self.random_process.sample()
                action_policy = np.clip(action_policy, self.env.action_space.low[0], self.env.action_space.high[0])
                observation, reward, done, info = self.env.step(action_policy)
                return_episode += reward

                X_next = observation.astype(np.float32)
                self.ddpg.replay_memory.append((X,
                                                action_policy,
                                                reward,
                                                X_next,
                                                done
                                                ))
                # training step
                if len(self.ddpg.replay_memory) > minibatch_size:
                    X_batch, action_batch, reward_batch, X_next_batch, done_batch = self.ddpg.replay_memory.get_batch(
                        minibatch_size)
                    loss_critic, loss_actor = self.ddpg.train(X_batch, action_batch, reward_batch, X_next_batch, done_batch)
                    losses.append((loss_critic, loss_actor))

                if done:
                    break

            episode_return.append(return_episode)
            verbose_return.append(return_episode)
            tr.set_description("%.4f" % (sum(episode_return) / len(episode_return)))

            if i_episode == 0 or ((i_episode + 1) % verbose == 0):
                if len(self.ddpg.replay_memory) <= minibatch_size:
                    stage_tooltip = "EXPLORATION"
                    print(Fore.RED + "[EPISODE %3d / STEP %5d] - %s" % (i_episode + 1, self.i_step, stage_tooltip))
                    print(Fore.GREEN + "Learned Step : %4d" % (self.ddpg.global_step))
                    print(Fore.BLUE + "AVG   Return         : %.4f" % (sum(verbose_return) / len(verbose_return)))
                    print(Fore.BLUE + "MAX   Return         : %.4f" % (max(verbose_return)))
                    continue
                else:
                    stage_tooltip = "TRAINING"
                losses_critic = [l[0] for l in losses]
                losses_actor = [l[1] for l in losses]

                # validation
                returns = []
                for epi_val in range(val_epi):
                    return_episode_val = 0
                    observation = self.env.reset()

                    for t in range(max_step):
                        if render:
                            self.env.render()

                        action_policy = self.ddpg.get_action(tf.convert_to_tensor(observation.astype(np.float32)))
                        observation, reward, done, info = self.env.step(action_policy)
                        return_episode_val += reward

                        if done:
                            # print(Fore.GREEN + "EPISODE %3d: REWARD: %s" % (i_episode, return_episode))
                            returns.append(return_episode_val)
                            break

                print(Fore.RED + "[EPISODE %3d / STEP %5d] - %s" % (i_episode + 1, self.i_step, stage_tooltip))
                print(Fore.GREEN + "Learned Step : %4d" % (self.ddpg.global_step))
                print(Fore.BLUE + "AVG   Return         : %.4f" % (sum(verbose_return) / len(verbose_return)))
                print(Fore.BLUE + "MAX   Return         : %.4f" % (max(verbose_return)))
                print(Fore.LIGHTYELLOW_EX + "AVG   LOSS Actor     :  %.4f" % (sum(losses_actor) / len(losses_actor)))
                print(Fore.LIGHTYELLOW_EX + "AVG   LOSS Critic    :  %.4f" % (sum(losses_critic) / len(losses_critic)))
                print(Fore.LIGHTRED_EX + "AVG VAL[%2d]   Return : %.4f" % (val_epi, sum(returns) / len(returns)))
                print(Fore.LIGHTRED_EX + "MAX VAL[%2d]   Return : %.4f" % (val_epi, max(returns)))
                verbose_return = []
                losses = []
                episode_return_val.append(sum(returns) / len(returns))

                if saving:
                    self.save()

                time.sleep(1)

        return episode_return

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
            observation = self.env.reset()

            for t in range(max_step):
                if render:
                    self.env.render()

                action_policy = self.ddpg.get_action(tf.convert_to_tensor(observation.astype(np.float32)))
                observation, reward, done, info = self.env.step(action_policy)
                return_episode += reward

                if done:
                    print(Fore.GREEN + "EPISODE %3d: REWARD: %s" % (i_episode, return_episode))
                    returns.append(return_episode)
                    break

        print(Fore.RED + "AVG REWARD : %s" % (sum(returns) / len(returns)))
        print(Fore.BLUE + "MAX REWARD : %s" % (max(returns)))
