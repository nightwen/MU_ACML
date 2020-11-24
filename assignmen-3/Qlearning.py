#coding=utf-8
import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import gym
import random
slow = True

#env = gym.make("MountainCar-v0")

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')
#env = gym.make('MountainCar-v0')
episode_nums=1000
n_states = 100
epsilon = 0.8
discount_factor = 0.9

initial_lr = 1.0 # Learning rate
min_lr = 0.003

def divide_state(observation, hill_high, hill_low,n_state = 100):
    env_gap = (hill_high - hill_low) / n_state
    x = int((observation[0] - hill_low[0]) / env_gap[0])
    y = int((observation[1] - hill_low[1]) / env_gap[1])
    return x, y


def policy_function(Q,x,y):
    rand_num = random.random()
    if rand_num > epsilon:
        action = random.randint(0, 2)
    else:
        Probility  = np.exp(Q[x][y])/np.sum(np.exp(Q[x][y]))
        action = np.random.choice(env.action_space.n, p=Probility)
    return action


def greedy_policy(Q,state):
    best_action = np.argmax(Q[state])
    return best_action


def plot(x,y):
    size = len(x)
    x = [x[i] for i in range(size) if i%10==0 ]
    y = [y[i] for i in range(size) if i%10==0 ]
    plt.plot(x, y, 'ro-')
    plt.ylim(-300, 0)
    plt.show()


def Q_learning(env,episode_nums,discount_factor=1.0, alpha=0.5,epsilon=0.1):

    Q = np.zeros((n_states, n_states, 3)) # three actions here
    rewards=[]



    for i_episode in range(1,1+episode_nums):
        observation = env.reset()

        if i_episode % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode, episode_nums))
            sys.stdout.flush()


        eta = max(min_lr, initial_lr * (0.85 ** (i_episode // 100)))

        sum_reward=0.0

        done = False

        while not done:
            #env.render()
            x,y = divide_state(observation,env.observation_space.high, env.observation_space.low)
            action = policy_function(Q,x,y)
            observation, reward, done, other = env.step(action)
            sum_reward += reward
            x1, y1 = divide_state(observation,env.observation_space.high, env.observation_space.low)
            Q[x][y][action] = Q[x][y][action] + eta * (
                        reward + discount_factor * np.max(Q[x1][y1]) - Q[x][y][action])
        rewards.append(sum_reward)

    return Q,rewards

Q,rewards = Q_learning(env,episode_nums)

average_rewards =[]
for i in range(10):
    _,rewards = Q_learning(env,episode_nums)
    if len(average_rewards)==0:
        average_rewards=np.array(rewards)
    else:
        average_rewards=average_rewards+np.array(rewards)
average_rewards=average_rewards*1.0/10
plot(range(1,1+episode_nums),average_rewards)