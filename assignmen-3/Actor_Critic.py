# -*- coding: utf-8 -*-
import gym
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt

epsilon = 0.5
max_c = 100

def policy_function(observation, theta):
    weight = np.dot(theta, observation)
    rand_num = random.random()
    if rand_num > epsilon:
        s = 0
        action = random.randint(0, 2)
    else :
        s = (np.exp(weight) - np.exp(-weight)) / (np.exp(-weight)+ np.exp(-weight))
        if s < -0.3:
            action = 0
        elif s > 0.3:
            action = 1 
        else :
            action = 2
    return s, action

def critic(pre_phi, phi, pre_observation, observation, reward, discount_factor, df_lambda):
    learning_rate = 0.001
    v = np.dot(phi, observation)
    v = 1 / (1 + np.exp(-v))
    pre_v = np.dot(pre_phi, pre_observation)
    pre_v = 1 / (1 + np.exp(-pre_v))
    delta = reward + discount_factor * v - pre_v
    pre_phi = phi
    phi += learning_rate * df_lambda * delta * pre_v * (1 - pre_v) * (-pre_observation)
    return delta, pre_phi, phi

def actor(env, observation, theta, pre_phi, phi, discount_factor, df_lambda):
    learning_rate = 0.001
    sum_rewards = 0
    current_counter = 0
    while True:
        s, action = policy_function(observation, theta)
        pre_observation = observation
        observation, reward, done, _ = env.step(action)
        sum_rewards += reward
        #env.render()
        delta, pre_phi, phi = critic(pre_phi, phi, pre_observation, observation, reward, discount_factor, df_lambda)
        theta += learning_rate * df_lambda * delta * (1 - s * s) * (-pre_observation)
        df_lambda *= discount_factor
        if done:
            env.reset()
            return sum_rewards



def actor_critic(env):
    observation = env.reset()
    theta = np.random.rand(2)
    phi = np.random.rand(2)
    pre_phi = phi
    discount_factor = 0.9
    df_lambda = 1
    rewards_list = []
    for i in range(max_c):
        rewards = actor(env, observation, theta, pre_phi, phi, discount_factor, df_lambda)
        rewards_list.append(rewards)
    return rewards_list


if __name__ == '__main__':
    gym.envs.register(
        id='MountainCarMyEasyVersion-v0',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=100000,  # MountainCar-v0 uses 200
    )
    env = gym.make('MountainCarMyEasyVersion-v0')
    rewards_list = actor_critic(env)
    plt.plot(rewards_list)
    plt.show()


