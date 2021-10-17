import random
import time
from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import KBinsDiscretizer

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 60000


def policy(_, __, pole_angle, pole_velocity): return int(pole_velocity > 0)


def cartpole():
    env = gym.make("CartPole-v1")
    print(env)
    print(env.observation_space)
    print(env.action_space)

    for _ in range(3):
        obs = env.reset()
        for _ in range(80):
            actions = policy(*obs)
            obs, reward, done, info = env.step(actions)
            env.render()
            time.sleep(0.05)

    env.close()


if __name__ == "__main__":
    cartpole()
