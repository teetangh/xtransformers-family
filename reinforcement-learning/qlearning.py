import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def cartpole():
    env = gym.make("CartPole-v1")

    def policy(obs): return 1

    for _ in range(3):
        obs = env.reset()
        for _ in range(80):
            actions = policy(obs)
            obs, reward, done, info = env.step(actions)
            env.render()
            time.sleep(0.05)

    env.close()


if __name__ == "__main__":
    cartpole()
