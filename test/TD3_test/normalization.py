# -*- coding: utf-8 -*-
import numpy as np

class Normalization:
    def __init__(self):
        self.low_state = None
        self.high_state = None
        self.normal_reward = None

        self.high_state = np.tile(np.array([4.6]), 3)
        self.high_state = np.append(self.high_state, np.tile(np.array([100]), 3))
        self.high_state = np.append(self.high_state, np.array([10]))
        self.high_state = np.append(self.high_state, np.array([1.4, 100, 8, 550]))
        self.high_state = np.append(self.high_state, np.array([100, 40, 2]))
        self.low_state = np.array([1.8, 1.8, 1.8, 20, 20, 20, 6, 0.2, 20, 5, 10, 10, 1, 0.5])

        self.high_reward = [1, 10, 20, 20]
        self.low_reward = [0, 0, 0, 0]
        self.max = 1.5
        self.min = 0

    def state_normal(self, state):
        return np.round((state - self.low_state) / (self.high_state - self.low_state), decimals=3)

    def reward_round(self, reward, delay, consume, cost, load, bal_load):
        return round(reward, 2), round(delay, 2), round(consume, 2), round(cost, 2), round(load, 2), round(bal_load, 2)

    def reward_normal(self, delay, consume, cost, load):
        return np.clip(
            round(
                10 * (self.max - self.min) * (delay - self.low_reward[0]) / (self.high_reward[0] - self.low_reward[0]),
                3),
            *[0, 5]), np.clip(
            round((self.max - self.min) * (consume - self.low_reward[1]) / (self.high_reward[1] - self.low_reward[1]),
                  3), *[0, 5]), np.clip(
            round((self.max - self.min) * (cost - self.low_reward[2]) / (self.high_reward[2] - self.low_reward[2]), 3),
            *[0, 5]), np.clip(
            round((self.max - self.min) * (load - self.low_reward[3]) / (self.high_reward[3] - self.low_reward[3]), 3),
            *[0, 5])
