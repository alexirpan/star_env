import gym
from gym import spaces
import numpy as np
from collections import Counter


class StarEnv(gym.Env):

    BACK = 0
    FORWARD = 1
    # Rest = branching choices

    def __init__(self, branches=4, length=20):
        self.branches = branches
        self.length = length
        self.n_states = 1 + self.branches * self.length
        self.n_actions = self.branches
        self.observation_space = spaces.Box(low=0, high=1, shape=[self.n_states])
        self.action_space = spaces.Discrete(self.n_actions)
        self.curr_state = 0
        # End of 1st branch.
        self.goal_state = self.length
        self.state_counter = Counter()
        self.track_counts = True

    @property
    def state(self):
        arr = np.zeros(self.n_states)
        arr[self.curr_state] = 1
        return arr

    def reset_count(self):
        self.state_counter = Counter()

    def reset(self):
        self.curr_state = 0
        self.reset_count()
        return self.state

    def step(self, action):
        # Action is 0 indexed
        assert self.action_space.contains(action)
        if self.track_counts:
            self.state_counter[self.curr_state] += 1
        if self.curr_state == 0:
            # Center, enter branch
            # 1, L + 1, 2L + 1, ...
            self.curr_state = self.length * action + 1
        else:
            # If action not in this set, do nothing.
            if action == self.BACK:
                self.curr_state -= 1
                if self.curr_state % self.length == 0:
                    # Back to center
                    self.curr_state = 0
            elif action == self.FORWARD:
                if self.curr_state % self.length != 0:
                    self.curr_state += 1
        reward = (1 if self.curr_state == self.goal_state else 0)
        if self.curr_state in self.state_counter and reward == 0:
            reward -= 0.05 * (self.state_counter[self.curr_state] ** 0.5)
        return self.state, reward, False, {}


if __name__ == '__main__':
    gym.envs.registration.register(
            id='Star-v0',
            entry_point='star_env:StarEnv')
