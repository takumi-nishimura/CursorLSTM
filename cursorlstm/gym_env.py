import gym
from gym import spaces
import numpy as np
from env import TaskEnv

class TaskEnvWrapper(gym.Env):
    def __init__(self):
        super(TaskEnvWrapper, self).__init__()
        self.env = TaskEnv()

        self.observation_space = spaces.Box(low=0, high=self.env.main_geometry.width, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self):
        self.env.init_env()
        state = self._get_state()
        return state

    def step(self, action):
        self._take_action(action)
        self.env.step()
        state = self._get_state()
        reward = self._compute_reward()
        done = self._check_done()
        return state, reward, done, {}

    def _get_state(self):
        pass

if __name__ == "__main__":
    env = TaskEnvWrapper()

