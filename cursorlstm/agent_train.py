import gym
import gym.spaces
import numpy as np
import torch
import stable_baselines3
from env import TaskEnv


class MyEnv(gym.Env):
    def __init__(self):
        self.env = TaskEnv()

        # ボタンの方角x2，操作者カーソルの方角
        self.observation_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32
        )

        self.ACTION_MAP = np.array(
            [
                [angle, speed]
                for speed in [1.0, 3.0, 5.0]
                for angle in np.arange(0, 2 * np.pi, np.pi / 6)
            ]
        )
        self.action_space = gym.spaces.Discrete(len(self.ACTION_MAP))

    def reset(self):
        self.env.init_env()
        state, distance = self._get_state()
        self.before_distance = distance

        return state

    def step(self, action):
        self._take_action(action)
        self.env.step()
        state, distance = self._get_state()
        reward = self._compute_reward(action)
        done = self._check_done()
        self.before_distance = distance

        return state, reward, done, {}

    def _get_state(self):
        agent_pos = self.env.agent_cursor.getPos()
        targets_pos = [
            [button.x + button.width, button.y + button.height]
            for button in self.env.target_buttons
            if not button.isChecked()
        ]
        while len(targets_pos) < 2:
            targets_pos.append(list(agent_pos))

        vector = agent_pos - targets_pos
        distance = np.array([np.linalg.norm(v) for v in vector])
        sorted_idx = np.argsort(distance)
        vector = vector[sorted_idx]
        state = np.array([np.arctan2(v[0], v[1]) for v in vector])

        operator_pos = self.env.operator_cursor.getPos()
        vector = agent_pos - operator_pos
        state = np.append(state, np.array([np.arctan2(vector[0], vector[1])]))

        return state, distance

    def _take_action(self, action_idx):
        action = self.ACTION_MAP[action_idx]
        action = np.array(
            [np.cos(action[0]) * action[1], np.sin(action[0]) * action[1]]
        )

        new_x = np.clip(
            self.env.agent_cursor.center_x + action[0],
            0,
            self.env.main_geometry.width,
        )
        new_y = np.clip(
            self.env.agent_cursor.center_y + action[1],
            0,
            self.env.main_geometry.height,
        )
        self.env.agent_cursor.setPos((new_x, new_y))

    def _compute_reward(self, action):
        reward = 0.0

        agent_cursor_pos = self.env.agent_cursor.getPos()
        target_button_pos = np.array(
            [
                [button.x + button.width / 2, button.y + button.height / 2]
                for button in self.env.target_buttons
                if not button.isChecked()
            ]
        )

        target_button_vector = agent_cursor_pos - target_button_pos
        target_distance = np.array(
            [np.linalg.norm(vec) for vec in target_button_vector]
        )
        reward = self.before_distance.min() - target_distance.min()

        return reward

    def _check_done(self):
        if self.before_distance.min() < 30:
            return True
        return False


if __name__ == "__main__":
    env = MyEnv()
    _device = "cuda" if torch.cuda.is_available() else "mps"
    model = stable_baselines3.DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="runs/agent",
        device=_device,
    )
    model.learn(total_timesteps=100000)
    model.save("model/cursor_agent_dqn")
