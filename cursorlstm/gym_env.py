import gym
import numpy as np
import torch
from env import TaskEnv
from gym import spaces
import stable_baselines3
from logger import logger


class TaskEnvWrapper(gym.Env):
    def __init__(self):
        super(TaskEnvWrapper, self).__init__()

        self.env = TaskEnv()
        self.observation_space = spaces.Dict(
            {
                "target_button": spaces.Box(
                    low=0, high=800, shape=(4,), dtype=np.float32
                ),
                "operator_cursor": spaces.Box(
                    low=0, high=800, shape=(2,), dtype=np.float32
                ),
                "agent_target": spaces.Box(
                    low=0, high=800, shape=(2,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        self.before_distance = 0

    def reset(self):
        self.env.init_env()
        state = self._get_state()
        self.before_distance = self._distance()
        return state

    def step(self, action):
        self._take_action(action)
        self.env.step()
        state = self._get_state()
        reward = self._compute_reward(action)
        done = self._check_done(action)
        return state, reward, done, {}

    def _get_state(self):
        target_button = [
            button
            for button in self.env.target_buttons
            if not button.isChecked()
        ]
        target_button = np.array(
            [pos for button in target_button for pos in [button.x, button.y]],
            dtype=np.float32,
        )
        if len(target_button) < 4:
            target_button = np.pad(target_button, (0, 4 - len(target_button)))

        operator_cursor = self.env.operator_cursor
        agent_cursor = self.env.agent_cursor
        return {
            "target_button": target_button,
            "operator_cursor": np.array(
                [operator_cursor.center_x, operator_cursor.center_y],
                dtype=np.float32,
            ),
            "agent_target": np.array(
                [
                    agent_cursor.current_target_button.x,
                    agent_cursor.current_target_button.y,
                ],
                dtype=np.float32,
            ),
        }

    def _take_action(self, action):
        new_x = np.clip(
            self.env.agent_cursor.center_x + action[0] * 10,
            0,
            self.env.main_geometry.width,
        )
        new_y = np.clip(
            self.env.agent_cursor.center_y + action[1] * 10,
            0,
            self.env.main_geometry.height,
        )
        self.env.agent_cursor.setPos((new_x, new_y))

        if action[2] > 0:
            self.env.agent_cursor.setClick(True)

        if action[3] > 0:
            self.env.change_target_button(self.env.agent_cursor)

    def _compute_distance(self):
        return np.linalg.norm(
            np.array(
                [
                    self.env.agent_cursor.center_x,
                    self.env.agent_cursor.center_y,
                ]
            )
            - np.array(
                [
                    self.env.agent_cursor.current_target_button.x
                    + self.env.agent_cursor.current_target_button.width / 2,
                    self.env.agent_cursor.current_target_button.y
                    + self.env.agent_cursor.current_target_button.height / 2,
                ]
            ),
        )

    def _distance(self):
        return np.array(
            [
                self.env.agent_cursor.center_x,
                self.env.agent_cursor.center_y,
            ]
        ) - np.array(
            [
                self.env.agent_cursor.current_target_button.x
                + self.env.agent_cursor.current_target_button.width / 2,
                self.env.agent_cursor.current_target_button.y
                + self.env.agent_cursor.current_target_button.height / 2,
            ]
        )

    def _compute_reward(self, action):
        reward = 0.0

        current_distance = self._distance()

        if current_distance[0] < self.before_distance[0]:
            reward += 1.0
        else:
            reward -= 1.0

        if current_distance[1] < self.before_distance[1]:
            reward += 1.0
        else:
            reward -= 1.0

        if action[2] > 0:
            if self.env.judge_overlap_cursor(
                self.env.agent_cursor,
                self.env.agent_cursor.current_target_button,
            ):
                reward += 3.0
            else:
                reward -= 2.0

        if action[3] > 0:
            if (
                self.env.agent_cursor.current_target_button
                != self.env.operator_cursor.current_target_button
            ):
                reward += 0.1
            else:
                reward -= 0.3

        self.before_distance = current_distance
        return reward

    def _check_done(self, action):
        if action[2] > 0 and not self.env.judge_overlap_cursor(
            self.env.agent_cursor, self.env.agent_cursor.current_target_button
        ):
            return True
        elif all(button.isChecked() for button in self.env.target_buttons):
            return True
        else:
            False

    def render(self):
        pass


def make_env():
    return TaskEnvWrapper()


if __name__ == "__main__":
    env = TaskEnvWrapper()
    device = "cuda:0" if torch.cuda.is_available() else "mps"
    model = stable_baselines3.PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device="cuda:0",
        tensorboard_log="runs/PPO",
    )

    model.learn(total_timesteps=100000)

    model.save("model/cursor_agent")
