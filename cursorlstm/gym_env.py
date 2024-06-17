import gym
from gym import spaces
import numpy as np
from env import TaskEnv
from stable_baselines3 import PPO


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
                "agent_cursor": spaces.Box(
                    low=0, high=800, shape=(2,), dtype=np.float32
                ),
                "agent_target": spaces.Box(
                    low=0, high=800, shape=(2,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(4,), dtype=np.float32
        )

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
            target_button = np.pad(target_button, ((0, 2)))
        operator_cursor = self.env.operator_cursor
        agent_cursor = self.env.agent_cursor
        return {
            "target_button": target_button,
            "operator_cursor": np.array(
                [operator_cursor.center_x, operator_cursor.center_y],
                dtype=np.float32,
            ),
            "agent_cursor": np.array(
                [agent_cursor.center_x, agent_cursor.center_y],
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

        if action[2] > 5:
            self.env.agent_cursor.setClick(True)

        if action[3] > 5:
            self.env.change_target_button(self.env.agent_cursor)

    def _compute_reward(self):
        if self.env.judge_overlap_cursor(
            self.env.agent_cursor, self.env.agent_cursor.current_target_button
        ):
            self.env.agent_cursor.setClick(True)
            return 1.0
        else:
            return -0.01

    def _check_done(self):
        return all(button.isChecked() for button in self.env.target_buttons)

    def render(self):
        pass


if __name__ == "__main__":
    env = TaskEnvWrapper()
    model = PPO("MultiInputPolicy", env, verbose=1)

    model.learn(total_timesteps=100000)

    model.save("model/cursor_agent")
