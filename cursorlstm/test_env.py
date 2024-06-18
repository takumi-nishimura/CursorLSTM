import cv2
import gym
import gym.spaces
import numpy as np
import torch
from stable_baselines3 import DQN, PPO


class MyEnv(gym.Env):
    def __init__(self):
        self.WINDOW_SIZE = 600  # 画面サイズの決定
        # self.ACTION_MAP = np.array(
        #     [
        #         [1, 0],
        #         [1, 1],
        #         [0, 1],
        #         [-1, 1],
        #         [-1, 0],
        #         [-1, -1],
        #         [0, -1],
        #         [1, -1],
        #     ]
        # )  # アクションの用意
        self.ACTION_MAP = np.array(
            [
                [0, 0.5],
                [np.pi / 6, 0.5],
                [np.pi / 4, 0.5],
                [np.pi / 3, 0.5],
                [np.pi / 2, 0.5],
                [np.pi * 2 / 3, 0.5],
                [3 * np.pi / 4, 0.5],
                [5 * np.pi / 6, 0.5],
                [np.pi, 0.5],
                [7 * np.pi / 6, 0.5],
                [5 * np.pi / 4, 0.5],
                [4 * np.pi / 3, 0.5],
                [3 * np.pi / 2, 0.5],
                [5 * np.pi / 3, 0.5],
                [7 * np.pi / 4, 0.5],
                [11 * np.pi / 6, 0.5],
                [0, 1],
                [np.pi / 6, 1],
                [np.pi / 4, 1],
                [np.pi / 3, 1],
                [np.pi / 2, 1],
                [np.pi * 2 / 3, 1],
                [3 * np.pi / 4, 1],
                [5 * np.pi / 6, 1],
                [np.pi, 1],
                [7 * np.pi / 6, 1],
                [5 * np.pi / 4, 1],
                [4 * np.pi / 3, 1],
                [3 * np.pi / 2, 1],
                [5 * np.pi / 3, 1],
                [7 * np.pi / 4, 1],
                [11 * np.pi / 6, 1],
                [0, 1.5],
                [np.pi / 6, 1.5],
                [np.pi / 4, 1.5],
                [np.pi / 3, 1.5],
                [np.pi / 2, 1.5],
                [np.pi * 2 / 3, 1.5],
                [3 * np.pi / 4, 1.5],
                [5 * np.pi / 6, 1.5],
                [np.pi, 1.5],
                [7 * np.pi / 6, 1.5],
                [5 * np.pi / 4, 1.5],
                [4 * np.pi / 3, 1.5],
                [3 * np.pi / 2, 1.5],
                [5 * np.pi / 3, 1.5],
                [7 * np.pi / 4, 1.5],
                [11 * np.pi / 6, 1.5],
            ]
        )
        self.GOAL_RANGE = 50  # ゴールの範囲設定

        # アクション数定義
        ACTION_NUM = len(self.ACTION_MAP)
        self.action_space = gym.spaces.Discrete(ACTION_NUM)

        # 状態の範囲を定義
        LOW = np.array([-np.pi])
        HIGH = np.array([np.pi])

        # self.action_space = gym.spaces.Box(
        #     low=LOW, high=HIGH, dtype=np.float32
        # )
        self.observation_space = gym.spaces.Box(
            low=LOW, high=HIGH, dtype=np.float32
        )

        self.reset()

    def reset(self):
        # ボールとゴールの位置をランダムで初期化
        self.ball_position = np.array(
            [
                np.random.randint(0, self.WINDOW_SIZE),
                np.random.randint(0, self.WINDOW_SIZE),
            ]
        )
        self.goal_position = np.array(
            [
                [
                    np.random.randint(0, self.WINDOW_SIZE),
                    np.random.randint(0, self.WINDOW_SIZE),
                ],
                [
                    np.random.randint(0, self.WINDOW_SIZE),
                    np.random.randint(0, self.WINDOW_SIZE),
                ],
            ]
        )

        # 状態の作成
        vec = self.ball_position - self.goal_position
        distance = np.array([np.linalg.norm(v) for v in vec])
        v = np.argmin(distance)
        observation = np.arctan2(vec[v][0], vec[v][1])  # 角度の計算
        observation = np.array([observation])

        self.before_distance = np.linalg.norm(vec)

        return observation

    def step(self, action_index):
        action = self.ACTION_MAP[action_index]
        action = np.array(
            [
                action[1] * np.cos(action[0]),
                action[1] * np.sin(action[0]),
            ]
        )

        self.ball_position = self.ball_position - action

        # 状態の作成
        vec = self.ball_position - self.goal_position
        distance = np.array([np.linalg.norm(v) for v in vec])
        v = np.argmin(distance)
        observation = np.arctan2(vec[v][0], vec[v][1])  # 角度の計算
        observation = np.array([observation])

        # 報酬の計算
        distance = np.linalg.norm(vec[v])  # 距離の計算
        reward = self.before_distance - distance  # どれだけゴールに近づいたか

        # 終了判定
        done = False
        if distance < self.GOAL_RANGE:
            done = True

        self.before_distance = distance

        return observation, reward, done, {}

    def render(self):
        # opencvで描画処理してます
        img = np.zeros((self.WINDOW_SIZE, self.WINDOW_SIZE, 3))  # 画面初期化

        for i in range(2):
            cv2.circle(
                img,
                tuple(self.goal_position[i]),
                10,
                (0, 255, 0),
                thickness=-1,
            )  # ゴールの描画
            cv2.circle(
                img,
                tuple(self.goal_position[i]),
                self.GOAL_RANGE,
                color=(0, 255, 0),
                thickness=5,
            )  # ゴールの範囲の描画

        cv2.circle(
            img,
            (int(self.ball_position[0]), int(self.ball_position[1])),
            10,
            (0, 0, 255),
            thickness=-1,
        )  # ボールの描画

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps"
    env = MyEnv()
    model = DQN(
        "MlpPolicy", env, verbose=1, tensorboard_log="runs/test", device=device
    )
    model.learn(total_timesteps=100000)
    model.save("model/test")
