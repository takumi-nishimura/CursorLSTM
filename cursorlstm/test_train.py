from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from test_env import MyEnv

env = MyEnv()

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="log", device="mps")
print("start learning")
checkpoint_callback = CheckpointCallback(
    save_freq=500, save_path="./save_weights/", name_prefix="rl_model"
)
model.learn(total_timesteps=10000, callback=checkpoint_callback)
print("finish learning")
