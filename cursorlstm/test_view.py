from stable_baselines3 import DQN, PPO
from test_env import MyEnv

env = MyEnv()

# pathを指定して任意の重みをロードする
model = DQN.load("model/test")

# 10回試行する
for i in range(10):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # env.render()
        if dones:
            print("done!")
            break
