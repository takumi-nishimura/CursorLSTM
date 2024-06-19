from stable_baselines3 import DQN
from agent_train import MyEnv

env = MyEnv()

model = DQN.load("model/cursor_agent_dqn")

for i in range(10):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # env.render()
        if dones:
            print("done!")
            break
