from gym_env import TaskEnvWrapper
from stable_baselines3 import PPO

model = PPO.load("model/cursor_agent", device="mps")

env = TaskEnvWrapper()

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    env.render()
