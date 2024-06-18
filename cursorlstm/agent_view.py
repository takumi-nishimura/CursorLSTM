import torch
from gym_env import TaskEnvWrapper
from stable_baselines3 import PPO

device = "cuda" if torch.cuda.is_available() else "mps"

model = PPO.load("model/cursor_agent", device=device)

env = TaskEnvWrapper()

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    print(
        f"Action: {action}, Agent cursor: {env.env.agent_cursor.center_x}, {env.env.agent_cursor.center_y}, Target: {obs['agent_target']}"
    )
    env.render()
