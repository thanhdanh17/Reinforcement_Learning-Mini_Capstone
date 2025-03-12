import gym
from cutting_stock_env import CuttingStockEnv  # Import môi trường đã tạo

gym.envs.register(
    id='CuttingStock-v0',
    entry_point='cutting_stock_env:CuttingStockEnv',  # Đúng định dạng
)

env = gym.make('CuttingStock-v0')

# Kiểm tra môi trường
obs = env.reset()
print("Environment loaded successfully!")
print("Initial state:", obs)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
print("Reward range:", env.reward_range)
print("")
