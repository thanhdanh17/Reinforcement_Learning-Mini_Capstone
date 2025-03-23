# main_dqn.py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import numpy as np
from env.environment import FlooringCuttingStockEnv

class DQNFlooringEnv(gym.Wrapper):
    """Bọc môi trường để DQN xử lý hành động rời rạc."""
    def __init__(self, env):
        super().__init__(env)
        self.discrete_actions = self._discretize_action_space()
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))

    def _discretize_action_space(self):
        """Rời rạc hóa không gian hành động."""
        grid_size = 5  # Giảm grid_size để phù hợp với tỉ lệ mới
        discrete_actions = []
        for stock_idx in range(self.env.num_stocks):
            for prod_idx, prod in enumerate(self.env._products):
                if prod["quantity"] > 0:
                    w, h = prod["size"]
                    stock_w, stock_h = self.env.scaled_stock_list[stock_idx]  # Sử dụng scaled_stock_list
                    for x in range(0, stock_w - w + 1, grid_size):
                        for y in range(0, stock_h - h + 1, grid_size):
                            discrete_actions.append([stock_idx, prod_idx, x, y])
        return discrete_actions

    def step(self, action_idx):
        action = self.discrete_actions[action_idx]
        return self.env.step(action, algorithm_name="dqn")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.discrete_actions = self._discretize_action_space()
        return obs, info

def train_dqn():
    env = FlooringCuttingStockEnv(
        stock_file="data/stocks.csv",
        product_file="data/products.csv",
        render_mode=None,
        scale_factor=10  # Thêm scale_factor
    )
    wrapped_env = DQNFlooringEnv(env)
    check_env(wrapped_env)

    model = DQN(
        "MultiInputPolicy",
        wrapped_env,
        learning_rate=1e-4,
        buffer_size=10000,
        batch_size=32,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1
    )

    model.learn(total_timesteps=50000)
    model.save("dqn_flooring_cutting")

    wrapped_env.env.render_mode = "human"
    obs, _ = wrapped_env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        wrapped_env.render()
    print(f"DQN - Total Reward: {total_reward:.4f}, Steps: {steps}, Info: {info}")
    wrapped_env.close()

if __name__ == "__main__":
    train_dqn()