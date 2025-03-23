# test.py
import gymnasium as gym
from stable_baselines3 import DQN, PPO
import numpy as np
from env.environment import FlooringCuttingStockEnv
import pygame

# Wrapper từ main_dqn.py để làm rời rạc không gian hành động cho DQN
class DQNFlooringEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.discrete_actions = self._discretize_action_space()
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))

    def _discretize_action_space(self):
        grid_size = 5
        discrete_actions = []
        for stock_idx in range(self.env.num_stocks):
            for prod_idx, prod in enumerate(self.env._products):
                if prod["quantity"] > 0:
                    w, h = prod["size"]
                    stock_w, stock_h = self.env.scaled_stock_list[stock_idx]
                    for x in range(0, max(1, stock_w - w + 1), grid_size):
                        for y in range(0, max(1, stock_h - h + 1), grid_size):
                            if x + w <= stock_w and y + h <= stock_h:
                                discrete_actions.append([stock_idx, prod_idx, x, y])
        return discrete_actions

    def step(self, action_idx):
        action = self.discrete_actions[action_idx]
        return self.env.step(action, algorithm_name="dqn")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.discrete_actions = self._discretize_action_space()
        return obs, info

    def render(self, mode="human"):
        return self.env.render()  # Chuyển tiếp render đến môi trường gốc

    def close(self):
        self.env.close()  # Chuyển tiếp close đến môi trường gốc

def test_model(model, env, algorithm_name, window_width=1000, window_height=600):
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"Testing {algorithm_name}")
    clock = pygame.time.Clock()

    # Thay đổi render_mode trên môi trường gốc (nếu là wrapper) hoặc trực tiếp
    if isinstance(env, DQNFlooringEnv):
        env.env.render_mode = "human"  # Truy cập môi trường gốc
    else:
        env.render_mode = "human"

    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    running = True
    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        env.render()
        pygame.display.flip()
        clock.tick(60)

    print(f"{algorithm_name} - Total Reward: {total_reward:.4f}, Steps: {steps}, Info: {info}")
    if done:
        env.close()
    pygame.quit()
    return total_reward, steps, info

def main():
    # Môi trường gốc cho PPO
    env_ppo = FlooringCuttingStockEnv(
        stock_file="data/stocks.csv",
        product_file="data/products.csv",
        render_mode=None,
        scale_factor=10
    )
    # Môi trường với wrapper cho DQN
    env_dqn_base = FlooringCuttingStockEnv(
        stock_file="data/stocks.csv",
        product_file="data/products.csv",
        render_mode=None,
        scale_factor=10
    )
    env_dqn = DQNFlooringEnv(env_dqn_base)

    # Tải mô hình
    dqn_model = DQN.load("dqn_flooring_cutting")
    ppo_model = PPO.load("ppo_flooring_cutting")

    # Thử nghiệm DQN
    print("Testing DQN...")
    dqn_reward, dqn_steps, dqn_info = test_model(dqn_model, env_dqn, "DQN")

    # Thử nghiệm PPO
    print("\nTesting PPO...")
    ppo_reward, ppo_steps, ppo_info = test_model(ppo_model, env_ppo, "PPO")

    # So sánh kết quả
    print("\nComparison:")
    print(f"DQN - Reward: {dqn_reward:.4f}, Steps: {dqn_steps}")
    print(f"PPO - Reward: {ppo_reward:.4f}, Steps: {ppo_steps}")

if __name__ == "__main__":
    main()