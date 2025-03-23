# main_ppo.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
from env.environment import FlooringCuttingStockEnv

def train_ppo():
    env = FlooringCuttingStockEnv(
        stock_file="data/stocks.csv",
        product_file="data/products.csv",
        render_mode=None,
        scale_factor=10
    )
    check_env(env)

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )

    model.learn(total_timesteps=50000)
    model.save("ppo_flooring_cutting")

    # Đóng môi trường sau khi huấn luyện để tránh xung đột
    env.close()

    # Khởi tạo lại môi trường với render_mode="human" để thử nghiệm
    env = FlooringCuttingStockEnv(
        stock_file="data/stocks.csv",
        product_file="data/products.csv",
        render_mode="human",
        scale_factor=10
    )
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        env.render()
    print(f"PPO - Total Reward: {total_reward:.4f}, Steps: {steps}, Info: {info}")
    env.close()

if __name__ == "__main__":
    train_ppo()