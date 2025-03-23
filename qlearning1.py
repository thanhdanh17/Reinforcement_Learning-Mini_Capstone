from env.cutting_stock import CuttingStockEnv
import numpy as np
import random
import pickle
import pandas as pd
import os

# Danh sách stocks (width, height)
stocks = [
    (50, 50),   (60, 40),   (70, 50),   (80, 60),   (90, 70),
    (100, 50),  (110, 60),  (120, 80),  (130, 90),  (140, 100),
    (150, 120), (160, 130), (170, 140), (180, 150), (200, 200)
]

# Danh sách products (width, height)
products = [
    (10, 5),  (15, 10), (20, 10), (25, 15), (30, 20),
    (35, 20), (40, 30), (45, 25), (50, 30), (55, 35),
    (20, 15), (25, 10), (30, 15), (35, 20), (40, 25),
    (45, 30), (50, 35), (55, 40), (60, 45), (65, 50),
    (70, 30), (75, 40), (80, 50), (85, 55), (90, 60),
    (15, 10), (20, 15), (25, 20), (30, 25), (35, 30)
]

# Khởi tạo môi trường với dữ liệu mới
env = CuttingStockEnv(
    render_mode="human",
    max_w=max([stock[0] for stock in stocks]),  # Lấy chiều rộng lớn nhất từ stocks
    max_h=max([stock[1] for stock in stocks]),  # Lấy chiều cao lớn nhất từ stocks
    seed=42,
    stock_list=stocks,
    product_list=products,
)

# Tham số học tập
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 500

# Kích thước Q-table
state_size = 10000
action_size = 500
Q_table = np.zeros((state_size, action_size))

def get_state(observation):
    stocks = observation["stocks"]
    products = observation["products"]
    empty_space = sum(np.sum(stock == -1) for stock in stocks)
    remaining_products = sum(prod["quantity"] for prod in products)
    state = (empty_space * 1000 + remaining_products) % state_size
    return state

def get_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_size - 1)
    else:
        return np.argmax(Q_table[state])

def get_env_action(action, observation):
    list_prods = observation["products"]
    list_stocks = observation["stocks"]

    if not list_prods or not list_stocks:
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

    prod_idx = action % len(list_prods)
    prod = list_prods[prod_idx]

    if prod["quantity"] == 0:
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

    prod_w, prod_h = prod["size"]
    stock_idx = (action // len(list_prods)) % len(list_stocks)
    stock = list_stocks[stock_idx]

    stock_w = np.sum(np.any(stock != -2, axis=1))
    stock_h = np.sum(np.any(stock != -2, axis=0))

    for x in range(stock_w - prod_w + 1):
        for y in range(stock_h - prod_h + 1):
            if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                return {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x, y)}

    return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

def get_reward(observation, info, action_successful):
    filled_ratio = info["filled_ratio"]
    trim_loss = info["trim_loss"]
    
    num_stocks_used = sum(1 for stock in observation["stocks"] if np.any(stock != -1))
    total_stocks = len(observation["stocks"])
    num_stocks_unused = total_stocks - num_stocks_used

    lambda_bonus = 0.2
    stock_bonus = lambda_bonus * (num_stocks_unused / total_stocks)
    
    penalty = -0.1 if not action_successful else 0
    
    reward = (filled_ratio - trim_loss) + stock_bonus + penalty
    return reward

# Biến theo dõi kết quả tốt nhất
max_ep_reward = -float('inf')
max_ep_action_list = []
max_start_state = None

# Huấn luyện
for episode in range(num_episodes):
    observation, info = env.reset(seed=42)
    state = get_state(observation)
    
    total_reward = 0
    ep_start_state = state
    action_list = []
    done = False

    while not done:
        action = get_action(state)
        env_action = get_env_action(action, observation)
        
        prev_observation = observation.copy()
        
        observation, step_reward, terminated, truncated, info = env.step(env_action)
        done = terminated
        
        action_successful = any(np.any(stock != prev_observation["stocks"][i]) 
                              for i, stock in enumerate(observation["stocks"]))
        
        reward = get_reward(observation, info, action_successful)
        total_reward += reward
        
        if done and total_reward > max_ep_reward:
            max_ep_reward = total_reward
            max_ep_action_list = action_list
            max_start_state = ep_start_state
        
        action_list.append(env_action)

        next_state = get_state(observation)
        Q_table[state, action] = (1 - alpha) * Q_table[state, action] + alpha * (
            reward + gamma * np.max(Q_table[next_state])
        )

        state = next_state

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Epsilon: {epsilon:.4f}")

# Hiển thị kết quả
print("Max reward = ", max_ep_reward)
print("Max action list = ", max_ep_action_list)

# Lưu kết quả
save_dir = r'D:\Final_Rel\save_out'
os.makedirs(save_dir, exist_ok=True)

try:
    with open(os.path.join(save_dir, "new_q_table_batch1.pkl"), "wb") as f:
        pickle.dump(Q_table, f)
    with open(os.path.join(save_dir, "new_best_actions_batch1.pkl"), "wb") as f:
        pickle.dump(max_ep_action_list, f)
    print("Files saved successfully!")
except PermissionError:
    print("Permission denied! Please run with admin rights or change save directory.")

env.close()

# Phát lại tập tốt nhất
observation, _ = env.reset()
for action in max_ep_action_list:
    env.step(action)
    env.render()