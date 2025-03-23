from env.cutting_stock import CuttingStockEnv
import numpy as np
import random
import pickle
import pandas as pd
import os

# Đọc dữ liệu từ file CSV
df = pd.read_csv('D:\Final_Rel\data.csv')
batch_1 = df[df['batch_id'] == 1]

# Tách stocks và products
stocks_df = batch_1[batch_1['type'] == 'stock']
products_df = batch_1[batch_1['type'] == 'product']
stocks = list(zip(stocks_df['width'], stocks_df['height']))
products = list(zip(products_df['width'], products_df['height']))

# Khởi tạo môi trường
env = CuttingStockEnv(
    render_mode="human",
    max_w=max(stocks_df['width'].max(), 137),
    max_h=max(stocks_df['height'].max(), 136),
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
num_episodes = 300
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
# Kích thước Q-table
state_size = 100000
action_size = 300
Q_table = np.zeros((state_size, action_size))

# chuyển đổi trạng thái hiện tại của môi trường thành một chỉ số trạng thái (state) trong Q-learning.
#observation là trạng thái hiện tại của môi trường, chứa thông tin về các tấm vật liệu (stocks) 
# và các sản phẩm cần cắt (products) stocks: danh sách các tấm vật liệu hiện có..
#products: danh sách các sản phẩm cần cắt từ các tấm vật liệu.

def get_state(observation):
    stocks = observation["stocks"]
    products = observation["products"]
#Tính tổng diện tích chưa sử dụng trong các tấm vật liệu    
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
    """Tính reward dựa trên trạng thái và thông tin"""
    filled_ratio = info["filled_ratio"]
    trim_loss = info["trim_loss"]
    
    num_stocks_used = sum(1 for stock in observation["stocks"] if np.any(stock != -1))
    total_stocks = len(observation["stocks"])
    num_stocks_unused = total_stocks - num_stocks_used

    lambda_bonus = 0.2
    stock_bonus = lambda_bonus * (num_stocks_unused / total_stocks)
    
    # Phạt nếu hành động không thành công
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
        
        # Lưu trạng thái trước khi thực hiện action
        prev_observation = observation.copy()
        
        observation, step_reward, terminated, truncated, info = env.step(env_action)
        done = terminated
        
        # Kiểm tra xem action có thành công không
        action_successful = any(np.any(stock != prev_observation["stocks"][i]) 
                              for i, stock in enumerate(observation["stocks"]))
        
        # Tính reward chi tiết
        reward = get_reward(observation, info, action_successful)
        total_reward += reward
        
        if done and total_reward > max_ep_reward:
            max_ep_reward = total_reward
            max_ep_action_list = action_list
            max_start_state = ep_start_state
        
        action_list.append(env_action)

        next_state = get_state(observation)
        
        # Cập nhật Q-table
        Q_table[state, action] = (1 - alpha) * Q_table[state, action] + alpha * (
            reward + gamma * np.max(Q_table[next_state])
        )

        state = next_state

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Epsilon: {epsilon:.4f}")

# Hiển thị kết quả
print("Max reward = ", max_ep_reward)
print("Max action list = ", max_ep_action_list)

# Lưu kết quả theo cách tham khảo
save_dir = r'D:\Final_Rel\save_out'
os.makedirs(save_dir, exist_ok=True)

try:
    with open(os.path.join(save_dir, "q_table_batch1.pkl"), "wb") as f:
        pickle.dump(Q_table, f)
    with open(os.path.join(save_dir, "best_actions_batch1.pkl"), "wb") as f:
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