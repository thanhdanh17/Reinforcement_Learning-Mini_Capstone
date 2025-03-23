import numpy as np
import pandas as pd
from env.cutting_stock import CuttingStockEnv
import random
import pickle
import os
import sys
import time



# Đường dẫn tới file dữ liệu
data_path = r'D:\git\REl\REL301m\Final_summit\data\data.csv'

# Đọc dữ liệu từ file CSV và lọc batch_id = 2
data = pd.read_csv(data_path)
batch_2 = data[data['batch_id'] == 2]  # Chỉ lấy batch_id = 2
stock_list = [(row['width'], row['height']) for _, row in batch_2[batch_2['type'] == 'stock'].iterrows()]
product_list = [(row['width'], row['height']) for _, row in batch_2[batch_2['type'] == 'product'].iterrows()]

# Khởi tạo môi trường
env = CuttingStockEnv(
    render_mode=None,  # None khi huấn luyện để tăng tốc độ
    stock_list=stock_list,
    product_list=product_list,
    max_w=max([w for w, _ in stock_list] + [w for w, _ in product_list]),
    max_h=max([h for _, h in stock_list] + [h for _, h in product_list]),
    num_stocks=len(stock_list)
)

# Tham số Q-learning
alpha = 0.1  # Tốc độ học
gamma = 0.9  # Hệ số giảm
epsilon = 1.0  # Khám phá ban đầu
epsilon_decay = 0.995  # Giảm epsilon
min_epsilon = 0.01  # Epsilon tối thiểu
num_episodes = 1000  # Số episode để học tốt hơn

# Q-table dạng dictionary
Q_table = {}

def state_to_key(observation):
    """Chuyển trạng thái thành tuple để dùng làm key trong Q-table."""
    stocks = observation["stocks"]
    products = observation["products"]
    # Tính tỷ lệ trống của mỗi stock
    fractions = tuple(
        np.sum(stock == -1) / (stock.shape[0] * stock.shape[1]) if np.any(stock != -2) else 1.0
        for stock in stocks
    )
    # Số lượng còn lại của mỗi product
    quantities = tuple(prod["quantity"] for prod in products)
    return (fractions, quantities)

def action_to_key(action):
    """Chuyển hành động thành tuple để dùng làm key."""
    return (action["stock_idx"], tuple(action["size"]), tuple(action["position"]))

def get_valid_actions(observation):
    """Lấy danh sách hành động khả thi."""
    actions = []
    list_prods = observation["products"]
    list_stocks = observation["stocks"]

    for stock_idx, stock in enumerate(list_stocks):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        for prod_idx, prod in enumerate(list_prods):
            if prod["quantity"] <= 0:
                continue
            prod_w, prod_h = prod["size"]
            if prod_w <= stock_w and prod_h <= stock_h:
                max_x = stock_w - prod_w
                max_y = stock_h - prod_h
                # Ưu tiên sát mép
                for x, y in [(0, 0), (0, random.randint(0, max_y)), (random.randint(0, max_x), 0)]:
                    if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                        actions.append({"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x, y)})
                        break
    return actions

def choose_action(state, observation, Q_table, epsilon):
    """Chọn hành động bằng epsilon-greedy."""
    valid_actions = get_valid_actions(observation)
    if not valid_actions:
        return None
    if random.uniform(0, 1) < epsilon:
        return random.choice(valid_actions)
    state_key = state_to_key(observation)
    if state_key not in Q_table or not Q_table[state_key]:
        return random.choice(valid_actions)
    action_values = Q_table[state_key]
    best_action_key = max(action_values, key=action_values.get)
    for action in valid_actions:
        if action_to_key(action) == best_action_key:
            return action
    return random.choice(valid_actions)

def get_reward(observation, info):
    """Tính phần thưởng."""
    filled_ratio = info["filled_ratio"]
    trim_loss = info["trim_loss"]
    num_stocks_used = sum(1 for stock in observation["stocks"] if np.any(stock != -2))
    total_stocks = len(observation["stocks"])
    stock_bonus = 0.2 * ((total_stocks - num_stocks_used) / total_stocks)
    reward = 2 * filled_ratio - trim_loss + stock_bonus + (1 if info.get("done", False) else 0)
    return reward

# Huấn luyện
best_reward = -float('inf')
best_actions = []

for episode in range(num_episodes):
    observation, info = env.reset()
    state = state_to_key(observation)
    done = False
    total_reward = 0
    actions = []

    while not done:
        action = choose_action(state, observation, Q_table, epsilon)
        if action is None:
            total_reward -= 0.1  # Phạt khi không còn hành động
            break

        next_observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        actions.append(action)

        next_state = state_to_key(next_observation)
        reward = get_reward(next_observation, info)  # Dùng trạng thái sau hành động

        # Cập nhật Q-table
        state_key = state
        action_key = action_to_key(action)
        if state_key not in Q_table:
            Q_table[state_key] = {}
        if action_key not in Q_table[state_key]:
            Q_table[state_key][action_key] = 0
        
        old_q = Q_table[state_key][action_key]
        next_max = max(Q_table.get(next_state, {}).values(), default=0)
        Q_table[state_key][action_key] = (1 - alpha) * old_q + alpha * (reward + gamma * next_max)

        total_reward += reward
        state = next_state
        observation = next_observation

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if total_reward > best_reward:
        best_reward = total_reward
        best_actions = actions

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

# Lưu kết quả
save_dir = r'D:\git\REl\REL301m\Final_summit\save_out'
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "q_table_batch2.pkl"), "wb") as f:
    pickle.dump(Q_table, f)
with open(os.path.join(save_dir, "best_actions_batch2.pkl"), "wb") as f:
    pickle.dump(best_actions, f)

print(f"Best Reward: {best_reward:.2f}")
print(f"Best Actions: {best_actions}")

# Phát lại với hiển thị dừng lại khi cắt xong
env = CuttingStockEnv(
    render_mode="human",
    stock_list=stock_list,
    product_list=product_list,
    max_w=max([w for w, _ in stock_list] + [w for w, _ in product_list]),
    max_h=max([h for _, h in stock_list] + [h for _, h in product_list]),
    num_stocks=len(stock_list)
)
observation, info = env.reset()
for action in best_actions:
    observation, reward, done, truncated, info = env.step(action)
    env.render()
    time.sleep(0.5)  # Độ trễ để nhìn rõ từng bước
    if done or truncated:
        print("Cutting completed! Displaying final state...")
        while True:
            env.render()
            time.sleep(0.1)  # Giữ trạng thái cuối, giảm tải CPU
            # Thoát bằng Ctrl+C hoặc thêm logic thoát (như nhấn phím)
env.close()