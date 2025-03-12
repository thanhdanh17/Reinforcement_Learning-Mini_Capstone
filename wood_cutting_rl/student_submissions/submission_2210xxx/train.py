from environment import WoodCuttingEnv
from policy_2210xxx import Policy2210xxx
from utils import save_q_table, load_q_table
import numpy as np
import random

# Khởi tạo môi trường và policy
env = WoodCuttingEnv()
policy = Policy2210xxx()
num_episodes = 5000
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Load Q-Table nếu đã có file lưu trước đó
Q_table = load_q_table("student_submissions/submission_2210xxx/q_table.npy")

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Chọn hành động theo epsilon-greedy
        if np.random.rand() < epsilon:
            action = (random.randint(0, 100), random.randint(0, 50))
        else:
            action = policy.select_action(state)

        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Cập nhật Q-Table theo công thức Q-learning
        old_value = Q_table[state[0], state[1], action[0], action[1]]
        next_max = np.max(Q_table[next_state[0], next_state[1]]) if not done else 0
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q_table[state[0], state[1], action[0], action[1]] = new_value

        state = next_state
    
    if episode % 500 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Lưu Q-Table sau khi train
save_q_table(Q_table, "student_submissions/submission_2210xxx/q_table.npy")
print("Training complete and Q-Table saved!")