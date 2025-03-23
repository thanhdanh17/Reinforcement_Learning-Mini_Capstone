import pickle
import matplotlib.pyplot as plt

# Đường dẫn đến file lịch sử huấn luyện
history_path = r'D:\Final_Rel\save_out\new_q_table_batch1.pkl'

# Tải dữ liệu từ file
with open(history_path, 'rb') as f:
    training_history = pickle.load(f)

# Lấy dữ liệu
episodes = training_history["episodes"]
total_rewards = training_history["total_rewards"]
trim_losses = training_history["trim_losses"]
filled_ratios = training_history["filled_ratios"]
epsilons = training_history["epsilons"]
steps_per_episode = training_history["steps_per_episode"]

# Tạo các biểu đồ
plt.figure(figsize=(15, 10))

# 1. Biểu đồ Total Reward
plt.subplot(2, 3, 1)
plt.plot(episodes, total_rewards, label="Total Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward qua các Episode")
plt.grid(True)
plt.legend()

# 2. Biểu đồ Trim Loss
plt.subplot(2, 3, 2)
plt.plot(episodes, trim_losses, label="Trim Loss", color="orange")
plt.xlabel("Episode")
plt.ylabel("Trim Loss")
plt.title("Trim Loss qua các Episode")
plt.grid(True)
plt.legend()

# 3. Biểu đồ Filled Ratio
plt.subplot(2, 3, 3)
plt.plot(episodes, filled_ratios, label="Filled Ratio", color="green")
plt.xlabel("Episode")
plt.ylabel("Filled Ratio")
plt.title("Filled Ratio qua các Episode")
plt.grid(True)
plt.legend()

# 4. Biểu đồ Epsilon
plt.subplot(2, 3, 4)
plt.plot(episodes, epsilons, label="Epsilon", color="red")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon qua các Episode")
plt.grid(True)
plt.legend()

# 5. Biểu đồ Steps per Episode
plt.subplot(2, 3, 5)
plt.plot(episodes, steps_per_episode, label="Steps", color="purple")
plt.xlabel("Episode")
plt.ylabel("Steps per Episode")
plt.title("Số bước mỗi Episode")
plt.grid(True)
plt.legend()

# Điều chỉnh bố cục và hiển thị
plt.tight_layout()
plt.show()

# Lưu biểu đồ thành file
plt.savefig(r'D:\Final_Rel\save_out\bieudo.png')