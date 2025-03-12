import numpy as np

class Policy2210xxx:
    def __init__(self):
        self.Q_table = np.zeros((301, 151, 101, 51))  # Q-table với trạng thái gỗ và hành động cắt

    def select_action(self, state):
        """Chọn hành động dựa trên Q-table hoặc exploration"""
        wood_length, wood_width = state
        
        if np.random.rand() < 0.1:  # Epsilon-greedy exploration
            cut_x = np.random.randint(0, max(1, wood_length))  # Đảm bảo có giá trị hợp lệ
            cut_y = np.random.randint(0, max(1, wood_width))
            cut_length = np.random.randint(1, max(2, min(wood_length, 50)))  # Đảm bảo high > low
            cut_width = np.random.randint(1, max(2, min(wood_width, 50)))
        else:
            best_action = np.unravel_index(np.argmax(self.Q_table[wood_length, wood_width]), (101, 51))
            cut_x, cut_y = best_action
            cut_length = np.random.randint(1, max(2, min(wood_length, 50)))  
            cut_width = np.random.randint(1, max(2, min(wood_width, 50)))

        return cut_x, cut_y, cut_length, cut_width  # Trả về đầy đủ 4 giá trị


    def update_policy(self, state, action, reward, next_state):
        """Cập nhật Q-table bằng công thức Q-learning"""
        wood_length, wood_width = state
        cut_x, cut_y, cut_length, cut_width = action
        old_value = self.Q_table[wood_length, wood_width, cut_x, cut_y]
        next_max = np.max(self.Q_table[next_state[0], next_state[1]])
        new_value = reward + 0.9 * next_max  # Sử dụng gamma = 0.9
        self.Q_table[wood_length, wood_width, cut_x, cut_y] = old_value + 0.1 * (new_value - old_value)

    def save(self):
        np.save("policy.npy", self.Q_table)

    def load(self):
        self.Q_table = np.load("policy.npy")

# Test policy
policy = Policy2210xxx()
state = (300, 150)
action = policy.select_action(state)
print("Selected action:", action)
policy.update_policy(state, action, 10, (200, 100))
print("Q-value after update:", policy.Q_table[300, 150, action[0], action[1]])
policy.save()
policy.load()
print("Q-value after load:", policy.Q_table[300, 150, action[0], action[1]])
# Expected output:
# Selected action: (0, 0, 1, 1)
# Q-value after update: 1.0
# Q-value after load: 1.0




