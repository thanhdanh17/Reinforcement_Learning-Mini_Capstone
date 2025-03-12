import numpy as np
import os

def save_q_table(Q_table, filename="q_table.npy"):
    """Lưu Q-Table vào file numpy"""
    np.save(filename, Q_table)
    print(f"✅ Q-Table saved to {filename}")

def load_q_table(filename="q_table.npy"):
    """Load Q-Table từ file numpy hoặc khởi tạo nếu không tồn tại"""
    if os.path.exists(filename):
        print(f"✅ Loading existing Q-Table from {filename}")
        return np.load(filename)
    else:
        print(f"⚠️ No Q-Table found at {filename}. Initializing a new one.")
        return np.zeros((301, 151, 101, 51))  # Tạo Q-Table mới nếu file không có
