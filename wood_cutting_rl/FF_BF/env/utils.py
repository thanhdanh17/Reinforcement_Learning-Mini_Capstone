# utils.py
import numpy as np

def find_first_fit_position(stock, size, stock_size):
    """Tìm vị trí đầu tiên trong stock mà sản phẩm có thể vừa."""
    width, height = size
    stock_w, stock_h = stock_size
    for x in range(stock_w - width + 1):
        for y in range(stock_h - height + 1):
            if np.all(stock[x:x + width, y:y + height] == -1):
                return x, y
    return None

def calculate_reward(stock_w, stock_h, width, height, cut_count, max_cuts):
    """Tính phần thưởng dựa trên diện tích sử dụng và số lần cắt."""
    used_area = width * height
    total_area = stock_w * stock_h
    reward = used_area / total_area
    if cut_count > max_cuts:
        reward -= 0.1
    return reward