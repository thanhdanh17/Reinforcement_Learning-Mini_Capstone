# utils.py
import numpy as np

def calculate_reward(stock_w, stock_h, width, height, cut_count, max_cuts):
    used_area = width * height
    total_area = stock_w * stock_h
    reward = used_area / total_area
    if cut_count >= max_cuts:
        reward -= 0.1  # Phạt nếu vượt quá số lần cắt tối đa
    return reward