# algorithms.py
import numpy as np
from .utils import find_first_fit_position

def first_fit(env):
    """Thuật toán First-Fit: Đặt sản phẩm vào tấm đầu tiên khả thi."""
    product_idx, size = env._get_next_product()
    if product_idx is None:
        return env._get_obs(), 0, True, False, env._get_info()

    for stock_idx, stock in enumerate(env._stocks):
        if env.cutted_stocks[stock_idx] < env.max_cuts_per_stock:
            stock_size = env.stock_list[stock_idx]
            position = find_first_fit_position(stock, size, stock_size)
            if position is not None:
                action = {"stock_idx": stock_idx, "size": size, "position": np.array(position)}
                return env.step(action)
    return env._get_obs(), 0, False, False, env._get_info()

def first_fit_decreasing(env):
    """Thuật toán First-Fit Decreasing: Sắp xếp sản phẩm theo diện tích giảm dần trước khi đặt."""
    # Lấy danh sách sản phẩm còn lại và sắp xếp theo diện tích giảm dần
    products = [(i, p["size"], p["quantity"]) for i, p in enumerate(env._products) if p["quantity"] > 0]
    if not products:
        return env._get_obs(), 0, True, False, env._get_info()
    
    # Sắp xếp theo diện tích (width * height) giảm dần
    products.sort(key=lambda x: x[1][0] * x[1][1], reverse=True)
    product_idx, size, _ = products[0]  # Lấy sản phẩm đầu tiên (lớn nhất)

    for stock_idx, stock in enumerate(env._stocks):
        if env.cutted_stocks[stock_idx] < env.max_cuts_per_stock:
            stock_size = env.stock_list[stock_idx]
            position = find_first_fit_position(stock, size, stock_size)
            if position is not None:
                action = {"stock_idx": stock_idx, "size": size, "position": np.array(position)}
                return env.step(action)
    return env._get_obs(), 0, False, False, env._get_info()

def best_fit(env):
    """Thuật toán Best-Fit: Đặt sản phẩm vào tấm có ít diện tích trống nhất sau khi đặt."""
    product_idx, size = env._get_next_product()
    if product_idx is None:
        return env._get_obs(), 0, True, False, env._get_info()

    best_stock_idx = None
    best_position = None
    min_remaining_area = float('inf')

    for stock_idx, stock in enumerate(env._stocks):
        if env.cutted_stocks[stock_idx] < env.max_cuts_per_stock:
            stock_size = env.stock_list[stock_idx]
            position = find_first_fit_position(stock, size, stock_size)
            if position is not None:
                # Tính diện tích trống còn lại sau khi đặt
                stock_copy = stock.copy()
                x, y = position
                w, h = size
                stock_copy[x:x+w, y:y+h] = product_idx + 1
                remaining_area = np.sum(stock_copy[:stock_size[0], :stock_size[1]] == -1)
                if remaining_area < min_remaining_area:
                    min_remaining_area = remaining_area
                    best_stock_idx = stock_idx
                    best_position = position

    if best_stock_idx is not None:
        action = {"stock_idx": best_stock_idx, "size": size, "position": np.array(best_position)}
        return env.step(action)
    return env._get_obs(), 0, False, False, env._get_info()