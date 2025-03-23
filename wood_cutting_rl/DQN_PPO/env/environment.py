# environment.py
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from .constants import METADATA, MAX_QUANTITY
from .data_loader import load_stock_list, load_product_list
from .utils import calculate_reward
from .renderer import render_frame, save_gif

class FlooringCuttingStockEnv(gym.Env):
    metadata = METADATA

    def __init__(self, stock_file=None, product_file=None, render_mode=None, max_cuts_per_stock=20, seed=42, scale_factor=10):
        super().__init__()
        self.seed = seed
        self.max_cuts_per_stock = max_cuts_per_stock
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.frames = []
        self.scale_factor = scale_factor

        self.stock_list = load_stock_list(stock_file)
        self.product_list = load_product_list(product_file)
        self._initialize_parameters()
        self._setup_spaces()
        self._reset_state()

    def _initialize_parameters(self):
        self.num_stocks = len(self.stock_list)
        self.max_stock_w = max(w // self.scale_factor for w, h in self.stock_list)
        self.max_stock_h = max(h // self.scale_factor for w, h in self.stock_list)
        self.max_product_types = len(self.product_list)
        self.scaled_stock_list = [(w // self.scale_factor, h // self.scale_factor) for w, h in self.stock_list]
        self.scaled_product_list = [(w // self.scale_factor, h // self.scale_factor, q) for w, h, q in self.product_list]

    def _setup_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "stocks": spaces.Box(
                    low=-2,
                    high=self.max_product_types + 2,
                    shape=(self.num_stocks, self.max_stock_w, self.max_stock_h),
                    dtype=np.int32
                ),
                "products": spaces.Box(
                    low=0,
                    high=np.tile(np.array([self.max_stock_w, self.max_stock_h, MAX_QUANTITY]), (self.max_product_types, 1)),
                    shape=(self.max_product_types, 3),
                    dtype=np.int32
                ),
            }
        )
        # Làm phẳng action_space thành MultiDiscrete
        self.action_space = spaces.MultiDiscrete([
            self.num_stocks,          # stock_idx: 0 đến num_stocks-1
            self.max_stock_w,         # width: 1 đến max_stock_w
            self.max_stock_h,         # height: 1 đến max_stock_h
            self.max_stock_w,         # x: 0 đến max_stock_w-1
            self.max_stock_h          # y: 0 đến max_stock_h-1
        ])

    def _reset_state(self):
        self.cutted_stocks = np.zeros((self.num_stocks,), dtype=int)
        self.used_stocks = np.zeros((self.num_stocks,), dtype=int)
        self._stocks = np.full((self.num_stocks, self.max_stock_w, self.max_stock_h), -2, dtype=np.int32)
        for i, (w, h) in enumerate(self.scaled_stock_list):
            self._stocks[i, :w, :h] = -1
        self._products = [{"size": np.array([w, h]), "quantity": q} for w, h, q in self.scaled_product_list]

    def _get_obs(self):
        products_array = np.array(
            [[p["size"][0], p["size"][1], p["quantity"]] for p in self._products],
            dtype=np.int32
        )
        return {"stocks": self._stocks, "products": products_array}

    def _get_info(self):
        waste = sum(int(np.sum(stock == -1)) for stock in self._stocks if stock[0, 0] >= 0)
        total_use = sum(int(np.sum(stock > -2)) for stock in self._stocks if stock[0, 0] >= 0)
        total_area = sum(w * h for w, h in self.scaled_stock_list)
        return {
            "used_ratio": total_use / total_area if total_area > 0 else 0,
            "waste_ratio": waste / (total_use + 1e-7),
            "total_cuts": int(np.sum(self.cutted_stocks)),
            "stocks_used": int(np.sum(self.used_stocks)),
        }

    def reset(self, seed=None, options=None):
        self.frames = []
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)
        self._reset_state()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action, algorithm_name="first_fit"):
        # Xử lý hành động MultiDiscrete
        if len(action) == 5:
            stock_idx, width, height, x, y = action
            width = width + 1  # Điều chỉnh vì MultiDiscrete bắt đầu từ 0
            height = height + 1
        # Xử lý hành động từ wrapper DQN
        else:
            stock_idx, product_idx, x, y = action
            width, height = self._products[product_idx]["size"] if product_idx < len(self._products) else np.array([1, 1])

        product_idx = self._find_product(np.array([width, height]))
        reward, terminated = 0, False

        if product_idx is not None and 0 <= stock_idx < self.num_stocks:
            stock = self._stocks[stock_idx]
            stock_w, stock_h = self.scaled_stock_list[stock_idx]
            if self._is_valid_action(stock, x, y, width, height, stock_w, stock_h, stock_idx):
                self._apply_action(stock, x, y, width, height, product_idx, stock_idx)
                reward = calculate_reward(stock_w, stock_h, width, height, self.cutted_stocks[stock_idx], self.max_cuts_per_stock)

        terminated = all(p["quantity"] == 0 for p in self._products)
        if terminated:
            reward += 1 - self._get_info()["waste_ratio"]

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            render_frame(self)
            if terminated:
                save_gif(self, algorithm_name)
        return observation, reward, terminated, False, info

    def _find_product(self, size):
        for i, product in enumerate(self._products):
            if np.array_equal(product["size"], size) and product["quantity"] > 0:
                return i
        return None

    def _is_valid_action(self, stock, x, y, width, height, stock_w, stock_h, stock_idx):
        return (
            x >= 0 and y >= 0 and
            x + width <= stock_w and y + height <= stock_h and
            self.cutted_stocks[stock_idx] < self.max_cuts_per_stock and
            np.all(stock[x:x + width, y:y + height] == -1)
        )

    def _apply_action(self, stock, x, y, width, height, product_idx, stock_idx):
        stock[x:x + width, y:y + height] = product_idx + 1
        self.cutted_stocks[stock_idx] += 1
        self.used_stocks[stock_idx] = 1
        self._products[product_idx]["quantity"] -= 1

    def render(self):
        if self.render_mode == "rgb_array":
            return render_frame(self, return_array=True)
        elif self.render_mode == "human":
            render_frame(self)

    def close(self):
        if hasattr(self, 'window') and self.window is not None:
            pygame.display.quit()
            pygame.font.quit()
            self.window = None
            self.clock = None