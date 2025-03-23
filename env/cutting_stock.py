import gymnasium as gym
import numpy as np
from .constants import METADATA
from .utils import set_seed, get_obs, get_info
from .actions import setup_spaces
from .renderer import render_frame, get_window_size

class CuttingStockEnv(gym.Env):
    metadata = METADATA

    def __init__(
        self,
        render_mode=None,
        min_w=50,
        min_h=50,
        max_w=100,
        max_h=100,
        num_stocks=100,
        max_product_type=25,
        max_product_per_type=20,
        seed=42,
        stock_list=None,
        product_list=None,
    ):
        self.seed = seed
        set_seed(seed)
        
        self.min_w = min_w
        self.min_h = min_h
        self.max_w = max_w
        self.max_h = max_h

        if stock_list is not None:
            self.stock_list = stock_list
            self.num_stocks = len(stock_list)
        else:
            self.stock_list = None
            self.num_stocks = num_stocks

        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type
        self.cutted_stocks = np.full((self.num_stocks,), fill_value=0, dtype=int)
        
        self.product_list = product_list

        self.observation_space, self.action_space = setup_spaces(self)

        self._stocks = []
        self._products = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        set_seed(seed)
        self.cutted_stocks = np.full((self.num_stocks,), fill_value=0, dtype=int)
        
        self._stocks = []
        if self.stock_list is not None:
            for (w, h) in self.stock_list:
                stock = np.full((self.max_w, self.max_h), fill_value=-2, dtype=int)
                stock[:w, :h] = -1
                self._stocks.append(stock)
        else:
            for _ in range(self.num_stocks):
                width = np.random.randint(low=self.min_w, high=self.max_w + 1)
                height = np.random.randint(low=self.min_h, high=self.max_h + 1)
                stock = np.full((self.max_w, self.max_h), fill_value=-2, dtype=int)
                stock[:width, :height] = -1
                self._stocks.append(stock)
        self._stocks = tuple(self._stocks)
        
        self._products = []
        if self.product_list is not None:
            for (w, h) in self.product_list:
                product = {"size": np.array([w, h]), "quantity": 1}
                self._products.append(product)
        else:
            num_type_products = np.random.randint(low=1, high=self.max_product_type)
            for _ in range(num_type_products):
                width = np.random.randint(low=1, high=self.min_w + 1)
                height = np.random.randint(low=1, high=self.min_h + 1)
                quantity = np.random.randint(low=1, high=self.max_product_per_type + 1)
                product = {"size": np.array([width, height]), "quantity": quantity}
                self._products.append(product)
        self._products = tuple(self._products)

        observation = get_obs(self)
        info = get_info(self)

        if self.render_mode == "human":
            render_frame(self)
        return observation, info
    
    def step(self, action):
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        width, height = size
        x, y = position

        product_idx = None
        for i, product in enumerate(self._products):
            if np.array_equal(product["size"], size) or np.array_equal(
                product["size"], size[::-1]
            ):
                if product["quantity"] == 0:
                    continue
                product_idx = i
                break

        if product_idx is not None:
            if 0 <= stock_idx < self.num_stocks:
                stock = self._stocks[stock_idx]
                stock_width = np.sum(np.any(stock != -2, axis=1))
                stock_height = np.sum(np.any(stock != -2, axis=0))
                if (
                    x >= 0
                    and y >= 0
                    and x + width <= stock_width
                    and y + height <= stock_height
                ):
                    if np.all(stock[x : x + width, y : y + height] == -1):
                        self.cutted_stocks[stock_idx] = 1
                        stock[x : x + width, y : y + height] = product_idx
                        self._products[product_idx]["quantity"] -= 1

        terminated = all([product["quantity"] == 0 for product in self._products])
        reward = 1 if terminated else 0

        observation = get_obs(self)
        info = get_info(self)

        if self.render_mode == "human":
            render_frame(self)

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return render_frame(self)

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            self.window = None
            self.clock = None