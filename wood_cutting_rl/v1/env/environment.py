# environment.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .constants import METADATA, MAX_QUANTITY
from .data_loader import load_stock_list, load_product_list
from .utils import find_first_fit_position, calculate_reward
from .renderer import render_frame, save_gif

class FlooringCuttingStockEnv(gym.Env):
    metadata = METADATA

    def __init__(self, stock_file=None, product_file=None, render_mode=None, max_cuts_per_stock=20, seed=42):
        super().__init__()
        self.seed = seed
        self.max_cuts_per_stock = max_cuts_per_stock
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.frames = []

        # Load data
        self.stock_list = load_stock_list(stock_file)
        self.product_list = load_product_list(product_file)
        self._initialize_parameters()
        self._setup_spaces()
        self._reset_state()

    def _initialize_parameters(self):
        self.num_stocks = len(self.stock_list)
        self.max_stock_w = max(w for w, h in self.stock_list)
        self.max_stock_h = max(h for w, h in self.stock_list)
        self.max_product_types = len(self.product_list)

    def _setup_spaces(self):
        upper = np.full(shape=(self.max_stock_w, self.max_stock_h), fill_value=self.max_product_types + 2, dtype=int)
        lower = np.full(shape=(self.max_stock_w, self.max_stock_h), fill_value=-2, dtype=int)
        self.observation_space = spaces.Dict(
            {
                "stocks": spaces.Tuple([spaces.MultiDiscrete(upper, start=lower)] * self.num_stocks, seed=self.seed),
                "products": spaces.Sequence(
                    spaces.Dict(
                        {
                            "size": spaces.MultiDiscrete(np.array([self.max_stock_w, self.max_stock_h]), start=np.array([1, 1])),
                            "quantity": spaces.Discrete(MAX_QUANTITY, start=0),
                        }
                    ),
                    seed=self.seed,
                ),
            }
        )
        self.action_space = spaces.Dict(
            {
                "stock_idx": spaces.Discrete(self.num_stocks),
                "size": spaces.Box(low=np.array([1, 1]), high=np.array([self.max_stock_w, self.max_stock_h]), shape=(2,), dtype=int),
                "position": spaces.Box(low=np.array([0, 0]), high=np.array([self.max_stock_w - 1, self.max_stock_h - 1]), shape=(2,), dtype=int),
            }
        )

    def _reset_state(self):
        self.cutted_stocks = np.zeros((self.num_stocks,), dtype=int)
        self.used_stocks = np.zeros((self.num_stocks,), dtype=int)
        self._stocks = [np.full((self.max_stock_w, self.max_stock_h), -2, dtype=int) for _ in range(self.num_stocks)]
        for i, (w, h) in enumerate(self.stock_list):
            self._stocks[i][:w, :h] = -1
        self._stocks = tuple(self._stocks)
        self._products = [{"size": np.array([w, h]), "quantity": q} for w, h, q in self.product_list]
        self._products = tuple(self._products)

    def _get_obs(self):
        return {"stocks": self._stocks, "products": self._products}

    def _get_info(self):
        waste = sum(int(np.sum(stock == -1)) for stock in self._stocks if stock[0, 0] >= 0)
        total_use = sum(int(np.sum(stock > -2)) for stock in self._stocks if stock[0, 0] >= 0)
        total_area = sum(w * h for w, h in self.stock_list)
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
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            render_frame(self)
        return observation, info

    def step(self, action):
        """Thực hiện một bước trong môi trường dựa trên hành động."""
        if "stock_idx" not in action:
            raise ValueError("Action thiếu 'stock_idx'. Action phải có dạng {'stock_idx': int, 'size': array, 'position': array}")
        
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]
        width, height = size
        x, y = position

        product_idx = self._find_product(size)
        reward, terminated = 0, False

        if product_idx is not None and 0 <= stock_idx < self.num_stocks:
            stock = self._stocks[stock_idx]
            stock_w, stock_h = self.stock_list[stock_idx]
            if self._is_valid_action(stock, x, y, width, height, stock_w, stock_h, stock_idx):
                self._apply_action(stock, x, y, width, height, product_idx, stock_idx)  # Truyền stock_idx vào đây
                reward = calculate_reward(stock_w, stock_h, width, height, self.cutted_stocks[stock_idx], self.max_cuts_per_stock)

        terminated = all(p["quantity"] == 0 for p in self._products)
        if terminated:
            reward += 1 - self._get_info()["waste_ratio"]

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            render_frame(self)
            if terminated:
                save_gif(self)
        return observation, reward, terminated, False, info

    def first_fit_step(self):
        """Thực hiện một bước theo thuật toán First-Fit."""
        product_idx, size = self._get_next_product()
        if product_idx is None:
            return self._get_obs(), 0, True, False, self._get_info()

        for stock_idx, stock in enumerate(self._stocks):
            if self.cutted_stocks[stock_idx] < self.max_cuts_per_stock:
                stock_size = self.stock_list[stock_idx]
                position = find_first_fit_position(stock, size, stock_size)
                if position is not None:
                    action = {"stock_idx": stock_idx, "size": size, "position": np.array(position)}
                    return self.step(action)
        return self._get_obs(), 0, False, False, self._get_info()

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
        """Áp dụng hành động cắt vào stock, với stock_idx được truyền vào."""
        stock[x:x + width, y:y + height] = product_idx + 1
        self.cutted_stocks[stock_idx] += 1
        self.used_stocks[stock_idx] = 1
        self._products[product_idx]["quantity"] -= 1

    def _get_next_product(self):
        for i, product in enumerate(self._products):
            if product["quantity"] > 0:
                return i, product["size"]
        return None, None

    def render(self):
        if self.render_mode == "rgb_array":
            return render_frame(self, return_array=True)
        elif self.render_mode == "human":
            render_frame(self)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.font.quit()
            self.window = None
            self.clock = None