import gymnasium as gym
import matplotlib as mpl
import numpy as np
import pygame
from pygame.locals import QUIT
from gymnasium import spaces
from matplotlib import colormaps
import pygame.locals
from PIL import Image

class FlooringCuttingStockEnv:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        render_mode=None,
        stock_size=(2440, 1220),  # Kích thước tấm gỗ lớn tiêu chuẩn (mm): 2440x1220 (8ft x 4ft)
        min_product_size=(300, 100),  # Kích thước tối thiểu của tấm sàn (mm)
        max_product_size=(1200, 200),  # Kích thước tối đa của tấm sàn (mm)
        num_stocks=10,  # Số tấm gỗ lớn
        max_product_types=5,  # Số loại tấm sàn khác nhau
        max_quantity_per_type=50,  # Số lượng tối đa mỗi loại tấm sàn
        max_cuts_per_stock=20,  # Số lần cắt tối đa trên mỗi tấm
        seed=42,
    ):
        self.frames = []
        self.namepoli = ["demo/flooring_combine.gif", "demo/flooring_bestfit.gif", "demo/flooring_firstfit.gif"]
        self.seed = seed
        self.stock_size = stock_size  # Kích thước cố định của tấm gỗ lớn
        self.min_product_size = min_product_size
        self.max_product_size = max_product_size
        self.num_stocks = num_stocks
        self.max_product_types = max_product_types
        self.max_quantity_per_type = max_quantity_per_type
        self.max_cuts_per_stock = max_cuts_per_stock  # Giới hạn số lần cắt
        self.cutted_stocks = np.zeros((num_stocks,), dtype=int)  # Theo dõi số lần cắt trên mỗi tấm
        self.used_stocks = np.zeros((num_stocks,), dtype=int)  # Theo dõi tấm đã sử dụng

        # Không gian quan sát
        stock_w, stock_h = stock_size
        upper = np.full(shape=(stock_w, stock_h), fill_value=max_product_types + 2, dtype=int)
        lower = np.full(shape=(stock_w, stock_h), fill_value=-2, dtype=int)
        self.observation_space = spaces.Dict(
            {
                "stocks": spaces.Tuple(
                    [spaces.MultiDiscrete(upper, start=lower)] * num_stocks, seed=seed
                ),
                "products": spaces.Sequence(
                    spaces.Dict(
                        {
                            "size": spaces.MultiDiscrete(
                                np.array([max_product_size[0], max_product_size[1]]),
                                start=np.array([min_product_size[0], min_product_size[1]]),
                            ),
                            "quantity": spaces.Discrete(max_quantity_per_type + 1, start=0),
                        }
                    ),
                    seed=seed,
                ),
            }
        )

        # Không gian hành động
        self.action_space = spaces.Dict(
            {
                "stock_idx": spaces.Discrete(num_stocks),
                "size": spaces.Box(
                    low=np.array(min_product_size),
                    high=np.array(max_product_size),
                    shape=(2,),
                    dtype=int,
                ),
                "position": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([stock_w - 1, stock_h - 1]),
                    shape=(2,),
                    dtype=int,
                ),
            }
        )

        self._stocks = []
        self._products = []
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"stocks": self._stocks, "products": self._products}

    def _get_info(self):
        waste = 0
        total_use = 0
        total_cuts = np.sum(self.cutted_stocks)
        for stock in self._stocks:
            if stock[0, 0] >= 0:  # Nếu tấm đã được sử dụng
                waste += int(np.sum(stock == -1))  # Diện tích lãng phí
                total_use += int(np.sum(stock > -2))  # Diện tích sử dụng
        total_area = self.stock_size[0] * self.stock_size[1] * self.num_stocks
        return {
            "used_ratio": total_use / total_area,
            "waste_ratio": waste / (total_use + 1e-7),
            "total_cuts": total_cuts,
            "stocks_used": np.sum(self.used_stocks),
        }

    def reset(self, seed=None, options=None):
        self.frames = []
        np.random.seed(seed)
        self.cutted_stocks = np.zeros((self.num_stocks,), dtype=int)
        self.used_stocks = np.zeros((self.num_stocks,), dtype=int)
        self._stocks = []

        # Khởi tạo các tấm gỗ lớn với kích thước cố định
        stock_w, stock_h = self.stock_size
        for _ in range(self.num_stocks):
            stock = np.full(shape=(stock_w, stock_h), fill_value=-2, dtype=int)
            stock[:stock_w, :stock_h] = -1  # Vùng trống trong tấm
            self._stocks.append(stock)
        self._stocks = tuple(self._stocks)

        # Khởi tạo các tấm sàn cần cắt
        self._products = []
        num_types = np.random.randint(low=1, high=self.max_product_types + 1)
        for _ in range(num_types):
            width = np.random.randint(self.min_product_size[0], self.max_product_size[0] + 1)
            height = np.random.randint(self.min_product_size[1], self.max_product_size[1] + 1)
            quantity = np.random.randint(1, self.max_quantity_per_type + 1)
            self._products.append({"size": np.array([width, height]), "quantity": quantity})
        self._products = tuple(self._products)

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]
        width, height = size
        x, y = position

        # Tìm sản phẩm phù hợp trong danh sách
        product_idx = None
        for i, product in enumerate(self._products):
            if np.array_equal(product["size"], size) and product["quantity"] > 0:
                product_idx = i
                break

        reward = 0
        terminated = False
        if product_idx is not None and 0 <= stock_idx < self.num_stocks:
            stock = self._stocks[stock_idx]
            stock_w, stock_h = self.stock_size

            # Kiểm tra xem sản phẩm có vừa tấm gỗ không
            if (
                x >= 0
                and y >= 0
                and x + width <= stock_w
                and y + height <= stock_h
                and self.cutted_stocks[stock_idx] < self.max_cuts_per_stock
            ):
                # Kiểm tra vùng trống
                if np.all(stock[x:x + width, y:y + height] == -1):
                    # Đặt sản phẩm vào tấm
                    stock[x:x + width, y:y + height] = product_idx + 1
                    self.cutted_stocks[stock_idx] += 1  # Tăng số lần cắt
                    self.used_stocks[stock_idx] = 1  # Đánh dấu tấm đã sử dụng
                    self._products[product_idx]["quantity"] -= 1

                    # Tính phần thưởng dựa trên diện tích sử dụng
                    used_area = width * height
                    total_area = stock_w * stock_h
                    reward = used_area / total_area  # Phần thưởng tỷ lệ với diện tích sử dụng
                    if self.cutted_stocks[stock_idx] > self.max_cuts_per_stock:
                        reward -= 0.1  # Phạt nếu vượt quá số lần cắt tối đa

        # Kiểm tra điều kiện kết thúc
        terminated = all(product["quantity"] == 0 for product in self._products)
        if terminated:
            info = self._get_info()
            reward += 1 - info["waste_ratio"]  # Thêm phần thưởng nếu lãng phí thấp

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        if terminated and self.render_mode == "human":
            for _ in range(100): 
                self.frames.append(self.frames[-1])
            self.frames[0].save(self.namepoli[0], save_all=True, append_images=self.frames[1:], duration=100, loop=0)
            self.namepoli.pop(0)
            self.frames.clear()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _get_window_size(self):
        width = int(np.ceil(np.sqrt(self.num_stocks)))
        height = int(np.ceil(self.num_stocks / width))
        return width * self.stock_size[0] // 10, height * self.stock_size[1] // 10  # Giảm tỷ lệ để hiển thị

    def _render_frame(self):
        window_size = self._get_window_size()
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Flooring Cutting Stock Environment")
            self.window = pygame.display.set_mode(window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(window_size)
        canvas.fill((0, 0, 0))
        pix_square_size = 1  # Kích thước pixel

        cmap = colormaps.get_cmap("hsv")
        norms = mpl.colors.Normalize(vmin=0, vmax=self.max_product_types - 1)
        list_colors = [cmap(norms(i)) for i in range(self.max_product_types + 1)]
        list_colors[0] = [1, 1, 1, 1]  # Màu viền

        for i, stock in enumerate(self._stocks):
            stock_w, stock_h = self.stock_size
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()

            # Vẽ tấm gỗ lớn
            pygame.draw.rect(
                canvas,
                (139, 69, 19),  # Màu gỗ nâu
                pygame.Rect(
                    (i % (window_size[0] // (stock_w // 10)) * (stock_w // 10)) * pix_square_size,
                    (i // (window_size[0] // (stock_w // 10)) * (stock_h // 10)) * pix_square_size,
                    (stock_w // 10) * pix_square_size,
                    (stock_h // 10) * pix_square_size,
                ),
            )

            # Vẽ các tấm sàn đã cắt
            for x in range(stock.shape[0]):
                for y in range(stock.shape[1]):
                    if stock[x, y] > -1:
                        color = list_colors[stock[x, y]][:3]
                        color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                        pygame.draw.rect(
                            canvas,
                            color,
                            pygame.Rect(
                                (i % (window_size[0] // (stock_w // 10)) * (stock_w // 10) + x // 10) * pix_square_size,
                                (i // (window_size[0] // (stock_w // 10)) * (stock_h // 10) + y // 10) * pix_square_size,
                                pix_square_size,
                                pix_square_size,
                            ),
                        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

        frame_data = pygame.surfarray.array3d(canvas)
        frame_data = frame_data.transpose((1, 0, 2))
        self.frames.append(Image.fromarray(frame_data))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.font.quit()