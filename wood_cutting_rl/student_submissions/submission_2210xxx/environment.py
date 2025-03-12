import numpy as np
import gym
from gym import spaces


class CuttingWoodEnv(gym.Env):
    def __init__(self, stock_size=(100, 50), demand=[(30, 20, 3), (20, 10, 4)]):
        """
        stock_size: (width, height) - Kích thước tấm gỗ lớn
        demand: [(w, h, qty), ...] - Danh sách các kích thước cần cắt
        """
        super(CuttingWoodEnv, self).__init__()

        self.stock_width, self.stock_height = stock_size
        self.demand = demand  # Danh sách các yêu cầu cắt [(w, h, qty)]
        self.remaining_stock = np.ones((self.stock_width, self.stock_height))  # Ma trận lưu trạng thái tấm gỗ
        
        # Action Space: Chọn một kích thước từ danh sách demand để cắt
        self.action_space = spaces.Discrete(len(demand))  

        # State Space: Vector mô tả tấm gỗ còn lại và số lượng cần cắt
        self.state_shape = (self.stock_width, self.stock_height, len(demand))  
        self.observation_space = spaces.Box(low=0, high=1, shape=self.state_shape, dtype=np.float32)

        self.reset()

    def reset(self):
        """ Reset lại trạng thái môi trường """
        self.remaining_stock = np.ones((self.stock_width, self.stock_height))  # Reset tấm gỗ
        self.current_demand = self.demand[:]  # Reset danh sách nhu cầu
        return self.get_state()

    def get_state(self):
        """ Trả về trạng thái môi trường dưới dạng ma trận """
        state = np.zeros(self.state_shape, dtype=np.float32)
        state[:, :, 0] = self.remaining_stock  # Ma trận trạng thái tấm gỗ
        for i, (w, h, qty) in enumerate(self.current_demand):
            state[0, 0, i] = qty  # Lưu số lượng miếng cần cắt
        return state

    def step(self, action):
        """ Thực hiện một hành động (cắt gỗ) """
        if action >= len(self.current_demand):
            return self.get_state(), -10, False, {}  # Hành động không hợp lệ
        
        w, h, qty = self.current_demand[action]

        # Tìm vị trí có thể cắt
        for x in range(self.stock_width - w + 1):
            for y in range(self.stock_height - h + 1):
                if np.all(self.remaining_stock[x:x+w, y:y+h] == 1):
                    self.remaining_stock[x:x+w, y:y+h] = 0  # Cắt xong, đánh dấu phần đã cắt
                    self.current_demand[action] = (w, h, qty - 1)  # Giảm số lượng yêu cầu
                    reward = 10 - (self.stock_width * self.stock_height - np.sum(self.remaining_stock)) * 0.1
                    done = all(qty <= 0 for _, _, qty in self.current_demand)
                    return self.get_state(), reward, done, {}

        return self.get_state(), -5, False, {}  # Không cắt được, bị phạt điểm

    def render(self):
        """ Hiển thị trạng thái tấm gỗ """
        print("Tấm gỗ còn lại:")
        print(self.remaining_stock)
