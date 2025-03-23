# test.py
from env import CuttingStockEnv

# Ví dụ sử dụng
env = CuttingStockEnv(
    render_mode="human",
    stock_list=[(100, 100), (80, 80)],
    product_list=[(20, 20), (30, 30)]
)
obs, info = env.reset()
print(obs, info)
env.close()