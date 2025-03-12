# constants.py

METADATA = {"render_modes": ["human", "rgb_array"], "render_fps": 5}
DEFAULT_STOCKS = [(2440, 1220)]  # Tấm gỗ lớn mặc định: 2440x1220 mm
DEFAULT_PRODUCTS = [(600, 200, 10)]  # Tấm cần cắt mặc định: 600x200 mm, 10 cái
MAX_QUANTITY = 1000  # Số lượng tối đa tạm thời cho không gian quan sát
GIF_PATHS = ["demo/flooring_combine.gif", "demo/flooring_bestfit.gif", "demo/flooring_firstfit.gif"]
WOOD_COLOR = (139, 69, 19)  # Màu nâu gỗ