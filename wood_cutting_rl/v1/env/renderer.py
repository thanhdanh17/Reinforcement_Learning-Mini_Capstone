# renderer.py
import pygame
from pygame.locals import QUIT
import matplotlib.pyplot as plt  # Import matplotlib để dùng plt
from matplotlib import colormaps  # Import colormaps
import matplotlib as mpl  # Import matplotlib dưới dạng mpl để dùng mpl.colors
import numpy as np
from PIL import Image
from .constants import WOOD_COLOR, GIF_PATHS

def render_frame(env, return_array=False):
    """Hiển thị trạng thái môi trường hoặc trả về mảng RGB."""
    window_size = _get_window_size(env)
    if env.window is None and env.render_mode == "human":
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("Flooring Cutting Stock Environment")
        env.window = pygame.display.set_mode(window_size)
    if env.clock is None and env.render_mode == "human":
        env.clock = pygame.time.Clock()

    canvas = pygame.Surface(window_size)
    canvas.fill((0, 0, 0))
    pix_square_size = 1

    cmap = colormaps.get_cmap("hsv")
    norms = mpl.colors.Normalize(vmin=0, vmax=env.max_product_types - 1)
    list_colors = [cmap(norms(i)) for i in range(env.max_product_types + 1)]
    list_colors[0] = [1, 1, 1, 1]

    for i, stock in enumerate(env._stocks):
        stock_w, stock_h = env.stock_list[i]
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()

        pygame.draw.rect(
            canvas,
            WOOD_COLOR,
            pygame.Rect(
                (i % (window_size[0] // (env.max_stock_w // 10)) * (env.max_stock_w // 10)) * pix_square_size,
                (i // (window_size[0] // (env.max_stock_w // 10)) * (env.max_stock_h // 10)) * pix_square_size,
                (stock_w // 10) * pix_square_size,
                (stock_h // 10) * pix_square_size,
            ),
        )

        for x in range(stock_w):
            for y in range(stock_h):
                if stock[x, y] > -1:
                    color = list_colors[stock[x, y]][:3]
                    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(
                            (i % (window_size[0] // (env.max_stock_w // 10)) * (env.max_stock_w // 10) + x // 10) * pix_square_size,
                            (i // (window_size[0] // (env.max_stock_w // 10)) * (env.max_stock_h // 10) + y // 10) * pix_square_size,
                            pix_square_size,
                            pix_square_size,
                        ),
                    )

    if env.render_mode == "human":
        env.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        env.clock.tick(env.metadata["render_fps"])
    if return_array:
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    frame_data = pygame.surfarray.array3d(canvas)
    frame_data = frame_data.transpose((1, 0, 2))
    env.frames.append(Image.fromarray(frame_data))

def save_gif(env):
    """Lưu GIF khi quá trình hoàn tất."""
    if env.frames and env.namepoli:
        for _ in range(100):
            env.frames.append(env.frames[-1])
        env.frames[0].save(env.namepoli[0], save_all=True, append_images=env.frames[1:], duration=100, loop=0)
        env.namepoli.pop(0)
        env.frames.clear()

def _get_window_size(env):
    width = int(np.ceil(np.sqrt(env.num_stocks)))
    height = int(np.ceil(env.num_stocks / width))
    return width * env.max_stock_w // 10, height * env.max_stock_h // 10