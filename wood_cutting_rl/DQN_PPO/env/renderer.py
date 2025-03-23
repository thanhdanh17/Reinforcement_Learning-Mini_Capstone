# renderer.py
import pygame
import numpy as np
import os

def render_frame(env, return_array=False):
    # Kiểm tra và khởi tạo lại Pygame nếu cần
    if not hasattr(env, 'window') or env.window is None:
        try:
            pygame.init()
            env.window = pygame.display.set_mode((1000, 600))
            env.clock = pygame.time.Clock()
        except pygame.error as e:
            print(f"Error initializing Pygame: {e}")
            return None

    screen = env.window
    if screen is None:
        print("Error: Pygame window is None")
        return None

    screen.fill((255, 255, 255))  # Màu nền trắng

    # Tính toán tỷ lệ để vẽ kích thước thu nhỏ lên màn hình
    scale_x = 1000 / (env.max_stock_w * env.num_stocks)
    scale_y = 600 / env.max_stock_h
    cell_size = min(scale_x, scale_y)

    # Vẽ các tấm gỗ
    for idx, stock in enumerate(env._stocks):
        for x in range(env.max_stock_w):
            for y in range(env.max_stock_h):
                if stock[x, y] > -1:
                    color = (0, 128, 255) if stock[x, y] > 0 else (200, 200, 200)
                    pygame.draw.rect(
                        screen,
                        color,
                        (
                            int(idx * env.max_stock_w * cell_size + x * cell_size),
                            int(y * cell_size),
                            int(cell_size),
                            int(cell_size)
                        )
                    )

    # Hiển thị thông tin sản phẩm
    if not hasattr(env, 'font') or env.font is None:
        env.font = pygame.font.SysFont("Arial", 20)
    for i, product in enumerate(env._products):
        text = f"Product {i+1}: {product['quantity']} left"
        text_surface = env.font.render(text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10 + i * 30))

    pygame.display.flip()
    env.clock.tick(60)

    if return_array:
        return np.transpose(pygame.surfarray.array3d(screen), axes=(1, 0, 2))
    env.frames.append(np.transpose(pygame.surfarray.array3d(screen), axes=(1, 0, 2)))

def save_gif(env, algorithm_name):
    import imageio
    output_dir = "demo"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/flooring_{algorithm_name}.gif"
    imageio.mimsave(filename, env.frames, fps=10)
    print(f"Saved GIF to {filename}")