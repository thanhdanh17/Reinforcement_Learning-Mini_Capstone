import pygame
import numpy as np
import time
from environment import WoodCuttingEnv
from policy_2210xxx import Policy2210xxx

# Cấu hình Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT, GRID_SIZE = 800, 600, 50
WHITE, BLACK, GRAY, ORANGE, RED = (255, 255, 255), (0, 0, 0), (150, 150, 150), (255, 100, 0), (200, 50, 50)

# Khởi tạo màn hình
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Wood Cutting Visualization")

# Khởi tạo môi trường và policy
env = WoodCuttingEnv()
policy = Policy2210xxx()

# Lưu trạng thái gỗ còn lại
wood_grid = np.ones((SCREEN_WIDTH // GRID_SIZE, SCREEN_HEIGHT // GRID_SIZE), dtype=int)

def draw_grid():
    """Vẽ lưới đại diện cho ván gỗ"""
    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            color = GRAY if wood_grid[x // GRID_SIZE, y // GRID_SIZE] == 1 else BLACK
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

def apply_cut(action):
    """Áp dụng một lần cắt vào lưới"""
    cut_x, cut_y, cut_length, cut_width = action
    for i in range(cut_length):
        for j in range(cut_width):
            if (cut_x + i) < wood_grid.shape[0] and (cut_y + j) < wood_grid.shape[1]:
                wood_grid[cut_x + i, cut_y + j] = 0  # Cắt gỗ (màu đen)

# Chạy vòng lặp hiển thị
running, state = True, env.reset()
cut_index = 0

while running:
    screen.fill(WHITE)
    draw_grid()

    # Áp dụng từng lần cắt với hiệu ứng động
    if cut_index < 10:
        action = policy.select_action(state)
        next_state, _, done = env.step(action)
        apply_cut(action)
        cut_index += 1
        time.sleep(0.5)
        state = next_state
        if done:
            print("Finished cutting.")

    # Xử lý sự kiện
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()
