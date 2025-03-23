# main.py
from env.environment import FlooringCuttingStockEnv
from env.policy import first_fit, first_fit_decreasing, best_fit

def run_algorithm(env, algorithm, name):
    """Chạy một thuật toán và trả về thông tin kết quả."""
    observation, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        observation, reward, terminated, truncated, info = algorithm(env)
        done = terminated or truncated
        total_reward += reward
        env.render()

    final_info = env._get_info()
    print(f"\n{name} Results:")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Used Ratio: {final_info['used_ratio']:.4f}")
    print(f"Waste Ratio: {final_info['waste_ratio']:.4f}")
    print(f"Total Cuts: {final_info['total_cuts']}")
    print(f"Stocks Used: {final_info['stocks_used']}")
    return final_info

def main():
    env = FlooringCuttingStockEnv(
        stock_file="data/stocks.csv",
        product_file="data/products.csv",
        render_mode="human"
    )

    algorithms = [
        (first_fit, "First-Fit"),
        (first_fit_decreasing, "First-Fit Decreasing"),
        (best_fit, "Best-Fit")
    ]
    results = {}

    for algo, name in algorithms:
        print(f"\nRunning {name}...")
        results[name] = run_algorithm(env, algo, name)

    # Đánh giá thuật toán tối ưu
    print("\nComparison of Algorithms:")
    for name, info in results.items():
        print(f"{name}: Used Ratio = {info['used_ratio']:.4f}, Waste Ratio = {info['waste_ratio']:.4f}")

    best_algo = max(results.items(), key=lambda x: x[1]['used_ratio'] - x[1]['waste_ratio'])
    print(f"\nBest Algorithm: {best_algo[0]} with Used Ratio = {best_algo[1]['used_ratio']:.4f} and Waste Ratio = {best_algo[1]['waste_ratio']:.4f}")

    env.close()

if __name__ == "__main__":
    main()