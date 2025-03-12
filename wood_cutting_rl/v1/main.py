# main.py

from env.environment import FlooringCuttingStockEnv

def main():
    env = FlooringCuttingStockEnv(
        stock_file="data/stocks.csv",
        product_file="data/products.csv",
        render_mode="human"
    )
    observation, info = env.reset()
    done = False

    while not done:
        observation, reward, terminated, truncated, info = env.first_fit_step()
        done = terminated or truncated
        env.render()
        print(f"Reward: {reward}, Info: {info}")

    env.close()

if __name__ == "__main__":
    main()