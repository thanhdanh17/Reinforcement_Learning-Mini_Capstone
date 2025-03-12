from environment import WoodCuttingEnv
from policy_2210xxx import Policy2210xxx
from utils import load_q_table



# Load Q-Table đã train trước đó
Q_table = load_q_table("student_submissions/submission_2210xxx/q_table.npy")

# Khởi tạo môi trường và policy
env = WoodCuttingEnv()
policy = Policy2210xxx()
policy.Q_table = Q_table  # Gán Q-Table đã load vào policy

def test_policy():
    for episode in range(5):
        state = env.reset()
        print(f"Initial wood: {state}")
        
        while True:
            action = policy.select_action(state)
            next_state, reward, done = env.step(action)
            print(f"Cutting {action}, Remaining: {next_state}, Reward: {reward}")
            
            if done:
                print("Finished cutting this wood.")
                break

if __name__ == "__main__":
    test_policy()
