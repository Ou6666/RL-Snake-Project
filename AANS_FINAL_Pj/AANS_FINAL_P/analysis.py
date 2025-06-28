
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from collections import defaultdict
from scipy import stats
import os
from tqdm import tqdm


from snake_gym_env import SnakeGymEnv
from agents import QLearningAgent, SARSAAgent

def analyze_q_table(q_table, agent_name):
    """
    分析给定的 Q-table 并打印统计信息。
    """
    if not q_table:
        print(f"Q-table for {agent_name} is empty.")
        return

    print(f"\n=== Q-table Analysis - {agent_name} ===")
    print(f"Number of visited states: {len(set(s for s, a in q_table.keys()))}")
    print(f"Number of state-action pairs: {len(q_table)}")

    q_values = list(q_table.values())
    print(f"Q-value Mean: {np.mean(q_values):.3f}")
    print(f"Q-value Max: {max(q_values):.3f}")
    print(f"Q-value Min: {min(q_values):.3f}")

def plot_q_value_distribution(q_table_q, q_table_s):
    """
    
    """
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(list(q_table_q.values()), label='Q-Learning', fill=True)
    sns.kdeplot(list(q_table_s.values()), label='SARSA', fill=True)
    
    plt.xlabel('Q-value')
    plt.ylabel('Density')
    plt.title('Distribution of Q-values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis_q_value_distribution.png', dpi=300)
    plt.show()

def analyze_state_patterns(q_table_q, q_table_s):
    """
    。
    """
    states_q = set(s for s, a in q_table_q.keys())
    states_s = set(s for s, a in q_table_s.keys())
    
    print(f"\nUnique states Q-Learning: {len(states_q)}")
    print(f"Unique states SARSA: {len(states_s)}")
    print(f"Common states: {len(states_q.intersection(states_s))}")
    
    danger_patterns_q = defaultdict(int)
    for state in states_q:
        danger_count = sum(state[2:6])  
        danger_patterns_q[danger_count] += 1
        
    danger_patterns_s = defaultdict(int)
    for state in states_s:
        danger_count = sum(state[2:6])
        danger_patterns_s[danger_count] += 1

    plt.figure(figsize=(10, 6))
    dangers = list(range(5))
    q_counts = [danger_patterns_q[d] for d in dangers]
    s_counts = [danger_patterns_s[d] for d in dangers]
    
    x = np.arange(len(dangers))
    width = 0.35
    
    plt.bar(x - width/2, q_counts, width, label='Q-Learning', alpha=0.7)
    plt.bar(x + width/2, s_counts, width, label='SARSA', alpha=0.7)
    
    plt.xlabel('Number of Adjacent Dangers')
    plt.ylabel('Number of Visited States')
    plt.title('Distribution of States by Danger Level')
    plt.xticks(x, dangers)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig('analysis_danger_patterns.png', dpi=300)
    plt.show()

def generate_performance_report(n_games=100):
    """
    。
    """
    if not (os.path.exists('q_learning_snake.pkl') and os.path.exists('sarsa_snake.pkl')):
        print("[Error] Model files not found. Please train first using run_project.py.")
        return

    q_agent = QLearningAgent()
    q_agent.load('q_learning_snake.pkl')
    q_agent.training = False

    sarsa_agent = SARSAAgent()
    sarsa_agent.load('sarsa_snake.pkl')
    sarsa_agent.training = False

    env = SnakeGymEnv()
    
    q_scores, s_scores = [], []

    print(f"\n--- Evaluating Q-Learning Agent ({n_games} games) ---")
    for _ in tqdm(range(n_games), desc="Q-Learning Eval"):
        state, info = env.reset()
        done = False
        while not done:
            action = q_agent.choose_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        q_scores.append(info['score'])

    print(f"\n--- Evaluating SARSA Agent ({n_games} games) ---")
    for _ in tqdm(range(n_games), desc="SARSA Eval"):
        state, info = env.reset()
        done = False
        while not done:
            action = sarsa_agent.choose_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        s_scores.append(info['score'])
    
    env.close()

    # 统计分析
    print("\n\n=== Final Performance Report ===")
    print(f"\nQ-Learning ({n_games} games):")
    print(f"  Mean Score: {np.mean(q_scores):.2f} ± {np.std(q_scores):.2f}")
    print(f"  Median: {np.median(q_scores):.2f}, Min: {min(q_scores)}, Max: {max(q_scores)}")
    
    print(f"\nSARSA ({n_games} games):")
    print(f"  Mean Score: {np.mean(s_scores):.2f} ± {np.std(s_scores):.2f}")
    print(f"  Median: {np.median(s_scores):.2f}, Min: {min(s_scores)}, Max: {max(s_scores)}")

    t_stat, p_value = stats.ttest_ind(q_scores, s_scores)
    print("\n--- Statistical Test (t-test) ---")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Conclusion: The difference is statistically significant.")
    else:
        print("  Conclusion: The difference is not statistically significant.")

   
    plt.figure(figsize=(10, 6))
    sns.histplot(q_scores, kde=True, label='Q-Learning', color='blue', bins=20)
    sns.histplot(s_scores, kde=True, label='SARSA', color='orange', bins=20)
    plt.title(f'Final Performance Distribution ({n_games} Games)')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis_final_performance.png', dpi=300)
    plt.show()

def run_analysis():
    """
    主函数，运行所有分析任务。
    """
    if not (os.path.exists('q_learning_snake.pkl') and os.path.exists('sarsa_snake.pkl')):
        print("[Error] Model files not found. Please train first using run_project.py.")
        return

    with open('q_learning_snake.pkl', 'rb') as f:
        q_table_q = pickle.load(f)
    with open('sarsa_snake.pkl', 'rb') as f:
        q_table_s = pickle.load(f)

    print("--- 1. Analyzing Q-tables ---")
    analyze_q_table(q_table_q, 'Q-Learning')
    analyze_q_table(q_table_s, 'SARSA')

    print("\n--- 2. Plotting Q-value Distributions ---")
    plot_q_value_distribution(q_table_q, q_table_s)

    print("\n--- 3. Analyzing State Patterns ---")
    analyze_state_patterns(q_table_q, q_table_s)
    
    print("\n--- 4. Generating Final Performance Report ---")
    try:
        n_games = int(input("Enter number of games to evaluate per agent (e.g., 100): "))
        generate_performance_report(n_games=n_games)
    except ValueError:
        print("Invalid input. Using default of 100 games.")
        generate_performance_report(n_games=100)

if __name__ == '__main__':
    
    try:
        import seaborn
    except ImportError:
        print("Seaborn not found. For better-looking plots, run: pip install seaborn")
    
    run_analysis()