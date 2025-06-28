# run_project.py

# 1. 修改导入语句
from training import compare_algorithms, play_game_with_agent, manual_play # <-- 添加 manual_play
from snake_gym_env import SnakeGymEnv
from agents import QLearningAgent, SARSAAgent
import os

def main_menu():
    """
    显示主菜单并处理用户输入。
    """
    while True:
        # 2. 修改菜单显示
        print("\n=============================================")
        print("   ISCTE Final Project: RL Snake Game")
        print("=============================================")
        print("1. Train & Compare Q-Learning vs. SARSA")
        print("2. Play game with trained Q-Learning agent")
        print("3. Play game with trained SARSA agent")
        print("4. Play game manually") # <-- 新增选项
        print("5. Run full analysis on trained models") # <-- 我建议把分析也加进来
        print("6. Exit") # <-- 退出选项变为 6
        print("---------------------------------------------")

        choice = input("Enter your choice (1-6): ") # <-- 修改范围

        if choice == '1':
            # ... (这部分代码不变)
            try:
                episodes = int(input("Enter number of episodes to train (e.g., 2000): "))
                compare_algorithms(episodes=episodes)
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == '2':
            # ... (这部分代码不变)
            if not os.path.exists('q_learning_snake.pkl'):
                print("\n[Error] Q-Learning model not found. Please train first (Option 1).")
                continue
            
            print("\n--- Playing with Q-Learning Agent ---")
            env = SnakeGymEnv(render_mode='human')
            agent = QLearningAgent()
            agent.load('q_learning_snake.pkl')
            play_game_with_agent(agent, env)

        elif choice == '3':
            # ... (这部分代码不变)
            if not os.path.exists('sarsa_snake.pkl'):
                print("\n[Error] SARSA model not found. Please train first (Option 1).")
                continue

            print("\n--- Playing with SARSA Agent ---")
            env = SnakeGymEnv(render_mode='human')
            agent = SARSAAgent()
            agent.load('sarsa_snake.pkl')
            play_game_with_agent(agent, env)

        # 3. 添加新的 elif 块
        elif choice == '4':
            manual_play()

        elif choice == '5':
            # 导入并运行分析脚本
            try:
                from analysis import run_analysis
                run_analysis()
            except ImportError:
                print("[Error] analysis.py not found or contains an error.")
        
        elif choice == '6': # <-- 修改退出选项
            print("Exiting project. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    # ... (这部分代码不变)
    try:
        import gymnasium
        import pygame
        import numpy
        import matplotlib
        import tqdm
        import seaborn # 添加 seaborn 检查
    except ImportError as e:
        print(f"Error: Missing required library - {e.name}")
        print(f"Please install it using: pip install {e.name}")
    else:
        main_menu()