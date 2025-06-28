# training.py

import pygame
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Uma biblioteca de barras de progresso muito útil


from snake_gym_env import SnakeGymEnv
from agents import QLearningAgent, SARSAAgent

def train_agent(agent, env, episodes, epsilon_decay=0.995, min_epsilon=0.01):
    """
    Função de formação genérica, adequada para agentes Q-Learning e SARSA.
    """
    scores = []
    steps_list = []
    initial_epsilon = agent.epsilon

    # usar tqdm
    for episode in tqdm(range(episodes), desc=f"Training {type(agent).__name__}"):
        state, info = env.reset()
        
        # O SARSA necessita de escolher a primeira ação antes do início do ciclo
        if isinstance(agent, SARSAAgent):
            action = agent.choose_action(state)
        
        done = False
        while not done:
            # Q-Learning seleciona ações dentro de um ciclo
            if isinstance(agent, QLearningAgent):
                action = agent.choose_action(state)
            
            # Interagir com o ambiente
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Atualização com base no tipo de agente
            if isinstance(agent, QLearningAgent):
                agent.update(state, action, reward, next_state)
            else:  # SARSA
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                action = next_action # A chave para a SARSA: Faça da próxima ação a ação atual
            
            state = next_state
        
        # Registe a pontuação e o número de passos para esta ronda
        scores.append(info['score'])
        steps_list.append(info['steps'])
        
        # Decaimento Epsilon
        if agent.epsilon > min_epsilon:
            agent.epsilon *= epsilon_decay
    
    # Redefinir o epsilon após a formação para facilitar a avaliação subsequente
    agent.epsilon = initial_epsilon
    return scores, steps_list

def play_game_with_agent(agent, env, num_games=1):
    """
    使用训练好的智能体玩游戏并显示。
    """
    agent.training = False # Desligue o modo de exploração e utilize apenas
    
    for i in range(num_games):
        state, info = env.reset()
        done = False
        print(f"\n--- Game {i+1} ---")
        while not done:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(f"Final Score: {info['score']}, Steps: {info['steps']}")
    
    env.close()

def compare_algorithms(episodes=2000):
    """
    Formar, comparar e visualizar o desempenho do Q-Learning e do SARSA.
    """
    print("--- 1. Training Q-Learning Agent ---")
    env = SnakeGymEnv() 
    q_agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.5)
    q_scores, q_steps = train_agent(q_agent, env, episodes)
    q_agent.save('q_learning_snake.pkl')
    print("Q-Learning agent trained and saved.")

    print("\n--- 2. Training SARSA Agent ---")
    
    sarsa_agent = SARSAAgent(alpha=0.1, gamma=0.9, epsilon=0.5)
    s_scores, s_steps = train_agent(sarsa_agent, env, episodes)
    sarsa_agent.save('sarsa_snake.pkl')
    print("SARSA agent trained and saved.")
    
    env.close() 
    
    print("\n--- 3. Plotting Results ---")
    plt.figure(figsize=(12, 5))
    
    # Desenhe um gráfico de comparação de pontuações
    plt.subplot(1, 2, 1)
    
    # Calcular a média móvel para suavizar a curva
    window = 100
    q_avg = [np.mean(q_scores[max(0, i-window):i+1]) for i in range(len(q_scores))]
    s_avg = [np.mean(s_scores[max(0, i-window):i+1]) for i in range(len(s_scores))]
    
    plt.plot(q_avg, label='Q-Learning (Avg)', linewidth=2)
    plt.plot(s_avg, label='SARSA (Avg)', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score Comparison: Q-Learning vs SARSA')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    # Desenhe um gráfico de comparação de etapas
    plt.subplot(1, 2, 2)
    q_steps_avg = [np.mean(q_steps[max(0, i-window):i+1]) for i in range(len(q_steps))]
    s_steps_avg = [np.mean(s_steps[max(0, i-window):i+1]) for i in range(len(s_steps))]
    
    plt.plot(q_steps_avg, label='Q-Learning (Avg)', linewidth=2)
    plt.plot(s_steps_avg, label='SARSA (Avg)', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Steps until Game Over')
    plt.title('Survival Comparison')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("Comparison plot saved as 'comparison_results.png'")
    plt.show()
    
    
    print("\n--- Final Statistics ---")
    print(f"Q-Learning - Avg Score (last 100 episodes): {np.mean(q_scores[-100:]):.2f}")
    print(f"Q-Learning - Best Score: {max(q_scores)}")
    print(f"SARSA - Avg Score (last 100 episodes): {np.mean(s_scores[-100:]):.2f}")
    print(f"SARSA - Best Score: {max(s_scores)}")

def manual_play():
    """
    允许用户手动玩 Snake 游戏。
    """
    # Crie um ambiente utilizando o modo 'humano' para apresentar a janela
    env = SnakeGymEnv(render_mode='human')
    state, info = env.reset()
    
    print("\n--- Manual Play Mode ---")
    print("Controls: Arrow Keys to move")
    print("Press 'R' to restart, 'Q' to quit.")
    
    # Mapeando os botões do Pygame para as nossas ações
    key_to_action = {
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
    }
    
    #Ação padrão para evitar que o jogo congele quando nenhuma tecla é pressionada
    last_action = 3 

    running = True
    while running:
        done = False
        while not done:
            action_taken = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in key_to_action:
                        last_action = key_to_action[event.key]
                    elif event.key == pygame.K_q:
                        done = True
                        running = False
            
            if not running: break
            
            # Execute ações e obtenha resultados
            state, reward, terminated, truncated, info = env.step(last_action)
            done = terminated or truncated

        if running:
            print(f"Game Over! Final Score: {info['score']}")
            # Aguarda que o utilizador reinicie ou efetue logout
            wait_for_input = True
            while wait_for_input and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_input = False
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            state, info = env.reset()
                            wait_for_input = False
                        elif event.key == pygame.K_q:
                            wait_for_input = False
                            running = False
    
    env.close()
    print("Exited manual play mode.")