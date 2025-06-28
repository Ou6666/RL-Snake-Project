# agents.py

import random
import pickle
import os

class QLearningAgent:
    """Agente Q-Learning para jogar Snake"""
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha      # Taxa de aprendizagem
        self.gamma = gamma      # Fator de desconto
        self.epsilon = epsilon  # Taxa de exploração
        self.training = True
        
    def get_q_value(self, state, action):
        """Retorna valor Q para um par estado-ação"""
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state):
        """Escolhe ação usando política ε-greedy"""
        if self.training and random.random() < self.epsilon:
            return random.randint(0, 3) # Assumes 4 actions [0, 1, 2, 3]
        
        q_values = [self.get_q_value(state, a) for a in range(4)]
        max_q = max(q_values)
        
        best_actions = [a for a in range(4) if q_values[a] == max_q]
        return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state):
        """Atualiza valor Q usando a equação Q-Learning"""
        current_q = self.get_q_value(state, action)
        
        next_q_values = [self.get_q_value(next_state, a) for a in range(4)]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def save(self, filename="q_learning.pkl"):
        """Salva Q-table em arquivo"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filename="q_learning.pkl"):
        """Carrega Q-table de arquivo"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
        else:
            print(f"No Q-table found at {filename}, starting fresh.")


class SARSAAgent:
    """Agente SARSA para jogar Snake"""
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = True
        
    def get_q_value(self, state, action):
        """Retorna valor Q para um par estado-ação"""
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state):
        """Escolhe ação usando política ε-greedy"""
        if self.training and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        q_values = [self.get_q_value(state, a) for a in range(4)]
        max_q = max(q_values)
        best_actions = [a for a in range(4) if q_values[a] == max_q]
        return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, next_action):
        """Atualiza valor Q usando a equação SARSA"""
        current_q = self.get_q_value(state, action)
        next_q = self.get_q_value(next_state, next_action)
        
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def save(self, filename="sarsa.pkl"):
        """Salva Q-table em arquivo"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filename="sarsa.pkl"):
        """Carrega Q-table de arquivo"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"SARSA Q-table loaded from {filename}")
        else:
            print(f"No SARSA Q-table found at {filename}, starting fresh.")