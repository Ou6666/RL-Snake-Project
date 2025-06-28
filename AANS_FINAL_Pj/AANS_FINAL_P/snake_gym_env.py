# snake_gym_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
import pygame

# ==============================================================================
#  Definição de constantes: Define todos os parâmetros fixos do mundo do jogo
# ==============================================================================
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 480
GRID_SIZE = 20
SCOPE_X = (0, SCREEN_WIDTH // GRID_SIZE - 1)
SCOPE_Y = (2, SCREEN_HEIGHT // GRID_SIZE - 1) # Deixar espaço na parte superior para a exibição da pontuação
FOOD_STYLES = [(10, (255, 100, 100)), (20, (100, 255, 100)), (30, (100, 100, 255))]
BG_COLOR = (40, 40, 60)
GRID_COLOR = (0, 0, 0)
SNAKE_COLOR = (200, 200, 200)
TEXT_COLOR = (255, 255, 255)
GAMEOVER_COLOR = (200, 30, 30)

# Vetores de direção, usados para calcular as coordenadas
UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)
# Mapeia as ações do agente (inteiros) para direções físicas
ACTIONS_MAP = {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}
# ==============================================================================


class SnakeGymEnv(gym.Env):
    """
    Ambiente personalizado do jogo da cobra.
    Esta classe herda de gymnasium.Env, tornando-a um ambiente padrão de aprendizagem por reforço.
    Esta é a pedra angular para o sucesso do projeto, pois desacopla a lógica do jogo do algoritmo do agente.
    """
    # Metadados do Gymnasium, que declaram os modos de renderização suportados e a taxa de fotogramas (frames) padrão
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, render_mode=None):
        """
        Inicializa o ambiente. Esta função é chamada apenas uma vez ao criar a instância do ambiente.
        """
        super().__init__()

        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        
        # 1. Definir o espaço de ações (Action Space): diz ao agente o que ele pode fazer
        #    Discrete(4) indica que existem 4 ações discretas, numeradas de 0, 1, 2, 3.
        self.action_space = spaces.Discrete(4)

        # 2. Definir o espaço de observação (Observation Space): diz ao agente o que ele vai ver
        #    Este é o ponto central de inovação do projeto, que utiliza engenharia de características (feature engineering) para criar uma representação de estado compacta e eficiente.
        self.observation_space = spaces.Tuple((
            spaces.Discrete(3, start=-1),  # Característica 1: Coordenada X relativa da comida {-1:esquerda, 0:mesma coluna, 1:direita}
            spaces.Discrete(3, start=-1),  # Característica 2: Coordenada Y relativa da comida {-1:cima, 0:mesma linha, 1:baixo}
            spaces.Discrete(2),            # Característica 3: Perigo em cima {0:Não, 1:Sim}
            spaces.Discrete(2),            # Característica 4: Perigo em baixo {0:Não, 1:Sim}
            spaces.Discrete(2),            # Característica 5: Perigo à esquerda {0:Não, 1:Sim}
            spaces.Discrete(2),            # Característica 6: Perigo à direita {0:Não, 1:Sim}
            spaces.Discrete(2),            # Característica 7: Atualmente a mover-se para cima {0:Não, 1:Sim}
            spaces.Discrete(2),            # Característica 8: Atualmente a mover-se para baixo {0:Não, 1:Sim}
            spaces.Discrete(2),            # Característica 9: Atualmente a mover-se para a esquerda {0:Não, 1:Sim}
            spaces.Discrete(2)             # Característica 10: Atualmente a mover-se para a direita {0:Não, 1:Sim}
        ))

        # Configurações relacionadas com a renderização
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

    def _get_obs(self):
        """
        Uma função auxiliar interna para calcular e devolver a observação atual do ambiente (ou seja, o tuplo de estado com 10 características).
        """
        head = self.snake[0]
        
        # Calcular a direção relativa da comida
        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]
        food_dir_x = 0 if food_dx == 0 else (1 if food_dx > 0 else -1)
        food_dir_y = 0 if food_dy == 0 else (1 if food_dy > 0 else -1)
        
        # Verificar perigos ao redor
        danger_up = 1 if self._is_danger((head[0], head[1] - 1)) else 0
        danger_down = 1 if self._is_danger((head[0], head[1] + 1)) else 0
        danger_left = 1 if self._is_danger((head[0] - 1, head[1])) else 0
        danger_right = 1 if self._is_danger((head[0] + 1, head[1])) else 0
        
        # Obter a direção de movimento atual
        dir_up = 1 if self.direction == UP else 0
        dir_down = 1 if self.direction == DOWN else 0
        dir_left = 1 if self.direction == LEFT else 0
        dir_right = 1 if self.direction == RIGHT else 0
        
        # Combinar as 10 características num tuplo e devolvê-lo
        return (food_dir_x, food_dir_y, danger_up, danger_down, danger_left, danger_right,
                dir_up, dir_down, dir_left, dir_right)

    def _get_info(self):
        """Devolve um dicionário com informações adicionais, para avaliação e depuração."""
        return {"score": self.score, "steps": self.steps}

    def reset(self, seed=None, options=None):
        """
        Repõe o ambiente para o estado inicial. É chamada no início de cada nova ronda.
        """
        super().reset(seed=seed)
        
        # Inicializar a posição da cobra, direção, pontuação e outras variáveis do jogo
        self.snake = deque([(2, SCOPE_Y[0]), (1, SCOPE_Y[0]), (0, SCOPE_Y[0])])
        self.direction = RIGHT
        self.food = self._create_food()
        self.food_style = self._get_food_style()
        self.score = 0
        self.steps = 0
        
        # Obter e devolver a observação inicial e as informações
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info

    def step(self, action):
        """
        Executa um passo no ambiente. Esta é a função principal de interação entre o ambiente e o agente.
        """
        self.steps += 1
        
        new_direction = ACTIONS_MAP[action]
        # Evita que a cobra inverta a direção em 180 graus (movimento inválido)
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction
        
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # --- Parte central da função de recompensa e da lógica do jogo ---
        terminated = False
        # Verificar se morreu
        if self._is_danger(new_head):
            terminated = True
            reward = -100  # Morte: penalização negativa enorme
        else:
            self.snake.appendleft(new_head)
            # Verificar se comeu a comida
            if new_head == self.food:
                self.score += self.food_style[0]
                reward = self.food_style[0]  # Comer a comida: recompensa positiva significativa
                self.food = self._create_food()
                self.food_style = self._get_food_style()
            else:
                self.snake.pop()
                reward = -1  # A cada passo: pequena penalização negativa para incentivar a eficiência
        
        # Verificar se atingiu o limite máximo de passos
        truncated = self.steps >= 1000

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            
        # Tem de devolver estes cinco valores, estritamente nesta ordem
        return observation, reward, terminated, truncated, info

    def render(self):
        """Renderiza o estado atual do ambiente, de acordo com o render_mode."""
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame(to_rgb_array=True)

    def _render_frame(self, to_rgb_array=False):
        """Função interna responsável pelas operações de desenho específicas do Pygame."""
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.font = pygame.font.SysFont('SimHei', 24)
            pygame.display.set_caption("Snake RL - Gymnasium Environment")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(BG_COLOR)
        
        # Desenhar a grelha, a comida e a cobra
        for x in range(GRID_SIZE, self.width, GRID_SIZE):
            pygame.draw.line(canvas, GRID_COLOR, (x, SCOPE_Y[0] * GRID_SIZE), (x, self.height), 1)
        for y in range(SCOPE_Y[0] * GRID_SIZE, self.height, GRID_SIZE):
            pygame.draw.line(canvas, GRID_COLOR, (0, y), (self.width, y), 1)
        pygame.draw.rect(canvas, self.food_style[1], (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE), 0)
        for s in self.snake:
            pygame.draw.rect(canvas, SNAKE_COLOR, (s[0] * GRID_SIZE + 1, s[1] * GRID_SIZE + 1, GRID_SIZE - 2, GRID_SIZE - 2), 0)

        if self.render_mode == "human":
            self.screen.blit(canvas, canvas.get_rect())
            score_text = self.font.render(f'Score: {self.score}', True, TEXT_COLOR)
            self.screen.blit(score_text, (30, 7))
            steps_text = self.font.render(f'Steps: {self.steps}', True, TEXT_COLOR)
            self.screen.blit(steps_text, (450, 7))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        if to_rgb_array:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        """Fecha o ambiente e liberta todos os recursos do Pygame."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    # ==============================================================================
    #  Funções auxiliares do ambiente
    # ==============================================================================
    def _create_food(self):
        """Cria comida numa posição aleatória que não esteja no corpo da cobra."""
        while True:
            # Usar self.np_random para garantir a reprodutibilidade após definir uma seed
            food_x = self.np_random.integers(SCOPE_X[0], SCOPE_X[1] + 1)
            food_y = self.np_random.integers(SCOPE_Y[0], SCOPE_Y[1] + 1)
            if (food_x, food_y) not in self.snake:
                return (food_x, food_y)

    def _get_food_style(self):
        """Escolhe aleatoriamente um estilo de comida (pontuação e cor)."""
        return random.choice(FOOD_STYLES)

    def _is_danger(self, pos):
        """Verifica se uma dada posição (x, y) é perigosa (colidir com a parede ou consigo mesma)."""
        x, y = pos
        if not (SCOPE_X[0] <= x <= SCOPE_X[1] and SCOPE_Y[0] <= y <= SCOPE_Y[1]):
            return True # Colidir com a parede
        if pos in self.snake:
            return True # Colidir consigo mesma
        return False