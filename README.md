# Project Title: Reinforcement Learning Agent for the Snake Game (ISCTE Final Project)

**Curricular Unit**: Unsupervised and Reinforcement Learning
**Author**: Ouhao Wu
**Student Number**: 123542

---

## 1. Project Overview

This is a custom Reinforcement Learning (RL) project. The central task consisted of designing and implementing an environment for the game "Snake", using the **Gymnasium** framework, and, in this environment, training and comparing two classic RL algorithms: **Q-Learning** and **SARSA**.

The project aims to demonstrate the understanding of the fundamental principles of Reinforcement Learning, including the construction of the environment, the design of the state space, the engineering of the reward function, the implementation of the algorithms and the evaluation of the models.

### Key Features:

* **Modular Architecture**: The project follows software engineering best practices, decoupling the environment, agent, training, and analysis logic into distinct modules.
* **Normalized Environment**: The `snake_gym_env.py` file implements a custom environment that fully complies with the Gymnasium standard API.
* **Feature Engineering**: To overcome the state space explosion problem, a compact state representation was designed with 10 key features, which dramatically increased the learning efficiency of tabular methods.
* **Algorithm Comparison**: Full implementation of the Q-Learning (Off-Policy) and SARSA (On-Policy) algorithms, with comparative visualization of their training and performance.
* **Complete Evaluation Process**: Provides a complete workflow, from visual training and testing to batch data analysis.

---

## 2. Project Structure

```
ISCTE_RL_Project/
├── run_project.py # (Entry Point) Run this file to launch the menu
├── snake_gym_env.py # (Environment) Implementation of the custom Gymnasium environment
├── agents.py # (Agents) Definition of the Q-Learning and SARSA algorithm classes
├── training.py # (Training Logic) Functions for training, comparing and playing manually
├── analysis.py # (Analysis Script) For in-depth analysis and visualization of the trained models
├── requirements.txt # (Dependencies) All required Python libraries
└── README.md # (Documentation) This file contains the following instructions
```

---

## 3. Installation Guide

1. **Clone or Download Project**: Extract the project files to a local folder.

2. **Install Python**: Ensure that you have Python 3.8 or higher installed.

3. **Installing Dependencies**: Open a terminal or command line, navigate to the project root folder and run the following command to install all required libraries:
```bash
pip install -r requirements.txt
```

---

## 4. How to Run

All operations are initiated through the `run_project.py` file. In the terminal, run:

```bash
python run_project.py
```

The program will present an interactive menu. You can select the different operations according to the instructions:

```
==============================================
ISCTE Final Project: RL Snake Game
===============================================
1. Train & Compare Q-Learning vs. SARSA
2. Play game with agent trained in Q-Learning
3. Play game with agent trained in SARSA
4. Play game manually
5. Perform full analysis on trained models
6. Exit
---------------------------------------------
Enter your choice (1-6):
```

### Description of Menu Options:

* **Option 1**: **Train and Compare Models**.
* You will be prompted to enter the number of episodes for training (2,000 or more is suggested).
* Trains the Q-Learning and SARSA agents sequentially. * After training, the `q_learning_snake.pkl` and `sarsa_snake.pkl` files will be generated in the project folder.
* Simultaneously, a Matplotlib graph will be displayed comparing the learning curves, which will also be saved as `comparison_results.png`.

* **Option 2 & 3**: **Watch a Trained Agent Play**.
* Loads the corresponding `.pkl` model file.
* Opens a Pygame window to demonstrate the agent's decision process in real time. This is the best way to evaluate the agent's final behavior.

* **Option 4**: **Play Manually**.
* Allows you to experience the game environment. Use the arrow keys to control, 'R' to start over and 'Q' to exit.

* **Option 5**: **Perform Deep Analysis**. * **(You must run option 1 first to generate the model files)**
* This option runs the `analysis.py` script to perform a detailed performance evaluation and generate several analysis graphs.

* **Option 6**: **Exit Program**.
