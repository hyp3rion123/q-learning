import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Sender:
    """
    A Q-learning agent that sends messages to a Receiver
    """
    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        self.num_sym = num_sym
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.actions = list(range(num_sym))
        self.q_vals = np.zeros((grid_rows, grid_cols, num_sym))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_vals[state[0], state[1], :])

    def update_q(self, old_state, action, reward):
        old_q = self.q_vals[old_state[0], old_state[1], action]
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_vals[old_state[0], old_state[1], action] = new_q

class Receiver:
    """
    A Q-learning agent that receives a message from a Sender and then navigates a grid
    """
    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        self.actions = [0, 1, 2, 3]
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.q_vals = np.zeros((num_sym, grid_rows, grid_cols, len(self.actions)))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_vals[state[0], state[1], state[2], :])

    def update_q(self, old_state, new_state, action, reward):
        old_q = self.q_vals[old_state[0], old_state[1], old_state[2], action]
        max_future_q = np.max(self.q_vals[new_state[0], new_state[1], new_state[2], :])
        new_q = old_q + self.alpha * (reward + self.discount * max_future_q - old_q)
        self.q_vals[old_state[0], old_state[1], old_state[2], action] = new_q

def get_grid(grid_name:str):
    grid = [[False for i in range(5)] for j in range(5)] # default case is 'empty'
    if grid_name == 'fourroom':
        grid[0][2] = True
        grid[2][0] = True
        grid[2][1] = True
        grid[2][3] = True
        grid[2][4] = True
        grid[4][2] = True
    elif grid_name == 'maze':
        grid[1][1] = True
        grid[1][2] = True
        grid[1][3] = True
        grid[2][3] = True
        grid[3][1] = True
        grid[4][1] = True
        grid[4][2] = True
        grid[4][3] = True
        grid[4][4] = True
    return grid

def legal_move(posn_x:int, posn_y:int, move_id:int, grid:list[list[bool]]):
    moves = [[0,-1],[0,1],[-1,0],[1,0]]
    new_x = posn_x + moves[move_id][0]
    new_y = posn_y + moves[move_id][1]
    result = (new_x,new_y)
    if new_x < 0 or new_y < 0 or new_x >= len(grid[0]) or new_y >= len(grid):
        result = (posn_x,posn_y)
    else:
        if grid[new_y][new_x]:
            result = (posn_x,posn_y)
    return result

def run_episodes(sender:Sender, receiver:Receiver, grid:list[list[bool]], num_ep:int, delta:float):
    reward_vals = []
    for ep in range(num_ep):
        receiver_x = 2
        receiver_y = 2
        prize_x = np.random.randint(len(grid[0]))
        prize_y = np.random.randint(len(grid))
        while grid[prize_y][prize_x] or (prize_x == receiver_x and prize_y == receiver_y):
            prize_x = np.random.randint(len(grid[0]))
            prize_y = np.random.randint(len(grid))
        prize_state = (prize_x, prize_y)
        sender_state = (prize_x, prize_y, 0)
        sender.q_vals[sender_state] = 0
        receiver_state = (0, receiver_x, receiver_y)
        reward = 0
        terminate = False
        while not terminate:
            message = sender.select_action(prize_state)
            new_state = (message, receiver_state[1], receiver_state[2])
            action = receiver.select_action(new_state)
            new_position = legal_move(receiver_state[1], receiver_state[2], action, grid)
            if new_position == prize_state:
                terminate = True
                reward = 1
            old_state = receiver_state
            receiver_state = (message, new_position[0], new_position[1])
            receiver.update_q(old_state, receiver_state, action, reward)
            if np.random.random() < delta:
                terminate = True
        sender.update_q(prize_state, message, reward)
        sender.alpha = max(sender.alpha_f, sender.alpha - (sender.alpha_i - sender.alpha_f)/num_ep)
        receiver.alpha = max(receiver.alpha_f, receiver.alpha - (receiver.alpha_i - receiver.alpha_f)/num_ep)
        reward_vals.append(reward)
    return reward_vals

def display_policy_grid(policy, N, grid):
    grid_size = 5
    action_arrows = {0: '←', 1: '↑', 2: '→', 3: '↓'}
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = ListedColormap(['white'])
    ax.matshow(np.zeros((grid_size, grid_size)), cmap=cmap)
    for i in range(grid_size):
        for j in range(grid_size):
            if not grid[i][j]:
                action = policy[i, j]
                ax.text(j, i, action_arrows[action], va='center', ha='center', fontsize=12)
            else:
                ax.text(j, i, "", va='center', ha='center', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Sender Policy when N = {N}")
    plt.show()

def run_simulations(signal_amount, num_tests, num_epsilons, num_N_episodes):
    rewards = np.zeros((num_tests, len(num_N_episodes), len(signal_amount)))
    epsilon = 0.1
    final_policies = []
    for j, signal in enumerate(signal_amount):
        for k, Nep in enumerate(num_N_episodes):
            print("Signal: ", signal, "Ep: ", Nep)
            for l in range(num_tests):
                grid = get_grid('fourroom')
                sender = Sender(signal, len(grid), len(grid[0]), 0.9, 0.01, Nep, epsilon, 0.95)
                receiver = Receiver(signal, len(grid), len(grid[0]), 0.9, 0.01, Nep, epsilon, 0.95)
                learn_rewards = run_episodes(sender, receiver, grid, Nep, 1 - 0.95)
                sender.epsilon = 0.0
                receiver.epsilon = 0.0
                test_rewards = run_episodes(sender, receiver, grid, 1000, 1 - 0.95)
                rewards[l, k, j] = np.mean(test_rewards)
                policy = np.argmax(sender.q_vals, axis=3)
                final_policies.append(policy)
                print(final_policies[0][0])
                display_policy_grid(final_policies[0][0], signal_amount, grid)
    return rewards

if __name__ == "__main__":
    num_signals = [4]
    num_tests = 10
    num_epsilons = [0.1]
    num_N_episodes = [100000]
    rewards = run_simulations(num_signals, num_tests, num_epsilons, num_N_episodes)
    fig, ax = plt.subplots(figsize=(6, 4))
    for j, signal in enumerate(num_signals):
        avg_rewards = np.mean(rewards[:, :, j], axis=0)
        std_errors = np.std(rewards[:, :, j], axis=0)
        ax.errorbar(num_N_episodes, avg_rewards, yerr=std_errors, fmt='-o', label=f' N={signal}')
    ax.set_xlabel('Nep')
    ax.set_ylabel('Average Discounted Reward')
    ax.set_title(f"Average Discounted Reward of Empty Room when e=0.1")
    ax.legend(loc='upper left')
    plt.show()