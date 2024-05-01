# grid_game_simulation_with_q_learning.py

import numpy as np
import random

from consts import COST_LEFT_RIGHT, EPSILON_DECAY, LAST_COLUMN, LAST_ROW, NUM_GAMES, REWARD_COIN
from helpers import log_game_stats, print_board, print_final_q_table, print_q_table


def init_game():
    """
    Initialize the game state.

    Returns:
    - tuple: The initial position of the player and the coin.
    """
    coin_position = (0, random.randint(0, LAST_COLUMN - 1))
    start_position = (LAST_ROW, random.randint(0, LAST_COLUMN - 1))
    return start_position, coin_position


def possible_moves(position):
    """
    Determine possible moves from the current position.

    Parameters:
    - position (tuple): The current position of the player.

    Returns:
    - dict: A dictionary mapping possible actions to their resulting positions and costs.
    """
    moves = {}
    x, y = position
    if x > 0:
        moves['up'] = [(x - 1, y), 0]
        if y - 1 >= 0:
            moves['up+left'] = [(x - 1, y - 1) ,COST_LEFT_RIGHT]
        if y + 1 < LAST_COLUMN:
            moves['up+right'] = [(x - 1, y + 1), COST_LEFT_RIGHT]
    return moves


def is_terminal(position):
    """
    Determine if the game has reached a terminal state, which in this context
    means that the position is on the last row of the grid.

    Parameters:
        position (tuple): The current position of the player on the grid.

    Returns:
        bool: True if the position is terminal, False otherwise.
    """
    return position[0] == 0  # Assuming 0 represents the last row in the game's context


def init_q_table():
    """
    Initialize the Q-table.

    Returns:
    - dict: The Q-table with all states initialized to zero.
    """
    q_table = {}
    for player_row in range(LAST_ROW + 1): # Player can be in the last row or the second last row
        for player_col in range(LAST_COLUMN): # Player can be in any column
            for coin_col in range(LAST_COLUMN): # Coin can be in any column of the first row
                state = ((player_row, player_col), (0, coin_col))
                q_table[state] = {'up': 0, 'up+left': 0, 'up+right': 0}
    return q_table


def simulate_game(num_games, epsilon_start=1.0, epsilon_end=0.001, epsilon_decay=EPSILON_DECAY, gamma=0.9, alpha=0.1):
    """
    Simulate a specified number of games using a Q-learning algorithm with an epsilon-greedy strategy.

    Parameters:
        num_games (int): The number of games to simulate.
        epsilon_start (float): The initial epsilon value for the epsilon-greedy strategy.
        epsilon_end (float): The minimum epsilon value.
        epsilon_decay (float): The rate at which epsilon decays.
        gamma (float): Discount factor for future rewards (how much future rewards are taken into account).
        alpha (float): Learning rate for updating the Q-table.
    """
    epsilon = epsilon_start

    for game_index in range(num_games):
        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        position, coin_position = init_game()
        total_cost = 0
        collected_coin = False

        while not is_terminal(position):
            state = (position, coin_position)
            moves = possible_moves(position)
            action = choose_action(state, moves, q_table, epsilon)

            next_position, move_cost = moves[action]
            total_cost += move_cost
            collected_coin = next_position == coin_position
            next_state = (next_position, coin_position)
            reward = calculate_reward(collected_coin, move_cost)

            update_q_table(q_table, state, action, reward, next_state, alpha, gamma)

            position = next_position

        log_game_stats(game_index, collected_coin, total_cost, epsilon)

    print_final_q_table(q_table, num_games)


def choose_action(state, moves, q_table, epsilon):
    """
    Choose an action using an epsilon-greedy strategy from the available moves.

    Parameters:
        state (tuple): Current state of the game.
        moves (dict): Possible moves from the current state.
        q_table (dict): Current Q-table.
        epsilon (float): Current epsilon value for exploration.

    Returns:
        str: Chosen action.
    """
    if random.random() < epsilon:
        return random.choice(list(moves.keys()))
    else:
        return max(moves, key=lambda action: q_table.get(state, {}).get(action, 0))


def calculate_reward(collected_coin, move_cost):
    """
    Calculate the reward based on the game's current state.

    Parameters:
        collected_coin (bool): Whether the coin was collected in the move.
        move_cost (int): Cost of the move.

    Returns:
        int: Calculated reward.
    """
    return REWARD_COIN if collected_coin else -move_cost


def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """
    Update the Q-table based on the action taken and the subsequent state.

    Parameters:
        q_table (dict): The Q-table to update.
        state (tuple): The current state.
        action (str): The action taken.
        reward (int): The reward received.
        next_state (tuple): The state after the action.
        alpha (float): Learning rate.
        gamma (float): Discount factor.

    """
    # Calculate the action that will return the Max Q ( e.g. the max sum of future rewoards )
    best_next_action = max(q_table[next_state], key=lambda a: q_table[next_state][a], default=0)

    # Update the Q table
    q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])
  

def print_game():
    """
    Simulates and prints one game using the Q-table to determine the next action.
    """
    print ("\nGame:")

    # Initialize the game
    position, coin_position = init_game()
    state = (position, coin_position)
    total_cost = 0
    collected_coin = False

    # Print the initial board
    print(f"Initial state: {state}")
    print_board(position, coin_position, LAST_COLUMN, LAST_ROW)

    while not is_terminal(position):
            state = (position, coin_position)
            moves = possible_moves(position)
            action = max(moves, key=lambda a: q_table[state][a])

            next_position, move_cost = moves[action]
            total_cost += move_cost
            collected_coin = next_position == coin_position
            
            position =  next_position

            # Print the state, action, and board
            print(f"State: {state}, Action: {action}")
            print_board(position, coin_position, LAST_COLUMN, LAST_ROW)

    # Calculate the score
    score = REWARD_COIN if collected_coin else 0
    score -= total_cost

    # Print the final state and whether the coin was collected
    print(f"Final State: {(position, coin_position)}, Coin Collected: {collected_coin}, Score: {score}")


# Init empty Q table
q_table = init_q_table()

# Example game run
if __name__ == "__main__":
    final_q_table = simulate_game(num_games=NUM_GAMES)

    print_game()
