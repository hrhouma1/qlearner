# grid_game_simulation_with_q_learning.py

import numpy as np
import random

# Constants for the game
#LAST_ROW = 2
#LAST_COLUMN = 2
#PRINT_SCORE_EVERY = 10
#NUM_GAMES=100
#EPSILON_DECAY=0.9

LAST_ROW = 5
LAST_COLUMN = 5
PRINT_SCORE_EVERY = 500
NUM_GAMES=10000
EPSILON_DECAY=0.9995

RANDOM=False

COST_LEFT_RIGHT = 1
REWARD_COIN = 10

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

def print_board(player_position, coin_position):
    """
    Print the current game board.

    Parameters:
    - player_position (tuple): The current position of the player.
    - coin_position (tuple): The current position of the coin.
    """
    # Initialize the board with empty spaces
    board = np.array([[" " for _ in range(LAST_COLUMN + 1)] for _ in range(LAST_ROW + 1)])
    
    # Place the coin at its position
    board[coin_position] = 'C'
    
    # Place the player at its position
    board[player_position] = 'P'
    
    # Print the board
    print("\n".join(["|".join(row) for row in board]))
    print()

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

def print_q_table(q_table):
    """
    Print the entire Q-table in a compact format.

    Parameters:
    - q_table (dict): The Q-table to print.
    """
    print("Q-table:")
    for state, actions in q_table.items():
        state_actions_str = f"State: {state} -> "
        for action, value in actions.items():
            state_actions_str += f"{action}: {value:.2f}, "
        # Remove the last comma and space, and print the state-action string
        print(state_actions_str[:-2])

def simulate_game(num_games, epsilon_start=1.0, epsilon_end=0.001, epsilon_decay=EPSILON_DECAY, gamma=0.9, alpha=0.1):
    """
    Simulates games of the grid game using Q-learning.

    Parameters:
    - num_games (int): The number of games to simulate.
    - epsilon_start (float): The initial epsilon value for the epsilon-greedy strategy.
    - epsilon_end (float): The minimum epsilon value.
    - epsilon_decay (float): The rate at which epsilon decays.
    - gamma (float): The discount factor for future rewards.
    - alpha (float): The learning rate for updating the Q-table.

    Returns:
    - q_table (dict): The updated Q-table after the game.
    """
    # Initialize epsilon
    epsilon = epsilon_start if epsilon_start is not None else 1.0

    for game_index in range(num_games):
        # Decay epsilon
        epsilon *= epsilon_decay
        epsilon = max(epsilon, epsilon_end)

        # Set to epsilon 1 for ALWAYS random action
        if RANDOM:
            epsilon = 1

        position, coin_position = init_game()
        total_cost = 0
        collected_coin = False
        
        while True:
            # Stop on the last row
            if position[0] == 0:
                break

            state = (position, coin_position)
            moves = possible_moves(position)

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(list(moves.keys()))
            else:
                action = max(moves, key=lambda a: q_table[state][a])

            next_position, move_cost = moves[action]
            total_cost += move_cost

            if next_position == coin_position and not collected_coin:
                collected_coin = True

            next_state = (next_position, coin_position)
            reward = REWARD_COIN if collected_coin else -move_cost

            position = next_position

            # Q-learning update with gamma and alpha as parameters
            best_next_action = max(q_table[next_state], key=lambda a: q_table[next_state][a], default=0)
            q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])

        score = REWARD_COIN if collected_coin else 0
        score -= total_cost

        if game_index % PRINT_SCORE_EVERY == 0:
            print (f"Game number {game_index}, score: {score}, epcilon {epsilon}")

    print(f"\nFinal Q-table after {num_games} games:")
    print_q_table(q_table)
    return q_table

def print_game():
    """
    Simulates and prints one game using the Q-table to determine the next action.
    """
    print ("\nGame:")

    # Initialize the game
    position, coin_position = init_game()
    total_cost = 0
    collected_coin = False
    
    while True:
        # Stop on the last row
        if position[0] == 0:
            break

        state = (position, coin_position)
        moves = possible_moves(position)

        # Use the Q-table to select the best action
        best_action = max(moves, key=lambda a: q_table[state][a])

        next_position, move_cost = moves[best_action]
        total_cost += move_cost

        if next_position == coin_position and not collected_coin:
            collected_coin = True

        # Print the state, action, and board
        print(f"State: {state}, Action: {best_action}")
        print_board(position, coin_position)

        position = next_position

    # Calculate the score
    score = REWARD_COIN if collected_coin else 0
    score -= total_cost

    # Print the last state, action, and board
    print(f"State: {state}, Action: {best_action}")
    print_board(position, coin_position)

    # Print the final state and whether the coin was collected
    print(f"Final State: {(position, coin_position)}, Coin Collected: {collected_coin}, Score: {score}")

# Init empty Q table
q_table = init_q_table()

# Example game run
if __name__ == "__main__":
    final_q_table = simulate_game(num_games=NUM_GAMES)

    print_game()
