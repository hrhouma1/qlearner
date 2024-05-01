# helpers.py

import numpy as np

from consts import PRINT_SCORE_EVERY, REWARD_COIN


def print_q_table(q_table):
    """
    Prints the entire Q-table in a compact, readable format.

    This function iterates over each state-action pair in the Q-table and prints
    the state, action, and corresponding Q-value in a formatted string. This is
    useful for debugging and understanding the learning process of the Q-learning
    algorithm.

    Parameters:
    - q_table (dict): A dictionary where each key is a state, and each value is
                      another dictionary mapping actions to their Q-values.
    """
    print("Q-table:")
    for state, actions in q_table.items():
        state_actions_str = f"State: {state} -> "
        for action, value in actions.items():
            state_actions_str += f"{action}: {value:.2f}, "
        # Remove the last comma and space, and print the state-action string
        print(state_actions_str[:-2])


def print_board(player_position, coin_position, columns, rows):
    """
    Prints the current game board with the player and coin positions marked.

    This function creates a 2D array representing the game board, places the player
    and coin at their respective positions, and then prints the board in a grid
    format. The board is represented as a grid of characters, where ' ' represents
    an empty space, 'P' represents the player, and 'C' represents the coin.

    Parameters:
    - player_position (tuple): A tuple representing the (row, column) position of the player.
    - coin_position (tuple): A tuple representing the (row, column) position of the coin.
    - columns (int): The number of columns in the game board.
    - rows (int): The number of rows in the game board.
    """
    # Initialize the board with empty spaces
    board = np.array([[" " for _ in range(columns + 1)] for _ in range(rows + 1)])
    
    # Place the coin at its position
    board[coin_position] = 'C'
    
    # Place the player at its position
    board[player_position] = 'P'
    
    # Print the board
    print("\n".join(["|".join(row) for row in board]))
    print()


def log_game_stats(game_index, collected_coin, total_cost, epsilon):
    """
    Log game statistics every few games.

    Parameters:
        game_index (int): Index of the current game.
        collected_coin (bool): Whether the coin was collected.
        total_cost (int): Total cost incurred during the game.
        epsilon (float): Current epsilon value.
    """
    if game_index % PRINT_SCORE_EVERY == 0:
        score = REWARD_COIN if collected_coin else 0 - total_cost
        print(f"Game number {game_index}, score: {score}, epsilon: {epsilon}")


def print_final_q_table(q_table, num_games):
    """
    Print the final Q-table after all games.

    Parameters:
        q_table (dict): The final Q-table.
        num_games (int): Total number of games simulated.
    """
    print(f"\nFinal Q-table after {num_games} games:")
    print_q_table(q_table)
