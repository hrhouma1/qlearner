# qlearner
Q learniner example

### Grid Game Overview
The grid game is a strategic simulation game played on an adjustable grid, with the default size being 2 rows by 3 columns. In this game, a player navigates to collect a coin while minimizing movement costs. The game showcases basic concepts of reinforcement learning through the use of a Q-learning algorithm.

### Game Rules and Mechanics

#### Board Layout:
- The default board consists of two rows and three columns, but the size is adjustable based on user preference.
- The player starts from any position in the bottom row and aims to reach the top row where a coin is placed at the beginning of each game.

#### Movements:
- The player has several movement options:
  - **Up**: Moves the player one square up vertically at no cost.
  - **Up+Left**: Moves the player one square up and one square to the left diagonally, costing 1 cent.
  - **Up+Right**: Moves the player one square up and one square to the right diagonally, also costing 1 cent.
- The available moves are restricted by the board's edges, preventing movement outside the grid.

#### Objectives:
- **Primary Objective**: The player's main goal is to collect the coin located in the top row.
- **Secondary Objective**: Minimize the cost of movements while attempting to collect the coin.

### Game Play:
- The game initiates with the player's position randomly selected in the bottom row and the coin always placed in the top row.
- The player selects moves based on the current situation, aiming to maximize rewards (coin collection) and minimize costs (movement costs).
- The game concludes when the player reaches the top row or when no further moves are possible from the current position.

### Scoring:
- The player earns a reward of 10 cents for collecting the coin.
- Each diagonal movement incurs a cost of 1 cent.
- The game's final score is the net result of subtracting the total movement costs from the coins collected during the game.

### Learning Component:
- A Q-learning algorithm is used to learn optimal strategies over multiple plays.
- A Q-table records expected future rewards for each action at each state, guiding decision-making.
- The Q-table is updated after each move based on the received reward and estimated future rewards, demonstrating principles of reinforcement learning.

### Purpose:
This grid game serves as an educational tool to illustrate the principles of Q-learning, a form of reinforcement learning, in a straightforward and interactive manner. It allows players and observers to understand how decisions are influenced and refined based on past experiences, providing a clear example of how artificial intelligence can learn and adapt to achieve specific objectives in a controlled environment.
