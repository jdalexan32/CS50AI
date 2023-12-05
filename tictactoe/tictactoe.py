"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board based on the board state.

    Args:
    - board (list of lists): the current state of the tic-tac-toe board.

    Returns:
    - str: The current player ('X' or 'O').
    """

    x_count = sum(row.count('X') for row in board)
    o_count = sum(row.count('O') for row in board)

    return 'X' if x_count == o_count else 'O'
   
    raise NotImplementedError

"""
# Example usage:
# Assume the initial state of the board is an empty 3x3 grid [['', '', ''], ['', '', ''], ['', '', '']]
from tictactoe import initial_state, player
board = initial_state()
player = player(board)
print(f"It's {current_player}'s turn.")
"""


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.

    Args:
    - board (list of lists): the current state of the tic-tac-toe board.

    Returns:
    - set: a set of possible actions represented as tuples (row, col).
    """
    return {(i,j ) for i in range(3) for j in range(3) if board[i][j] == EMPTY}

    raise NotImplementedError

"""
# Example usage:
from tictactoe import actions

## Assume the initial state of the board is an empty 3x3 grid [['', '', ''], ['', '', ''], ['', '', '']]
#initial_state = [['', '', ''], ['', '', ''], ['', '', '']]
#board = initial_state

# Or Assume a board like below
board = [['X', 'O', 'X'], ['', 'O', ''], ['', '', '']]

actions = actions(board)
print("Possible actions:", actions)
"""


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.

    Args:
    - board (list of lists): the current state of the tic-tac-toe board.
    - action (tuple): the action to be applied, represented as a tuple (row, col).

    Returns:
    - list of lists: the new board state after applying the action.
    """
    if board[action[0]][action[1]] != EMPTY:
        raise Exception("Invalid move")

    # Make a deep copy of the original board to avoid modifying it
    #new_board = deepcopy(board)
    new_board = [row.copy() for row in board]
    
    new_board[action[0]][action[1]] = player(board)
    return new_board

    raise NotImplementedError

"""
# Example usage:
# Assume the initial state of the board is an empty 3x3 grid.
from tictactoe import result
initial_state = [['', '', ''], ['', '', ''], ['', '', '']]
action = (1, 1)  # Example action: Player 'X' places their symbol at (1, 1)
new_board_state = result(initial_state, action)

# Print the initial and resulting board states
print("Initial Board State:")
for row in initial_state:
    print(row)

print("\nResulting Board State:")
for row in new_board_state:
    print(row)
"""


def winner(board):
    """
    Returns the winner of the game, if there is one.

    Args:
    - board (list of lists): the current state of the tic-tac-toe board.

    Returns:
    - str: the winner ('X', 'O'), or None if there is no winner.
    """
    for i in range(3):
        # Check rows
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        # Check columns
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]
    
    # No winner
    return None

    raise NotImplementedError

"""
# Example usage:
# Assume a board with a winning condition for 'X'
from tictactoe import winner
winning_board_X = [['X', 'O', ''], ['O', 'X', ''], ['O', '', 'X']]
winner_X = winner(winning_board_X)
print("Winner:", winner_X)
# Or 
# Assume a board with no winner
no_winner_board = [['X', 'O', 'X'], ['O', 'X', 'O'], ['O', 'X', 'O']]
no_winner = winner(no_winner_board)
print("Winner:", no_winner)
"""


def terminal(board):
    """
    Returns True if game is over, False otherwise.

    Args:
    - board (list of lists): the current state of the tic-tac-toe board.

    Returns:
    - bool: True if the game is over, False otherwise.
    """
    return winner(board) is not None or all(all(cell != EMPTY for cell in row) for row in board)

    raise NotImplementedError

"""
# Example usage:
# Assume a board with a winning condition for 'X'
from tictactoe import winner, terminal
winning_board_X = [['X', 'O', ''], ['O', 'X', ''], ['O', '', 'X']]
terminal = terminal(winning_board_X)
print("Game over:", terminal)

# Assume a board with no winner but still empty cells
from tictactoe import winner, terminal
incomplete_board = [['X', 'O', ''], ['O', 'X', ''], ['O', '', '']]
terminal = terminal(incomplete_board)
print("Game over:", terminal)

# Assume a board with no winner and all cells filled (a tie)
from tictactoe import winner, terminal
tie_board = [['X', 'O', 'X'], ['O', 'X', 'O'], ['O', 'X', 'O']]
terminal = terminal(tie_board)
print("Game over:", terminal)

# Assume a board at initial_state = [['', '', ''], ['', '', ''], ['', '', '']]
from tictactoe import winner, terminal
initial_state = [['', '', ''], ['', '', ''], ['', '', '']]
terminal = terminal(initial_state)
print("Game over:", terminal)
"""


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.

    Args:
    - terminal_board (list of lists): the terminal state of the tic-tac-toe board.

    Returns:
    - int: The utility of the board (1 for 'X' win, -1 for 'O' win, 0 for a tie).
    """

    winner_player = winner(board)

    if winner_player == 'X':
        return 1
    elif winner_player == 'O':
        return -1
    else:
        return 0

    raise NotImplementedError

"""
# Example usage:
# Assume a board with a winning condition for 'X'
from tictactoe import result, winner, utility
winning_board_X = [['X', 'O', ''], ['O', 'X', ''], ['O', '', 'X']]
result = utility(winning_board_X)
print("Utility of the board:", result)

# Assume a board with no winner but still empty cells
from tictactoe import result, winner, utility
incomplete_board = [['X', 'O', ''], ['O', 'X', ''], ['O', '', '']]
result = utility(incomplete_board)
print("Utility of the board:", result)

# Assume a board with no winner and all cells filled (a tie)
from tictactoe import result, winner, utility
tie_board = [['X', 'O', 'X'], ['O', 'X', 'O'], ['O', 'X', 'O']]
result = utility(tie_board)
print("Utility of the board:", result)

"""


def max_value(board):
    """
    Helper function for the Max player in the Minimax algorithm.

    Args:
    - board (list of lists): the current state of the tic-tac-toe board.

    Returns:
    - tuple: A tuple (score, move) representing the maximum score and corresponding move.
    """
    if terminal(board):
        return utility(board)

    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))

    return v


def min_value(board):
    """
    Helper function for the Min player in the Minimax algorithm.

    Args:
    - board (list of lists): the current state of the tic-tac-toe board.

    Returns:
    - tuple: a tuple (score, move) representing the minimum score and corresponding move.
    """
    if terminal(board):
        return utility(board)

    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))

    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.

    Args:
    - board (list of lists): the current state of the tic-tac-toe board.

    Returns:
    - tuple: the optimal move represented as a tuple (row, col).
    """
    if terminal(board):
        return None

    player_turn = player(board)
    if player_turn == 'X':
        v = -math.inf
        best_move = None
        for action in actions(board):
            k = min_value(result(board, action))
            if k > v:
                v = k
                best_move = action

    else:
        v = math.inf
        best_move = None
        for action in actions(board):
            k = max_value(result(board, action))
            if k < v:
                v = k
                best_move = action

    return best_move

    raise NotImplementedError

"""
# Example usage:
from tictactoe import player, actions, result, winner, terminal, utility, max_value, min_value, minimax
# Assume the initial state of the board is an empty 3x3 grid.
initial_board = [['', '', ''], ['', '', ''], ['', '', '']]
optimal_move = minimax(initial_board)
print("Optimal move:", optimal_move)
"""
