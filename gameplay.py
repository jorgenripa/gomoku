"""
Utility functions for playing n-in-a-row, where n is 3, 4 or 5 
 n = 3 and a 3x3 board corresponds to the classic tic-tac-toe game.
 n = 5 and a 15x15 or larger board corresponds to five-in-a-row, or gomoku.
 n = 4 with a board size 5x5 is also an interesting variant.

 The board is coded as two matrices of binary (0/1) numbers. 
 Each matrix the size of the board and represents the markers set by one of the players.
 A 0 is an empty square and a 1 is a marker.
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Global parameters:
board_size = 15 # The board is 15x15
seq_to_win = 5  # how many marks in a row counts as a win


def check_winner(boards, player, moves):
    """ 
    Count consequitive markers, starting at the latest move.
     boards : a batch of boards, one per ongoing game, size: (m,b,b,2) (b=board_size)
     player : 0 or 1
     moves : an array of the last moves (row, col), one per ongoing game, size: (m,2)

    returns: 
     game_over : a 1D array of booleans = True for each winning move

    """
    # boards is (m,b,b,2)
    m = boards.shape[0]
    game_over = np.full((m,),False)
    # check rows:
    for i in range(m):
        count = 1
        row = moves[i,0]
        col = moves[i,1]
        while col>0 and boards[i,row,col-1,player]==1:
            col = col-1
            count = count + 1
        col = moves[i,1]
        while col<board_size-1 and boards[i,row,col+1,player]==1:
            col = col+1
            count = count + 1
        game_over[i] = count >= seq_to_win

    # check cols:
    for i in range(m):
        if ~game_over[i]:
            count = 1
            row = moves[i,0]
            col = moves[i,1]
            while row>0 and boards[i,row-1,col,player]==1:
                row = row-1
                count = count + 1
            row = moves[i,0]
            while row<board_size-1 and boards[i,row+1,col,player]==1:
                row = row+1
                count = count + 1
            game_over[i] = count >= seq_to_win
    
    # check diagonal 1:
    for i in range(m):
        if ~game_over[i]:
            count = 1
            row = moves[i,0]
            col = moves[i,1]
            while row>0 and col>0 and boards[i,row-1,col-1,player]==1:
                row = row-1
                col = col-1
                count = count + 1
            row = moves[i,0]
            col = moves[i,1]
            while row<board_size-1 and col<board_size-1 and boards[i,row+1,col+1,player]==1:
                row = row + 1
                col = col + 1
                count = count + 1
            game_over[i] = count >= seq_to_win
    
    # check diagonal 2:
    for i in range(m):
        if ~game_over[i]:
            count = 1
            row = moves[i,0]
            col = moves[i,1]
            while row>0 and col<board_size-1 and boards[i,row-1,col+1,player]==1:
                row = row-1
                col = col+1
                count = count + 1
            row = moves[i,0]
            col = moves[i,1]
            while row<board_size-1 and col>0 and boards[i,row+1,col-1,player]==1:
                row = row + 1
                col = col - 1
                count = count + 1
            game_over[i] = count >= seq_to_win
    
    return game_over


def check_game_over(boards, player, moves):
    """ 
    Check if the last move was a win, or if the board is full (it's a tie)
    The reward to player 0(!) is:
        1 for a win, 
        0 for a tie, and 
        -1 for a loss (if player 1 made a winning move)
    """
    game_over = check_winner(boards, player, moves)
    rewards = np.double(game_over)  # win=1, tie=0, not game over = don't care
    if player==1:
        rewards = -rewards
    game_over = game_over | np.all(np.sum(boards, axis=3) > 0, axis=(1, 2))  # check for ties
    return game_over, rewards

def choose_exploring_move(states, player):
    """ 
    Choose a move randomly from the empty squares
    """    
    moves = np.zeros((states.shape[0], 2), dtype=int)
    for i, s in enumerate(states):
        possible_moves = np.argwhere(np.sum(s, axis=-1) == 0)  # each row is (row, col)
        choice = np.random.randint(possible_moves.shape[0])
        moves[i, :] = possible_moves[choice, :]
    return moves


def choose_greedy_move(states, model):
    """ 
    Choose the best move available according to model
    This is only used by player = 0 (the learning Player)
    states : a batch of boards. A numpy array or tensor with shape (m,b,b,2), 
             where b=board_size and m is board index
    model:  typically a tensorflow.keras model 

    returns a numpy array of chosen_moves and estimated values (returns) of those moves
    """

    # find possible moves:
    poss_moves = np.argwhere(np.sum(states, axis=3) == 0)  # each row contains (board, row, col)
    n_moves = poss_moves.shape[0]
    # choose most valuable afterstate
    # create a batch of possibe moves:
    poss_states = states[poss_moves[:, 0], ...].copy()
    for i, pos in enumerate(poss_moves):
        poss_states[i, pos[1], pos[2], 0] = 1

    # evaluate the possibilities:
    poss_values = model(poss_states).numpy()  # faster than predict(), according to documentation
    # choose best action:
    m = states.shape[0]
    chosen_moves = np.zeros((m, 2), dtype=int)
    chosen_values = np.zeros((m,))
    for i in range(m):
        sel = np.flatnonzero(poss_moves[:, 0] == i)
        choice = np.argmax(poss_values[sel])
        chosen_moves[i, :] = poss_moves[sel[choice], 1:3]
        chosen_values[i] = poss_values[sel[choice]]

    return chosen_moves, chosen_values

# Utility function:
def choose_softmax(x):
    ex = np.exp(x-np.amax(x))  # x can be large. Subtract max(x) to avoid overflow
    cumex = np.cumsum(ex)
    choice = np.nonzero(np.random.rand() * cumex[-1] <= cumex)[0][0]
    return choice

def choose_softmax_move(states, player, max_factor, model):
    """ 
    Choose a move with probabilities given by their values according to model
    The max_factor parameter controls the extent of 'maximization'
      max_factor = 0 : choose a move completely randomly
      max_factor = inf : choose the best move, always
    This can be used by both players
    """    
    m = states.shape[0]
    poss_moves = np.argwhere(np.sum(states, axis=-1) == 0)  # each row is (i, row, col)
    
    # create a batch of possibe moves for player:
    poss_states = states[poss_moves[:, 0], ...].copy()
    for i, pos in enumerate(poss_moves):
        poss_states[i, pos[1], pos[2], player] = 1

    if player==1:
        # switch roles for evaluation:
        poss_states = poss_states[:, :, :, [1, 0]]  
    # evaluate the possibilities:
    poss_values = model(poss_states).numpy() 

    # choose best action:
    chosen_moves = np.zeros((m, 2), dtype=int)
    chosen_values = np.zeros((m,))
    for i in range(m):
        sel = np.flatnonzero(poss_moves[:, 0] == i)
        choice = choose_softmax(max_factor * poss_values[sel, 0])
        chosen_moves[i, :] = poss_moves[sel[choice], 1:3]
        chosen_values[i] = poss_values[sel[choice]]

    return chosen_moves, chosen_values

# Update the boards and check for game over
def update_boards(boards, player, moves):
    m = boards.shape[0]
    # Make the moves:
    boards[range(m), moves[:, 0], moves[:, 1], player] = 1
    # check for game_over
    game_over, rewards = check_game_over(boards, player, moves)
    return boards, rewards, game_over

# Make moves on all boards, sometimes exploring
def make_player_move(states, player, exploration_rate, max_factor, model):
    m = states.shape[0]
    moves = np.zeros((m, 2), dtype=int)
    values = np.zeros((m,)) # estimated values of chosen moves
    if exploration_rate > 0:
        exploring = np.random.rand(m, ) < exploration_rate
        moves[exploring,...] = choose_exploring_move( states[exploring,...], player )
    else:
        exploring = np.full((m,), False)
    
    moves[~exploring,...], values[~exploring] = choose_softmax_move( states[~exploring,...], player, max_factor, model )
                                                #choose_greedy_move( states[~exploring,...], model )
    
    states, rewards, game_over = update_boards(states, player, moves)
    
    if np.any(exploring):
        values[exploring] = model(states[exploring])[:,0]
        
    return states, values, rewards, game_over, moves
    

################################################
# A couple of evaluation and graphics functions 

# Plot the board in the current figure
def plot_board(b, last_move, gomoku_style=False):
    b = b[0]
    plt.cla()
    plt.axis("equal")
    if gomoku_style:
        # gomoku style:
        ax = plt.gca()
        ax.set_facecolor("xkcd:sandy brown")
        ax.set_xticks([])
        ax.set_yticks([])

        for i in range(board_size):
            plt.plot([0, board_size-1], [i, i], 'k-', zorder=0)
            plt.plot([i, i], [0, board_size-1], 'k-', zorder=0)

        for row in range(board_size):
            for col in range(board_size):
                if b[row, col, 0] == 1: # black player
                    circle = plt.Circle((col, row), 0.45, color='k')
                    plt.gca().add_patch(circle)
                elif b[row, col, 1] == 1:
                    circle = plt.Circle((col, row), 0.45, color='w')
                    plt.gca().add_patch(circle)

        # mark last move
        if not last_move is None:
            r = last_move[0,0]
            c = last_move[0,1]
            circle = plt.Circle((c, r), 0.49, color='xkcd:light grey', lw=2, fill=False)
            plt.gca().add_patch(circle)

    else: # luffarschack style
        plt.gca().set_axis_off()
        # plot grid
        for i in range(board_size+1):
            plt.plot([0, board_size], [i, i], 'k-')
            plt.plot([i, i], [0, board_size], 'k-')

        for row in range(board_size):
            for col in range(board_size):
                if b[row, col, 0] == 1:
                    margin = 0.1
                    plt.plot([col + margin, col + 1 - margin], [row + margin, row + 1 - margin], 'r', linewidth=2)
                    plt.plot([col + margin, col + 1 - margin], [row + 1 - margin, row + margin], 'r', linewidth=2)
                elif b[row, col, 1] == 1:
                    phi = np.linspace(0, 2 * np.pi, 100)
                    mx = col + 0.5
                    my = row + 0.5
                    radius = 0.4
                    plt.plot(mx + radius * np.cos(phi), my + radius * np.sin(phi), 'b', linewidth=2)
        # mark last move
        if not last_move is None:
            r = last_move[0,0]
            c = last_move[0,1]
            plt.plot([c,c+1,c+1,c,c], [r,r,r+1,r+1,r], '-', linewidth=3, color='k')


def play_a_game(model_1, model_2, player_exploration, player_max_factor, opp_exploration, opp_max_factor):
    """
    Play a game between two models and plot each move:
    """    
    if model_2 is None:
        model_2 = model_1
        
    # play a game:
    board = np.zeros((1, board_size, board_size, 2))
    game_over = False
    plt.figure(1).set_size_inches(20,60)
    plt.clf()
    n_move = 0
    while n_move<75 and not game_over:
        player = (n_move) % 2
        if player==0:
            board, _, reward, game_over, move = make_player_move(
                board, player, exploration_rate=player_exploration, max_factor=player_max_factor, model=model_1)
        elif player==1:
            board, _, reward, game_over, move = make_player_move(
                board, player, exploration_rate=opp_exploration, max_factor=opp_max_factor, model=model_2)
        n_move += 1
        plt.subplot(15, 5, n_move)
        plot_board(board, move, gomoku_style=True)
    plt.show()
