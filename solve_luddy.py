#!/usr/local/bin/python3
# solve_luddy.py : Sliding tile puzzle solver
#
# Code by: [Aneri Shah, Hely Modi, Dhruva Bhavsar||annishah, helymodi,dbhavsar]
#
# Based on skeleton code by D. Crandall, September 2019
#
from queue import PriorityQueue
import sys
import numpy as np


#Possible moves where the slides can be moved during each state in case of original and circular configuration.
MOVES = { "R": (0, -1), "L": (0, 1), "D": (-1, 0), "U": (1,0) }
#Possible moves where the slides can be moved during each state in case of Luddy (L letter fashion).
MOVES_LUDDY = {"A": (2, 1), "B": (2, -1), "C": (-2, 1), "D": (-2,-1), "E": (1, 2), "F": (1, -2), "G": (-1, 2), "H": (-1,-2)}

HammingHeuristic = "hamming"
ManhattanHeuristic = "manhattan"
Blank=0

def rowcol2ind(row, col):
    return row*4 + col

def ind2rowcol(ind):
    return (int(ind/4), ind % 4)

def valid_index(row, col):
    return 0 <= row <= 3 and 0 <= col <= 3

def swap_ind(list, ind1, ind2):
    return list[0:ind1] + (list[ind2],) + list[ind1+1:ind2] + (list[ind1],) + list[ind2+1:]

def swap_tiles(state, row1, col1, row2, col2):
    return swap_ind(state, *(sorted((rowcol2ind(row1,col1), rowcol2ind(row2,col2)))))

def printable_board(row):
    return [ '%3d %3d %3d %3d'  % (row[j:(j+4)]) for j in range(0, 16, 4) ]


#Applying hamming distance and manhattan distance heuristic function to obtain the optimal solution.
def hamming_dist(state):
  """Return the number of misplaced tiles."""
  return len([i for i, v in enumerate(state) if v != i+1 and v != len(state)])    


def manhattan(state):
    target=sorted(state, key=lambda x: float('inf') if x is 0 else x)
    cost = 0
    for node in state:
            if node != 0:
                gdist = abs(target.index(node) - state.index(node))
                (jumps, steps) = (gdist // 4, gdist % 4)
                cost += jumps + steps
    return cost/3



# return a list of possible successor states
# def successors(state):
#     (empty_row, empty_col) = ind2rowcol(state.index(0))
#     return [ (swap_tiles(state, empty_row, empty_col, empty_row+i, empty_col+j), c) \
#              for (c, (i, j)) in MOVES.items() if valid_index(empty_row+i, empty_col+j) ]


# return a list of possible successor states
def successors(state, choice):
    solution = []
    (empty_row, empty_col) = ind2rowcol(state.index(0))
    
    if(choice == 'circular'):    
        for (c, (i, j)) in MOVES.items():
            newI, newJ = empty_row+i, empty_col+j
            if newI == 4:
                newI = 0
            elif newI == -1:
                newI = 3
            if newJ == 4:
                newJ = 0
            elif newJ == -1:
                newJ = 3
            if valid_index(newI, newJ):
                solution.append((swap_tiles(state, empty_row, empty_col, newI, newJ), c))
#        return [ (swap_tiles(state, empty_row, empty_col, empty_row+i, empty_col+j), c) \
#                 for (c, (i, j)) in MOVES.items() if valid_index(empty_row+i, empty_col+j) ]
    
    elif choice == 'original':
        for (c, (i, j)) in MOVES.items():
            if valid_index(empty_row+i, empty_col+j):
                solution.append((swap_tiles(state, empty_row, empty_col, empty_row+i, empty_col+j), c)) 
                
    elif choice == 'luddy':
        for (c, (i, j)) in MOVES_LUDDY.items():
            if valid_index(empty_row+i, empty_col+j):
                solution.append((swap_tiles(state, empty_row, empty_col, empty_row+i, empty_col+j), c)) 
        
    return solution

# check if we've reached the goal
def is_goal(state):
    return sorted(state[:-1]) == list(state[:-1]) and state[-1]==0
  

def solve1(initial_board):
    fringe = [ (initial_board, "") ]
    while len(fringe) > 0:
        (state, route_so_far) = fringe.pop()
        for (succ, move) in successors( state,choice ):
            if is_goal(succ):
                return( route_so_far + move )
            fringe.insert(0, (succ, route_so_far + move ) )
    return False

#Solving the puzzle using A star Search.
def solve(initial_board):
    frontier = PriorityQueue()
    visited=[]
    frontier.put((0, 0, initial_board, ""))

    while not frontier.empty():
        _, level, board, current_answer = frontier.get()
        if is_goal(board):
            return (current_answer)
            break
        for (succ,move) in successors(board,choice):
            if succ not in visited:
                heuristic_cost = hamming(succ)
                visited.append(succ)
                frontier.put((heuristic_cost+level+1,
                               level+1,
                               succ,
                               current_answer + move
                               ))
    return current_answer,move

# For calculating permutation inversion referred from https://github.com/snehalvartak/AI-as-Search/blob/master/solver16.py
def is_solvable(initial_board):
    #Get the elements in the matrix as a list while preserving the order of numbers
    board = np.array(initial_board).flatten()
    permutation_inversions = 0
    n = board.shape[0]
    
    for i in range(0,n):
        for j in range(i+1,n):
            if board[i]>board[j] and board[i]!=0 and board[j]!= 0:
                permutation_inversions +=1
    
    #Adding the row number of the empty tile
    zero_row = np.where(np.array(initial_board) == 0)[0][0] + 1 
    
    permutation_inversions = permutation_inversions + zero_row
    #print permutation_inversions
    # If parity of the initial board is odd the puzzle cannot be solved.
    if permutation_inversions%2 == 1:
        return False
    else:
        return True


# test cases
if __name__ == "__main__":
    if(len(sys.argv) != 3):
       raise(Exception("Error: expected 2 arguments"))

    start_state = []
    
    choice = str(sys.argv[2])
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state+=( [ int(i) for i in line.split() ])

#    if(choice != "original"):
#        raise(Exception("Error: only 'original' puzzle currently supported -- you need to implement the other two!"))
    
    if len(start_state) != 16:
        raise(Exception("Error: couldn't parse start state file"))

    print("Start state: \n" +"\n".join(printable_board(tuple(start_state))))

    print("Solving...")
    if is_solvable(start_state):
        route = solve1(tuple(start_state))
        if route:
    
            print("Solution found in " + str(len(route)) + " moves:" + "\n" + route )
        else:
            print("INF")
    else:
        print ("INF")

