# Sliding-Tile-Puzzle


The Luddy Puzzle:
➔ This problem has been implemented using the A* star Algorithm with Priority queue and heuristic function.We have used the heuristic function as manhattan distance and hamming distance to calculate the heuristic cost and obtain the optimal solution.

➔ The search abstractions for this problem are as stated below:

◆ State Space: A 4\*4 puzzle board with 15 titles numbered from 1 to 15 and 0 indicating an empty tile.
◆ Successor Function: Search for all possible moves and make an optimal move based on heuristic cost.
◆ Cost: It is equal to the cost of the path from initial to goal state g(n) and heuristic function cost h(n)[h(n):- Optimal path from current state to goal state]
◆ Goal state: Arrange the tiles on the board in sorted manner with empty space at the end.
