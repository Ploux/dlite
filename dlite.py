# Compares the performance of three search algorithms on a variable-sized grid, with both stationary and moving obstacles
# The three search algorithms are: breadth-first search, A*, and D* Lite

# we run the algorithms as follows
# search(grid, algorithm)

# where grid is an GridProblem object, consisting of a two-dimensional array of cells with 1 = occupied, 0 = empty, the start cell, the goal cell, and the cell to block. 
# For stationary grid problems where no cell is blocked and no replanning is required, the block cell is None
# and algorithm = "bfs", "astar", "dlite"

# we expect the following output
# the path from the start to the goal, or None if no path exists
# the number of nodes expanded
# the time taken to find the path
# the cost of the path
# We also plot the grid and the path

import sys
import math
import random
import time
import heapq
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors, cm, patches, gridspec, rc, rcParams, ticker, transforms, patheffects, lines, collections, markers, text, image, patches

# the grid problem class
class GridProblem:
    def __init__(self, grid, start, goal, block=None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.block = block
    
    def get_grid(self):
        return self.grid
        
    def get_start(self):
        return self.start
        
    def get_goal(self):
        return self.goal
        
    def get_block(self):
        return self.block
     
    def get_grid_size(self):
        return len(self.grid), len(self.grid[0])
        
    def get_grid_height(self):
        return len(self.grid)
        
    def get_grid_width(self):
        return len(self.grid[0])
            
    def get_neighbors(self, cell):
        neighbors = []
        if cell[0] > 0:
            neighbors.append((cell[0]-1, cell[1]))
        if cell[0] < self.get_grid_height()-1:
            neighbors.append((cell[0]+1, cell[1]))
        if cell[1] > 0:
            neighbors.append((cell[0], cell[1]-1))
        if cell[1] < self.get_grid_width()-1:
            neighbors.append((cell[0], cell[1]+1))
        return neighbors
        
    
   # returns the cost of the path from the start to the goal
    def get_path_cost(self, path):
        cost = 0
        for i in range(len(path)-1):
            cost += self.get_cost(path[i], path[i+1])
        return cost
        
# the breadth-first search algorithm
# returns the path from the start to the goal, or None if no path exists
# the path is a list of cells
# the grid is a GridProblem object
def bfs(grid):
    start = grid.get_start()
    goal = grid.get_goal()
    block = grid.get_block()
    grid_size = grid.get_grid_size()
    grid_height = grid.get_grid_height()
    grid_width = grid.get_grid_width()
    grid = grid.get_grid()
    visited = set()
    queue = []
    queue.append(start)
    parent = {}
    parent[start] = None
    while len(queue) > 0:
        cell = queue.pop(0)
        if cell == goal:
            path = []
            while cell != start:
                path.append(cell)
                cell = parent[cell]
            path.append(start)
            path.reverse()
            return path
        if cell not in visited:
            visited.add(cell)
            neighbors = grid.get_neighbors(cell)
            for neighbor in neighbors:
                if neighbor not in visited and grid[neighbor[0]][neighbor[1]] == 0:
                    queue.append(neighbor)
                    parent[neighbor] = cell
    return None
    

    
# a small grid, only 5x4 cells
# The goal is in the lower left corner (3, 0)
# The start is on the right side, three cells up from the bottom (2, 4)
# The obstacles are at (2,1), (2,2), and (3,2)
# There is no block

grid = [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0]]

grid_problem = GridProblem(grid, (3, 0), (2, 4), None)

# run bfs on this simple grid
path = bfs(grid_problem)

# plot the path
# use a different color for the start and goal, and for the block cell(if any)

# plot the grid
plt.imshow(grid, cmap='Greys',  interpolation='nearest')

# plot the path
if path is not None:
    path = np.array(path)
    plt.plot(path[:,1], path[:,0], 'b', linewidth=3)
    
# plot the start and goal
plt.plot(grid_problem.get_start()[1], grid_problem.get_start()[0], 'go', markersize=14)
plt.plot(grid_problem.get_goal()[1], grid_problem.get_goal()[0], 'ro', markersize=14)

# plot the block (if any)
if grid_problem.get_block() is not None:
    plt.plot(grid_problem.get_block()[1], grid_problem.get_block()[0], 'bo', markersize=14)
    
plt.show()




        
    
             