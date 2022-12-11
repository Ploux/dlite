%matplotlib inline
import matplotlib.pyplot as plt
import random
import heapq
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations
import numpy as np

# D* Lite

def dstarlite(height, width, start, goal, obstacles):
    # the start and goal are tuples of (x,y) coordinates
    # the obstacle is a lists of tuples of (x,y) coordinates

    # planning phase
    
    # the grid is a 2D array of cells, where 1 = occupied, 0 = empty
    grid = np.zeros((height, width))
    for obstacle in obstacles:
        grid[obstacle[0], obstacle[1]] = 1
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0
    
    # draw the initial grid
    draw_grid("D* Lite", height, width, start, goal, obstacles)

        
        
        
    # navigation phase

def draw_grid(title, height, width, start, goal, obstacles):
    "Use matplotlib to draw the grid."
    # set up the plot
    # 0,0 is in the top left corner
    # bottom left corner is (height-1, 0)
    # top right corner is (0, width-1)
    # bottom right corner is (height-1, width-1)
    
    # set up the plot
    fig = plt.figure(figsize=(width, height))
    # draw the grid
    plt.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'k-')
    plt.axis('off'); plt.axis('equal')
    # divide the grid into cells
    for x in range(width):
        for y in range(height):
            plt.plot([x, x+1, x+1, x, x], [y, y, y+1, y+1, y], 'k-')
    # color the obstacles
    for obstacle in obstacles:
        # 0,0 is in the top left corner
        # bottom left corner is (height-1, 0)
        # top right corner is (0, width-1)
        # bottom right corner is (height-1, width-1)
        plt.fill([obstacle[1], obstacle[1]+1, obstacle[1]+1, obstacle[1]], [obstacle[0], obstacle[0], obstacle[0]+1, obstacle[0]+1], 'k')
        
        
        
        
    # color the start red
    plt.fill([start[1], start[1]+1, start[1]+1, start[1]], [start[0], start[0], start[0]+1, start[0]+1], 'r')
    # color the goal green
    # remember that if the goal is (3,0) that should be the bottom left corner in a 4x5 grid
    print(goal)
    print(goal[0])
    print(goal[1])
    plt.fill([goal[1], goal[1], goal[1]+1, goal[1]+1], [goal[0], goal[0], goal[0]+1, goal[0]+1], 'g')
    
    # this is plotting at 0,0, so the goal is in the upper left corner
    
    
    
    
    # title the plot
    plt.title(title)
    plt.show()
    
    
        
# test

# a small grid, only 4x5 cells
# The start is on the right side, three cells up from the bottom
# The goal is in the lower left corner
# The 1 obstacle is in the bottom right corner

dstarlite(4, 5, (1, 4), (3, 0), [])