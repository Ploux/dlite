%matplotlib inline
import matplotlib.pyplot as plt
import random
import heapq
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations
import numpy as np

# Queues

FIFOQueue = deque

LIFOQueue = list

class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x): 
        self.key = key
        self.items = [] # a heap of (score, item) pairs
        for item in items:
            self.add(item)
         
    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]
    
    def top(self): return self.items[0][1]

    def __len__(self): return len(self.items)    

# straight-line distance between two points
def straight_line_distance(A, B):
    "Straight-line distance between two points."
    return sum(abs(a - b)**2 for (a, b) in zip(A, B)) ** 0.5

# D* Lite

def dstarlite(height, width, start, goal, obstacles, block=None, trigger=None):
    # the start and goal are tuples of (x,y) coordinates
    # the obstacles and block are lists of tuples of (x,y) coordinates
    # the trigger is a tuple of (x,y) coordinates, that when reached, will turn the block cells into obstacles
    
    # planning phase
       
   
    # draw the initial grid
    draw_grid("initial", height, width, start, goal, obstacles)
    
    # the grid is a 2D array of cells
    # each cell contains occupancy, g, rhs, and h values
    # in the form of a list [occupancy, g, rhs, h]
    # occupancy = 1 = occupied, 0 = empty
    # g = the cost of the cheapest path from the start to the cell
    # rhs = the cost of the cheapest path from the cell to the goal
    # h = the heuristic cost of the cheapest path from the cell to the goal
    
    # initialize the grid
    grid = [[0 for x in range(width)] for y in range(height)]
    # set the g and rhs values to infinity
    for y in range(height):
        for x in range(width):
            grid[y][x] = [0, math.inf, math.inf, math.inf]
    
    # set the h values to straight-line distance from the cell to the goal
    for y in range(height):
        for x in range(width):
            grid[y][x][3] = straight_line_distance((x, y), goal)
    
    # set the obstacles
    for obstacle in obstacles:
        grid[obstacle[1]][obstacle[0]] = [1, math.inf, math.inf, math.inf]
    
    # make sure the start is not occupied
    grid[start[1]][start[0]][0]= 0
         
    # set the robot location to the start
    robot = start
    
    # set the rhs value of the goal to 0
    grid[goal[1]][goal[0]][2] = 0
    
    # initialize the open list
    # The open list is a priority queue
    
    # The key of a node on the open list is min(g, rhs) + h
    # secondary key for tie-breaking is min(g, rhs)
    
    openlist = PriorityQueue(key=lambda x: min(x[1][1], x[1][2]) + x[1][3])
    
    # while U.TopKey() < calculatekey(start) or rhs(start) > g(start) 
    while openlist.top()[0] < calculatekey(grid, start) or grid[start[1]][start[0]][2] > grid[start[1]][start[0]][1]:
        # U.TopKey() is the key of the top node on the open list
        # calculatekey(start) is the key of the start node
        # rhs(start) is the rhs value of the start node
        # g(start) is the g value of the start node
        
        # get the top node on the open list
        u = openlist.pop()
        # get the key of the top node
        k_old = u[0]
        # get the cell of the top node
        u = u[1]
        
        # if g(u) > rhs(u)
        if grid[u[1]][u[0]][1] > grid[u[1]][u[0]][2]:
            # g(u) = rhs(u)
            grid[u[1]][u[0]][1] = grid[u[1]][u[0]][2]
            # for each s in SUCC(u)
            for s in SUCC(grid, u):
                # update_vertex(s)
                update_vertex(grid, openlist, s)
        # else
        else:
            # g(u) = infinity
            grid[u[1]][u[0]][1] = math.inf
            # for each s in SUCC(u) U {u}
            for s in SUCC(grid, u) + [u]:
                # update_vertex(s)
                update_vertex(grid, openlist, s)
    
    
    

        
        
        
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
        plt.fill([obstacle[1], obstacle[1]+1, obstacle[1]+1, obstacle[1]], [height-1-obstacle[0], height-1-obstacle[0], height-obstacle[0], height-obstacle[0]], 'k')
   
    # color the start red
    plt.fill([start[1], start[1]+1, start[1]+1, start[1]], [height-1-start[0], height-1-start[0], height-start[0], height-start[0]], 'r')
    # color the goal green
    plt.fill([goal[1], goal[1]+1, goal[1]+1, goal[1]], [height-1-goal[0], height-1-goal[0], height-goal[0], height-goal[0]], 'g')

    # title the plot
    plt.title(title)
    plt.show()
    
 
        
# test

# a small grid, only 4x5 cells
# The start is on the right side, three cells up from the bottom
# The goal is in the lower left corner
# The 1 obstacle is in the bottom right corner

dstarlite(4, 5, (1,4), (3,0), [(3,4)])