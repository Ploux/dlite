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
        """Add item to the queue."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]
    
    def top(self): 
        """Return the item with min f(item) value without popping."""
        return self.items[0][1]
        
    def topkey(self): 
        """Return the key of the item with min f(item) value without popping."""
        return self.items[0][0]

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
    # top left corner is (0,0)
    # bottom left corner is (height-1, 0)
    # top right corner is (0, width-1)
    # bottom right corner is (height-1, width-1)
    
    grid = [[0 for x in range(width)] for y in range(height)]
    
    # let's check this and be sure that we can access (3,4) as grid[4][3]
    # print(grid[4][3])
    # that doesn't work, try this instead
    #print(grid[3][4])
   
    # set the g and rhs values to infinity
    for x in range(height):
        for y in range(width):
            grid[x][y] = [0, math.inf, math.inf, math.inf]
    
    # set the h values to straight-line distance from the cell to the goal
    for x in range(height):
        for y in range(width):
            grid[x][y][3] = straight_line_distance((x, y), goal)
    
    # set the obstacles
    for obstacle in obstacles:
        grid[obstacle[0]][obstacle[1]] = [1, math.inf, math.inf, math.inf]
    
    # make sure the start is not occupied
    grid[start[0]][start[1]][0]= 0
         
    # set the robot location to the start
    robot = start
    
    # set the rhs value of the goal to 0
    grid[goal[0]][goal[1]][2] = 0
    
    # initialize the open list
    # The open list is a priority queue
    
    # The key of a node on the open list is min(g, rhs) + h
    # secondary key for tie-breaking is min(g, rhs)
    def calculate_key(cell):
        return min(grid[cell[0]][cell[1]][1], grid[cell[0]][cell[1]][2]) + grid[cell[0]][cell[1]][3]
  
        
    openlist = PriorityQueue( key = lambda x: calculate_key(x))
    

    
  
    # test priority queue
    cell = (1, 2)
    # set g to 1, h to 2, rhs to 3
    grid[1][2] = [0, 1, 3, 2]
    openlist.add(cell)
    
   
    print(openlist.top())
    print(openlist.topkey())   
    print("should be (1, 2) and 3") 
    
    cell = (2, 3)
    # set g to 1, h to 1, rhs to 1
    grid[2][3] = [0, 1, 1, 1]

    openlist.add(cell)
    
    print(openlist.top())
    print(openlist.topkey())
    print("should be (2, 3) and 2")
  
        
    
   
    

        
        
        
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