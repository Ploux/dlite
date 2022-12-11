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
    """A queue in which the item with minimum key is always popped first."""

    def __init__(self, items=(), key=lambda x: x): 
        self.key = key
        self.items = [] # a heap of (key, item) pairs
        for item in items:
            self.add(item)
         
    def add(self, item):
        """Add item to the queue."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min key."""
        return heapq.heappop(self.items)[1]
    
    def top(self): 
        """Return the item with min key without popping."""
        return self.items[0][1]
        
    def topkey(self): 
        """Return the key of the item with min key without popping."""
        # if the queue is empty, return infinity
        if len(self.items) == 0:
            return math.inf
        return self.items[0][0]
        
    # check for a specific item in the queue
    def __contains__(self, item):
        return any(item == pair[1] for pair in self.items)
        

    def __len__(self): return len(self.items)    

# straight-line distance between two points
def straight_line_distance(A, B):
    "Straight-line distance between two points."
    return sum(abs(a - b)**2 for (a, b) in zip(A, B)) ** 0.5
    
# The key of a node on the open list is min(g, rhs) + h
def calculate_key(cell):
    return min(grid[cell[0]][cell[1]][1], grid[cell[0]][cell[1]][2]) + grid[cell[0]][cell[1]][3]

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
    def calculate_key(cell):
        return min(grid[cell[0]][cell[1]][1], grid[cell[0]][cell[1]][2]) + grid[cell[0]][cell[1]][3]
        
    def get_successors(cell):
        # return the list of valid neighbors of the cell (8-connected)
        # a neighbor is valid if it is not occupied and it is within the grid
        # a neighbor is a tuple of (x,y) coordinates
        neighbors = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                if (x != 0 or y != 0) and (cell[0] + x >= 0 and cell[0] + x < height) and (cell[1] + y >= 0 and cell[1] + y < width) and grid[cell[0] + x][cell[1] + y][0] == 0:
                    neighbors.append((cell[0] + x, cell[1] + y))
        return neighbors
    
    def update_vertex(cell):
        # if the cell is not the goal
        if cell != goal:
            # set rhs(cell) = min cost of all successors of cell
            # [occupancy, g, rhs, h]
            #   0         1   2   3
            # grid[cell[0]][cell[1]][2] = math.inf
            min_g = math.inf
            for s in get_successors(cell):
                if grid[s[0]][s[1]][1] < min_g:
                    min_g = grid[s[0]][s[1]][1]
                grid[cell[0]][cell[1]][2] = min_g + straight_line_distance(cell, s)
        # if the cell is in the open list, remove it
        if cell in openlist:
            openlist.remove(cell)
                

            
        
  
    def compute_shortest_path():
        # while topkey < key(start) or rhs(start) != g(start)
        while((openlist.topkey() < calculate_key(start) or grid[start[0]][start[1]][2] != grid[start[0]][start[1]][1])):
            # u = pop the top item from the open list
            u = openlist.pop()
            # if g(u) > rhs(u)
            if grid[u[0]][u[1]][1] > grid[u[0]][u[1]][2]:
                # set g(u) = rhs(u)
                grid[u[0]][u[1]][1] = grid[u[0]][u[1]][2]
                # for each successor (neighbor) of u
                for s in get_successors(u):
                    update_vertex(s)
            else:
                # set g(u) = infinity
                grid[u[0]][u[1]][1] = math.inf
                # for each successor (neighbor) of u
                for p in get_successors(u):
                    print()
                    # ??? 
        
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
    
    # test checking for a cell
    print(openlist.contains((1, 2)))
    print("should be True")
    print(openlist.contains((2, 3)))
    print("should be True")
    print(openlist.contains((3, 4)))
    print("should be False")
    
   
   
   
"""
    # test get_successors
    print(get_successors((1, 2)))
    # should be [(0, 1), (0, 2), (0, 3), (1, 1), (1, 3), (2, 1), (2, 2), (2, 3)]
    print("should be [(0, 1), (0, 2), (0, 3), (1, 1), (1, 3), (2, 1), (2, 2), (2, 3)]")
    print(get_successors((0, 0)))
    # should be [(0, 1), (1, 0), (1, 1)]
    print("should be [(0, 1), (1, 0), (1, 1)]")
    
    # assert that (3,4) is not a neighbor to (3,3) bc obstacle
    print(get_successors((3, 3)))
    print("should be [(2, 2), (2, 3), (2, 4), (3, 2), ]")
    
    # while the robot is not at the goal
    # while robot != goal:
    # compute_shortest_path()
        
        
    # check values for testing
    # topkey
    print(openlist.topkey())
    print("should be inf")
    
    # start cell's key
    print(calculate_key(start))
    print("should be inf")
    
    # rhs of start cell
    print(grid[start[0]][start[1]][2])
    print("should be inf")
    
    # g of start cell
    print(grid[start[0]][start[1]][1])
    print("should be inf")
    
    
    

"""

  
        
    
   
    

        
        
        
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