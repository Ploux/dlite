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
    
    # remove a specific item from the queue
    def remove(self, item):
        for pair in self.items:
            if pair[1] == item:
                self.items.remove(pair)
                heapq.heapify(self.items)
                return True
        return False
    
    # update the key of an item in the queue
    def update(self, item, newkey):
        for pair in self.items:
            if pair[1] == item:
                self.items.remove(pair)
                heapq.heapify(self.items)
                self.add(item)
                return True
        return False
    
        
    # print the queue
    def __repr__(self):
        return 'PriorityQueue({0})'.format(self.items)
        
        

    def __len__(self): return len(self.items)    

# straight-line distance between two points
def straight_line_distance(A, B):
    "Straight-line distance between two points."
    return sum(abs(a - b)**2 for (a, b) in zip(A, B)) ** 0.5
    
def draw_grid(title, height, width, start, goal, obstacles, path = [], traveled = [], blocked = None, robot = None):
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
    
    
    # draw the path cyan
    for cell in path:
        if path != goal:
            plt.fill([cell[1], cell[1]+1, cell[1]+1, cell[1]], [height-1-cell[0], height-1-cell[0], height-cell[0], height-cell[0]], 'c')
            
    # draw the traveled path blue
    for cell in traveled:
        plt.fill([cell[1], cell[1]+1, cell[1]+1, cell[1]], [height-1-cell[0], height-1-cell[0], height-cell[0], height-cell[0]], 'b')
            
    # draw the detected blocked cell gray
    if blocked != None:
        plt.fill([blocked[1], blocked[1]+1, blocked[1]+1, blocked[1]], [height-1-blocked[0], height-1-blocked[0], height-blocked[0], height-blocked[0]], 'gray')
        
    # color the goal green if not reached, yellow if reached
    if goal not in traveled:
        plt.fill([goal[1], goal[1]+1, goal[1]+1, goal[1]], [height-1-goal[0], height-1-goal[0], height-goal[0], height-goal[0]], 'g')
    else:
        plt.fill([goal[1], goal[1]+1, goal[1]+1, goal[1]], [height-1-goal[0], height-1-goal[0], height-goal[0], height-goal[0]], 'y')
        
    
        
    # draw the robot
    # the robot is a circle with diameter = cell size
    # with an X in the middle
    if robot is not None:
        # circle
        circle = plt.Circle((robot[1]+0.5, height-1-robot[0]+0.5), 0.5, color='k', fill=False)
        plt.gca().add_patch(circle)
        # X        
        plt.plot([robot[1]+0.25, robot[1]+0.75], [height-1-robot[0]+0.25, height-1-robot[0]+0.75], 'k-')
        plt.plot([robot[1]+0.25, robot[1]+0.75], [height-1-robot[0]+0.75, height-1-robot[0]+0.25], 'k-')
        
    # title the plot
    plt.title(title)
 
    # show the plot
    plt.show()

# A* Search
def astar(height, width, start, goal, obstacles, blocks=[]):
    # the start and goal are tuples of (x,y) coordinates
    # the obstacles and block are lists of tuples of (x,y) coordinates
    
    
    # The key of a node on the open list is f = g + h
    def calculate_key(cell):
        # return g(cell) + h(cell)
        return grid[cell[0]][cell[1]][1] + straight_line_distance(cell, start)
        
    
    
        
    def get_successors(cell):
        # return the list of valid neighbors of the cell (8-connected)
        # a neighbor is valid if it is not occupied and it is within the grid
        # a neighbor is a tuple of (x,y) coordinates
        neighbors = []
        for x in range(cell[0]-1, cell[0]+2):
            for y in range(cell[1]-1, cell[1]+2):
                if (x,y) != cell and x >= 0 and x < height and y >= 0 and y < width and (x,y) not in obstacles:
                    neighbors.append((x,y))
        return neighbors
        
    def compute_shortest_path():
        # keep track of number of pops and expansions
        pops = 0
        expansions = 0
        # [occupancy, g,  h]
        #   0         1   2  
        # while priority queue is not empty
        
        # for i in range(7):
        while len(open_list) > 0:
            # print topkey
            # print("Topkey is", open_list.topkey())
            # print(open_list)
            # pop the cell with the lowest f value
            u = open_list.pop()
            pops += 1
            '''
            print("Popping", u)
            # print g value of popped cell
            # print("g value of popped cell is", grid[u[0]][u[1]][1])
            # print g and h values for each cell
            # rounded to the nearest tenth
            for x in range(height):
                for y in range(width):
                    print ("[", round(grid[x][y][1],1), round(grid[x][y][2],1), "]", end = " ")
                print()
            print()
            print(get_successors(u))
            print()
            '''
            for s in get_successors(u):
                
                # if is the start, return the path
                if s == start:
                    grid[s[0]][s[1]][1] = grid[u[0]][u[1]][1] + straight_line_distance(u, s)
                    print("start reached!")
                    # return how many cells were popped and how many cells were expanded
                    return [pops, expansions]
                # update g value
                # if g is infinity, then set it to the g value of the current cell + the cost of the edge
                
                if grid[s[0]][s[1]][1] == float('inf'):
                    expansions += 1
                    # print("infinity")
                    grid[s[0]][s[1]][1] = grid[u[0]][u[1]][1] + straight_line_distance(u, s)
                    # print("Updating g value of", s, "to", grid[s[0]][s[1]][1])
                    # update the f value
                    open_list.update(s, calculate_key(s))
                    # print("Updating f value of", s, "to", calculate_key(s))
                    
                    # if the cell is not in the open list
                    if s not in open_list:
                        # add it to the open list
                        open_list.add(s)
                        # print("Adding", s, "to the open list")
                
                # if the g value of the neighbor is greater than the g value of the current cell + the cost of the edge
                if grid[s[0]][s[1]][1] > grid[u[0]][u[1]][1] + straight_line_distance(u, s):
                    # update the g value
                    grid[s[0]][s[1]][1] = grid[u[0]][u[1]][1] + straight_line_distance(u, s)
                    # print("Updating g value of", s, "to", grid[s[0]][s[1]][1])
                    # update the f value
                    open_list.update(s, calculate_key(s))
                    # print("Updating f value of", s, "to", calculate_key(s))
                    
        print("No path found")            
                    
    def trace_path():   
        # trace the path
        
        # print g and h values for each cell
        # rounded to the nearest tenth
        for x in range(height):
            for y in range(width):
                print ("[", round(grid[x][y][1],1), round(grid[x][y][2],1), "]", end = " ")
            print()
        print()
        
        if grid[robot[0]][robot[1]][1] == math.inf:
            print("No path found")
            return
        path = []
        cell = robot
        # check each neighbor of the robot
        while cell != goal:
            # find the neighbor with the lowest g value
            min_g = math.inf
            for s in get_successors(cell):
                if grid[s[0]][s[1]][1] < min_g:
                    min_g = grid[s[0]][s[1]][1]
                    next_cell = s
                    
            # add the neighbor to the path
            # print ("Next cell ", next_cell, " G value ", min_g)
            if next_cell != goal:
                path.append(next_cell)
            # move the robot to the neighbor
            cell = next_cell
            
        # add goal to path at the end
        path.append(goal)
        return path                   

     
    # initialization
    # keep track of pops and expansions
    pops = 0
    expansions = 0
    
    # initialize the grid
    grid = [[0 for x in range(width)] for y in range(height)]
    
    # the grid is a 2D array of cells
    # each cell contains occupancy, g, and h values
    # in the form of a list [occupancy, g, h]
    # occupancy = 1 = occupied, 0 = empty
    # g = the cost of the cheapest path from the start to the cell
    # h = the heuristic cost of the cheapest path from the cell to the goal
        
    # set the g values to infinity
    for x in range(height):
        for y in range(width):
            grid[x][y] = [0, math.inf, math.inf]
            
    # set the h values to straight-line distance from the cell to the goal
    for x in range(height):
        for y in range(width):
            grid[x][y][2] = straight_line_distance((x, y), goal)
    
    # set the obstacles
    for obstacle in obstacles:
        grid[obstacle[0]][obstacle[1]] = [1, math.inf, math.inf]
    
    # make sure the start is not occupied
    grid[start[0]][start[1]][0]= 0
    
    # set the g value of the goal to 0
    grid[goal[0]][goal[1]][1] = 0
         
    # set the robot location to the start
    robot = start
    
    # draw the initial grid
    draw_grid("Initial A*", height, width, start, goal, obstacles)
    
    '''
    # print g and h values for each cell
    # rounded to the nearest tenth
    for x in range(height):
        for y in range(width):
            print ("[", round(grid[x][y][1],1), round(grid[x][y][2],1), "]", end = " ")
        print()
    '''
    
    
    # initialize a priority queue and put the goal cell on it
    open_list = PriorityQueue( key = lambda x: calculate_key(x))
        
    # add the goal to the open list
    open_list.add(goal)
    
    # compute the shortest path
    # adding to our pops and expansions
    temp_count = compute_shortest_path()
    pops += temp_count[0]
    expansions += temp_count[1]
    
    # trace the path
    path = trace_path()
    
    # draw the grid
    draw_grid("Plan", height, width, start, goal, obstacles, path)
            
    # navigation phase
   

# D* Lite

def dstarlite(height, width, start, goal, obstacles, blocks=[]):
    # the start and goal are tuples of (x,y) coordinates
    # the obstacles and block are lists of tuples of (x,y) coordinates
    
    # The key of a node on the open list is min(g, rhs) + h
    def calculate_key(cell):
        # return min(g(cell), rhs(cell)) + h(cell)      
        return min(grid[cell[0]][cell[1]][1], grid[cell[0]][cell[1]][2]) + straight_line_distance(cell, start)
        
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
            #print("Updating", cell)
            # set rhs(cell) = min cost of all successors of cell
            min_g = math.inf
            # set old_rhs = cell's rhs
            old_rhs = grid[cell[0]][cell[1]][2]
            # print(get_successors(cell))
            for s in get_successors(cell):
                # print("s:", s, "g:", grid[s[0]][s[1]][1], "rhs:", grid[s[0]][s[1]][2])
                if grid[s[0]][s[1]][1] < min_g:
                    min_g = grid[s[0]][s[1]][1]
                    # print("new min_g:", min_g, "at:", s)
                    temp_rhs = min_g + straight_line_distance(cell, s)
                # print("straight line distance between cell ", cell, "and s", s, "is", straight_line_distance(cell, s))
            if temp_rhs != old_rhs:
                grid[cell[0]][cell[1]][2] = temp_rhs
                #print("Updating", cell, "rhs:", grid[cell[0]][cell[1]][2])
        # if the cell is in the open list, remove it
        # if cell in openlist:
            # openlist.remove(cell)
            # print("Removing", cell)
        # if g(cell) != rhs(cell), insert cell into the open list
        if grid[cell[0]][cell[1]][1] != grid[cell[0]][cell[1]][2]:
            # if cell not in openlist:
            if cell not in openlist:
                openlist.add(cell)
                # print("Adding", cell)
  
    def compute_shortest_path():
        # keep track of number of pops and expansions
        pops = 0
        expansions = 0
        # [occupancy, g, rhs, h]
        #   0         1   2   3
        # while topkey < key(robot) or rhs(robot) != g(robot)
        #for i in range(3):
        while((openlist.topkey() < calculate_key(robot) or grid[robot[0]][robot[1]][2] != grid[robot[0]][robot[1]][1])):
            # u = pop the top item from the open list
            #print ("openlist:", openlist)
            u = openlist.pop()
            pops += 1
            #print("Popping", u)
            # if g(u) > rhs(u) (overconsistent)
            if grid[u[0]][u[1]][1] > grid[u[0]][u[1]][2]:
                #print("Overconsistent")
                # set g(u) = rhs(u)
                grid[u[0]][u[1]][1] = grid[u[0]][u[1]][2]
                #print("Updating", u, "g:", grid[u[0]][u[1]][1])
                # for each successor (neighbor) of u
                for s in get_successors(u):
                    expansions += 1
                    update_vertex(s)
            else:
                # set g(u) = infinity
                grid[u[0]][u[1]][1] = math.inf
                
                # expand the popped cell
                for p in get_successors(u):
                    # if p is inconsistent g != rhs
                    expansions += 1
                    if grid[p[0]][p[1]][1] != grid[p[0]][p[1]][2]:
                        # put p into the open list
                        if p not in openlist:
                            openlist.add(p)
                            # print("Adding", p)
                # call update_vertex on the popped cell
                update_vertex(u)
        # return how many cells were popped and how many cells were expanded
        return [pops, expansions]
                
    def trace_path():   
        # trace the path
        if grid[robot[0]][robot[1]][1] == math.inf:
            print("No path found")
            return
        path = []
        cell = robot
        # check each neighbor of the robot
        while cell != goal:
            # find the neighbor with the lowest g value
            min_g = math.inf
            for s in get_successors(cell):
                if grid[s[0]][s[1]][1] < min_g:
                    min_g = grid[s[0]][s[1]][1]
                    next_cell = s
                    
            # add the neighbor to the path
            # print ("Next cell ", next_cell, " G value ", min_g)
            if next_cell != goal:
                path.append(next_cell)
            # move the robot to the neighbor
            cell = next_cell
            
        # add goal to path at the end
        path.append(goal)
        return path
    

       
    # initialization
    # keep track of pops and expansions
    pops = 0
    expansions = 0
    
    # draw the initial grid
    draw_grid("Initial", height, width, start, goal, obstacles)
    
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
    # print("Initializing open list")
    openlist = PriorityQueue( key = lambda x: calculate_key(x))
    # add the goal to the open list
    openlist.add(goal)
    
    # print("Adding goal", goal)
    
    # compute the shortest path
    # adding to our pops and expansions
    temp_count = compute_shortest_path()
    pops += temp_count[0]
    expansions += temp_count[1]
    
    '''
    # print g, rhs, and h values for each cell
    # rounded to the nearest tenth
    for x in range(height):
        for y in range(width):
            print ("[", round(grid[x][y][1],1), round(grid[x][y][2],1), round(grid[x][y][3],1), "]", end = " ")
        print()
    '''
    path = trace_path()
    
    # draw the grid
    draw_grid("Plan", height, width, start, goal, obstacles, path)
            
    # navigation phase
    
    # examine the path
    # print("Path:", path)
    
    
    
            
    # move robot along path
    traveled = []
    blocked = None
    while robot != goal:
        # find the next cell in the path
        next_cell = path[0]
        # if it's not blocked, move there
        if next_cell not in blocks:
            # move the robot
            robot = next_cell
            # add the cell to the traveled list
            traveled.append(robot)
            # remove the cell from the path
            path.remove(robot)
        else:
            # we have to replan
            # mark the cell as blocked
            blocked = next_cell
            # print grid
            draw_grid("Blocked!", height, width, start, goal, obstacles, path, traveled, blocked, robot)
            # remove the cell from the path
            path.remove(next_cell)
            # flag it as occupied
            grid[next_cell[0]][next_cell[1]][0] = 1
            # raise the blocked cell's rhs value to infinity
            grid[next_cell[0]][next_cell[1]][2] = math.inf
            # put the blocked cell in the open list
            openlist.add(next_cell)
            # print("Adding", next_cell)
            # print open list
            # print("Open list:", openlist)
            # run update vertex on the neighbors of the blocked cell
            # if they are not obstacles
            for s in get_successors(next_cell):
                if grid[s[0]][s[1]][0] == 0:
                    update_vertex(s)
            # compute the shortest path
            # print("Replanning")
            temp_count = compute_shortest_path()
            pops += temp_count[0]
            expansions += temp_count[1]
            # clear planned path
            path = []
            # trace the path
            path = trace_path()
            # move blocked cell to obstacles
            obstacles.append(blocked)
            # flag it as occupied
            grid[next_cell[0]][next_cell[1]][0] = 1
            # clear blocked cell
            blocked = None
            # print grid
            draw_grid("Replanned", height, width, start, goal, obstacles, path, traveled, blocked, robot)
     
            
    # print grid
    draw_grid("Goal Reached", height, width, start, goal, obstacles, path, traveled, blocked, robot)
    
    # print the report
    print("Path Length:", len(traveled))
    print("Pops:", pops)
    print("Expansions:", expansions)
    
          
# test

# a small grid, only 4x5 cells
# The start is on the right side, three cells up from the bottom
# The goal is in the lower left corner
# 3 obstacles
# block in (1,2)

# dstarlite(4, 5, (1,4), (3,0), [(2,1), (2,2), (3,2)], [(1,2)])
astar(4, 5, (1,4), (3,0), [(2,1), (2,2), (3,2)], [(1,2)])
