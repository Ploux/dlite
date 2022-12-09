# Compares the performance of three search algorithms on a variable-sized grid, with both stationary and moving obstacles
# The three search algorithms are: breadth-first search, A*, and D* Lite

# we run the algorithms as follows
# search(grid, heuristic)

# where grid is an array, consisting of a two-dimensional array of cells with 1 = occupied, 0 = empty, the start cell, the goal cell, the cell to block, and the time to block it. 
# For stationary grid problems where no cell is blocked, the block cell and time are both None
# and heuristics = "bfs", "astar", "dlite"

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



