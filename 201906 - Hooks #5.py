'''
This code is terrible; don't bother looking at it

Notes for myself
- took 4 days to run + extra manual work on good configs
- "one island" constraint is instead a "one neighbor" constraint
- "every 2-by-2 region must contain at least one unfilled square" constraint is not coded
'''

import pickle
import os
import sys
from z3 import *
from itertools import combinations_with_replacement
from collections import namedtuple
from functools import reduce
import operator
from enum import Enum, auto
sys.path.append('/utils/z3/bin/python')

# define utility classes
point = namedtuple('Point', 'x y')

class Corner(Enum):
    TOP_LEFT = auto(),
    TOP_RIGHT = auto(),
    BOTTOM_LEFT = auto(),
    BOTTOM_RIGHT = auto(),
    
class Side(Enum):
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()

# define utility variables
board_size = 9
cell=[[Int(f"cell{x}{y}") for y in range(board_size)] for x in range(board_size)]

def all_hook_configs():
    starting_corners = {
        Corner.TOP_LEFT:point(0,0),
        Corner.TOP_RIGHT:point(0,board_size-1),
        Corner.BOTTOM_LEFT:point(board_size-1,0),
        Corner.BOTTOM_RIGHT:point(board_size-1, board_size-1),
    }
    curr_configs = [({}, starting_corners)]
    
    for i in range(board_size, 0, -1):
        new_configs = []
        while curr_configs:
            curr_config, corners = curr_configs.pop()
            # generate the four i-hooks and add to new_configs
            top_left_centered_hook = []
            for j in range(corners[Corner.TOP_LEFT].y, corners[Corner.TOP_RIGHT].y+1):
                top_left_centered_hook.append(cell[corners[Corner.TOP_LEFT].x][j])
            for j in range(corners[Corner.TOP_LEFT].x+1, corners[Corner.BOTTOM_LEFT].x+1):
                top_left_centered_hook.append(cell[j][corners[Corner.TOP_LEFT].y])
            top_left_centered_hook_new_corners = {
                Corner.TOP_LEFT: point(corners[Corner.TOP_LEFT].x+1, corners[Corner.TOP_LEFT].y+1),
                Corner.TOP_RIGHT: point(corners[Corner.TOP_RIGHT].x+1, corners[Corner.TOP_RIGHT].y),
                Corner.BOTTOM_LEFT: point(corners[Corner.BOTTOM_LEFT].x, corners[Corner.BOTTOM_LEFT].y+1),
                Corner.BOTTOM_RIGHT: corners[Corner.BOTTOM_RIGHT],
            }
            new_dict = curr_config.copy()
            new_dict[i] = top_left_centered_hook
            new_configs.append((new_dict, top_left_centered_hook_new_corners))

            top_right_centered_hook = []
            for j in range(corners[Corner.TOP_LEFT].y, corners[Corner.TOP_RIGHT].y+1):
                top_right_centered_hook.append(cell[corners[Corner.TOP_LEFT].x][j])
            for j in range(corners[Corner.TOP_RIGHT].x+1, corners[Corner.BOTTOM_RIGHT].x+1):
                top_right_centered_hook.append(cell[j][corners[Corner.TOP_RIGHT].y])
            top_right_centered_hook_new_corners = {
                Corner.TOP_LEFT: point(corners[Corner.TOP_LEFT].x+1, corners[Corner.TOP_LEFT].y),
                Corner.TOP_RIGHT: point(corners[Corner.TOP_RIGHT].x+1, corners[Corner.TOP_RIGHT].y-1),
                Corner.BOTTOM_LEFT: corners[Corner.BOTTOM_LEFT],
                Corner.BOTTOM_RIGHT: point(corners[Corner.BOTTOM_RIGHT].x, corners[Corner.BOTTOM_RIGHT].y-1),
            }
            new_dict = curr_config.copy()
            new_dict[i] = top_right_centered_hook
            new_configs.append((new_dict, top_right_centered_hook_new_corners))

            bottom_left_centered_hook = []
            for j in range(corners[Corner.TOP_LEFT].x, corners[Corner.BOTTOM_LEFT].x+1):
                bottom_left_centered_hook.append(cell[j][corners[Corner.BOTTOM_LEFT].y])
            for j in range(corners[Corner.BOTTOM_LEFT].y+1, corners[Corner.BOTTOM_RIGHT].y+1):
                bottom_left_centered_hook.append(cell[corners[Corner.BOTTOM_LEFT].x][j])
            bottom_left_centered_hook_new_corners = {
                Corner.TOP_LEFT: point(corners[Corner.TOP_LEFT].x, corners[Corner.TOP_LEFT].y+1),
                Corner.TOP_RIGHT: corners[Corner.TOP_RIGHT],
                Corner.BOTTOM_LEFT: point(corners[Corner.BOTTOM_LEFT].x-1, corners[Corner.BOTTOM_LEFT].y+1),
                Corner.BOTTOM_RIGHT: point(corners[Corner.BOTTOM_RIGHT].x-1, corners[Corner.BOTTOM_RIGHT].y),
            }
            new_dict = curr_config.copy()
            new_dict[i] = bottom_left_centered_hook
            new_configs.append((new_dict, bottom_left_centered_hook_new_corners))

            bottom_right_centered_hook = []
            for j in range(corners[Corner.BOTTOM_LEFT].y, corners[Corner.BOTTOM_RIGHT].y+1):
                bottom_right_centered_hook.append(cell[corners[Corner.BOTTOM_RIGHT].x][j])
            for j in range(corners[Corner.TOP_RIGHT].x, corners[Corner.BOTTOM_RIGHT].x):
                bottom_right_centered_hook.append(cell[j][corners[Corner.BOTTOM_RIGHT].y])
            bottom_right_centered_hook_new_corners = {
                Corner.TOP_LEFT: corners[Corner.TOP_LEFT],
                Corner.TOP_RIGHT: point(corners[Corner.TOP_RIGHT].x, corners[Corner.TOP_RIGHT].y-1),
                Corner.BOTTOM_LEFT: point(corners[Corner.BOTTOM_LEFT].x-1, corners[Corner.BOTTOM_LEFT].y),
                Corner.BOTTOM_RIGHT: point(corners[Corner.BOTTOM_RIGHT].x-1, corners[Corner.BOTTOM_RIGHT].y-1),
            }
            new_dict = curr_config.copy()
            new_dict[i] = bottom_right_centered_hook
            new_configs.append((new_dict, bottom_right_centered_hook_new_corners))

        curr_configs = new_configs
    
    return curr_configs

def is_one_island(cells):
    # get the first non-zero island, then sink them all
    j = next((j for j, x in enumerate(cells[0]) if x), None)
    sink_islands(cells, 0, j)
    return all(cells[i][j]==0 for i in range(board_size) for j in range(board_size))
    
def sink_islands(cells, i, j):
    if i<0 or j<0 or i>=board_size or j>=board_size or cells[i][j]==0: return
    cells[i][j]=0
    
    for a,b in ((-1,0), (1,0), (0,-1), (0,1)):
        sink_islands(cells, i+a, j+b)

def get_zero_islands(cells):
    output = []
    
    # visit every island
    for i in range(board_size):
        for j in range(board_size):
            # if is 0, then DFS to expand the island and flag as *
            if cells[i][j]==0:
                count = 0
                stack = [(i,j)]
                while stack:
                    x,y = stack.pop()
                    if x<0 or y<0 or x>=board_size or y>=board_size or cells[x][y]!=0:
                        continue
                    count += 1
                    cells[x][y]='*'
                    for a,b in ((-1,0), (1,0), (0,-1), (0,1)):
                        stack.append((x+a,y+b))
            
                output.append(count)

    return output

def solve_config(config):
    # given a configuration, set Z3 constraints
    s = Solver()
    
    # must be <int> or 0
    for k,v in config.items():
        for c in v:
            s.add(Or(c==k, c==0))
            
    # must sum to <int>^2
    for k,v in config.items():
        s.add(Sum([a for a in v])==k**2)
    
    # must satisfy outside sums 
    def sum_constraint(i, n, side):
        if side == Side.LEFT:
            condition = Or([And(
                *[cell[i][x]==0 for x in range(a)],
                Sum([cell[i][x] for x in range(a, b+1)])==n,
                *[cell[i][x]!=0 for x in range(a, b+1)],
                *[cell[i][x]==0 for x in range(b+1, min(b+2,board_size))],
            ) for a,b in combinations_with_replacement(range(board_size), 2)])
        elif side == Side.RIGHT:
            condition = Or([And(
                *[cell[i][x]==0 for x in range(board_size-1,a,-1)],
                Sum([cell[i][x] for x in range(a, b-1, -1)])==n,
                *[cell[i][x]!=0 for x in range(a, b-1, -1)],
                *[cell[i][x]==0 for x in range(b-1, max(b-2,-1), -1)],
            ) for a,b in combinations_with_replacement(range(board_size-1,-1,-1), 2)])
        elif side == Side.TOP:
            condition = Or([And(
                *[cell[x][i]==0 for x in range(a)],
                Sum([cell[x][i] for x in range(a, b+1)])==n,
                *[cell[x][i]!=0 for x in range(a, b+1)],
                *[cell[x][i]==0 for x in range(b+1, min(b+2,board_size))],
            ) for a,b in combinations_with_replacement(range(board_size), 2)])
        elif side == Side.BOTTOM:
            condition = Or([And(
                *[cell[x][i]==0 for x in range(board_size-1,a,-1)],
                Sum([cell[x][i] for x in range(a, b-1, -1)])==n,
                *[cell[x][i]!=0 for x in range(a, b-1, -1)],
                *[cell[x][i]==0 for x in range(b-1, max(b-2,-1), -1)],
            ) for a,b in combinations_with_replacement(range(board_size-1,-1,-1), 2)])
        else:
            raise ValueError(f"passed value of slide is invalid: {side}")
        return condition
    
    sum_constraints = [
        sum_constraint(0, 41, Side.TOP),
        sum_constraint(1, 8, Side.TOP),
        sum_constraint(4, 14, Side.TOP),
        sum_constraint(6, 15, Side.TOP),
        
        sum_constraint(1, 9, Side.BOTTOM),
        sum_constraint(3, 17, Side.BOTTOM),
        sum_constraint(5, 15, Side.BOTTOM),
        sum_constraint(7, 35, Side.BOTTOM),
        
        sum_constraint(2, 25, Side.LEFT),
        sum_constraint(4, 15, Side.LEFT),
        sum_constraint(6, 26, Side.LEFT),
        
        sum_constraint(0, 25, Side.RIGHT),
        sum_constraint(4, 10, Side.RIGHT),
        sum_constraint(8, 27, Side.RIGHT),
    ]
    s.add(sum_constraints)
    
    # must be one island
    for x in range(board_size):
        for y in range(board_size):
            s.add(If(cell[x][y] != 0,
                     # must have a non-zero neighbor
                     Or(
                         # broken because multiple islands
                         And(x>0, cell[x-1][y]!=0),
                         And(y>0, cell[x][y-1]!=0),
                         And(x<board_size-1, cell[min(board_size-1,x+1)][y]!=0),
                         And(y<board_size-1, cell[x][min(board_size-1,y+1)]!=0),
                     ),
                     True))
    
    # print first solution
    # fix ugly printing of solution
#    if s.check() == sat:
#        m = s.model()
#        r = [[m.evaluate(cell[i][j]) for j in range(board_size)] for i in range(board_size)]
#        print_matrix(r)
#        print(50*'-')
        
    # enumerate all possible solutions:
    results=[]
    while True:
        if s.check() == sat:
            m = s.model()
            r = [[m.evaluate(cell[i][j]).as_long() for j in range(board_size)] for i in range(board_size)]
            
            if is_one_island(r):
                r = [[m.evaluate(cell[i][j]).as_long() for j in range(board_size)] for i in range(board_size)]
                print_matrix(r)
                
                # print the solution, the product of the 0-islands
                zero_islands = get_zero_islands(r)
#                print_matrix(r)
                print(f"zero_islands: {'*'.join(list(map(str,zero_islands)))} ==> {reduce(operator.mul, zero_islands, 1)}")
                
                print('\n')
                results.append(m)
                
            block = []
            for d in m:
                c=d()
                block.append(c != m[d])
            s.add(Or(block))
        else:
            print(f"results total={len(results)}")
            break

all_configs = all_hook_configs()

print(f"len(all_configs): {len(all_configs)}")

i = 0
for config,_ in all_configs:
    if i not in range(247684, 247688):
        i +=1 
        continue
#    if not i % 1000: print(f"Processing config #{i}")
    print(f"Processing config #{i}")
    solve_config(config)
    i += 1
