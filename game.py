import sys

import pygame

from lib import helper
from tetromino import Tetromino
import random
import numpy as np
from gui import Gui
import time
from common import *

INITIAL_EX_WIGHT = 0.0
SPIN_SHIFT_FOR_NON_T = [(1, 0, 0), (-1, 0, 0),
                        (0, 1, 0), (0, -1, 0),
                        (1, 1, 0), (-1, 1, 0),
                        (1, -1, 0), (-1, -1, 0),
                        (0, 2, 0), (1, 2, 0), (-1, 2, 0)]

# if you don't want to see some spurious t-spin moves
SPIN_SHIFT_FOR_T = [(1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (1, 1, 0), (-1, 1, 0),
                    (1, -1, 0), (-1, -1, 0),
                    (0, 2, 0), (1, 2, 0), (-1, 2, 0)]  # disable triple t-spin

# if you allow t spins in various funky ways
# SPIN_SHIFT_FOR_T = [(1, 0, 0), (-1, 0, 0),
#                     (0, 1, 0), (1, 1, 0), (-1, 1, 0),
#                     (0, 2, 0), (1, 2, 0), (-1, 2, 0),
#                     (0, -1, 0), (1, -1, 0), (-1, -1, 0)]  # enable triple t-spin

ACTIONS = [
    "left", "right", "down", "turn left", "turn right", "drop"
]

IDLE_MAX = 9999


class Gamestate:
    def __init__(self, grid=None, seed=None, rd=None, height=0):
        if seed is None:
            self.seed = random.randint(0, round(9e9))
        else:
            self.seed = seed
        self.rand_count = 0

        if rd is None:
            self.rd = random.Random(seed)
        else:
            self.rd = rd

        if grid is None:
            self.grid = self.initial_grid(height)
        else:
            self.grid = list()
            for row in grid:
                self.grid.append(list(row))

        self.tetromino = Tetromino.new_tetromino_fl(self.get_random().random())
        self.hold_type = None
        self.next = list()
        for i in range(5):
            self.next.append(Tetromino.random_type_str(self.get_random().random()))
        self.next_next = Tetromino.random_type_str(self.get_random().random())
        self.n_lines = [0, 0, 0, 0]
        self.t_spins = [0, 0, 0, 0]
        self.game_status = "playing"
        self.is_hold_last = False
        self.ex_weight = INITIAL_EX_WIGHT
        self.score = 0
        self.lines = 0
        self.pieces = 0
        self.idle = 0
    
 def start(self):
        self.tetromino = Tetromino.new_tetromino(self.next[0])
        self.next[:-1] = self.next[1:]
        self.next[-1] = self.next_next
        self.next_next = Tetromino.random_type_str(self.get_random().random())
