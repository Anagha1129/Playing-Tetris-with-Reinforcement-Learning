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

def initial_grid(self, height=0):
        grid = list()
        for _ in range(GAME_BOARD_HEIGHT):
            grid.append([0] * GAME_BOARD_WIDTH)

        if height == 0: return grid

        # if height = 15, range(6, 20), saving the first row for random generation
        for i in range(GAME_BOARD_HEIGHT - height, GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                grid[i][j] = self.get_random().randint(0, Tetromino.pool_size())
            grid[i][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        # for j in range(GAME_BOARD_WIDTH):
        #     grid[GAME_BOARD_HEIGHT - height][j] = self.get_random().randint(0, 1)

        return grid

 def get_random_grid(self):
        grid = list()
        for i in range(GAME_BOARD_HEIGHT):
            row = list()
            for j in range(GAME_BOARD_WIDTH):
                row.append(0)
            grid.append(row)

        height = self.get_random().randint(0, min(15, GAME_BOARD_HEIGHT - 2))

        # if height = 15, range(6, 20), saving the first row for random generation
        for i in range(GAME_BOARD_HEIGHT - height, GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                grid[i][j] = self.get_random().randint(0, len(Tetromino.pool_size()))
            grid[i][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        all_brick = True
        for j in range(GAME_BOARD_WIDTH):
            grid[GAME_BOARD_HEIGHT - height - 1][j] = self.get_random().randint(0, 1)
            if grid[GAME_BOARD_HEIGHT - height - 1][j] == 0: all_brick = False
        if all_brick: grid[GAME_BOARD_HEIGHT - height - 1][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        return grid

    @staticmethod
    def random_gamestate(seed=None):
        if seed is None:
            large_int = 999999999
            seed = random.randint(0, large_int)
        gamestate = Gamestate(seed=seed)
        gamestate.grid = gamestate.get_random_grid()
        return gamestate

    def copy(self):
        state_copy = Gamestate(self.grid, rd=self.rd)

        state_copy.seed = self.seed
        state_copy.tetromino = self.tetromino.copy()
        state_copy.hold_type = self.hold_type
        state_copy.next = list()
        for s in self.next:
            state_copy.next.append(s)
        state_copy.next_next = self.next_next
        state_copy.n_lines = list(self.n_lines)
        state_copy.t_spins = list(self.t_spins)
        state_copy.game_status = self.game_status
        state_copy.is_hold_last = self.is_hold_last
        state_copy.ex_weight = self.ex_weight
        state_copy.score = self.score
        state_copy.lines = self.lines
        state_copy.pieces = self.pieces
        state_copy.rand_count = self.rand_count
        state_copy.idle = self.idle
        state_copy.combo = self.combo

        return state_copy

    def copy_value(self, state_original):
        for i in range(GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                self.grid[i][j] = state_original.grid[i][j]

        self.seed = state_original.seed
        self.tetromino = state_original.tetromino.copy()
        self.hold_type = state_original.hold_type
        for i in range(len(self.next)):
            self.next[i] = state_original.next[i]
        self.next_next = state_original.next_next
        self.n_lines = list(state_original.n_lines)
        self.t_spins = list(state_original.t_spins)
        self.game_status = state_original.game_status
        self.is_hold_last = state_original.is_hold_last
        self.ex_weight = state_original.ex_weight
        self.score = state_original.score
        self.lines = state_original.lines
        self.pieces = state_original.pieces
        self.rand_count = state_original.rand_count
        self.idle = state_original.idle
        self.combo = state_original.combo

    def put_tet_to_grid(self, tetro=None):
        grid_copy = helper.copy_2d(self.grid)
        if tetro is None:
            tetro = self.tetromino

        disp = tetro.get_displaced()
        for sq in disp:
            x = sq[0]
            y = sq[1]
            if x < 0 or x > GAME_BOARD_WIDTH or y > GAME_BOARD_HEIGHT:
                continue
            if y < 0:
                continue
            grid_copy[y][x] = tetro.to_num()
        return grid_copy

    def check_collision(self, tetro=None):
        if tetro is None:
            tetro = self.tetromino

        disp = tetro.get_displaced()
        for sq in disp:
            x = sq[0]
            y = sq[1]
            if x < 0 or x >= GAME_BOARD_WIDTH or y >= GAME_BOARD_HEIGHT:
                return True
            if y < 0:
                continue
            if self.grid[y][x] != 0:
                return True
        return False

    def check_t_spin(self):
        if self.tetromino.type_str != "T" or self.tetromino.rot != 2: return False
        check_mov = [(0, -1, 0),
                     (1, 0, 0),
                     (-1, 0, 0)]
        for mov in check_mov:
            tetro = self.tetromino.copy().move(mov)
            if not self.check_collision(tetro): return False

        return True
