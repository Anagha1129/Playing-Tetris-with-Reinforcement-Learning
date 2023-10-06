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

 def check_completed_lines(self, above_grid=None):
        completed_lines = 0
        row_num = 0
        for row in self.grid:
            complete = True
            for sq in row:
                if sq == 0:
                    complete = False
                    break
            if complete:
                self.remove_line(row_num, above_grid=above_grid)
                completed_lines += 1
            row_num += 1

        return completed_lines

    def remove_line(self, row_num, above_grid=None):
        self.grid[1:row_num + 1] = self.grid[:row_num]
        if above_grid is None:
            new_row = [0] * GAME_BOARD_WIDTH
        else:
            new_row = above_grid[:]
        self.grid[0] = new_row

    def check_clear_board(self):
        for i in reversed(range(GAME_BOARD_HEIGHT)):
            for block in self.grid[i]:
                if block != 0:
                    return False
        return True

    def update_score(self, lines, is_t_spin, is_clear, combo):
        if is_t_spin:
            if lines == 1:
                score_lines = 2
            elif lines == 2:
                score_lines = 4
            elif lines == 3:
                score_lines = 5
            else:
                score_lines = 0
            self.t_spins[lines] += 1
        else:
            score_lines = lines

        add_score = (score_lines + 1) * score_lines / 2 * 10

        if is_clear:
            add_score += 60

        if T_SPIN_MARK and is_t_spin:
            self.score = int(self.score) + add_score + 0.1
            add_score += 0.1
        else:
            self.score += add_score
        self.lines += lines

        if add_score != 0:
            if 1 < combo <= 3:
                self.score += 10
            elif 3 < combo <= 5:
                self.score += 20
            elif 5 < combo <= 8:
                self.score += 30
            elif combo > 8:
                self.score += 40

        if lines != 0: self.n_lines[lines - 1] += 1
        self.pieces += 1
        return add_score

    def get_score_text(self):
        s = "score:  " + str(int(self.score)) + "\n"
        s += "lines:  " + str(int(self.lines)) + "\n"
        s += "pieces: " + str(self.pieces) + "\n"
        one_line = ''
        for num in self.n_lines:
            one_line += f'{num} '
        s += "n_lines: " + one_line + '\n'
        one_line = ''
        for num in self.t_spins:
            one_line += f'{num} '
        s += "t_spins: " + one_line + '\n'
        s += "combo: " + f'{self.combo}\n'
        return s

    def get_info_text(self):
        # s = "unfinished info text \n"
        s = "seed: " + str(self.seed)
        return s

    def soft_drop(self):
        tetro = self.tetromino
        down = 0
        while not self.check_collision(tetro.move((0, 1, 0))): down += 1
        tetro.move((0, -1, 0))
        return down

    def hard_drop(self):
        self.soft_drop()
        return self.process_down_collision()

    def process_down_collision(self):
        is_t_spin = self.check_t_spin()
        is_above_grid = self.tetromino.check_above_grid()
        above_grid = self.tetromino.to_above_grid()
        self.freeze()
        completed_lines = self.check_completed_lines(above_grid=above_grid)
        is_clear = self.check_clear_board()
        add_score = self.update_score(completed_lines, is_t_spin, is_clear, self.combo)
        if add_score == 0:
            self.combo = 0
        else:
            self.combo += 1

        if self.check_collision() or (is_above_grid and completed_lines == 0):
            self.game_status = "gameover"
            done = True
        else:
            done = False
        return add_score, done

    def process_turn(self):  # return true if turn is successful
        if self.check_collision():
            success = False
            shifted = None
            if self.tetromino.type_str.lower() == 't':
                spin_moves = SPIN_SHIFT_FOR_T
            else:
                spin_moves = SPIN_SHIFT_FOR_NON_T
            for mov in spin_moves:
                shifted = self.tetromino.copy().move(mov)
                if not self.check_collision(shifted):
                    success = True
                    break
            if success:
                self.tetromino = shifted
            return success
        else:
            return True

    def process_left_right(self):
        if self.check_collision():
            return False
        else:
            return True

    def check_equal(self, gamestate):
        if self.is_hold_last != gamestate.is_hold_last or self.hold_type != gamestate.hold_type:
            return False
        if self.tetromino.type_str != gamestate.tetromino.type_str:
            return False
        for i in range(4):
            if self.next[i] != gamestate.next[i]:
                return False
        for r in range(GAME_BOARD_HEIGHT):
            for c in range(GAME_BOARD_WIDTH):
                if self.grid[r][c] != gamestate.grid[r][c]:
                    return False
        return True

    @classmethod
    def cls_put_tet_to_grid(cls, grid, tetro):
        grid_copy = helper.copy_2d(grid)
        disp = tetro.get_displaced()
        collide = False
        for sq in disp:
            x = sq[0]
            y = sq[1]
            if grid_copy[y][x] != 0:
                collide = True
            grid_copy[y][x] = tetro.to_num()
        return grid_copy, collide

    def hold(self):
        if self.is_hold_last: return False

        new_hold_type = self.tetromino.type_str
        if self.hold_type is None:
            self.tetromino = Tetromino.new_tetromino(self.next[0])
            self.next[:-1] = self.next[1:]
            self.next[-1] = self.next_next
            self.next_next = Tetromino.random_type_str(self.get_random().random())
        else:
            self.tetromino = Tetromino.new_tetromino(self.hold_type)

        self.hold_type = new_hold_type
        self.is_hold_last = True
        self.pieces += 1

        if self.check_collision():
            self.game_status = "gameover"
        return True

