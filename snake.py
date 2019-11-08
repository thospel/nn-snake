# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
"""Snakes

Usage:
  snake.py [-f <file>] [--snakes=<snakes>] [--stepping] [--fps=<fps>]
           [--width=<width>] [--height=<height>]
           [--columns=columns] [--rows=rows] [--block=<block_size>]
  snake.py (-h | --help)
  snake..py --version

Options:
  -h --help             Show this screen
  --version             Show version
  --stepping            Don't start running immediately, wait for the user to press ENTER
  --fps=<fps>           Frames per second (0 is no delays) [default: 40]
  --snakes=<snakes>     How many snakes to run at the same time [default: 0]
                        0 means use rows * colums or 1
  --width=<width>       Pit width [default: 40]
  --height=<height>     Pit height [default: 40]
  --columns=<columns>   Pit height [default: 2]
  --block=<block_size>  Block size in pixels [default: 20]
  --rows=<rows>         Pit width [default: 1]
  -f <file>:            Used by jupyter, ignored

"""
from docopt import docopt

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Snake 1.0')
arguments
# -

# %matplotlib inline
from matplotlib import pyplot as plt

from dataclasses import dataclass
from typing import List
import random
import math
import itertools
import timeit
import numpy as np

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pygame.freetype
from pygame.locals import *


# +
@dataclass
class TextField:
    name:   str
    prefix: str
    width:  int

@dataclass
class TextRow:
    x:           int
    y:           int
    max_width:   int
    text_fields: List[TextField]

@dataclass
class _TextRows:
    font:        pygame.freetype.Font
    skip:        str = "  "

@dataclass
class TextData:
    y:          int
    prefix_x:   int
    prefix:     str
    format_x:   int
    format:     str
    max_width:  int
    old_rect:   List[pygame.Rect]

class TextRows(_TextRows):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        rect = self.font.get_rect(self.skip)
        self.skip_width = rect.width
        # assume 8 is widest
        rect = self.font.get_rect("8")
        self.digit_width = rect.width
        self._lookup = {}

    def add(self, text_row, windows=1):
        x = text_row.x
        y = text_row.y
        for text_field in text_row.text_fields:
            pos_prefix = x
            rect = self.font.get_rect(text_field.prefix)
            x += rect.width
            pos_format = x
            x += self.digit_width * text_field.width + self.skip_width

            if text_field.name in self._lookup:
                raise(AssertionError("Duplicate name %s", text_field.name))
            self._lookup[text_field.name] = TextData(
                y         = y,
                prefix_x  = pos_prefix,
                prefix    = text_field.prefix,
                format_x  = pos_format,
                format    = "%%%du" % text_field.width,
                max_width = text_row.max_width,
                old_rect  = [None]*windows)

    def lookup(self, name):
        return self._lookup[name]


# +
ROW_TOP = [TextField("score", "Score: ", 4),
           TextField("game",  "Game: ",  4),
           TextField("moves", "Moves: ", 6),
           TextField("snake", "Id: ",    2),
           # TextField("x",     "x: ",     2),
           # TextField("y",     "y: ",     2),
]

# This must be signed because window offsets can be negative
TYPE_PIXELS = np.int32

class Display:
    WALL  = 255,255,255
    BODY  = 160,160,160
    HEAD  = 200,200,0
    BACKGROUND = 0,0,0
    APPLE = 0,255,0
    COLLISION = 255,0,0

    BLOCK_DEFAULT=20
    EDGE=1

    # we test these at the start of some functions
    # Make sure they have a "nothing to see here" value in case __init__ fails
    screen = None
    updates = []

    # You can only have one pygame instance in one process,
    # so make display related variables into class variables
    def __init__(self, snakes, columns=0, rows=1, block_size=0, caption="Snakes"):
        self.windows = rows*columns
        if not self.windows:
            return

        self.caption = caption

        self.BLOCK = block_size or Display.BLOCK_DEFAULT
        self.DRAW_BLOCK = self.BLOCK-2*Display.EDGE

        # coordinates relative to the upper left corner of the window
        self.TOP_TEXT_X  = self.BLOCK
        self.TOP_TEXT_Y  = self.DRAW_BLOCK

        self.WINDOW_X = (snakes.WIDTH +2) * self.BLOCK
        self.WINDOW_Y = (snakes.HEIGHT+2) * self.BLOCK
        self.OFFSET_X = (1-snakes.VIEW_X) * self.BLOCK
        self.OFFSET_Y = (1-snakes.VIEW_Y) * self.BLOCK

        self._window_x = np.tile  (np.arange(self.OFFSET_X, columns*self.WINDOW_X+self.OFFSET_X, self.WINDOW_X, dtype=TYPE_PIXELS), rows)
        self._window_y = np.repeat(np.arange(self.OFFSET_Y, rows   *self.WINDOW_Y+self.OFFSET_Y, self.WINDOW_Y, dtype=TYPE_PIXELS), columns)
        # print("window_x", self._window_x)
        # print("window_y", self._window_y)

        # self.last_collision_x = np.zeros(self.windows, dtype=TYPE_PIXELS)
        # self.last_collision_y = np.zeros(self.windows, dtype=TYPE_PIXELS)

    def start(self):
        if not self.windows:
            return
        # Avoid pygame.init() since the init of the mixer component leads to 100% CPU
        pygame.display.init()
        pygame.display.set_caption(self.caption)
        # pygame.mouse.set_visible(1)
        pygame.key.set_repeat(KEY_DELAY, KEY_INTERVAL)

        pygame.freetype.init()
        self._font = pygame.freetype.Font(None, self.BLOCK)
        self._font.origin = True

        self._textrows = TextRows(font=self._font)
        self._textrows.add(TextRow(self.TOP_TEXT_X - self.OFFSET_X,
                                   self.TOP_TEXT_Y - self.OFFSET_Y,
                                   self.WINDOW_X, ROW_TOP), self.windows)

        Display.screen = pygame.display.set_mode((self.WINDOW_X * columns, self.WINDOW_Y * rows))
        rect = 0, 0, self.WINDOW_X * columns, self.WINDOW_Y * rows
        Display.updates = [rect]
        pygame.draw.rect(Display.screen, Display.WALL, rect)

    def stop(self):
        if not Display.screen:
            return

        Display.screen  = None
        Display.updates = []
        pygame.quit()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __del__(self):
        self.stop()

    def update(self):
        if Display.updates:
            self._time_start  = timeit.default_timer()
            pygame.display.update(Display.updates)
            # print("Update took", int((timeit.default_timer() - self._time_start)*1000), "updates\n", Display.updates)
            Display.updates = []

    def draw_text(self, w, name, value=None,
                       fg_color=BACKGROUND, bg_color=WALL):
        text_data = self._textrows.lookup(name)

        if value is None:
            text = text_data.prefix
            x = text_data.prefix_x
        else:
            text = text_data.format % value
            # Erase old text
            old_rect = text_data.old_rect[w]
            if old_rect:
                pygame.draw.rect(Display.screen, bg_color, old_rect)
                Display.updates.append(old_rect)
            x = text_data.format_x
        y = text_data.y

        # Draw new text
        rect = self._font.get_rect(text)
        rect.x += x
        if rect.x + rect.width > text_data.max_width:
            return None
        x += self._window_x[w]
        y += self._window_y[w]
        rect.x += self._window_x[w]
        rect.y = y - rect.y
        # print("Draw text", w, x, y, '"%s"' % text, x + self._window_x[w], y + self._window_y[w], rect, old_rect)
        self._font.render_to(Display.screen, (x, y), None, fg_color, bg_color)
        Display.updates.append(rect)
        if value is not None:
            # Remember what we updated
            text_data.old_rect[w] = rect

    def draw_field_empty(self, w):
        rect = (self._window_x[w] - self.OFFSET_X + self.BLOCK,
                self._window_y[w] - self.OFFSET_Y + self.BLOCK,
                self.WINDOW_X - 2 * self.BLOCK,
                self.WINDOW_Y - 2 * self.BLOCK)
        pygame.draw.rect(Display.screen, Display.BACKGROUND, rect)
        Display.updates.append(rect)

    def draw_block(self, w, x, y, color):
        rect = (x * self.BLOCK + self._window_x[w] + Display.EDGE,
                y * self.BLOCK + self._window_y[w] + Display.EDGE,
                self.DRAW_BLOCK,
                self.DRAW_BLOCK)

        # print("Draw %d (%d,%d): %d,%d,%d: [%d %d %d %d]" % ((w, x, y)+color+(rect)))
        Display.updates.append(rect)
        pygame.draw.rect(Display.screen, color, rect)

    def draw_collisions(self, all_windows, w_index, x, y, nr_games):
        for w in w_index:
            i = all_windows[w]
            if False:
                # This test was for when the idea was to freeze some snakes
                # if self._nr_moves[w] > self._cur_move:
                self.draw_block(w, x[i], y[i], Display.COLLISION)
            else:
                #self.draw_block(w,
                #                self.last_collision_x[w],
                #                self.last_collision_y[w],
                #                Display.WALL)
                self.draw_text(w, "score")
                self.draw_text(w, "game")
                self.draw_text(w, "game", nr_games[i])
                self.draw_text(w, "moves")
                self.draw_text(w, "snake")
                self.draw_text(w, "snake", i)
                # self.draw_text(w, "x")
                # self.draw_text(w, "y")
                self.draw_field_empty(w)

    def draw_move(self,
                  all_windows, head_x_new, head_y_new,
                  is_collision, head_x_old, head_y_old,
                  eat, tail_x, tail_y,
                  w_nr_moves):
        for w in range(all_windows.size):
            i = all_windows[w]
            if not is_collision[i]:
                # The current head becomes body
                # (For length 1 snakes the following tail erase will undo this)
                self.draw_block(w, head_x_old[i], head_y_old[i], Display.BODY)
            if not eat[i]:
                # Drop the tail if we didn't eat an apple then
                self.draw_block(w, tail_x[i], tail_y[i], Display.BACKGROUND)
            self.draw_block(w, head_x_new[i], head_y_new[i], Display.HEAD)
            self.draw_text(w, "moves", w_nr_moves[w])
            # self.draw_text(w, "x", head_x_new[i])
            # self.draw_text(w, "y", head_y_new[i])

    def draw_apples(self, all_windows, w_index, apple_x, apple_y, score):
        for w in w_index:
            i = all_windows[w]
            self.draw_block(w, apple_x[i], apple_y[i], Display.APPLE)
            self.draw_text(w, "score", score[i])


# -

def np_empty(shape, type):
    # return np.empty(shape, type)
    # Fill with garbage for debug
    return np.random.randint(100, size=shape, dtype=type)


# +
TEXT_SCORE   = "Score: "
TEXT_SCORE_X = 1
TEXT_GAME    = "Game: "
TEXT_GAME_X  = 7
TEXT_MOVE    = "Move: "
TEXT_MOVE_X  = 13
TEXT_SNAKE   = "Snake: "
TEXT_SNAKE_X = 20

POLL_SLOW_DEFAULT = 1/25
POLL_FAST_DEFAULT = 1/40
KEY_INTERVAL = int(1000 / 20)  # twenty per second
KEY_DELAY = 500           # Start repeating after half a second

WIDTH_DEFAULT  = 40
HEIGHT_DEFAULT = 40
E_POS = np.uint8
TYPE_POS   = np.int8
TYPE_BOOL  = np.bool
TYPE_INDEX = np.intp
TYPE_FLAG  = np.uint8
TYPE_SCORE = np.uint32
TYPE_MOVES = np.uint32
TYPE_GAMES = np.uint32

VIEW_X0 = 2
VIEW_Y0 = 2
VIEW_X2 = VIEW_X0+2
VIEW_Y2 = VIEW_Y0+2
VIEW_WIDTH  = 2*VIEW_X0+1
VIEW_HEIGHT = 2*VIEW_Y0+1

INDEX_SNAKE = 0
INDEX_APPLE = 1
INDEX_WALL  = 2
INDEX_MAX   = 3

class Snakes:
    START_MOVE = 1

    DIRECTIONS = [[1,0],[-1,0],[0,1],[0,-1]]
    DIRECTION_PERMUTATIONS = np.array(list(itertools.permutations(DIRECTIONS)), dtype=TYPE_POS)
    NR_DIRECTION_PERMUTATIONS = len(DIRECTION_PERMUTATIONS)
    DIRECTION_PERMUTATIONS_X=DIRECTION_PERMUTATIONS[:,:,0]
    DIRECTION_PERMUTATIONS_Y=DIRECTION_PERMUTATIONS[:,:,1]

    def __init__(self, nr_snakes=1, width=0, height=0, view_x=0, view_y=0):
        self.windows = None

        self._nr_snakes = nr_snakes
        self._all_snakes = np.arange(nr_snakes, dtype=TYPE_INDEX)

        self.VIEW_X = view_x or VIEW_X0
        self.VIEW_Y = view_y or VIEW_Y0
        self.WIDTH  = width  or WIDTH_DEFAULT
        self.HEIGHT = height or HEIGHT_DEFAULT
        self.AREA   = self.WIDTH * self.HEIGHT
        # First power of 2 greater or equal to AREA for fast modular arithmetic
        self.AREA2 = 1 << (self.AREA-1).bit_length()
        self.MASK  = self.AREA2 - 1

        # Notice that we store in row major order, so use field[y,x]
        self._field = np.ones((nr_snakes,
                               self.HEIGHT+2*self.VIEW_Y,
                               self.WIDTH +2*self.VIEW_X), dtype=TYPE_FLAG)
        # Position arrays are split in x and y so we can do fast _field indexing
        self._snake_body_x = np_empty((nr_snakes, self.AREA2), TYPE_POS)
        self._snake_body_y = np_empty((nr_snakes, self.AREA2), TYPE_POS)

        # Very first run: body_length =0 and offset (cur_move) = START_MOVE-1
        # So the very first tail_set() will access (offset-cur_move) & MASK
        # So we need to make sure this is a coordinate inside _field so the
        # border won't get destroyed. At all later times the field before the
        # head will hav been set by previous runs
        start_move = (Snakes.START_MOVE-1) & self.MASK
        self._snake_body_x[:, start_move] = 1
        self._snake_body_y[:, start_move] = 1

        # Body length measures the snake *without* the head
        # This is therefore also the score (if we start with length 0 snakes)
        self._body_length = np_empty(nr_snakes, TYPE_INDEX)
        self._head_x   = np_empty(nr_snakes, TYPE_POS)
        self._head_y   = np_empty(nr_snakes, TYPE_POS)
        self._apple_x  = np_empty(nr_snakes, TYPE_POS)
        self._apple_y  = np_empty(nr_snakes, TYPE_POS)
        self._nr_moves = np_empty(nr_snakes, TYPE_MOVES)
        self._nr_games = np_empty(nr_snakes, TYPE_GAMES)

    def rand_x(self, nr):
        offset = self.VIEW_X
        return np.random.randint(offset, offset+self.WIDTH,  size=nr, dtype=TYPE_POS)

    def rand_y(self, nr):
        offset = self.VIEW_Y
        return np.random.randint(offset, offset+self.HEIGHT, size=nr, dtype=TYPE_POS)

    def scores(self):
        return self._body_length

    def score(self, i):
        return self._body_length[i]

    def score_max(self):
        return self._score_max

    # def nr_moves(self):
    #    return self._cur_move - self._nr_moves

    def nr_moves(self, i):
        return self._cur_move - self._nr_moves[i]

    def nr_moves_max(self):
        return self._moves_max

    def nr_games(self, i):
        return self._nr_games[i]

    def nr_games_max(self):
        return self._nr_games_max

    def head_x(self):
        return self._head_x

    def head_y(self):
        return self._head_y

    def head_set(self, x, y):
        self._head_x = x
        self._head_y = y
        offset = self._cur_move & self.MASK
        self._snake_body_x[self._all_snakes, offset] = x
        self._snake_body_y[self._all_snakes, offset] = y
        self._field[self._all_snakes, y, x] = 1

    def tail_set(self, values):
        # print("Eat", values)
        # print("body length", self._body_length)

        # Bring potentially large cur_move into a reasonable range
        # so tails will not use some large integer type
        offset = self._cur_move & self.MASK
        # print("Offset", offset)
        tail_offset = (offset - self._body_length) & self.MASK
        # print("tail offset", tail_offset)
        x = self._snake_body_x[self._all_snakes, tail_offset]
        y = self._snake_body_y[self._all_snakes, tail_offset]
        # print("tail pos")
        # print(np.dstack((x, y)))
        self._field[self._all_snakes, y, x] = values
        return x, y

    """
    def view_string(self):
        port = self.view_port()
        str = ""
        for y in range(VIEW_self.HEIGHT):
            str = str + "|"
            for x in range(VIEW_self.WIDTH):
                v = port[:,x,y]
                sum = v.sum()
                # print("v=", v, "sum=", sum)
                if sum == 0:
                    str += " "
                elif sum > 1:
                    # raise(AssertionError("Too many ones in viewport"))
                    str += "*"
                elif v[0]:
                    str += "O"
                elif v[1]:
                    str += "@"
                else:
                    str += "X"
            str += "|\n"
        return str

    def view_port(self):
        index,x,y = self.head()
        return self._field[:,x-VIEW_X0:x+VIEW_X2,y-VIEW_Y0:y+VIEW_Y2]
    """

    def new_apples(self, todo):
        too_large = self._body_length[todo] >= self.AREA-1
        if too_large.any():
            raise(AssertionError("No place for apples"))

        # print("todo", todo)
        while todo.size:
            rand_x = self.rand_x(todo.size)
            rand_y = self.rand_y(todo.size)
            self._apple_x[todo] = rand_x
            self._apple_y[todo] = rand_y
            fail = self._field[todo, rand_y, rand_x]
            # Index with boolean is grep
            todo = todo[fail != 0]
            # print("todo", todo)

    def draw_heads(self):
        self.draw_blocks(self.head_x(), self.head_y(), Display.HEAD)

    def plan_greedy(self):
        x = self.head_x()
        y = self.head_y()

        dx = self._apple_x - x
        dy = self._apple_y - y
        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)
        dir_x = abs_dx > abs_dy
        dx = np.where(dir_x, np.sign(dx), 0)
        dy = np.where(dir_x, 0,          np.sign(dy))
        # Diag is mainly meant to detect dx == dy == 0
        # But is also debugs *some* diagonal moves
        diag = dx == dy
        if np.count_nonzero(diag):
            raise(AssertionError("Impossible apple direction"))
        return x+dx, y+dy

    def plan_random(self, collided):
        # different permutation for each collision
        direction_index = np.random.randint(Snakes.NR_DIRECTION_PERMUTATIONS,
                                            size=collided.size)
        # different permutation of directions for each collision
        dx = Snakes.DIRECTION_PERMUTATIONS_X[direction_index].transpose()
        dy = Snakes.DIRECTION_PERMUTATIONS_Y[direction_index].transpose()
        # different permutation of test coordinates for each collision
        x = self.head_x()[collided] + dx
        y = self.head_y()[collided] + dy
        # Is there nothing on the new coordinate ?
        empty = self._field[collided, y, x] ^ 1
        # which permutation (select) for which snake(i) is empty
        select, i = empty.nonzero()
        # Fill result with a random direction for each snake
        # (fallback for if the head is completely surrounded)
        pos_x = x[0].copy()
        pos_y = y[0].copy()
        # Copy coordinates of empty neighbours
        # Each snake can get coordinates assigned multiple times
        # I assume some assignment wins and there is no tearing
        # (I do not know if numpy enforces anything like this)
        pos_x[i] = x[select, i]
        pos_y[i] = y[select, i]
        return pos_x, pos_y

    def frame(self):
        return self._cur_move - Snakes.START_MOVE

    def elapsed(self):
        return self._time_end - self._time_start

    def paused(self):
        return self._paused

    def frames_skipped(self):
        return self._frames_skipped

    def frame_rate(self):
        elapsed = self.elapsed() - self.paused()
        frames  = self.frame() - self.frames_skipped()
        if elapsed == 0:
            # This properly handles positive, negative and 0 (+inf, -inf, nan)
            return math.inf * frames
        return frames / elapsed

    def move_collisions(self, display, x, y):
        is_collision = self._field[self._all_snakes, y, x]
        collided = is_collision.nonzero()[0]
        # print("Collided", collided)
        if collided.size:
            if self._score_max < 0:
                self._score_max = 0
                self._moves_max = 0
            else:
                self._nr_games[collided] += 1
                nr_games_max = np.amax(self._nr_games[collided])
                if nr_games_max > self._nr_games_max:
                    self._nr_games_max = nr_games_max
                score_max = np.amax(self._body_length[collided])
                if score_max > self._score_max:
                    self._score_max = score_max
                moves_max = self._cur_move - np.amin(self._nr_moves[collided])
                if moves_max > self._moves_max:
                    self._moves_max = moves_max

            # After the test because it skips the first _nr_games update
            w_index = is_collision[self._all_windows].nonzero()[0]
            display.draw_collisions(self._all_windows, w_index, x, y, self._nr_games)

            self._field[collided, self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH] = 0

            # We are currently doing setup, so this move doesn't count
            self._nr_moves[collided] = self._cur_move + 1

            self._body_length[collided]  = 0
            rand_x = self.rand_x(collided.size)
            rand_y = self.rand_y(collided.size)
            x[collided] = rand_x
            y[collided] = rand_y
        return is_collision, collided

    def move_execute(self, display, x, y, is_collision, collided):
        eat = (x == self._apple_x) & (y == self._apple_y)
        tail_pos_x, tail_pos_y = self.tail_set(eat)
        self._body_length += eat
        if collided.size:
            eat[collided] = True

        # cur_move must be updated before head_set for head progress
        # Also before draw_pre_move so nr moves will be correct
        self._cur_move += 1
        display.draw_move(self._all_windows, x, y,
                          is_collision, self.head_x(), self.head_y(),
                          eat, tail_pos_x, tail_pos_y,
                          self._cur_move - self._nr_moves[self._all_windows])
        self.head_set(x, y)

        eaten = eat.nonzero()[0]
        if eaten.size:
            self.new_apples(eaten)
            w_index = eat[self._all_windows].nonzero()[0]
            display.draw_apples(self._all_windows, w_index, self._apple_x, self._apple_y, self._body_length)

        # print(self.view_string())

        # print("Field\n", self._field)
        #print("body_x", self._snake_body_x)
        #print("body_y", self._snake_body_y)
        # print("body")
        # print(np.dstack((self._snake_body_x, self._snake_body_y)))
        # print("Head")
        # print(np.dstack((self.head_x(), self.head_y())))
        # print("Apple")
        # print(np.dstack((self._apple_x, self._apple_y)))
        # print("-------------------")

    def move_select(self):
        x, y = self.plan_greedy()
        collided = self._field[self._all_snakes, y, x].nonzero()[0]
        # print("Greedy Collided", collided)
        if collided.size:
            rand_x, rand_y = self.plan_random(collided)
            x[collided] = rand_x
            y[collided] = rand_y
        # print("Move")
        # print(np.dstack((x, y)))
        return x, y

    def move_evaluate(self, display):
        display.update()
        return self.wait(display)

    def run_start(self, display, fps=None, stepping=False):
        nr_windows = min(self._nr_snakes, display.windows)
        self._all_windows = np.arange(nr_windows-1, -1, -1, dtype=TYPE_INDEX)

        # Fake a collision for all snakes so all snakes will reset
        x = np.zeros(self._nr_snakes, TYPE_POS)
        y = np.zeros(self._nr_snakes, TYPE_POS)

        # Make sure we won't hit an apple left from a previous run
        self._apple_x.fill(0)
        self._apple_y.fill(0)

        if fps is None:
            self._poll_fast = POLL_FAST_DEFAULT
        elif fps > 0:
            self._poll_fast = 1 / fps
        else:
            self._poll_fast = 0
        self._nr_games.fill(0)
        self._nr_games_max = 0
        self._score_max = -1
        self._moves_max = -1
        self._cur_move = Snakes.START_MOVE-1
        self._stepping = stepping
        self._paused = 0
        # print("Start at time 0, frame", self.frame())
        self._time_start  = timeit.default_timer()
        self._time_target = self._time_start
        if self._stepping:
            self._pause_start = self._time_start
            self._frame_start = self.frame()
            # When stepping the first frame doesn't count
            self._frames_skipped = -1
        else:
            self._pause_start = 0
            self._frames_skipped = 0
        return x, y

    def run_finish(self):
        score_max = np.amax(self._body_length)
        if score_max > self._score_max:
            self._score_max = score_max
        moves_max = self._cur_move - np.amin(self._nr_moves)
        if moves_max > self._moves_max:
            self._moves_max = moves_max
        # We could only measure nr_games_max here, but then
        # we wouldn't have a running update
        # self._nr_games_max = np.amax(self._nr_games)
        self._all_windows = None

        #print("Quit at", self._time_end - self._time_start,
        #      "Paused", self.paused(),
        #      "frame", self.frame(),
        #      "Skipped", self.frames_skipped())

    def wait(self, display):
        waiting = True
        while waiting:
            if self._stepping:
                self._stepping = False
            elif self._pause_start or self._poll_fast:
                left = int((self._time_target - timeit.default_timer())*1000)
                if left > 0:
                    pygame.time.wait(left)
            for event in pygame.event.get():
                if event.type == QUIT or event.type == KEYDOWN and event.key == K_q:
                    self._time_end  = timeit.default_timer()
                    if self._pause_start:
                        self._paused += self._time_end - self._pause_start
                        self._frames_skipped += self.frame() - self._frame_start

                    return False
                elif event.type == KEYDOWN:
                    # events seem to come without time
                    time = timeit.default_timer()
                    if event.key == K_SPACE or event.key == K_r:
                        # Stop/start running
                        self._time_target = time
                        if self._pause_start:
                            self._paused += time - self._pause_start
                            self._frames_skipped += self.frame() - self._frame_start
                            # print("Start running at", time - self._time_start, "frame", self.frame())
                            self._pause_start = 0
                        else:
                            # print("Stop running at", time-self._time_start, "frame", self.frame())
                            self._pause_start = time
                            self._frame_start = self.frame()
                    elif event.key == K_s:
                        # Single step
                        self._stepping = True
                        self._time_target = time
                        if not self._pause_start:
                            self._pause_start = time
                            self._frame_start = self.frame()
            if self._pause_start:
                waiting = not self._stepping
                self._time_target += POLL_SLOW_DEFAULT
            else:
                waiting = False
                self._time_target += self._poll_fast
        return True

    def draw_run(self, display, fps=None, stepping=False):
        # print("New game, head=%d [%d, %d]" % self.head())
        # print(self.view_string())

        x, y = self.run_start(display, fps=fps, stepping=stepping)
        while True:
            is_collision, collided = self.move_collisions(display, x, y)
            self.move_execute(display, x, y, is_collision, collided)
            if not self.move_evaluate(display):
                self.run_finish()
                return
            x, y = self.move_select()


# +
columns = int(arguments["--columns"])
rows = int(arguments["--rows"])
nr_snakes = int(arguments["--snakes"]) or rows*columns
block_size = int(arguments["--block"])

snakes = Snakes(nr_snakes=nr_snakes,
                width=int(arguments["--width"]),
                height=int(arguments["--height"]))

with Display(snakes,
             columns=columns,
             rows=rows,
             block_size = block_size) as display:
    while snakes.draw_run(display,
                          fps=float(arguments["--fps"]),
                          stepping=arguments["--stepping"]):
        pass
print("Framerate", snakes.frame_rate())
print("Max: Score:", snakes.score_max(), "Moves:", snakes.nr_moves_max(), "Lost Games:", snakes.nr_games_max())
# -
