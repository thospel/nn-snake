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

import random
import math
import itertools
import timeit
import numpy as np

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *


def np_empty(shape, type):
    # return np.empty(shape, type)
    # Fill with garbage for debug
    return np.random.randint(100, size=shape, dtype=type)


# +
POLL_DEFAULT = 1/25
KEY_INTERVAL = int(1000 / 20)  # twenty per second
KEY_DELAY = 500           # Start repeating after half a second

WIDTH_DEFAULT  = 40
HEIGHT_DEFAULT = 40
E_POS = np.uint8
TYPE_POS   = np.int8
TYPE_PIXELS = np.uint16
TYPE_BOOL  = np.bool
TYPE_INDEX = np.intp
TYPE_FLAG  = np.uint8
TYPE_SCORE = np.uint32
TYPE_MOVES = np.uint32
EDGE=1
BLOCK_DEFAULT=20

VIEW_X0 = 1
VIEW_Y0 = 1
VIEW_X2 = VIEW_X0+2
VIEW_Y2 = VIEW_Y0+2
VIEW_WIDTH  = 2*VIEW_X0+1
VIEW_HEIGHT = 2*VIEW_Y0+1

INDEX_SNAKE = 0
INDEX_APPLE = 1
INDEX_WALL  = 2
INDEX_MAX   = 3

class Snakes:
    WALL  = 255,255,255
    BODY  = 160,160,160
    HEAD  = 200,200,0
    BACKGROUND = 0,0,0
    APPLE = 0,255,0
    COLLISION = 255,0,0

    START_MOVE = 1

    DIRECTIONS = [[1,0],[-1,0],[0,1],[0,-1]]
    DIRECTION_PERMUTATIONS = np.array(list(itertools.permutations(DIRECTIONS)), dtype=TYPE_POS)
    NR_DIRECTION_PERMUTATIONS = len(DIRECTION_PERMUTATIONS)
    DIRECTION_PERMUTATIONS_X=DIRECTION_PERMUTATIONS[:,:,0]
    DIRECTION_PERMUTATIONS_Y=DIRECTION_PERMUTATIONS[:,:,1]

    def __init__(self, nr_snakes=1, width=0, height=0, view_x=0, view_y=0):
        self._windows = None

        self._nr_snakes = nr_snakes
        self._all_snakes = np.arange(nr_snakes, dtype=TYPE_INDEX)

        self._view_x = view_x or VIEW_X0
        self._view_y = view_y or VIEW_Y0
        self.WIDTH  = width  or WIDTH_DEFAULT
        self.HEIGHT = height or HEIGHT_DEFAULT
        self.AREA   = self.WIDTH * self.HEIGHT
        # First power of 2 greater or equal to AREA for fast modular arithmetic
        self.AREA2 = 1 << (self.AREA-1).bit_length()
        self.MASK  = self.AREA2 - 1

        # Notice that we store in row major order, so use field[y,x]
        self._field = np.ones((nr_snakes,
                               self.HEIGHT+2*self._view_y,
                               self.WIDTH +2*self._view_x), dtype=TYPE_FLAG)
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

    # You can only have one pygame instance in one process,
    # so make display related variables into class variables
    def display_start(self, columns=0, rows=1, block_size=0):
        self.BLOCK = block_size or BLOCK_DEFAULT
        self.DRAW_BLOCK = self.BLOCK-2*EDGE

        self._windows = rows*columns
        if self._windows:
            # Avoid pygame.init() since the init of the mixer component leads to 100% CPU
            pygame.display.init()
            pygame.display.set_caption('Snakes')
            # pygame.mouse.set_visible(1)
            pygame.key.set_repeat(KEY_DELAY, KEY_INTERVAL)

            self.last_collision_x = np.zeros(self._windows, dtype=TYPE_PIXELS)
            self.last_collision_y = np.zeros(self._windows, dtype=TYPE_PIXELS)

            WINDOW_X = self.WIDTH+2
            WINDOW_Y = self.HEIGHT+2
            OFFSET_X =  self._view_x-1
            OFFSET_Y =  self._view_y-1
            self._window_x = np.tile  (np.arange(OFFSET_X, columns*WINDOW_X+OFFSET_X, WINDOW_X, dtype=np.uint32), rows)
            self._window_y = np.repeat(np.arange(OFFSET_Y, rows   *WINDOW_Y+OFFSET_Y, WINDOW_Y, dtype=np.uint32), columns)

            Snakes.screen = pygame.display.set_mode((WINDOW_X * self.BLOCK * columns, WINDOW_Y * self.BLOCK * rows))
            rect = 0, 0, WINDOW_X * self.BLOCK * columns, WINDOW_Y * self.BLOCK * rows
            Snakes.updates = [rect]
            pygame.draw.rect(Snakes.screen, Snakes.WALL, rect)

    def display_stop(self):
        if self._windows:
            Snakes.screen = None
            pygame.quit()

    def rand_x(self, nr):
        offset = self._view_x
        return np.random.randint(offset, offset+self.WIDTH,  size=nr, dtype=TYPE_POS)

    def rand_y(self, nr):
        offset = self._view_y
        return np.random.randint(offset, offset+self.HEIGHT, size=nr, dtype=TYPE_POS)

    def score(self):
        return self._body_length

    def nr_moves(self):
        return self._cur_move - self._nr_moves

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

    def draw_start(self):
        self.draw_blocks(self.last_collision_x, self.last_collision_y, Snakes.WALL)
        self.last_collision_x.fill(0)
        self.last_collision_y.fill(0)

    def draw_block(self, x, y, color):
        # print("Draw (%d,%d,%d): %d,%d,%d" % (pos+color))
        rect = x*self.BLOCK+EDGE, y*self.BLOCK+EDGE, self.DRAW_BLOCK, self.DRAW_BLOCK
        Snakes.updates.append(rect)
        pygame.draw.rect(Snakes.screen, color, rect)

    def draw_blocks(self, x, y, color):
        # print("X", x)
        # print("Y", y)
        for i in range(self._windows):
            self.draw_block(x[i] + self._window_x[i], y[i] + self._window_y[i], color)

    def draw_apples(self):
        self.draw_blocks(self._apple_x, self._apple_y, Snakes.APPLE)

    def draw_heads(self):
        self.draw_blocks(self.head_x(), self.head_y(), Snakes.HEAD)

    def draw_collisions(self, collided, x, y):
        # Does numpy have a simple lazy loop ?
        for i in range(collided.size):
            w = collided[i]
            if w >= self._windows:
                break
            if False:
                if self._nr_moves[w] != self._cur_move:
                    self.draw_block(x[w] + self._window_x[w],
                                    y[w] + self._window_y[w], Snakes.COLLISION)
            else:
                rect = (self._window_x[w]+1) * self.BLOCK, (self._window_y[w]+1) * self.BLOCK, self.WIDTH*self.BLOCK, self.HEIGHT*self.BLOCK
                Snakes.updates.append(rect)
                pygame.draw.rect(Snakes.screen, Snakes.BACKGROUND, rect)

    def draw_pre_move(self, is_collision, eat, x, y):
        head_x = self.head_x()
        head_y = self.head_y()
        for w in range(self._windows):
            if not is_collision[w]:
                self.draw_block(head_x[w] + self._window_x[w],
                                head_y[w] + self._window_y[w], Snakes.BODY)
            if not eat[w]:
                self.draw_block(x[w] + self._window_x[w],
                                y[w] + self._window_y[w], Snakes.BACKGROUND)

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

    def update(self):
        # pygame.display.update()
        if Snakes.updates:
            # print("Real update")
            pygame.display.update(Snakes.updates)
            Snakes.updates = []

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

    def draw_run(self, fps=40, stepping=False):
        self.draw_start()
        # print("New game, head=%d [%d, %d]" % self.head())
        # print(self.view_string())

        # Fake a collision for all snakes so all snakes will reset
        is_collision = np.ones(self._nr_snakes, dtype=TYPE_FLAG)
        x = np_empty(self._nr_snakes, TYPE_POS)
        y = np_empty(self._nr_snakes, TYPE_POS)

        # Make sure we won't hit an apple left from a previous run
        self._apple_x.fill(0)
        self._apple_y.fill(0)

        self._cur_move = Snakes.START_MOVE-1
        self._paused = 0
        time_step = 1/fps if fps > 0 else 0
        # print("Start at time 0, frame", self.frame())
        self._time_start  = timeit.default_timer()
        time_target = self._time_start
        if stepping:
            pause_start = self._time_start
            time_step = POLL_DEFAULT
            frame_start = self.frame()
            # When stepping the first frame doesn't count
            self._frames_skipped = -1
        else:
            pause_start = 0
            self._frames_skipped = 0

        while True:
            collided = is_collision.nonzero()[0]
            # print("Collided", collided)
            if collided.size:
                self._field[collided, self._view_y:self._view_y+self.HEIGHT, self._view_x:self._view_x+self.WIDTH] = 0

                self._nr_moves[collided] = self._cur_move
                self._body_length[collided]  = 0
                self.draw_collisions(collided, x, y)
                rand_x = self.rand_x(collided.size)
                rand_y = self.rand_y(collided.size)
                x[collided] = rand_x
                y[collided] = rand_y

            eat = (x == self._apple_x) & (y == self._apple_y)
            tail_pos_x, tail_pos_y = self.tail_set(eat)
            self._body_length += eat
            if collided.size:
                eat[collided] = True
            self.draw_pre_move(is_collision, eat, tail_pos_x, tail_pos_y)

            # cur_move must be updated before head_set for head progress
            self._cur_move += 1
            self.head_set(x, y)
            self.draw_heads()

            eaten = eat.nonzero()[0]
            if eaten.size:
                self.new_apples(eaten)
                self.draw_apples()

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

            self.update()
            waiting = True
            while waiting:
                if stepping:
                    stepping = False
                elif time_step:
                    left = int((time_target - timeit.default_timer())*1000)
                    if left > 0:
                        pygame.time.wait(left)
                for event in pygame.event.get():
                    if event.type == QUIT or event.type == KEYDOWN and event.key == K_q:
                        self._time_end  = timeit.default_timer()
                        if pause_start:
                            self._paused += self._time_end - pause_start
                            self._frames_skipped += self.frame() - frame_start
                        #print("Quit at", self._time_end - self._time_start,
                        #      "Paused", self.paused(),
                        #      "frame", self.frame(),
                        #      "Skipped", self.frames_skipped())
                        return False
                    elif event.type == KEYDOWN:
                        # events seem to come without time
                        time = timeit.default_timer()
                        if event.key == K_RETURN or event.key == K_r:
                            # Stop/start running
                            time_target = time
                            if pause_start:
                                self._paused += time - pause_start
                                self._frames_skipped += self.frame() - frame_start
                                # print("Start running at", time - self._time_start, "frame", self.frame())
                                time_step = 1/fps if fps > 0 else 0
                                pause_start = 0
                            else:
                                # print("Stop running at", time-self._time_start, "frame", self.frame())
                                pause_start = time
                                frame_start = self.frame()
                                time_step = POLL_DEFAULT
                        elif event.key == K_s:
                            # Single step
                            stepping = True
                            time_target = time
                            if not pause_start:
                                pause_start = time
                                frame_start = self.frame()
                                time_step = POLL_DEFAULT
                waiting = pause_start and not stepping
                time_target += time_step

            x, y = self.plan_greedy()
            collided = self._field[self._all_snakes, y, x].nonzero()[0]
            # print("Greedy Collided", collided)
            if collided.size:
                rand_x, rand_y = self.plan_random(collided)
                x[collided] = rand_x
                y[collided] = rand_y
            # print("Move")
            # print(np.dstack((x, y)))

            is_collision = self._field[self._all_snakes, y, x]


# +
columns = int(arguments["--columns"])
rows = int(arguments["--rows"])
nr_snakes = int(arguments["--snakes"]) or rows*columns
block_size = int(arguments["--block"])

snakes = Snakes(nr_snakes=nr_snakes,
                width=int(arguments["--width"]),
                height=int(arguments["--height"]))
snakes.display_start(columns=columns, rows=rows, block_size = block_size)

while snakes.draw_run(fps=float(arguments["--fps"]), stepping=arguments["--stepping"]):
    pass
print("Score", snakes.score(), "Framerate", snakes.frame_rate())

snakes.display_stop()
