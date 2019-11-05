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
  snake.py [-f <file>] [--fps=<fps>] [--pause=<pause>]
  snake.py (-h | --help)
  snake..py --version

Options:
  -h --help        Show this screen
  --version        Show version
  --fps=<fps>      Frames per second (0 is no delays) [default: 40]
  --pause=<pause>  Pause for <pause> seconds after death [default: 5]
  -f <file>:       Used by jupyter, ignored

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
import timeit
import numpy as np

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *

a=[1]
if a:
    print(8)
else:
    print(9)

# +
WIDTH  = 10
HEIGHT =  5
# TYPE_POS = np.uint8
TYPE_POS   = np.int8
TYPE_SNAKE = np.int8
EDGE=1
AREA=WIDTH*HEIGHT
BLOCK=20
DRAW_BLOCK = BLOCK-2*EDGE
# First power of 2 above greater or equal to AREA
AREA2 = 1<<(AREA-1).bit_length()
MASK = SIZE2-1

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

    def __init__(self, nr_snakes=1, view_x=0, view_y=0):
        self._windows = None

        self._nr_snakes = nr_snakes
        self._all_snakes = np.arange(nr_snakes, dtype=TYPE_SNAKE)
        # Notice that "active" can be in a different order from "all_snakes"
        # However, if active.size == nr_snakes) we must always behave as if active == all_snakes.
        # This allow us to skip indexing in several cases
        self._active = self._all_snakes.copy()

        self._view_x = view_x or VIEW_X0
        self._view_y = view_y or VIEW_Y0
        # Notice that we store in row major order, so use field[y][x]
        self._field = np.ones((nr_snakes, HEIGHT+2*self._view_y, WIDTH+2*self._view_x), np.float32)
        # Position arrays are split in x and y so we can do fast field indexing
        self._snake_body_x = np.empty((nr_snakes, AREA2), TYPE_POS)
        self._snake_body_y = np.empty((nr_snakes, AREA2), TYPE_POS)
        self._snake_head = np.zeros(nr_snakes, np.uint32)
        # Body length measures the snake without a head
        self._body_length = np.zeros(nr_snakes, np.uint32)

    # You can only have one pygame instance in one process,
    # so make display related variables into class variables
    def display_start(self, columns=0, rows=1):
        self._windows = rows*columns
        if self._windows:
            # Avoid pygame.init() since the init of the mixer component leads to 100% CPU
            pygame.display.init()
            pygame.display.set_caption('Snakes')
            # pygame.mouse.set_visible(1)

            self.last_collision_x = np.zeros(self._windows, TYPE_POS)
            self.last_collision_y = np.zeros(self._windows, TYPE_POS)

            WINDOW_X = WIDTH+2
            WINDOW_Y = HEIGHT+2
            OFFSET_X =  self._view_x-1
            OFFSET_Y =  self._view_y-1
            self._window_x = np.tile  (np.arange(OFFSET_X, columns*WINDOW_X+OFFSET_X, WINDOW_X, dtype=np.uint32), rows)
            self._window_y = np.repeat(np.arange(OFFSET_Y, rows   *WINDOW_Y+OFFSET_Y, WINDOW_Y, dtype=np.uint32), columns)

            Snakes.screen = pygame.display.set_mode((WINDOW_X * BLOCK * columns, WINDOW_Y * BLOCK * rows))
            rect = 0, 0, WINDOW_X * BLOCK * columns, WINDOW_Y * BLOCK * rows
            Snakes.updates = [rect]
            pygame.draw.rect(Snakes.screen, Snakes.WALL, rect)

    def restart(self):
        self._score = np.zeros(self._nr_snakes, np.uint32)
        self._body_length.fill(0)
        self.head_set_x(self.rands_pos(self._nr_snakes, WIDTH,  self._view_x))
        self.head_set_y(self.rands_pos(self._nr_snakes, HEIGHT, self._view_y))
        self._field[:, self._view_y:self._view_y+HEIGHT, self._view_x:self._view_x+WIDTH] = 0
        # print("_body_x", self._snake_body_x)
        # print("_body_y", self._snake_body_y)
        # print("head_x()", self.head_x())
        # print("head_y()", self.head_y())
        self._field[self._all_snakes, self.head_y(), self.head_x()] = 1

        self._apple_x = self.rands_pos(self._nr_snakes, WIDTH,  self._view_x)
        self._apple_y = self.rands_pos(self._nr_snakes, HEIGHT, self._view_y)
        # print("_apple_x", self._apple_x)
        # print("_apple_y", self._apple_y)

    def display_stop(self):
        if self._windows:
            Snakes.screen = None
            pygame.display.quit()

    def rand_pos(self):
        return random.randrange(WIDTH)+VIEW_X0, random.randrange(HEIGHT)+VIEW_Y0

    def rands_pos(self, nr, range, offset=0):
        return np.random.randint(offset, offset+range, nr, TYPE_POS)

    def score(self):
        return self._score

    def head_x(self):
        return self._snake_head_x
        # return self._snake_head_x[self._all_snakes, self._snake_head & MASK]

    def head_y(self):
        return self._snake_head_y
        # return self._snake_body_y[self._all_snakes, self._snake_head & MASK]

    def head_set_x(self, value):
        self._snake_head_x = value
        self._snake_body_x[self._all_snakes, self._snake_head] = value;

    def head_set_y(self, value):
        self._snake_head_y = value
        self._snake_body_y[self._all_snakes, self._snake_head] = value;

    def tail(self):
        return self._snake_body[(self._snake_head - self._body_length) & MASK]

    def view_string(self):
        port = self.view_port()
        str = ""
        for y in range(VIEW_HEIGHT):
            str = str + "|"
            for x in range(VIEW_WIDTH):
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

    def new_apple(self):
        if self._body_length+1 >= AREA:
            raise(AssertionError("No place for apples"))
        self._field[INDEX_APPLE, self._apple[1], self._apple[2]] = 0
        # If we ever get good enough to almost fill the screen this will be slow
        while True:
            self._apple = self.rand_pos()
            if (self._field[self._apple] == 0):
                break
        self._field[INDEX_APPLE, self._apple[1], self._apple[2]] = 1
        # print("apple at [%d, %d, %d]" % self._apple)

    def draw_start(self):
        self.draw_blocks(self.last_collision_x, self.last_collision_y, Snakes.WALL)
        self.last_collision = INDEX_SNAKE,VIEW_X0-1,VIEW_Y0-1
        for i in range(self._windows):
            rect = (self._window_x[i]+1) * BLOCK, (self._window_y[i]+1) * BLOCK, WIDTH*BLOCK, HEIGHT*BLOCK
            Snakes.updates.append(rect)
            pygame.draw.rect(Snakes.screen, Snakes.BACKGROUND, rect)
        self.draw_heads()
        self.draw_apples()

    def draw_block(self, x, y, color):
        # print("Draw (%d,%d,%d): %d,%d,%d" % (pos+color))
        rect = x*BLOCK+EDGE, y*BLOCK+EDGE, DRAW_BLOCK, DRAW_BLOCK
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

    def draw_body(self, pos):
        self.draw_block(pos, Snakes.BODY)

    def draw_collision(self, pos):
        self.last_collision = pos
        self.draw_block(pos, Snakes.COLLISION)

    def draw_pre_move(self):
        self.draw_block(self.head(), Snakes.BODY)
        self.draw_block(self.tail(), Snakes.BACKGROUND)

    def draw_pre_eat(self):
        self.draw_block(self.head(), Snakes.BODY)

    def move(self, pos):
        self._field[self.tail()] = 0
        self._snake_head = self._snake_head+1 & MASK
        self.head_set(pos)
        self._field[self.head()] = 1

    def eat(self, pos):
        self._snake_head = self._snake_head+1 & MASK
        self._body_length = self._body_length +1
        self.head_set(pos)
        self._field[self.head()] = 1
        self._score = self._score+1

    def collision_indices(self, pos, indices):
        x, y = pos
        if indices.size == self._nr_snakes:
            return self._field[self._all_snakes, y, x].nonzero()[0]
        return self._field[indices, y, x].nonzero()[0]

    def plan_greedy(self):
        x = self.head_x()
        y = self.head_y()
        if self._active.size != self._nr_snakes:
            x = x[self._active]
            y = y[self._active]

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

    def plan_random(self):
        index,x, y = self.head()
        directions = [
            (index, x+1, y),
            (index, x-1, y),
            (index, x, y+1),
            (index, x,y-1)]
        random.shuffle(directions)
        for new_pos in directions:
            if not self.collision(new_pos):
                return new_pos
        return directions[0]

    def update(self):
        # pygame.display.update()
        if Snakes.updates:
            pygame.display.update(Snakes.updates)
            Snakes.updates = []

    def frames(self):
        return self._frames

    def elapsed(self):
        return self._elapsed

    def frame_rate(self):
        return self.frames() / self.elapsed()

    def draw_run(self, fps=40):
        self.restart()
        self.draw_start()
        # print("New game, head=%d [%d, %d]" % self.head())
        # print(self.view_string())

        clock = pygame.time.Clock()
        frames = 0
        start_time = timeit.default_timer()
        while True:
            self.update()
            waiting = True
            # while waiting:
            if waiting:
                clock.tick(fps)
                frames += 1
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            self._elapsed = timeit.default_timer() - start_time
                            self._frames  = frames
                            return False
                        waiting = False
            # continue
            print("Head_x ", self.head_x())
            print("Head_y ", self.head_y())
            print("Apple_x", self._apple_x)
            print("Apple_y", self._apple_y)
            new_pos = self.plan_greedy()
            collisions = self.collision_indices(new_pos, self._active)
            # print(collisions)
            if collisions.size:
                new_pos[collisions] = self.plan_random(collisions if self._active.size == self._nr_snakes else self._active[collisions])
            print("Move x ", new_pos[0])
            print("Move y ", new_pos[1])
            collisions = self.collision_indices(new_pos, self._active)
            if collisions.size:
                self.draw_collision(new_pos)
                self.update()
                clock.tick(fps)
                self._elapsed = timeit.default_timer() - start_time
                self._frames  = frames +1
                # print(self.view_string())
                return True

            if new_pos == self._apple:
                self.draw_pre_eat()
                self.eat(new_pos)
                self.new_apple()
                self.draw_apple()
                # print("Score", self.score())
            elif self.collision(new_pos):
            else:
                self.draw_pre_move()
                self.move(new_pos)
            self.draw_head()

            #print("Head at %d [%d,%d]" % self.head())
            #print(self._field.swapaxes(1,2))
            #print(self.view_string())

    # Don't set fps to 0
    # I tried with set_timer, but the first trigger seems to be immediate
    # Try again when pygame 2 is released (timers get a new "once" option)
    def wait_escape(self, period=0, fps=40):
        target_time = timeit.default_timer() + period
        clock = pygame.time.Clock()
        while timeit.default_timer() < target_time:
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    return True
        return False


# +
snakes = Snakes(nr_snakes=6)
snakes.display_start(3,2)

pause = float(arguments["--pause"])
while snakes.draw_run(fps=float(arguments["--fps"])):
    pass
print("Score", snakes.score(), "Framerate", snake.frame_rate())

snake.display_stop()
# -
