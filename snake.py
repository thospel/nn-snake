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

# + {"language": "html"}
# <style>.container {width:100% !important;}</style>

# +
"""Snake

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



# +
clock = pygame.time.Clock()

WIDTH  = 40
HEIGHT = 40
EDGE=1
AREA=WIDTH*HEIGHT
BLOCK=20
DRAW_BLOCK = BLOCK-2*EDGE
SIZE2 = 1<<(AREA-1).bit_length()
MASK = SIZE2-1

VIEW_X0 = 9
VIEW_Y0 = 9
VIEW_X2 = VIEW_X0+2
VIEW_Y2 = VIEW_Y0+2
VIEW_WIDTH  = 2*VIEW_X0+1
VIEW_HEIGHT = 2*VIEW_Y0+1

INDEX_SNAKE = 0
INDEX_APPLE = 1
INDEX_WALL  = 2
INDEX_MAX   = 3

class Snake:
    WALL  = 255,255,255
    BODY  = 160,160,160
    HEAD  = 200,200,0
    BACKGROUND = 0,0,0
    APPLE = 0,255,0
    COLLISION = 255,0,0

    def __init__(self):
        # Notice that we store in column major order
        self.field = np.zeros((INDEX_MAX, WIDTH+2*VIEW_X0, HEIGHT+2*VIEW_Y0), np.float32)
        self.field[INDEX_WALL] = 1
        self.field[INDEX_WALL, VIEW_X0:WIDTH+VIEW_X0,VIEW_Y0:HEIGHT+VIEW_Y0] = 0
        self.snake_body = [None]*SIZE2
        self.snake_head  = 0
        # Body length measures the snake without a head
        self.body_length = 0
        self.apple = INDEX_SNAKE,0,0

    # You can only have one pygame instance in one process,
    # so make display related variables into class variables
    def display_start(self):
        # Avoid pygame.init() since the init of the mixer component leads to 100% CPU
        pygame.display.init()
        Snake.last_collision = INDEX_SNAKE,VIEW_X0-1,VIEW_Y0-1
        Snake.screen = pygame.display.set_mode(((WIDTH+2)*BLOCK, (HEIGHT+2)*BLOCK))
        rect = 0, 0, (WIDTH+2)*BLOCK, (HEIGHT+2)*BLOCK
        Snake.updates = [rect]
        pygame.draw.rect(Snake.screen, Snake.WALL, rect)
        pygame.display.set_caption('Snake')
        # pygame.mouse.set_visible(1)

    def display_stop(self):
        Snake.screen = None
        pygame.display.quit()

    def rand_pos(self):
        return INDEX_SNAKE, random.randrange(WIDTH)+VIEW_X0, random.randrange(HEIGHT)+VIEW_Y0

    def score(self):
        return self._score

    def head(self):
        return self.snake_body[self.snake_head & MASK]

    def head_set(self, value):
        self.snake_body[self.snake_head & MASK] = value;

    def tail(self):
        return self.snake_body[(self.snake_head - self.body_length) & MASK]

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
                elif v[0] != 0:
                    str += "O"
                elif v[1] != 0:
                    str += "@"
                else:
                    str += "X"
            str += "|\n"
        return str

    def view_port(self):
        index,x,y = self.head()
        return self.field[:,x-VIEW_X0:x+VIEW_X2,y-VIEW_Y0:y+VIEW_Y2]

    def restart(self):
        self._score = 0
        self.body_length = 0
        self.head_set(self.rand_pos())
        self.field[INDEX_SNAKE, VIEW_X0:WIDTH+VIEW_X0,VIEW_Y0:HEIGHT+VIEW_Y0] = 0
        self.field[self.head()] = 1

        self.new_apple()

    def new_apple(self):
        if self.body_length+1 >= AREA:
            raise(AssertionError("No place for apples"))
        self.field[INDEX_APPLE, self.apple[1], self.apple[2]] = 0
        # If we ever get good enough to almost fill the screen this will be slow
        while True:
            self.apple = self.rand_pos()
            if (self.field[self.apple] == 0):
                break
        self.field[INDEX_APPLE, self.apple[1], self.apple[2]] = 1
        # print("apple at [%d, %d, %d]" % self.apple)

    def draw_start(self):
        self.draw_block(Snake.last_collision, Snake.WALL)
        Snake.last_collision = INDEX_SNAKE,VIEW_X0-1,VIEW_Y0-1
        rect = BLOCK, BLOCK, WIDTH*BLOCK, HEIGHT*BLOCK
        Snake.updates.append(rect)
        pygame.draw.rect(Snake.screen, Snake.BACKGROUND, rect)
        self.draw_head()
        self.draw_apple()

    def draw_block(self, pos, color):
        # print("Draw (%d,%d,%d): %d,%d,%d" % (pos+color))
        rect = ((pos[1]-VIEW_X0+1)*BLOCK+EDGE,
                (pos[2]-VIEW_Y0+1)*BLOCK+EDGE,
                DRAW_BLOCK, DRAW_BLOCK)
        Snake.updates.append(rect)
        pygame.draw.rect(Snake.screen, color, rect)

    def draw_apple(self):
        self.draw_block(self.apple, Snake.APPLE)

    def draw_head(self):
        self.draw_block(self.head(), Snake.HEAD)

    def draw_body(self, pos):
        self.draw_block(pos, Snake.BODY)

    def draw_collision(self, pos):
        Snake.last_collision = pos
        self.draw_block(pos, Snake.COLLISION)

    def draw_pre_move(self):
        self.draw_block(self.head(), Snake.BODY)
        self.draw_block(self.tail(), Snake.BACKGROUND)

    def draw_pre_eat(self):
        self.draw_block(self.head(), Snake.BODY)

    def move(self, pos):
        self.field[self.tail()] = 0
        self.snake_head = self.snake_head+1 & MASK
        self.head_set(pos)
        self.field[self.head()] = 1

    def eat(self, pos):
        self.snake_head = self.snake_head+1 & MASK
        self.body_length = self.body_length +1
        self.head_set(pos)
        self.field[self.head()] = 1
        self._score = self._score+1

    def collision(self, pos):
        return self.field[pos] or self.field[INDEX_WALL, pos[1], pos[2]] != 0

    def plan_greedy(self):
        index,x,y = self.head()
        dx = self.apple[1] - x
        dy = self.apple[2] - y
        if dx == 0 and dy == 0:
            raise(AssertionError("Head is on apple"))
        if abs(dx) > abs(dy):
            if dx > 0:
                return index,x+1,y
            else:
                return index,x-1,y
        else:
            if dy > 0:
                return index,x,y+1
            else:
                return index,x,y-1

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
        pygame.display.update(Snake.updates)
        Snake.updates = []

    def frames(self):
        return self._frames

    def elapsed(self):
        return self._elapsed

    def frame_rate(self):
        return self.frames() / self.elapsed()

    def draw_run(self, fps=20):
        self.restart()
        self.draw_start()
        # print("New game, head=%d [%d, %d]" % self.head())
        # print(self.view_string())

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
            new_pos = self.plan_greedy()
            if self.collision(new_pos):
                new_pos = self.plan_random()
            # print("Move to %d,%d" % new_pos)
            if new_pos == self.apple:
                self.draw_pre_eat()
                self.eat(new_pos)
                self.new_apple()
                self.draw_apple()
                # print("Score", self.score())
            elif self.collision(new_pos):
                self.draw_collision(new_pos)
                self.update()
                clock.tick(fps)
                self._elapsed = timeit.default_timer() - start_time
                self._frames  = frames +1
                # print(self.view_string())
                return True
            else:
                self.draw_pre_move()
                self.move(new_pos)
            self.draw_head()

            #print("Head at %d [%d,%d]" % self.head())
            #print(self.field.swapaxes(1,2))
            #print(self.view_string())

# +
snake = Snake()
snake.display_start()

pause = float(arguments["--pause"])
while snake.draw_run(fps=float(arguments["--fps"])):
    print("Score", snake.score(), "Framerate", snake.frame_rate())
    pygame.time.wait(int(pause*1000))

snake.display_stop()
