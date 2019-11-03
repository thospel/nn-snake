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
# -

# %matplotlib inline
from matplotlib import pyplot as plt

import random
import math
import numpy as np
import pygame
from pygame.locals import *

abs(-6)

# +
clock = pygame.time.Clock()

WIDTH=40
HEIGHT=40
EDGE=1
AREA=WIDTH*HEIGHT
BLOCK=20
DRAW_BLOCK = BLOCK-2*EDGE
SIZE2 = 1<<(AREA-1).bit_length()
MASK = SIZE2-1

class Snake:
    WALL  = 255,255,255
    BODY  = 160,160,160
    HEAD  = 200,200,0
    BACKGROUND = 0,0,0
    APPLE = 0,255,0
    COLLISION = 255,0,0

    def __init__(self):
        self.field = np.ones((HEIGHT+2, WIDTH+2), np.float32)
        self.snake_body = [None]*SIZE2
        self.snake_head  = 0
        # Body length measures the snake without a head
        self.body_length = 0
        self.apple = (0,0)

    def display_start(self):
        self.last_collision = 0,0
        self.screen = pygame.display.set_mode(((WIDTH+2)*BLOCK, (HEIGHT+2)*BLOCK))
        pygame.draw.rect(self.screen, self.WALL, (0, 0, (WIDTH+2)*BLOCK, (HEIGHT+2)*BLOCK), 0)
        pygame.display.set_caption('Snake')
        # pygame.mouse.set_visible(1)
        pygame.init()

    def display_stop(self):
        pygame.display.quit()

    def rand_pos(self):
        return random.randrange(WIDTH)+1, random.randrange(HEIGHT)+1

    def score(self):
        return self._score

    def head(self):
        return self.snake_body[self.snake_head & MASK]

    def head_set(self, value):
        self.snake_body[self.snake_head & MASK] = value;

    def tail(self):
        return self.snake_body[(self.snake_head - self.body_length) & MASK]

    def new_snake(self):
        self._score = 0
        self.body_length = 0
        self.head_set(self.rand_pos())
        self.field[1:HEIGHT+1,1:WIDTH+1] = 0
        self.field[self.head()] = 1

    def new_apple(self):
        if self.body_length+1 >= AREA:
            raise(AssertionError("No place for apples"))
        while True:
            self.apple = self.rand_pos()
            if (self.field[self.apple] == 0): break
        # print("apple at [%d, %d]" % self.apple)

    def draw_start(self):
        self.draw_block(self.last_collision, self.WALL)
        self.last_collision = 0,0
        pygame.draw.rect(self.screen, self.BACKGROUND, (BLOCK, BLOCK, WIDTH*BLOCK, HEIGHT*BLOCK), 0)
        self.draw_head()
        self.draw_apple()

    def draw_block(self, pos, color):
        # print("Draw (%d,%d): %d,%d,%d" % (pos+color))
        pygame.draw.rect(self.screen, color,
                         (pos[0]*BLOCK+EDGE, pos[1]*BLOCK+EDGE, DRAW_BLOCK, DRAW_BLOCK),0)

    def draw_apple(self):
        self.draw_block(self.apple, self.APPLE)

    def draw_head(self):
        self.draw_block(self.head(), self.HEAD)

    def draw_body(self, pos):
        self.draw_block(pos, self.BODY)

    def draw_collision(self, pos):
        self.last_collision = pos
        self.draw_block(pos, self.COLLISION)

    def draw_pre_move(self):
        self.draw_block(self.head(), self.BODY)
        self.draw_block(self.tail(), self.BACKGROUND)

    def draw_pre_eat(self):
        self.draw_block(self.head(), self.BODY)

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
        return self.field[pos] != 0

    def plan_greedy(self):
        x,y = self.head()
        dx = self.apple[0] - x
        dy = self.apple[1] - y
        if dx == 0 and dy == 0:
            raise(AssertionError("Head is on apple"))
        if abs(dx) > abs(dy):
            if dx > 0:
                return x+1,y
            else:
                return x-1,y
        else:
            if dy > 0:
                return x,y+1
            else:
                return x,y-1

    def plan_random(self):
        x, y = self.head()
        directions = [
            (x+1, y),
            (x-1, y),
            (x, y+1),
            (x,y-1)]
        random.shuffle(directions)
        for new_pos in directions:
            if not self.collision(new_pos):
                return new_pos
        return directions[0]

    def update(self):
        pygame.display.update()

    def run(self, fps=20):
        #print("New game")
        self.new_snake()
        self.new_apple()
        self.draw_start()

        while True:
            # print("Head at %d,%d" % self.head())
            self.update()
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    return False
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
                return True
            else:
                self.draw_pre_move()
                self.move(new_pos)
            self.draw_head()
# + {}
snake = Snake()
snake.display_start()

while snake.run(fps=40):
    print("Score was", snake.score())


snake.display_stop()
# -

not snake.field[3,4] == 0
