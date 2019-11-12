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
           [--width=<width>] [--height=<height>] [--frames=<frames>]
           [--columns=columns] [--rows=rows] [--block=<block_size>]
  snake.py --benchmark
  snake.py (-h | --help)
  snake..py --version

Options:
  -h --help             Show this screen
  --version             Show version
  --stepping            Start in paused mode, wait for the user to press SPACE
  --fps=<fps>           Frames per second (0 is no delays) [default: 40]
  --snakes=<snakes>     How many snakes to run at the same time [default: 0]
                        0 means use rows * colums or 1
  --block=<block_size>  Block size in pixels [default: 20]
  --width=<width>       Pit width  in blocks [default: 40]
  --height=<height>     Pit height in blocks [default: 40]
  --columns=<columns>   Columns of pits to display [default: 2]
  --rows=<rows>         Rows of pits to display [default: 1]
  --frames=<frames>     Stop automatically at this frames number [Default: -1]
  --benchmark           Run a simple speed benchmark
  -f <file>:            Used by jupyter, ignored

Display key actions:
  s:          enter pause mode after doing a single step
  r, SPACE:   toggle run/pause mode
  q, <close>: quit
  +:          More frames per second (wait time /= 2)
  -:          Less frames per second (wait time *= 2)
  =:          Restore the original frames per second

"""
from docopt import docopt

DEFAULTS = docopt(__doc__, [])
# print(DEFAULTS)

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
import time
import numpy as np

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
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
    skip:        str = " "

@dataclass
class TextData:
    y:          int
    prefix_x:   int
    prefix:     str
    format_x:   int
    format:     str
    max_width:  int
    old_text:   str
    old_rect:   List[pygame.Rect]

class TextRows(_TextRows):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        rect = self.font.get_rect(self.skip)
        self.skip_width = rect.width
        # assume 8 is widest
        rect = self.font.get_rect("8")
        self.digit_width = rect.width
        self._lookup     = {}
        self._lookup_row = {}

    def add(self, row_name, text_row, windows=1):
        if row_name in self._lookup_row:
            raise(AssertionError("Duplicate row name %s", row_name))
        for text_field in text_row.text_fields:
            if text_field.name in self._lookup:
                raise(AssertionError("Duplicate field name %s", text_field.name))

        x = text_row.x
        y = text_row.y
        row_data = []
        for text_field in text_row.text_fields:
            pos_prefix = x
            rect = self.font.get_rect(text_field.prefix)
            x += rect.width
            pos_format = x
            x += self.digit_width * text_field.width + self.skip_width

            text_data = TextData(
                y         = y,
                prefix_x  = pos_prefix,
                prefix    = text_field.prefix,
                format_x  = pos_format,
                format    = "%%%du" % text_field.width,
                max_width = text_row.max_width,
                old_text  = [None]*windows,
                old_rect  = [None]*windows)
            self._lookup[text_field.name] = text_data
            row_data.append(text_data)
        self._lookup_row[row_name] = row_data

    def lookup(self, name):
        return self._lookup[name]

    def lookup_row(self, name):
        return self._lookup_row[name]

# +
ROW_TOP = [
    TextField("score", "Score:", 5),
    TextField("game",  "Game:",  5),
    TextField("moves", "Moves:", 7),
    TextField("snake", "Id:",    3),
    # TextField("x",     "x:",     3),
    # TextField("y",     "y:",     3),
]

ROW_BOTTOM = [
    TextField("step",        "Step:", 7),
    TextField("score_max",   "Max Score:", 4),
    TextField("moves_max",   "Max Moves:", 7),
    TextField("game_max",    "Max Game:",  5),
    TextField("score_per_snake", "Score/Snake:", 4),
    TextField("score_per_game",  "Score/Game:",  4),
    TextField("moves_per_game",  "Moves/Game:",  7),
    TextField("moves_per_apple", "Moves/Apple:", 4),
    TextField("time",        "Time:", 7),
    # Put games last. If you have a lot of snakes this can go up very fast
    TextField("games",       "Games:",  7),
]

# This must be signed because window offsets can be negative
TYPE_PIXELS = np.int32

class Display:
    KEY_INTERVAL = int(1000 / 20)  # twenty per second
    KEY_DELAY = 500                # Start repeating after half a second

    WALL  = 255,255,255
    BODY  = 160,160,160
    HEAD  = 200,200,0
    BACKGROUND = 0,0,0
    APPLE = 0,255,0
    COLLISION = 255,0,0

    EDGE=1

    _updates_count = 0
    _updates_time  = 0

    # we test these at the start of some functions
    # Make sure they have a "nothing to see here" value in case __init__ fails
    _screen = None
    _updates = []

    # You can only have one pygame instance in one process,
    # so make display related variables into class variables
    def __init__(self, snakes,
                 columns    = int(DEFAULTS["--columns"]),
                 rows       = int(DEFAULTS["--rows"]),
                 block_size = int(DEFAULTS["--block"]),
                 caption="Snakes", slow_updates=0):
        self.windows = rows*columns
        if not self.windows:
            return

        self.caption = caption
        self._slow_updates = slow_updates

        self.BLOCK = block_size
        self.DRAW_BLOCK = self.BLOCK-2*Display.EDGE

        # coordinates relative to the upper left corner of the window
        self.TOP_TEXT_X  = self.BLOCK
        self.TOP_TEXT_Y  = self.DRAW_BLOCK

        # coordinates relative to the bottom left corner of the screen
        self.BOTTOM_TEXT_X  = self.BLOCK
        self.BOTTOM_TEXT_Y  = self.DRAW_BLOCK - self.BLOCK
        # self.BOTTOM_TEXT_Y  = -Display.EDGE

        self.WINDOW_X = (snakes.WIDTH +2) * self.BLOCK
        self.WINDOW_Y = (snakes.HEIGHT+2) * self.BLOCK
        self.OFFSET_X = (1-snakes.VIEW_X) * self.BLOCK
        self.OFFSET_Y = (1-snakes.VIEW_Y) * self.BLOCK

        self._window_x = np.tile  (np.arange(self.OFFSET_X, columns*self.WINDOW_X+self.OFFSET_X, self.WINDOW_X, dtype=TYPE_PIXELS), rows)
        self._window_y = np.repeat(np.arange(self.OFFSET_Y, rows   *self.WINDOW_Y+self.OFFSET_Y, self.WINDOW_Y, dtype=TYPE_PIXELS), columns)
        # print("window_x", self._window_x)
        # print("window_y", self._window_y)

        # Fixup for window offset
        self.TOP_WIDTH = self.WINDOW_X-self.TOP_TEXT_X
        self.TOP_TEXT_X -= self.OFFSET_X
        self.TOP_TEXT_Y -= self.OFFSET_Y
        self.BOTTOM_WIDTH = self.WINDOW_X * columns -self.BOTTOM_TEXT_X
        self.BOTTOM_TEXT_X -= self.OFFSET_X
        self.BOTTOM_TEXT_Y -= self.OFFSET_Y
        self.BOTTOM_TEXT_Y += rows * self.WINDOW_Y

        # self.last_collision_x = np.zeros(self.windows, dtype=TYPE_PIXELS)
        # self.last_collision_y = np.zeros(self.windows, dtype=TYPE_PIXELS)

    def start(self):
        if not self.windows:
            return

        # Avoid pygame.init() since the init of the mixer component leads to 100% CPU
        pygame.display.init()
        pygame.display.set_caption(self.caption)
        # pygame.mouse.set_visible(1)
        pygame.key.set_repeat(Display.KEY_DELAY, Display.KEY_INTERVAL)

        pygame.freetype.init()
        self._font = pygame.freetype.Font(None, self.BLOCK)
        self._font.origin = True

        self._textrows = TextRows(font=self._font)
        self._textrows.add("top",
                           TextRow(self.TOP_TEXT_X,
                                   self.TOP_TEXT_Y,
                                   self.TOP_WIDTH, ROW_TOP), self.windows)

        self._textrows.add("bottom",
                           TextRow(self.BOTTOM_TEXT_X,
                                   self.BOTTOM_TEXT_Y,
                                   self.BOTTOM_WIDTH, ROW_BOTTOM))

        Display._screen = pygame.display.set_mode((self.WINDOW_X * columns, self.WINDOW_Y * rows))
        rect = 0, 0, self.WINDOW_X * columns, self.WINDOW_Y * rows
        rect = pygame.draw.rect(Display._screen, Display.WALL, rect)
        for w in range(self.windows):
            self.draw_text_row("top", w)
        self.draw_text_row("bottom", 0)
        Display._updates = [rect]

    def stop(self):
        if not Display._screen:
            return

        Display._screen  = None
        Display._updates = []
        pygame.quit()
        if self._slow_updates:
            if Display._updates_count:
                print("Average disply update time: %.6f" %
                      (Display._updates_time / Display._updates_count))
            else:
                print("No display updates")

    def __enter__(self):
        if Display._screen:
            raise(AssertionError("Attempt to start multiple displays at the same time"))
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # Don't do a stop() in __del__ since object desctruction can be delayed
    # and a new display can have started

    def update(self):
        if Display._updates:
            if self._slow_updates:
                self._time_start  = time.monotonic()
                pygame.display.update(Display._updates)
                period = time.monotonic() - self._time_start
                if period > self._slow_updates:
                    print("Update took %.4f" % period)
                    for rect in Display._updates:
                        print("    ", rect)
                Display._updates_count +=1
                Display._updates_time  += period
            else:
                pygame.display.update(Display._updates)
            Display._updates = []

    def events_get(self, to_sleep):
        if to_sleep > 0:
            time.sleep(to_sleep)
        keys = []
        if Display._screen:
            events = pygame.event.get()
            if events:
                for event in events:
                    if event.type == QUIT:
                        keys.append("q")
                    elif event.type == KEYDOWN:
                        keys.append(event.unicode)
        return keys

    def draw_text_row(self, row_name, w, fg_color=BACKGROUND, bg_color=WALL):
        if not Display._screen:
            return

        for text_data in self._textrows.lookup_row(row_name):
            text = text_data.prefix
            x = text_data.prefix_x
            y = text_data.y

            # Draw new text
            rect = self._font.get_rect(text)
            rect.x += x
            if rect.x + rect.width > text_data.max_width:
                return
            x += self._window_x[w]
            y += self._window_y[w]
            rect.x += self._window_x[w]
            rect.y = y - rect.y
            # print("Draw text", w, x, y, '"%s"' % text, x + self._window_x[w], y + self._window_y[w], rect, old_rect)
            self._font.render_to(Display._screen, (x, y), None, fg_color, bg_color)
            # We could union all rects, but for now this is only called on
            # start() and start() already forces its own single rect
            Display._updates.append(rect)

    def draw_text(self, w, name, value=None,
                       fg_color=BACKGROUND, bg_color=WALL):
        if not Display._screen:
            return

        text_data = self._textrows.lookup(name)

        if value is None:
            text = text_data.prefix
            old_rect = None
            x = text_data.prefix_x
        else:
            text = text_data.format % value
            old_text = text_data.old_text[w]
            if old_text is not None and text == old_text:
                # Update does nothing
                return
            # Erase old text
            old_rect = text_data.old_rect[w]
            if old_rect:
                pygame.draw.rect(Display._screen, bg_color, old_rect)
            x = text_data.format_x
        y = text_data.y

        # Draw new text
        rect = self._font.get_rect(text)
        rect.x += x
        if rect.x + rect.width > text_data.max_width:
            if value is not None and old_rect is not None:
                Display._updates.append(old_rect)
                text_data.old_rect[w] = None
            return
        x += self._window_x[w]
        y += self._window_y[w]
        rect.x += self._window_x[w]
        rect.y = y - rect.y
        # print("Draw text", w, x, y, '"%s"' % text, x + self._window_x[w], y + self._window_y[w], rect, old_rect)
        self._font.render_to(Display._screen, (x, y), None, fg_color, bg_color)
        if value is None:
            Display._updates.append(rect)
        else:
            if old_rect is None:
                Display._updates.append(rect)
            else:
                Display._updates.append(rect.union(old_rect))
            # Remember what we updated
            text_data.old_rect[w] = rect
            text_data.old_text[w] = text

    def draw_field_empty(self, w):
        rect = (self._window_x[w] - self.OFFSET_X + self.BLOCK,
                self._window_y[w] - self.OFFSET_Y + self.BLOCK,
                self.WINDOW_X - 2 * self.BLOCK,
                self.WINDOW_Y - 2 * self.BLOCK)
        rect = pygame.draw.rect(Display._screen, Display.BACKGROUND, rect)
        Display._updates.append(rect)

    def draw_block(self, w, x, y, color, update=True):
        rect = (x * self.BLOCK + self._window_x[w] + Display.EDGE,
                y * self.BLOCK + self._window_y[w] + Display.EDGE,
                self.DRAW_BLOCK,
                self.DRAW_BLOCK)

        # print("Draw %d (%d,%d): %d,%d,%d: [%d %d %d %d]" % ((w, x, y)+color+(rect)))
        rect = pygame.draw.rect(Display._screen, color, rect)
        if update:
            Display._updates.append(rect)
        return rect

    def draw_collisions(self, i_index, w_index, pos, nr_games):
        y , x = pos
        for w, i, x, y in zip(w_index, i_index, pos[1], pos[0]):
            if False:
                # This test was for when the idea was to freeze some snakes
                # if self._nr_moves[w] > self._cur_move:
                self.draw_block(w, x, y, Display.COLLISION)
            else:
                #self.draw_block(w,
                #                self.last_collision_x[w],
                #                self.last_collision_y[w],
                #                Display.WALL)
                self.draw_text(w, "game", nr_games[i])
                self.draw_text(w, "snake", i)
                # self.draw_text(w, "x")
                # self.draw_text(w, "y")
                self.draw_field_empty(w)

    def draw_move(self,
                  all_windows, w_head_new,
                  is_collision, w_head_old,
                  eat, w_tail,
                  w_nr_moves):
        w_head_y_new, w_head_x_new = w_head_new
        w_head_y_old, w_head_x_old = w_head_old
        w_tail_y, w_tail_x = w_tail
        for w in range(all_windows.size):
            i = all_windows[w]
            if is_collision[i]:
                body_rect = None
            else:
                # The current head becomes body
                # (For length 1 snakes the following tail erase will undo this)
                body_rect = self.draw_block(w, w_head_x_old[w], w_head_y_old[w], Display.BODY, update=False)
            if not eat[i]:
                # Drop the tail if we didn't eat an apple then
                self.draw_block(w, w_tail_x[w], w_tail_y[w], Display.BACKGROUND)
            if body_rect:
                head_rect = self.draw_block(w, w_head_x_new[w], w_head_y_new[w], Display.HEAD, update=False)
                Display._updates.append(head_rect.union(body_rect))
            else:
                self.draw_block(w, w_head_x_new[w], w_head_y_new[w], Display.HEAD)
            self.draw_text(w, "moves", w_nr_moves[w])
            # self.draw_text(w, "x", w_head_x_new[w])
            # self.draw_text(w, "y", w_head_y_new[w])

    def draw_apples(self, i_index, w_index, apple, score):
        apple_y, apple_x = apple
        # print("draw_apples", i_index, w_index, score)
        # print(np.dstack((apple_x, apple_y)))
        for i, w, x, y in zip(i_index, w_index, apple_x, apple_y):
            self.draw_block(w, x, y, Display.APPLE)
            self.draw_text(w, "score", score[i])


# +
"""
import tensorflow as tf

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        # return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        # sample random categorical actions from given logits
        return tf.random.categorical(logits, 1)

class ActorCriticModel(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(100, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(100, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
        return logits, values

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        # return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
        return action, value

3
"""

# -

def np_empty(shape, type):
    # return np.empty(shape, type)
    # Fill with garbage for debug
    return np.random.randint(100, size=shape, dtype=type)

def print_xy(text, x, y):
    print(text)
    print(np.dstack((x, y)))

def print_yx(text, pos):
    print_xy(text, pos[1], pos[0])


# +

# TYPE_POS = np.uint8
TYPE_POS   = np.int16
TYPE_BOOL  = np.bool
TYPE_INDEX = np.intp
# Using bool for TYPE_FLAG is about 5% faster
# TYPE_FLAG  = np.uint8
TYPE_FLAG  = np.bool
TYPE_SCORE = np.uint32
TYPE_MOVES = np.uint32
TYPE_GAMES = np.uint32

VIEW_X0 = 1
VIEW_Y0 = 1
VIEW_X2 = VIEW_X0+2
VIEW_Y2 = VIEW_Y0+2
VIEW_WIDTH  = 2*VIEW_X0+1
VIEW_HEIGHT = 2*VIEW_Y0+1

class Snakes:
    # Event polling time in paused mode.
    # Avoid too much CPU waste or even a busy loop in case fps == 0
    POLL_SLOW = 1/25
    POLL_MAX = POLL_SLOW * 1.5
    WAIT_MIN = 1/1000

    # Possible directions for a random walk
    DIRECTIONS0 = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    DIRECTIONS0_X = DIRECTIONS0[:,0]
    DIRECTIONS0_Y = DIRECTIONS0[:,1]
    NR_DIRECTIONS = len(DIRECTIONS0)

    DIRECTION_PERMUTATIONS0 = np.array(list(itertools.permutations(DIRECTIONS0)), dtype=TYPE_POS)
    NR_DIRECTION_PERMUTATIONS = len(DIRECTION_PERMUTATIONS0)
    DIRECTION_PERMUTATIONS0_X=DIRECTION_PERMUTATIONS0[:,:,0]
    DIRECTION_PERMUTATIONS0_Y=DIRECTION_PERMUTATIONS0[:,:,1]

    def __init__(self, nr_snakes=1,
                 width     = int(DEFAULTS["--width"]),
                 height    = int(DEFAULTS["--height"]),
                 frame_max = int(DEFAULTS["--frames"]),
                 view_x=0, view_y=0):
        if nr_snakes <= 0:
            raise(ValueError("Number of snakes must be positive"))

        self.windows = None

        self._nr_snakes = nr_snakes
        self._all_snakes = np.arange(nr_snakes, dtype=TYPE_INDEX)
        self._frame_max = frame_max

        self.VIEW_X = view_x or VIEW_X0
        self.VIEW_Y = view_y or VIEW_Y0
        self.WIDTH  = width
        self.HEIGHT = height
        self.AREA   = self.WIDTH * self.HEIGHT
        # First power of 2 greater or equal to AREA for fast modular arithmetic
        self.AREA2 = 1 << (self.AREA-1).bit_length()
        self.MASK  = self.AREA2 - 1

        self.HEIGHT1 = self.HEIGHT+2*self.VIEW_Y
        self.WIDTH1  = self.WIDTH +2*self.VIEW_X

        self.DIRECTIONS = Snakes.DIRECTIONS0_X + Snakes.DIRECTIONS0_Y * self.WIDTH1
        self.DIRECTION_PERMUTATIONS = Snakes.DIRECTION_PERMUTATIONS0_X + Snakes.DIRECTION_PERMUTATIONS0_Y * self.WIDTH1

        # Pit is just the edges
        self._empty_pit = np.ones((self.HEIGHT1, self.WIDTH1), dtype=TYPE_FLAG)
        self._empty_pit[self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH] = 0
        # self._field1 = np.ones((nr_snakes, self.HEIGHT1, self.WIDTH1), dtype=TYPE_FLAG)

        # The playing field starts out as nr_snakes copies of the empty pit
        # Notice that we store in row major order, so use field[y,x]
        self._field1 = self._empty_pit.reshape(1,self.HEIGHT1,self.WIDTH1).repeat(nr_snakes, axis=0)
        self._field0 = self._field1[:, self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH]
        if self._field0.base is not self._field1:
            raise(AssertionError("field0 is a copy instead of a view"))
        self._field = self._field1.reshape(nr_snakes, self.HEIGHT1*self.WIDTH1)
        if self._field.base is not self._field1:
            raise(AssertionError("field0 is a copy instead of a view"))

        # self._apple_pit = np.zero((self.HEIGHT, self.WIDTH, self.HEIGHT, self.WIDTH),

        self._snake_body = np_empty((nr_snakes, self.AREA2), TYPE_POS)

        # Body length measures the snake *without* the head
        # This is therefore also the score (if we start with length 0 snakes)
        self._body_length = np_empty(nr_snakes, TYPE_INDEX)
        self._head     = np_empty(nr_snakes, TYPE_POS)
        self._apple    = np_empty(nr_snakes, TYPE_POS)
        self._nr_moves = np_empty(nr_snakes, TYPE_MOVES)
        self._nr_games = np_empty(nr_snakes, TYPE_GAMES)

    def rand(self, nr):
        offset_x = self.VIEW_X
        rand_x = np.random.randint(offset_x, offset_x+self.WIDTH,  size=nr, dtype=TYPE_POS)
        offset_y = self.VIEW_Y
        rand_y = np.random.randint(offset_y, offset_y+self.HEIGHT, size=nr, dtype=TYPE_POS)
        # print("Rand:", self.WIDTH1)
        # print(np.dstack((rand_x, rand_y)))
        return rand_x + rand_y * self.WIDTH1

    def nr_snakes(self):
        return self._nr_snakes

    def scores(self):
        return self._body_length

    def score(self, i):
        return self._body_length[i]

    def score_max(self):
        return self._score_max

    def score_total_snakes(self):
        return self._score_total_snakes

    def score_total_games(self):
        return self._score_total_games

    def score_per_game(self):
        if self.nr_games_total() <= 0:
            return math.inf * snakes.score_total_games()
        return snakes.score_total_games() / self.nr_games_total()

    # def nr_moves(self):
    #    return self._cur_move - self._nr_moves

    def nr_moves(self, i):
        return self._cur_move - self._nr_moves[i]

    def nr_moves_max(self):
        return self._moves_max

    def nr_moves_total_games(self):
        return self._moves_total_games

    def nr_moves_per_game(self):
        if self.nr_games_total() <= 0:
            return math.inf * snakes.nr_moves_total_games()
        return snakes.nr_moves_total_games() / self.nr_games_total()

    def nr_moves_per_apple(self):
        if self.score_total_games() <= 0:
            return math.inf * snakes.nr_moves_total_games()
        return snakes.nr_moves_total_games() / self.score_total_games()

    def nr_games(self, i):
        return self._nr_games[i]

    def nr_games_max(self):
        return self._nr_games_max

    def nr_games_total(self):
        return self._nr_games_total

    def head(self):
        return self._head

    def head_set(self, head_new):
        self._head = head_new
        offset = self._cur_move & self.MASK
        self._snake_body[self._all_snakes, offset] = head_new
        self._field[self._all_snakes, head_new] = 1

    def tail_set(self, values):
        # print("Eat", values)
        # print("body length", self._body_length)

        # Bring potentially large cur_move into a reasonable range
        # so tail_offset will not use some large integer type
        offset = self._cur_move & self.MASK
        # print("Offset", offset)
        tail_offset = (offset - self._body_length) & self.MASK
        # print("tail offset", tail_offset)
        pos = self._snake_body[self._all_snakes, tail_offset]
        # print_xy("tail pos", x, y))
        self._field[self._all_snakes, pos] = values
        return pos

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
        return self._field1[:,x-VIEW_X0:x+VIEW_X2,y-VIEW_Y0:y+VIEW_Y2]
    """

    def yx(self, array):
        y, x = np.divmod(array, self.WIDTH1)
        # print_xy("yx", x, y)
        return y, x

    # Sprinkle new apples in all pits where the snake ate them (todo)
    def new_apples(self, todo):
        too_large = self._body_length[todo] >= self.AREA-1
        if too_large.any():
            raise(AssertionError("No place for apples"))

        # print("New apples", todo)
        # old_todo = todo.copy()
        while todo.size:
            rand = self.rand(todo.size)
            self._apple[todo] = rand
            fail = self._field[todo, rand]
            # Index with boolean is grep
            todo = todo[fail != 0]
            # print("New apples todo", todo)
        # print_yx("Placed apples", self.yx(self._apple[old_todo]))

    # Plot the shortest course to the apple completely ignoring any snake body
    def plan_greedy(self):
        head = self.head()
        y, x             = self.yx(head)
        apple_y, apple_x = self.yx(self._apple)

        # print_xy("Greedy Heads:", x, y))
        # print_xy("Apples:", apple_x, apple_y))

        dx = apple_x - x
        dy = apple_y - y
        # print_xy("Delta:", np.dstack((dx, dy)))
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
        delta = dx + dy * self.WIDTH1
        return head+delta

    # Pick a completely random direction
    def plan_random(self):
        rand_dir = np.random.randint(Snakes.NR_DIRECTION, size=self._nr_snakes)
        return self.head() + self.DIRECTION[rand_dir]

    # Pick a random direction that isn't blocked
    # Or just a random direction if all are blocked
    # But only for snakes with an index in collided
    def plan_random_unblocked(self, collided):
        # different permutation for each collision
        direction_index = np.random.randint(Snakes.NR_DIRECTION_PERMUTATIONS,
                                            size=collided.size)

        # Currently we randomly generate the whole set of directions from
        # which we will pick first that is not blocked
        # That is a complete waste of work for later directions
        # So instead we could do a loop over the 4 directions further
        # restricting collided each time. That may well be faster
        # (and avoids the awkward transpose)

        # different permutation of directions for each collision
        delta = self.DIRECTION_PERMUTATIONS[direction_index].transpose()

        # different permutation of test coordinates for each collision
        p = self.head()[collided] + delta

        # Is there nothing on the new coordinate ?
        empty = self._field[collided, p] ^ 1
        # which permutation (select) for which snake(i) is empty
        select, i = empty.nonzero()

        # Fill result with a random direction for each snake
        # (fallback for if the head is completely surrounded)
        pos = p[0].copy()

        # Copy coordinates of empty neighbours
        # Each snake can get coordinates assigned multiple times
        # I assume some assignment wins and there is no tearing
        # (I do not know if numpy enforces anything like this)
        pos[i] = p[select, i]

        return pos

    def frame(self):
        return self._cur_move

    def elapsed(self):
        return self._time_end - self._time_start

    def elapsed_process(self):
        return self._time_process_end - self._time_process_start

    # How long we didn't run
    def paused(self):
        return self._paused

    # How many frames we manually single-stepped
    def frames_skipped(self):
        return self._frames_skipped

    def frame_rate(self):
        elapsed = self.elapsed() - self.paused()
        frames  = self.frame() - self.frames_skipped()
        if elapsed == 0:
            # This properly handles positive, negative and 0 (+inf, -inf, nan)
            return math.inf * frames
        return frames / elapsed

    # In all pits where the snake lost we need to restart the game
    def move_collisions(self, display, pos):
        is_collision = self._field[self._all_snakes, pos]
        collided = is_collision.nonzero()[0]
        # print("Collided", collided)
        if collided.size:
            self._nr_games[collided] += 1
            self._nr_games_total += collided.size
            nr_games_max = np.amax(self._nr_games[collided])
            if nr_games_max > self._nr_games_max:
                self._nr_games_max = nr_games_max
            body_collided = self._body_length[collided]
            body_total = body_collided.sum()
            self._score_total_snakes -= body_total
            self._score_total_games  += body_total
            score_max = np.amax(body_collided)
            if score_max > self._score_max:
                self._score_max = score_max
            nr_moves = self._cur_move - self._nr_moves[collided]
            self._moves_total_games += nr_moves.sum()
            moves_max = np.amax(nr_moves)
            if moves_max > self._moves_max:
                self._moves_max = moves_max

            # After the test because it skips the first _nr_games update
            w_index = is_collision[self._all_windows].nonzero()[0]
            i_index = self._all_windows[w_index]
            display.draw_collisions(i_index, w_index, self.yx(pos[i_index]), self._nr_games)

            self._field0[collided] = 0

            # We are currently doing setup, so this move doesn't count
            self._nr_moves[collided] = self._cur_move + 1

            self._body_length[collided]  = 0
            # print("New Heads after collision")
            pos[collided] = self.rand(collided.size)
        return is_collision, collided

    def move_execute(self, display, pos, is_collision, collided):
        eat = pos == self._apple
        tail_pos = self.tail_set(eat)
        self._body_length += eat
        if collided.size:
            eat[collided] = True

        # cur_move must be updated before head_set for head progress
        # Also before draw_pre_move so nr moves will be correct
        self._cur_move += 1
        display.draw_move(self._all_windows, self.yx(pos[self._all_windows]),
                          is_collision, self.yx(self.head()[self._all_windows]),
                          eat, self.yx(tail_pos[self._all_windows]),
                          self._cur_move - self._nr_moves[self._all_windows])
        self.head_set(pos)

        eaten = eat.nonzero()[0]
        if eaten.size:
            self._score_total_snakes += eaten.size - collided.size
            self.new_apples(eaten)
            w_index = eat[self._all_windows].nonzero()[0]
            if w_index.size:
                i_index = self._all_windows[w_index]
                display.draw_apples(i_index, w_index,
                                    self.yx(self._apple[i_index]),
                                    self._body_length)
                #i0 = self._all_windows[0]
                #if eat[i0]:
                #    print(np.array(self._field0[i0], dtype=np.uint8))

        # print(self.view_string())

        # print_yx("body", self.yx(self._snake_body))
        # print_yx("Head", self.yx(self.head()))
        # print_yx("Apple", self.yx(self._apple))
        # print("-------------------")

    def move_select(self):
        # return self.plan_random_unblocked(self._all_snakes)
        # return self.plan_random()
        pos = self.plan_greedy()
        collided = self._field[self._all_snakes, pos].nonzero()[0]
        # print("Greedy Collided", collided)
        if collided.size:
            pos[collided] = self.plan_random_unblocked(collided)
        # print_xy("Move", x, y))
        return pos

    # Do whatever needs to be done after moving the snake
    # In this case we dislay the current state and wait for a bit
    # while handling events
    # Return True if you want to continue running
    def move_finish(self, display):
        frame = self.frame()
        if frame == self._frame_max:
            return False

        # draw_text is optimized to not draw any value that didn't change
        display.draw_text(0, "step", frame)
        display.draw_text(0, "score_per_snake", self.score_total_snakes() / self._nr_snakes)
        display.draw_text(0, "score_max", self.score_max())
        display.draw_text(0, "moves_max", self.nr_moves_max())
        display.draw_text(0, "game_max",  self.nr_games_max())
        display.draw_text(0, "moves_max", self.nr_moves_max())
        display.draw_text(0, "games", self.nr_games_total())
        if self.nr_games_total():
            display.draw_text(0, "score_per_game",  self.score_per_game())
            display.draw_text(0, "moves_per_game",  self.nr_moves_per_game())
        if self.score_total_games():
            display.draw_text(0, "moves_per_apple", self.nr_moves_per_apple())

        elapsed = time.monotonic() - self._time_start;
        display.draw_text(0, "time", int(elapsed+0.5))
        display.update()
        return self.wait(display)

    # Setup initial variables for moving snakes
    def run_start(self, display,
                  fps      = int(DEFAULTS["--fps"]),
                  stepping = DEFAULTS["--stepping"]):
        nr_windows = min(self.nr_snakes(), display.windows)
        self._all_windows = np.arange(nr_windows-1, -1, -1, dtype=TYPE_INDEX)

        if fps > 0:
            self._poll_fast = 1 / fps
        elif fps == 0:
            self._poll_fast = 0
        else:
            raise(ValueError("fps must not be negative"))
        self._poll_fast0 = self._poll_fast

        self._nr_games_max = 0
        self._nr_games_total = 0
        self._score_max = 0
        self._score_total_snakes = 0
        self._score_total_games  = 0
        self._moves_max = 0
        self._moves_total_games = 0
        self._cur_move = 0

        self._nr_games.fill(0)
        self._field0.fill(0)
        self._nr_moves.fill(self._cur_move)
        self._body_length.fill(0)
        # print("Initial heads")
        head = self.rand(self.nr_snakes())
        self.head_set(head)
        self.new_apples(self._all_snakes)

        w_head_y, w_head_x = self.yx(head[self._all_windows])
        # print_xy("Initial head:", w_head_x, w_head_y))
        for w in range(nr_windows):
            i = self._all_windows[w]
            display.draw_text(w, "moves", 0)
            display.draw_text(w, "game", 0)
            display.draw_text(w, "snake", i)
            display.draw_field_empty(w)
            display.draw_block(w, w_head_x[w], w_head_y[w], Display.HEAD)
        display.draw_apples(self._all_windows, range(nr_windows),
                            self.yx(self._apple[self._all_windows]),
                            self._body_length)

        self._stepping = stepping
        self._paused = 0
        # print("Start at time 0, frame", self.frame())
        self._time_process_start = time.process_time()
        self._time_start  = time.monotonic()
        self._time_target = self._time_start
        if self._stepping:
            self._pause_time = self._time_start
            self._pause_frame = self.frame()
            # When stepping the first frame doesn't count
            self._frames_skipped = -1
        else:
            self._pause_time = 0
            self._frames_skipped = 0

    # We are done moving snake. Report some statistics and cleanup
    def run_finish(self):
        self._time_process_end = time.process_time()
        self._time_end = time.monotonic()
        if self._pause_time:
            self._paused += self._time_end - self._pause_time
            self._frames_skipped += self.frame() - self._pause_frame

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

    # Wait for timeout/events
    def wait(self, display):
        # print("Wait", self._time_target)
        while True:
            now = time.monotonic()
            if self._stepping:
                self._stepping = False
                to_sleep = 0
            else:
                to_sleep = self._time_target - now
                # print("To_Sleep", to_sleep)
                if to_sleep > 0:
                    # Don't become unresponsive
                    if to_sleep > Snakes.POLL_MAX:
                        to_sleep = Snakes.POLL_SLOW
                    if to_sleep < Snakes.WAIT_MIN:
                        to_sleep = 0

            events = display.events_get(to_sleep)
            if to_sleep > 0:
                now = time.monotonic()
            # events seem to come without timestamp, so just assume "now"
            for key in events:
                if key == " " or key == "r":
                    # Stop/start running
                    if self._pause_time:
                        self._time_target = now
                        self._paused += now - self._pause_time
                        self._frames_skipped += self.frame() - self._pause_frame
                        # print("Start running at", now - self._time_start, "frame", self.frame())
                        self._pause_time = 0
                    else:
                        # print("Stop running at", time-self._time_start, "frame", self.frame())
                        self._pause_time = now
                        self._pause_frame = self.frame()
                elif key == "s":
                    # Single step
                    self._stepping = True
                    if not self._pause_time:
                        self._pause_time = now
                        self._pause_frame = self.frame()
                elif key == "+":
                    self._time_target -= self._poll_fast
                    self._poll_fast /= 2
                    self._time_target = max(now, self._time_target + self._poll_fast)
                elif key == "-":
                    self._time_target -= self._poll_fast
                    self._poll_fast *= 2
                    self._time_target = max(now, self._time_target + self._poll_fast)
                elif key == "=":
                    self._time_target -= self._poll_fast
                    self._poll_fast = self._poll_fast0
                    self._time_target = max(now, self._time_target + self._poll_fast)
                elif key == "q":
                    return False

            if self._pause_time:
                if self._stepping:
                    break
                self._time_target = now + Snakes.POLL_SLOW
            elif now >= self._time_target - Snakes.WAIT_MIN:
                break
        #print("elapsed=%.3f, target=%.3f, frame=%d" %
        #      (time.monotonic()-self._time_start,
        #       self._time_target-self._time_start,
        #       self.frame()))
        self._time_target += self._poll_fast
        return True

    def draw_run(self, display, fps=None, stepping=False):
        # print("New game, head=%d [%d, %d]" % self.head())
        # print(self.view_string())

        self.run_start(display, fps=fps, stepping=stepping)
        while True:
            if not self.move_finish(display):
                self.run_finish()
                return
            pos = self.move_select()
            is_collision, collided = self.move_collisions(display, pos)
            self.move_execute(display, pos, is_collision, collided)

# +
if arguments["--benchmark"]:
    speed = 0
    for i in range(1):
        np.random.seed(1)
        snakes = Snakes(nr_snakes = 100000,
                        width     = 40,
                        height    = 40,
                        frame_max = 1000)
        with Display(snakes, rows=0) as display:
            while snakes.draw_run(display,
                                  fps=0,
                                  stepping=False):
                pass
            speed = max(speed, snakes.frame() * snakes.nr_snakes() / snakes.elapsed())
    print("%.0f" % speed)
    sys.exit()

# +
columns    = int(arguments["--columns"])
rows       = int(arguments["--rows"])
nr_snakes  = int(arguments["--snakes"]) or rows*columns
block_size = int(arguments["--block"])

snakes = Snakes(nr_snakes = nr_snakes,
                width     = int(arguments["--width"]),
                height    = int(arguments["--height"]),
                frame_max = int(arguments["--frames"]))

# +
with Display(snakes,
             columns=columns,
             rows=rows,
             block_size = block_size,
             slow_updates =0 ) as display:
    while snakes.draw_run(display,
                          fps=float(arguments["--fps"]),
                          stepping=arguments["--stepping"]):
        pass

print("Elapsed %.3f s (%.3fs used), Frames: %d, Frame Rate %.3f" %
      (snakes.elapsed(), snakes.elapsed_process(), snakes.frame(), snakes.frame_rate()))
print("Max Score: %d, Score/Game: %.3f, Max Moves: %d, Moves/Game: %.3f, Moves/Apple: %.3f" %
      (snakes.score_max(), snakes.score_per_game(), snakes.nr_moves_max(), snakes.nr_moves_per_game(), snakes.nr_moves_per_apple()))
print("Total Lost Games: %d, Lost Game Max: %d" %
      (snakes.nr_games_total(), snakes.nr_games_max()))
