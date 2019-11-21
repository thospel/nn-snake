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
  snake.py [-f <file>] [--snakes=<snakes>] [--debug] [--stepping] [--fps=<fps>]
           [--width=<width>] [--height=<height>] [--frames=<frames>]
           [--columns=columns] [--rows=rows] [--block=<block_size>]
           [--wall=<wall>]
           [--vision-file=<file>] [--dump-file=<file>] [--log-file=<log>]
           [--learning-rate=<r>] [--discount <ratio>] [--accelerated]
  snake.py --benchmark
  snake.py (-h | --help)
  snake..py --version

Options:
  -h --help               Show this screen
  --version               Show version
  --stepping              Start in paused mode, wait for the user to press SPACE
  --fps=<fps>             Frames per second (0 is no delays) [default: 40]
  --snakes=<snakes>       How many snakes to run at the same time [default: 0]
                          0 means use rows * colums or 1
  --block=<block_size>    Block size in pixels [default: 20]
  --width=<width>         Pit width  in blocks [default: 40]
  --height=<height>       Pit height in blocks [default: 40]
  --columns=<columns>     Columns of pits to display [default: 2]
  --rows=<rows>           Rows of pits to display [default: 1]
  --frames=<frames>       Stop automatically at this frames number [Default: -1]
  --benchmark             Run a simple speed benchmark
  --wall=<wall>           Have state for distance from wall up to <wall>
                          [Default: 2]
  --vision-file=<file>    Read snake vision from file
  --log-file=<file>       Write to logfile. Use an empty string if you
                          explicitely don't want any logging
                          [Default: snakes.log.txt]
  --dump-file=<file>      Which file to dump to on keypress
                          [Default: snakes.dump.txt]
  -l --learning-rate=<r>  Learning rate (will get divided by number of snakes)
                          [Default: 0.1]
  --discount <ratio>      state to state Discount [Default: 0.99]
  --debug                 Run debug code
  --accelerated           Prefill the Q table with walls
                          It will learn this by itself but takes a long time if
                          there are very many states
  -f <file>:              Used by jupyter, ignored

Display key actions:
  s:          enter pause mode after doing a single step
  r, SPACE:   toggle run/pause mode
  Q, <close>: quit
  +:          More frames per second (wait time /= 2)
  -:          Less frames per second (wait time *= 2)
  =:          Restore the original frames per second
  d:          Toggle debug
  D:          Dump current state (without snake state)

"""
from docopt import docopt

DEFAULTS = docopt(__doc__, [])
# print(DEFAULTS)

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Snake 1.0')
    if arguments["--debug"]:
        print(arguments)

# For jupyter
arguments
# -

# %matplotlib inline
from matplotlib import pyplot as plt

from dataclasses import dataclass
from typing import List,Dict
import random
import math
import itertools
import collections.abc
import time
import numpy as np
import numpy.ma as ma

np.set_printoptions(floatmode="fixed", precision=10, suppress=True)

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
import platform

import pygame
import pygame.freetype
from pygame.locals import *


# +
import hashlib

def script():
    try:
        return __file__
    except:
        return None

def read_bin(file = script()):
    if file is None:
        return None
    with open(file, "rb") as fh:
        return fh.read()

# Calculate object hash like git
def file_object_hash(file = script()):
    body = read_bin(file)
    if body is None:
        return None
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(body))
    h.update(body)
    return h.hexdigest()


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
    TextField("win",   "Won:",   2),
    TextField("snake", "Id:",    3),
    # TextField("x",     "x:",     3),
    # TextField("y",     "y:",     3),
]

ROW_BOTTOM = [
    TextField("step",        "Step:", 7),
    TextField("score_max",   "Max Score:", 4),
    TextField("moves_max",   "Max Moves:", 7),
    TextField("game_max",    "Max Game:",  5),
#   TextField("score_per_snake", "Score/Snake:", 4),
    TextField("score_per_game",  "Score/Game:",  4),
    TextField("moves_per_game",  "Moves/Game:",  7),
    TextField("moves_per_apple", "Moves/Apple:", 4),
    TextField("time",        "Time:", 7),
    TextField("wins",        "Won:", 3),
    # Put games last. If you have a lot of snakes this can go up very fast
    TextField("games",       "Games:", 7),
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
                 rows       = int(DEFAULTS["--rows"]),
                 columns    = int(DEFAULTS["--columns"]),
                 block_size = int(DEFAULTS["--block"]),
                 caption="Snakes", slow_updates = 0):
        self.rows    = rows
        self.columns = columns
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
                self._time_monotone_start  = time.monotonic()
                pygame.display.update(Display._updates)
                period = time.monotonic() - self._time_monotone_start
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
                        keys.append("Q")
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

    def draw_pit_empty(self, w):
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

    def draw_collisions(self, i_index, w_index, pos, nr_games, nr_games_won):
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
                self.draw_text(w, "win",  nr_games_won[i])
                # self.draw_text(w, "snake", i)
                # self.draw_text(w, "x")
                # self.draw_text(w, "y")
                self.draw_pit_empty(w)

    def draw_move(self,
                  all_windows, w_head_new,
                  is_collision, w_head_old,
                  is_eat, w_tail,
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
            if not is_eat[i]:
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
        # print(np.stack((apple_x, apple_y), axis=-1))
        for i, w, x, y in zip(i_index, w_index, apple_x, apple_y):
            self.draw_block(w, x, y, Display.APPLE)
            self.draw_text(w, "score", score[i])


# -

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


# +
def np_empty(shape, dtype):
    # return np.empty(shape, dtype)
    # Fill with garbage for debug
    return np.random.randint(100, size=shape, dtype=dtype)

# +
def print_xy(text, x, y):
    print(text)
    print(np.stack((x, y), axis=-1))

def print_yx(text, pos):
    print_xy(text, pos[1], pos[0])


# +
# (Intentionally) crashes if any length is 0
def shape_any(object):
    shape = []
    while isinstance(object, collections.abc.Sized) and not isinstance(object, str):
        shape.append(len(object))
        object = object[0]
    return shape


# -
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

# Parse drawings like:
#      #
#    # O #
#      #
# and returns a pair of coordinates arrays split into an x and y column:
#  X  Y
#  0 -1
# -1  0
#  1  0
#  0  1

@dataclass
class Vision(dict):
    VISION4       = """
      *
    * O *
      *
    """

    VISION8       = """
    * * *
    * O *
    * * *
    """

    VISION12       = """
      *
    * * *
  * * O * *
    * * *
      *
    """

    x:     np.ndarray
    y:     np.ndarray
    min_x: TYPE_POS
    max_x: TYPE_POS
    min_y: TYPE_POS
    max_y: TYPE_POS

    def __init__(self, str, shuffle = False):
        super().__init__()
        head_x = None
        head_y = None
        see_x = []
        see_y = []
        x = -1
        y = 0
        comment = False
        for ch in str:
            x += 1
            if comment:
                if ch != "\n":
                    continue
                comment = False

            if ch == " ":
                pass
            elif ch == "*":
                see_x.append(x)
                see_y.append(y)
            elif ch == "O" or ch == "0":
                if head_x is not None:
                    raise(ValueError("Multiple heads in vision string"))
                head_x = x
                head_y = y
            elif ch == "\n":
                y += 1
                x = -1
            elif ch == "#":
                comment = True
                if x == 0:
                    y -= 1
            else:
                raise(ValueError("Unknown character '%s'" % ch))

        if head_x is None:
            raise(ValueError("No heads in vision string"))

        vision_x = np.array(see_x, TYPE_POS)
        vision_y = np.array(see_y, TYPE_POS)
        vision_x -= head_x
        vision_y -= head_y
        odd_x = vision_x % 2
        if odd_x.any():
            raise(ValueError("Field has odd distance from head"))
        vision_x //= 2

        if shuffle:
            order = np.arange(vision_x.shape[0])
            np.random.shuffle(order)
        else:
            # Order by distance and angle
            distance2  = vision_x * vision_x + vision_y * vision_y
            angle = np.arctan2(-vision_y, vision_x) % (2 * np.pi)
            order = np.lexsort((angle, distance2))
            # order = np.lexsort((angle, distance2 > 2))

        vision_x = vision_x[order]
        vision_y = vision_y[order]
        self.x = vision_x
        self.y = vision_y

        self.min_x = np.amin(vision_x)
        self.max_x = np.amax(vision_x)
        self.min_y = np.amin(vision_y)
        self.max_y = np.amax(vision_y)

        for i,(x,y) in enumerate(zip(np.nditer(vision_x), np.nditer(vision_y))):
            self[y+0, x+0] = i


    def string(self, final_newline = True, values = None, indent = "", separator = "    "):
        if values is None:
            rows = 1
            cols = 1
        else:
            shape = shape_any(values)
            if len(shape) == 1:
                rows = 1
                cols = 1
                values = [[values]]
            elif len(shape) == 2:
                rows = 1
                cols = shape[0]
                values = [values]
            elif len(shape) == 3:
                rows = shape[0]
                cols = shape[1]
            else:
                raise(AssertionError("Invalid shape"))

        out = ""
        for r in range(rows):
            for y in range(self.min_y, self.max_y + 1):
                if y != self.min_y:
                    out += "\n"
                spaces = indent
                for c in range(cols):
                    for x in range(self.min_x, self.max_x+1):
                        if x == 0 and y == 0:
                            if values is None:
                                out += spaces + "O"
                            else:
                                out += spaces + "#"
                        elif (y, x) in self:
                            out += spaces
                            if values is None:
                                out += "*"
                            else:
                                v = str(values[r][c][self[y, x]])
                                if v == "False":
                                    v = "0"
                                elif v == "True":
                                    v = "1"
                                out += v
                            spaces = " "
                        else:
                            spaces += "  "
                    spaces = spaces[:-1] + separator
        if final_newline:
            out += "\n";
        return out

class VisionFile(Vision):
    def __init__(self, file):
        with open(file) as fh:
            super().__init__(fh.read())

# +
@dataclass
class MoveResult:
    is_win:       np.ndarray = None
    won:          np.ndarray = None
    collided:     np.ndarray = None
    is_collision: np.ndarray = None
    is_eat:       np.ndarray = None
    eaten:        np.ndarray = None

    def print(self):
        print("win: ", None if self.is_win       is None else self.is_win+0)
        print("col: ", None if self.is_collision is None else self.is_collision+0)
        print("eat: ", None if self.is_eat       is None else self.is_eat+0)


# +
class Snakes:
    # Event polling time in paused mode.
    # Avoid too much CPU waste or even a busy loop in case fps == 0
    POLL_SLOW = 1/25
    POLL_MAX = POLL_SLOW * 1.5
    WAIT_MIN = 1/1000

    # Possible directions for a random walk
    ROTATION = np.array([[0,1],[-1,0]], dtype=TYPE_POS)
    # In python 3.8 we will get the ability to use "initial" so the
    # awkward "append" can be avoided
    # ROTATIONS will be 4 rotation matrices each rotating one more to the left
    # (negative y is up). The first one is the identity
    ROTATIONS = np.array(list(itertools.accumulate(
        np.append(np.expand_dims(np.identity(2), axis=0),
                  np.expand_dims(ROTATION, axis=0).repeat(3,axis=0), axis=0),
        np.matmul)), dtype=TYPE_POS)
    # Inverse ROTATIONS
    ROTATIONS_INV = ROTATIONS[-np.arange(4)]
    # Next will set DIRECTIONS0 to all directions:
    # [[ 1  0]
    #  [ 0 -1]
    #  [-1  0]
    #  [ 0  1]]
    DIRECTIONS0 = np.matmul(ROTATIONS, np.array([1,0], dtype=TYPE_POS))
    DIRECTIONS0_X = DIRECTIONS0[:,0]
    DIRECTIONS0_Y = DIRECTIONS0[:,1]
    NR_DIRECTIONS = len(DIRECTIONS0)

    DIAGONALS0 = np.matmul(ROTATIONS, np.array([1,-1], dtype=TYPE_POS))
    DIAGONALS0_X = DIAGONALS0[:,0]
    DIAGONALS0_Y = DIAGONALS0[:,1]

    DIRECTION_PERMUTATIONS0   = np.array(list(itertools.permutations(DIRECTIONS0)), dtype=TYPE_POS)
    NR_DIRECTION_PERMUTATIONS = len(DIRECTION_PERMUTATIONS0)
    DIRECTION_PERMUTATIONS0_X = DIRECTION_PERMUTATIONS0[:,:,0]
    DIRECTION_PERMUTATIONS0_Y = DIRECTION_PERMUTATIONS0[:,:,1]

    DIRECTION_INDEX_PERMUTATIONS = np.array(list(itertools.permutations(range(NR_DIRECTIONS))), dtype=TYPE_POS)

    # Construct the left array with -1 based indexing, so really the right one
    # 5 1 4          8 0 2
    # 2 8 0    so    3 7 6
    # 6 3 7          1 4 5
    DIRECTION8_ID = np.empty((3,3), dtype=np.uint8)
    DIRECTION8_X = np.empty(DIRECTION8_ID.size, dtype=TYPE_POS)
    DIRECTION8_Y = np.empty(DIRECTION8_ID.size, dtype=TYPE_POS)
    DIRECTION8_X[0:4] = DIRECTIONS0_X
    DIRECTION8_Y[0:4] = DIRECTIONS0_Y
    DIRECTION8_X[4:8] = DIAGONALS0_X
    DIRECTION8_Y[4:8] = DIAGONALS0_Y
    DIRECTION8_X[8]   = 0
    DIRECTION8_Y[8]   = 0
    DIRECTION8_ID[DIRECTION8_Y, DIRECTION8_X] = np.arange(DIRECTION8_ID.size)

    # Does it need to mirror about the x-axis to get into quadrant 1 and 2
    # (-1, 0) is assigned to "mirror" for symmetry, negative y is "up"
    MIRROR8 = (DIRECTION8_Y > 0) | (DIRECTION8_Y == 0) & (DIRECTION8_X < 0)
    # Rotation lookup table:
    #     Rows are snake directions as defined by DIRECTIONS0
    #     Columns are (x,y) deltas for directions defined by DIRECTION8_ID
    DIRECTION8_ROTATIONS = ROTATIONS[-np.arange(NR_DIRECTIONS)] @ np.stack((DIRECTION8_X, DIRECTION8_Y), axis=0)
    # dihedral lookup table: Do we need to mirror to put the apple to the left ?
    #     Rows    are snake directions as defined by DIRECTIONS0
    #     Columns are apple directions as defined by DIRECTION8_ID
    MIRROR_DIRECTION8 = MIRROR8[DIRECTION8_ID[DIRECTION8_ROTATIONS[:,1],
                                              DIRECTION8_ROTATIONS[:,0]]]

    MIRROR_Y = np.diag(np.array([1,-1], dtype=TYPE_POS))
    # Symmetry group of the square (dihedral group).
    # First 4 rotations, then 4 mirrored versions
    ROTATIONS_DH = np.append(ROTATIONS, ROTATIONS @ MIRROR_Y, axis=0)
    # Inverse of ROTATIONS_DH
    ROTATIONS_DH_INV = np.append(ROTATIONS_INV, MIRROR_Y @ ROTATIONS_INV, axis=0)
    # Check that the rotations are each others inverse
    # print(np.einsum("ijk,ikl -> ijl", ROTATIONS_DH, ROTATIONS_DH_INV))
    # print(np.einsum("ijk,ikl -> ijl", ROTATIONS_DH_INV, ROTATIONS_DH))

    # ROTATIONS_DIRECTION8_ID[snake dir, apple dir Y, apple dir X] gives
    # the id in ROTATIONS_DH that must be aplied to rightward facing snake
    # with the apple to the left to get to that configuration
    # Contents will be:
    #     0 0 4     1 5 1     2 6 2     3 3 7
    #     4 4 4     5 5 1     2 2 2     3 3 7
    #     0 0 0     1 5 1     6 6 6     7 3 7
    ROTATIONS_DIRECTION8_ID = np.arange(NR_DIRECTIONS).reshape(-1,1,1) + MIRROR_DIRECTION8[:,DIRECTION8_ID]*4
    # print(ROTATIONS_DH_INV)
    # print(ROTATIONS_DIRECTION8_ID)
    # print(ROTATIONS_DH_INV[ROTATIONS_DIRECTION8_ID])
    # ROTATED_DIRECTION8 maps apple directions from before to after rotation
    #     Axis 0: select x or y coordinate of output direction
    #     Axis 1: snake direction as defined by DIRECTIONS0
    #     Axis 2: y of apple direction
    #     Axis 2: x of apple direction
    ROTATED_DIRECTION8 = np.einsum(
        "nyxij,yxj -> inyx",
        ROTATIONS_DH_INV[ROTATIONS_DIRECTION8_ID],
        np.stack((DIRECTION8_X, DIRECTION8_Y), axis=1)[DIRECTION8_ID])

    # LEFT8 will start with the 5 directions facing left
    # LEFT8 = [0 1 2 4 5 3 6 7 8]
    TAXI8  = abs(DIRECTION8_X) + abs(DIRECTION8_Y)
    ANGLE8 = np.arctan2(-DIRECTION8_Y, DIRECTION8_X) % (2 * np.pi)
    LEFT8  = np.lexsort((ANGLE8, TAXI8, DIRECTION8_Y > 0, TAXI8 == 0))

    # LEFT8_INV is the inverted permutation of LEFT8,
    # Given a direction id it returns the index in LEFT
    LEFT8_INV = np.lexsort((LEFT8,))

    # ROTATED_LEFT8 maps apple directions to apple state id (index in LEFT8)
    #     Axis 0: snake direction as defined by DIRECTIONS0
    #     Axis 1: y of apple direction
    #     Axis 2: x of apple direction
    # Actual result:
    #   8 0 2     8 1 1     8 2 0     8 1 1
    #   1 3 4     2 4 4     1 4 3     0 3 3
    #   1 3 4     0 3 3     1 4 3     2 4 4
    ROTATED_LEFT8 = LEFT8_INV[DIRECTION8_ID[ROTATED_DIRECTION8[1],ROTATED_DIRECTION8[0]]]
    # print(ROTATED_LEFT8)

    # The state id will be restricted to [0..4] if we did everything right
    # (3 directions to the left, forward and backward)
    ROTATED_LEFT8_MASK = np.zeros(ROTATED_LEFT8.shape, dtype=np.bool)
    ROTATED_LEFT8_MASK[:,0,0] = True
    # So any direction result strictly to the right is a bug
    # Except that index (0,0) is not a direction we should ever see, so mask it
    if (ma.masked_array(ROTATED_LEFT8, ROTATED_LEFT8_MASK) > 4).any():
        raise(AssertionError("dihedral symmetry left something right"))
    if True:
        # Check that all matrices work by displaying all symmetries
        # for manual inspection
        # VISION8 is good enough to show all symmetries, but then the
        # number of points is equal to the number of directions making it
        # much easier to mix up indices. So I use VISION12 instead
        V = Vision(Vision.VISION12, shuffle = True)

        # Make order different from the standard order in DIRECTION8
        # This debugs that we do the proper extra indexing
        if False:
            # Random
            order = np.arange(8)
            np.random.shuffle(order)
        else:
            # Counter clockwise
            order = np.lexsort((ANGLE8[0:8],))
        Dx = np.sign(DIRECTION8_X[order])
        Dy = np.sign(DIRECTION8_Y[order])
        # print_xy("Delta", Dx, Dy)
        D = np.full((8,V.max_y-V.min_y+1,V.max_x-V.min_x+1), ".", dtype=str)
        D[:,V.y, V.x] = "*"
        D[:,0,0] = "E"
        D[:,np.array([0,1,-1]).reshape(3,1),[0,1,-1]] = DIRECTION8_ID.astype(str)
        D[range(8), Dy % D.shape[1], Dx % D.shape[2]] = "@"
        print(V.string(values=D[:,V.y,V.x]))

        C = np.stack((V.x, V.y),axis=0)
        # matmul gives (rotation id, xy, vision i)
        P = ROTATIONS_DH @ C
        for n in range(4):
            pts = P[ROTATIONS_DIRECTION8_ID[n, Dy, Dx]]
            print(V.string(values=D[np.arange(8).reshape(-1,1), pts[:,1], pts[:,0]]))
            print(LEFT8[ROTATED_LEFT8[n,Dy,Dx]])

    VIEW_X0 = 1
    VIEW_Y0 = 1
    VIEW_X2 = VIEW_X0+2
    VIEW_Y2 = VIEW_Y0+2
    VIEW_WIDTH  = 2*VIEW_X0+1
    VIEW_HEIGHT = 2*VIEW_Y0+1

    def pos_from_yx(self, y, x):
        return x + y * self.WIDTH1

    def pos_from_xy(self, x, y):
        return x + y * self.WIDTH1

    def __init__(self, nr_snakes=1,
                 width     = int(DEFAULTS["--width"]),
                 height    = int(DEFAULTS["--height"]),
                 frame_max = int(DEFAULTS["--frames"]),
                 view_x = None, view_y = None,
                 dump_file = DEFAULTS["--dump-file"],
                 log_file  = DEFAULTS["--log-file"],
                 debug = False, xy_apple = True, xy_head = True):
        if nr_snakes <= 0:
            raise(ValueError("Number of snakes must be positive"))

        self._dump_file = dump_file
        self._log_file  = log_file
        self._log_fh = None
        self.debug = debug
        # Do we keep a cache of apple coordinates ?
        # This helps if e.g. we need the coordinates on every move decission
        self._xy_apple = xy_apple
        # xy_head is currently a hack and not implemented for all planners
        # Check by turning debug on for a bit
        self._xy_head  = xy_head

        self.windows = None

        self._nr_snakes = nr_snakes
        self._all_snakes = np.arange(nr_snakes, dtype=TYPE_INDEX)
        self._frame_max = frame_max

        if view_x is None:
            view_x = Snakes.VIEW_X0
        if view_y is None:
            view_y = Snakes.VIEW_Y0
        if view_x < 1:
            raise(ValueError("view_x must be positive to provide an edge"))
        if view_y < 1:
            raise(ValueError("view_y must be positive to provide an edge"))
        self.VIEW_X = view_x
        self.VIEW_Y = view_y

        self.WIDTH  = width
        self.HEIGHT = height
        self.AREA   = self.WIDTH * self.HEIGHT
        if self.AREA < 2:
            # Making area 1 work is too much bother since you immediately win
            # So you never get to move which is awkward for this implementation
            raise(ValueError("No space to put both head and apple"))
        # First power of 2 greater or equal to AREA for fast modular arithmetic
        self.AREA2 = 1 << (self.AREA-1).bit_length()
        self.MASK  = self.AREA2 - 1

        self.HEIGHT1 = self.HEIGHT+2*self.VIEW_Y
        self.WIDTH1  = self.WIDTH +2*self.VIEW_X

        # Set up offset based versions of DIRECTIONS and PERMUTATIONS
        self.DIRECTIONS = self.pos_from_xy(Snakes.DIRECTIONS0_X,
                                           Snakes.DIRECTIONS0_Y)
        self.DIRECTION_PERMUTATIONS = self.pos_from_xy(
            Snakes.DIRECTION_PERMUTATIONS0_X,
            Snakes.DIRECTION_PERMUTATIONS0_Y)

        # Table of all positions inside the pit
        x0 = np.arange(self.VIEW_X, self.VIEW_X+self.WIDTH,  dtype=TYPE_POS)
        y0 = np.arange(self.VIEW_Y, self.VIEW_Y+self.HEIGHT, dtype=TYPE_POS)
        all0 = self.pos_from_xy(x0, y0.reshape(self.HEIGHT,1))
        # self.print_pos("All", all0)
        self._all0 = all0.flatten()
        # self.print_pos("All", self._all0)

        # empty_pit is just the edges with a permanently empty playing field
        self._pit_empty = np.ones((self.HEIGHT1, self.WIDTH1), dtype=TYPE_FLAG)
        self._pit_empty[self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH] = 0
        # Easy way to check if a position is inside the pit
        self._pit_flag = np.logical_not(self._pit_empty)
        # self._field1 = np.ones((nr_snakes, self.HEIGHT1, self.WIDTH1), dtype=TYPE_FLAG)

        # The playing field starts out as nr_snakes copies of the empty pit
        # Notice that we store in row major order, so use field[snake,y,x]
        # (This makes interpreting printouts a lot easier)
        self._field1 = self._pit_empty.reshape(1,self.HEIGHT1,self.WIDTH1).repeat(nr_snakes, axis=0)
        self._field0 = self._field1[:, self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH]
        if self._field0.base is not self._field1:
            raise(AssertionError("field0 is a copy instead of a view"))
        self._field = self._field1.reshape(nr_snakes, self.HEIGHT1*self.WIDTH1)
        if self._field.base is not self._field1:
            raise(AssertionError("field is a copy instead of a view"))

        # self._apple_pit = np.zero((self.HEIGHT, self.WIDTH, self.HEIGHT, self.WIDTH),

        self._snake_body = np_empty((nr_snakes, self.AREA2), TYPE_POS)

        # Body length measures the snake *without* the head
        # This is therefore also the score (if we start with length 0 snakes)
        self._body_length = np_empty(nr_snakes, TYPE_INDEX)
        # Don't need to pre-allocate _head.
        # run_start will implicitely create it
        self._apple    = np_empty(nr_snakes, TYPE_POS)
        if self._xy_apple:
            self._apple_x  = np_empty(nr_snakes, TYPE_POS)
            self._apple_y  = np_empty(nr_snakes, TYPE_POS)
        self._nr_moves = np_empty(nr_snakes, TYPE_MOVES)
        self._nr_games_won = np_empty(nr_snakes, TYPE_GAMES)
        self._nr_games = np_empty(nr_snakes, TYPE_GAMES)

    def log_open(self):
        if self._log_file is not None and self._log_file != "":
            self._log_fh = open(self._log_file, "w")

    def log_close(self):
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def log_constants(self, fh):
        print("Script:", script(), file=fh)
        print("Script Hash:", file_object_hash(), file=fh)
        print("Type:", type(self).__name__, file=fh)
        print("Hostname:", platform.node(),   file=fh)
        print("Width:%8d" % self.WIDTH,   file=fh)
        print("Height:%7d" % self.HEIGHT,  file=fh)
        print("Snakes:%7d"  % self.nr_snakes, file=fh)
        print("View X:%7d" % self.VIEW_X,  file=fh)
        print("View_Y:%7d" % self.VIEW_Y,  file=fh)

    def log_start(self):
        if not self._log_fh:
            return

        fh = self._log_fh
        print(time.strftime("time_start: %Y-%m-%d %H:%M:%S %z",
                            time.localtime(self._time_start)),
              file=fh)
        print("time_start_epoch: %.3f" % self._time_start, file=fh)
        print("time_start_monotone:", self._time_monotone_start,
              file=fh)
        print("time_start_process:",  self._time_process_start,
              file=fh)

        self.log_constants(fh)

        print("Frame:", self.frame(), file=fh)

    def log_stop(self):
        if not self._log_fh:
            return

        self.log_frame()
        fh = self._log_fh
        print(time.strftime("time_end: %Y-%m-%d %H:%M:%S %z",
                            time.localtime(self._time_end)),
              file=fh)
        print("time_end_epoch: %.3f" % self._time_end, file=fh)
        print("time_end_monotone:", self._time_monotone_end, file=fh)
        print("time_end_process:",  self._time_process_end,  file=fh)

    def log_frame(self):
        if not self._log_fh:
            return

        fh = self._log_fh
        print("#----", file=fh)
        print("Frame:%12d"         % self.frame(), file=fh)
        print("Elapsed:%14.3f"     % self.elapsed(), file=fh)
        print("Paused:%15.3f"      % self.paused(),  file=fh)
        print("Frames skipped:%3d" % self.frames_skipped(), file=fh)
        print("Frame rate:%11.3f"  % self.frame_rate(), file=fh)
        print("Games Total:%6d"    % self.nr_games_total(), file=fh)
        print("Games Won:%8d"      % self.nr_games_won_total(), file=fh)
        print("Score Max:%8d"      % self.score_max(), file=fh)
        print("Score Total:%6d"    % self.score_total_games(), file=fh)
        print("Moves Max:%8d"      % self.nr_moves_max(), file=fh)
        print("Moves total:%6d"    % self.nr_moves_total_games(), file=fh)
        print("Moves/Game:%11.3f"  % self.nr_moves_per_game(), file=fh)
        print("Moves/Apple:%10.3f" % self.nr_moves_per_apple(), file=fh)

    def dump(self, file):
        self.timestamp()
        with open(file, "w") as fh:
            self.dump_fh(fh)

    def dump_fh(self, fh):
        self.log_constants(fh)

        print("Elapsed:%8.3f" % self.elapsed(), file=fh)
        print("Paused: %8.3f" % self.paused(),  file=fh)
        print("Used:   %8.3f" % self.elapsed_process(), file=fh)
        print("Frame:%6d" % self.frame(), file=fh)
        print("Frames skipped:%3d" % self.frames_skipped(), file=fh)
        print("Frame rate:%11.3f"  % self.frame_rate(), file=fh)
        print("Games Total:%6d"    % self.nr_games_total(), file=fh)
        print("Games Won:%8d"      % self.nr_games_won_total(), file=fh)
        print("Score Max:%8d"      % self.score_max(), file=fh)
        print("Score Total:%6d"    % self.score_total_games(), file=fh)
        print("Moves Max:%8d"      % self.nr_moves_max(), file=fh)
        print("Moves total:%6d"    % self.nr_moves_total_games(), file=fh)
        print("Moves/Game:%11.3f"  % self.nr_moves_per_game(), file=fh)
        print("Moves/Apple:%10.3f" % self.nr_moves_per_apple(), file=fh)

    def rand_x(self, nr):
        offset_x = self.VIEW_X
        return np.random.randint(offset_x, offset_x+self.WIDTH,  size=nr, dtype=TYPE_POS)

    def rand_y(self, nr):
        offset_y = self.VIEW_Y
        return np.random.randint(offset_y, offset_y+self.HEIGHT, size=nr, dtype=TYPE_POS)

    def rand(self, nr):
        # Use lookup table
        return self._all0[np.random.randint(self._all0.size, size=nr, dtype=TYPE_POS)]
        # Or combine rand_x and rand_y
        rand_x = self.rand_x(nr)
        rand_y = self.rand_y(nr)
        return rand_x + rand_y * self.WIDTH1

    @property
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
            return math.inf * int(self.score_total_games())
        return self.score_total_games() / self.nr_games_total()

    def nr_moves(self, i):
        return self._cur_move - self._nr_moves[i]

    def nr_moves_max(self):
        return self._moves_max

    def nr_moves_total_games(self):
        return self._moves_total_games

    def nr_moves_per_game(self):
        if self.nr_games_total() <= 0:
            return math.inf * int(self.nr_moves_total_games())
        return self.nr_moves_total_games() / self.nr_games_total()

    def nr_moves_per_apple(self):
        if self.score_total_games() <= 0:
            return math.inf * int(self.nr_moves_total_games())
        return self.nr_moves_total_games() / self.score_total_games()

    def nr_games(self, i):
        return self._nr_games[i]

    def nr_games_max(self):
        return self._nr_games_max

    def nr_games_total(self):
        return self._nr_games_total

    def nr_games_won(self, i):
        return self._nr_games_won[i]

    def nr_games_won_total(self):
        return self._nr_games_won_total

    def head_x(self):
        return self._head_x

    def head_y(self):
        return self._head_y

    def head(self):
        return self._head

    def head_set(self, head_new):
        self._head = head_new
        offset = self._cur_move & self.MASK
        self._snake_body[self._all_snakes, offset] = head_new
        # print_xy("Head coordinates", self._head_x, self._head_y)
        # print(head_new)
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

    def snake_string(self, shape):
        apple_y, apple_x = self.yx0(self._apple[shape])
        head_y, head_x = self.yx0(self.head()[shape])
        rows, columns = shape.shape
        horizontal = "+" + "-" * (2*self.WIDTH-1) + "+"
        horizontal = horizontal + (" " + horizontal) * (columns-1) + "\n"
        str = ""
        for r in range(rows):
            str += horizontal
            for y in range(self.HEIGHT):
                for c in range(columns):
                    if c != 0:
                        str += " "
                    i = shape[r,c]
                    field = self._field0[i]
                    str = str + "|"
                    for x in range(self.WIDTH):
                        if x != 0:
                            str += " "
                        if field[y][x]:
                            if y == head_y[r,c] and x == head_x[r,c]:
                                str += "O"
                            else:
                                str += "#"
                        elif y == apple_y[r,c] and x == apple_x[r,c]:
                            str += "@"
                        else:
                            str += " "
                    str = str + "|"
                str += "\n"
            str += horizontal
        return str

    def snakes_string(self, rows, columns):
        return self.snake_string(np.arange(rows*columns).reshape(rows, columns))

    # Get coordinates in the pit WITH the edges
    # Does not work on negative numbers since divmod rounds toward 0
    def yx(self, array):
        y, x = np.divmod(array, self.WIDTH1)
        # print_xy("yx", x, y)
        return y, x

    # Get coordinates in the pit WITHOUT the edges
    # Does not work on negative numbers since divmod rounds toward 0
    def yx0(self, array):
        y, x = np.divmod(array, self.WIDTH1)
        return y - self.VIEW_Y, x - self.VIEW_X

    def print_pos(self, text, pos):
        print_yx(text, self.yx(pos))

    # Sprinkle new apples in all pits where the snake ate them (todo)
    # On a 40x40 pit with the greedy algorithm about 3.5% of snakes need apples
    def new_apples(self, todo):
        if self.debug:
            too_large = self._body_length[todo] >= self.AREA-1
            if too_large.any():
                raise(AssertionError("No place for apples"))

        # print("New apples", todo.size)
        # print("New apples", todo)
        # old_todo = todo.copy()
        # Simple retry strategy. Will get slow once a snake grows very large
        old_todo = todo
        while todo.size:
            # rand_x = self.rand_x(todo.size)
            # rand_y = self.rand_y(todo.size)
            rand = self.rand(todo.size)
            # rand = rand_x + rand_y * self.WIDTH1
            # self._apple_x[todo] = rand_x
            # self._apple_y[todo] = rand_y
            self._apple  [todo] = rand
            fail = self._field[todo, rand]
            # Index with boolean is grep
            todo = todo[fail != 0]
            # print("New apples todo", todo)
            # print("Still need", todo.size)
        if self._xy_apple:
            self._apple_y[old_todo], self._apple_x[old_todo] = self.yx(self._apple[old_todo])
        # self.print_pos("Placed apples", self._apple[old_todo])

    # Plot the shortest course to the apple completely ignoring any snake body
    def plan_greedy(self):
        if self._xy_head:
            x = self.head_x()
            y = self.head_y()
        else:
            head = self.head()
            y, x = self.yx(head)

        if self._xy_apple:
            apple_x = self._apple_x
            apple_y = self._apple_y
        else:
            apple_y, apple_x = self.yx(self._apple)

        # print_xy("Greedy Heads:", x, y))
        # print_xy("Apples:", apple_x, apple_y))

        dx = apple_x - x
        dy = apple_y - y
        # print_xy("Delta:", dx, dy)
        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)
        dir_x = abs_dx > abs_dy
        dx = np.where(dir_x, np.sign(dx), 0)
        dy = np.where(dir_x, 0,           np.sign(dy))
        # Diag is mainly meant to detect dx == dy == 0
        # But is also debugs *some* diagonal moves
        diag = dx == dy
        if np.count_nonzero(diag):
            raise(AssertionError("Impossible apple direction"))
        if self._xy_head:
            # This updates self._head_x since x IS self._head_x. same for y
            x += dx
            y += dy
            return x + y * self.WIDTH1
        else:
            delta = dx + dy * self.WIDTH1
            return head+delta

    def plan_greedy_unblocked(self):
        pos = self.plan_greedy()
        collided = self._field[self._all_snakes, pos].nonzero()[0]
        # print("Move Collided", collided)
        if collided.size:
            pos_new = self.plan_random_unblocked(collided)
            if self._xy_head:
                y, x = self.yx(pos_new)
                self._head_x[collided] = x
                self._head_y[collided] = y
            pos[collided] = pos_new
        return pos

    # Pick a completely random direction
    def plan_random(self):
        rand_dir = np.random.randint(Snakes.NR_DIRECTIONS, size=self._nr_snakes)
        return self.head() + self.DIRECTIONS[rand_dir]

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
        return self._time_monotone_end - self._time_monotone_start

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
            return math.inf * int(frames)
        return frames / elapsed

    def move_evaluate(self, pos):
        is_eat       = pos == self._apple
        is_collision = self._field[self._all_snakes, pos]
        is_win       = self._body_length >= self.AREA-2
        # This really checked for "about to win" instead of "won"
        # So the pit is completely filled except for one spot which has
        # the apple. So we still need to check if the snake gets the apple
        # And in fact any non-collision move MUST get the apple
        won = is_win.nonzero()[0]
        if won.size:
            # Wins will be rare so it's no problem if this is a bit slow
            lost_index = is_collision[won].nonzero()[0]
            if lost_index.size:
                # You didn't win but crashed on the last move. Sooo close
                is_win[won[lost_index]] = False
                won = won[is_collision[won] == 0]
            # Handle a win as a collision so the board will be reset
            is_collision[won] = True
            # However we won't actually get to eat the apple
            # Important because otherwise "move_execute" will grow the new body
            # "move_execute" itself will later set this flag to True again
            is_eat[won]       = False

        return MoveResult(
            is_win       = is_win,
            won          = won,
            is_collision = is_collision,
            collided     = is_collision.nonzero()[0],
            is_eat       = is_eat,
        )

    # In all pits where the snake lost we need to restart the game
    def move_collisions(self, display, pos, move_result):
        collided = move_result.collided
        won      = move_result.won
        # print("Collided", collided)
        if collided.size == 0:
            return

        if won.size:
            self._nr_games_won_total += won.size
            self._nr_games_won[won] += 1
            # We are not going to actually do the move that eats the apple
            self._moves_total_games += won.size
            moves_max = np.amax(self.nr_moves(won))+1
            if moves_max > self._moves_max:
                self._moves_max = moves_max
            self._score_total_games += won.size
            score_max = np.amax(self._body_length[won])+1
            if score_max > self._score_max:
                self._score_max = score_max

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

        nr_moves = self.nr_moves(collided)
        self._moves_total_games += nr_moves.sum()
        moves_max = np.amax(nr_moves)
        if moves_max > self._moves_max:
            self._moves_max = moves_max

        # After the test because it skips the first _nr_games update
        w_index = move_result.is_collision[self._all_windows].nonzero()[0]
        i_index = self._all_windows[w_index]
        display.draw_collisions(i_index, w_index, self.yx(pos[i_index]),
                                self._nr_games, self._nr_games_won)

        self._field0[collided] = 0

        # We are currently doing setup, so this move doesn't count
        self._nr_moves[collided] = self._cur_move + 1

        self._body_length[collided]  = 0
        # print("New Heads after collision")
        if self._xy_head:
            head_x = self.rand_x(collided.size)
            head_y = self.rand_y(collided.size)
            self._head_x[collided] = head_x
            self._head_y[collided] = head_y
            pos[collided] = head_x + head_y * self.WIDTH1
        else:
            pos[collided] = self.rand(collided.size)

    def move_execute(self, display, pos, move_result):
        is_eat   = move_result.is_eat
        collided = move_result.collided

        tail_pos = self.tail_set(is_eat)
        self._body_length += is_eat
        if collided.size:
            is_eat[collided] = True

        # cur_move must be updated before head_set for head progress
        # Also before draw_pre_move so nr moves will be correct
        self._cur_move += 1
        display.draw_move(self._all_windows, self.yx(pos[self._all_windows]),
                          move_result.is_collision,
                          self.yx(self.head()[self._all_windows]),
                          is_eat, self.yx(tail_pos[self._all_windows]),
                          self.nr_moves(self._all_windows))
        self.head_set(pos)

        eaten = is_eat.nonzero()[0]
        if eaten.size:
            self._score_total_snakes += eaten.size - collided.size
            self.new_apples(eaten)
            w_index = is_eat[self._all_windows].nonzero()[0]
            if w_index.size:
                i_index = self._all_windows[w_index]
                display.draw_apples(i_index, w_index,
                                    self.yx(self._apple[i_index]),
                                    self._body_length)
                #i0 = self._all_windows[0]
                #if is_eat[i0]:
                #    print(np.array(self._field0[i0], dtype=np.uint8))
        move_result.eaten = eaten

        # print(self.view_string())

        # self.print_pos("body", self._snake_body)
        # self.print_pos("Head", self.head())
        # self.print_pos("Apple", self._apple)
        # print("-------------------")

    def move_debug(self):
        if self._xy_apple:
            y, x = self.yx(self._apple)
            if not np.array_equal(x, self._apple_x):
                raise(AssertionError("Bad apple"))
            if not np.array_equal(y, self._apple_y):
                raise(AssertionError("Bad apple"))

        if self._xy_head:
            y, x = self.yx(self._head)
            if not np.array_equal(x, self._head_x):
                raise(AssertionError("Bad head"))
            if not np.array_equal(y, self._head_y):
                raise(AssertionError("Bad head"))

    def move_select(self, display, move_result):
        pos = self.plan_greedy_unblocked()
        # self.print_pos("Move", pos)
        return pos

    # Dislay the current state and wait for a bit while handling events
    # Return True if you want to continue running
    def move_show(self, display, pos):
        frame = self.frame()
        if frame == self._frame_max:
            return False

        # draw_text is optimized to not draw any value that didn't change
        display.draw_text(0, "step", frame)
        # display.draw_text(0, "score_per_snake", self.score_total_snakes() / self._nr_snakes)
        display.draw_text(0, "score_max", self.score_max())
        display.draw_text(0, "moves_max", self.nr_moves_max())
        display.draw_text(0, "game_max",  self.nr_games_max())
        display.draw_text(0, "moves_max", self.nr_moves_max())
        display.draw_text(0, "games",     self.nr_games_total())
        display.draw_text(0, "wins",      self.nr_games_won_total())
        if self.nr_games_total():
            display.draw_text(0, "score_per_game",  self.score_per_game())
            display.draw_text(0, "moves_per_game",  self.nr_moves_per_game())
        if self.score_total_games():
            display.draw_text(0, "moves_per_apple", self.nr_moves_per_apple())

        elapsed_sec = int(self.elapsed())
        if elapsed_sec != self._elapsed_sec:
            self._elapsed_sec = elapsed_sec
            self.log_frame()
            display.draw_text(0, "time", elapsed_sec)

        display.update()
        return self.wait(display)

    # Setup initial variables for moving snakes
    def run_start(self, display,
                  fps      = int(DEFAULTS["--fps"]),
                  stepping = DEFAULTS["--stepping"]):
        nr_windows = min(self.nr_snakes, display.windows)
        # self._all_windows = np.arange(nr_windows-1, -1, -1, dtype=TYPE_INDEX)
        self._all_windows = np.arange(nr_windows, dtype=TYPE_INDEX)
        self._debug_index = self._all_windows[0] if nr_windows else None

        if fps > 0:
            self._poll_fast = 1 / fps
        elif fps == 0:
            self._poll_fast = 0
        else:
            raise(ValueError("fps must not be negative"))
        self._poll_fast0 = self._poll_fast

        self._nr_games_won_total = 0
        self._nr_games_max = 0
        self._nr_games_total = 0
        self._score_max = 0
        self._score_total_snakes = 0
        self._score_total_games  = 0
        self._moves_max = 0
        self._moves_total_games = 0
        self._cur_move = 0

        self._nr_games.fill(0)
        self._nr_games_won.fill(0)
        self._field0.fill(0)
        self._body_length.fill(0)
        self._nr_moves.fill(self._cur_move)

        # print("Initial heads")
        if self._xy_head:
            self._head_x = self.rand_x(self.nr_snakes)
            self._head_y = self.rand_y(self.nr_snakes)
            head = self._head_x + self._head_y * self.WIDTH1
        else:
            head = self.rand(self.nr_snakes)
        self.head_set(head)
        self.new_apples(self._all_snakes)

        w_head_y, w_head_x = self.yx(head[self._all_windows])
        # print_xy("Initial head:", w_head_x, w_head_y))
        for w in range(nr_windows):
            i = self._all_windows[w]
            display.draw_text(w, "moves", self.nr_moves(i))
            display.draw_text(w, "game",  self.nr_games(i))
            display.draw_text(w, "win",   self.nr_games_won(i))
            display.draw_text(w, "snake", i)
            display.draw_pit_empty(w)
            display.draw_block(w, w_head_x[w], w_head_y[w], Display.HEAD)
        display.draw_apples(self._all_windows, range(nr_windows),
                            self.yx(self._apple[self._all_windows]),
                            self._body_length)

        self._stepping = stepping
        self._paused = 0
        self._elapsed_sec = 0

    def run_start_extra(self, display):
        pass

    def run_start_results(self):
        # Initial values for move_result
        # This is for planners that want to know about what happened on
        # the previous move. But there was no previous move the first time...
        # Planners that care will have to test for <None> values or override
        # "run_start_results" with something that makes sense to them
        return MoveResult()

    def run_timers(self, display):
        # print("Start at time 0, frame", self.frame())
        self._time_start = time.time()
        self._time_process_start = time.process_time()
        self._time_monotone_start  = time.monotonic()
        self._time_target = self._time_monotone_start
        self._frames_skipped = 0
        if self._stepping:
            self._pause_time = self._time_monotone_start
            self._pause_frame = self.frame()
            # When stepping the first frame doesn't count
        else:
            self._pause_time = 0

    def timestamp(self):
        self._time_end = time.time()
        self._time_process_end = time.process_time()
        self._time_monotone_end = time.monotonic()
        if self._pause_time:
            self._paused += self._time_monotone_end - self._pause_time
            self._pause_time = self._time_monotone_end
            self._frames_skipped += self.frame() - self._pause_frame
            self._pause_frame = self.frame()

    # We are done moving snake. Report some statistics and cleanup
    def run_finish(self):
        self.timestamp()

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

        #print("Quit at", self._time_monotone_end - self._time_monotone_start,
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
                        # print("Start running at", now - self._time_monotone_start, "frame", self.frame())
                        self._pause_time = 0
                    else:
                        # print("Stop running at", time-self._time_monotone_start, "frame", self.frame())
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
                elif key == "d":
                    self.debug = not self.debug
                elif key == "D":
                    self.dump(self._dump_file)
                    if self.debug:
                        print("Dumped to", self._dump_file, flush=True)
                elif key == "Q":
                    return False
                elif self.debug:
                    print("Unknown key <%s>" % key)

            if self._pause_time:
                if self._stepping:
                    break
                self._time_target = now + Snakes.POLL_SLOW
            elif now >= self._time_target - Snakes.WAIT_MIN:
                break
        #print("elapsed=%.3f, target=%.3f, frame=%d" %
        #      (time.monotonic()-self._time_monotone_start,
        #       self._time_target-self._time_monotone_start,
        #       self.frame()))
        self._time_target += self._poll_fast
        return True

    def draw_run(self, display, fps=None, stepping=False):
        # print("New game, head=%d [%d, %d]" % self.head())
        # print(self.view_string())

        self.run_start(display, fps=fps, stepping=stepping)
        self.run_start_extra(display)
        move_result = self.run_start_results()
        self.run_timers(display)
        self.log_open()
        self.log_start()
        while True:
            # Decide what we will do before showing the current state
            # This allows us to show the current plan.
            # It also makes debug messages during planning less confusing
            # since screen output represents the state during planning
            pos = self.move_select(display, move_result)
            # Forgetting to return is an easy bug leading to confusing errors
            if pos is None:
                raise(AssertionError("pos is None"))

            self.timestamp()
            if not self.move_show(display, pos):
                self.run_finish()
                self.log_stop()
                self.log_close()
                return
            if self.debug:
                self.move_debug()

            move_result = self.move_evaluate(pos)
            # Initial move_result has no "eaten"
            self.move_collisions(display, pos, move_result)
            # Modifies "is_eat" with collided and sets "eaten in move_result
            self.move_execute(display, pos, move_result)


# -

class SnakesRandom(Snakes):
    def __init__(self, *args, xy_head=False, xy_apple=False, **kwargs):
        super().__init__(*args, xy_head=False, xy_apple=False, **kwargs)

    def move_select(self, display, move_result):
        return self.plan_random()


class SnakesRandomUnblocked(Snakes):
    def move_select(self, display, move_result):
        return self.plan_random_unblocked(self._all_snakes)


class SnakesQ(Snakes):
    TYPE_FLOAT    = np.float32
    TYPE_QSTATE   = np.uint32
    EPSILON_INV   = 10000
    REWARD_APPLE  = TYPE_FLOAT(1)
    REWARD_CRASH  = TYPE_FLOAT(-100)
    # A small penalty for taking too long to get to an apple
    REWARD_MOVE   = TYPE_FLOAT(-0.001)
    # Small random disturbance to escape from loops
    REWARD_RAND   =  TYPE_FLOAT(0.001)
    # Prefill with 0 is enough to encourage some early exploration since
    # we have a negative reward for moving
    REWARD0       = TYPE_FLOAT(0)
    # LOOP is in units of AREA
    LOOP_MAX      = 2
    LOOP_ESCAPE   = 1
    NR_STATES_APPLE = 8
    NR_STATES_MAX = np.iinfo(TYPE_QSTATE).max

    def __init__(self, *args,
                 view_x = None, view_y = None,
                 xy_head = False,
                 vision = None,
                 wall_left = 0, wall_right = 0, wall_up = 0, wall_down = 0,
                 learning_rate = float(DEFAULTS["--learning-rate"]),
                 discount      = float(DEFAULTS["--discount"]),
                 accelerated = False,
                 **kwargs):

        self._accelerated = accelerated
        if vision is None:
            vision = Vision(Vision.VISION4)
        if view_x is None:
            view_x = max(Snakes.VIEW_X0, -np.amin(vision.x), np.amax(vision.x))
        if view_y is None:
            view_y = max(Snakes.VIEW_Y0, -np.amin(vision.y), np.amax(vision.y))

        nr_neighbours = len(vision)
        if nr_neighbours <= 0:
            raise(AssertionError("Your snake is blind"))
        self.NR_STATES_NEIGHBOUR = 2 ** nr_neighbours

        super().__init__(*args,
                         xy_head = xy_head,
                         view_x = view_x,
                         view_y = view_y,
                         **kwargs)

        # self._rewards = np.empty(self._nr_snakes, dtype=SnakesQ.TYPE_FLOAT)
        self._learning_rate = SnakesQ.TYPE_FLOAT(learning_rate / self.nr_snakes)
        self._discount      = SnakesQ.TYPE_FLOAT(discount)
        if (vision.min_x < -self.VIEW_X or
            vision.max_x > +self.VIEW_X):
            raise(ValueError("X View not wide enough for vision"))
        if (vision.min_y < -self.VIEW_Y or
            vision.max_y > +self.VIEW_Y):
            raise(ValueError("Y View not high enough for vision"))
        self._vision_obj = vision
        self._vision_pos = self.pos_from_xy(vision.x, vision.y)
        self._eat_frame = np_empty(self.nr_snakes, TYPE_MOVES)

        # Make sure the user entered sane values
        if wall_left  < 0: raise(ValueError("wall_left must not be negative"))
        if wall_right < 0: raise(ValueError("wall_right must not be negative"))
        if wall_up    < 0: raise(ValueError("wall_up must not be negative"))
        if wall_down  < 0: raise(ValueError("wall_down must not be negative"))

        # No need to look at walls farther than the pit size
        wall_left  = min(wall_left,  self.WIDTH)
        wall_right = min(wall_right, self.WIDTH)
        wall_up    = min(wall_up,    self.HEIGHT)
        wall_down  = min(wall_down,  self.HEIGHT)

        # No need to look at walls if their distance is known
        if self.WIDTH == 1:
            wall_left  = 0
            wall_right = 0
        if self.HEIGHT == 1:
            wall_up   = 0
            wall_down = 0

        self._wall_left  = wall_left
        self._wall_right = wall_right
        self._wall_up    = wall_up
        self._wall_down  = wall_down

        wall_x = np.zeros(self.WIDTH1, SnakesQ.TYPE_QSTATE)
        # If we VERY carefully consider all case we don't need the dict() method
        # But that would be much less obvious with all the corner cases
        wall_seen_x = dict()
        # Try to give a spot out of view of the walls index 0
        normal_left  = min(wall_left, self.WIDTH-1)
        normal_right = max(self.WIDTH-1-wall_right, 0)
        for x in [(normal_left + normal_right) // 2] + list(range(self.WIDTH)):
            left  = min(x+1, wall_left+1)
            right = min(self.WIDTH-x, wall_right+1)
            if (left, right) not in wall_seen_x:
                n = len(wall_seen_x)
                wall_seen_x[left, right] = n
            wall_x[self.VIEW_X + x] = wall_seen_x[left, right]
        # print(wall_x)

        wall_y = np.zeros(self.HEIGHT1, SnakesQ.TYPE_QSTATE)
        wall_seen_y = dict()
        # Try to give a spot out of view of the walls index 0
        normal_up  = min(wall_up, self.HEIGHT-1)
        normal_down = max(self.HEIGHT-1-wall_down, 0)
        for y in [(normal_up + normal_down) // 2] + list(range(self.HEIGHT)):
            up  = min(y+1, wall_up+1)
            down = min(self.HEIGHT-y, wall_down+1)
            if (up, down) not in wall_seen_y:
                n = len(wall_seen_y)
                wall_seen_y[up, down] = n
            wall_y[self.VIEW_Y + y] = wall_seen_y[up, down]
        # print(wall_y)

        self.NR_STATES_WALL = len(wall_seen_x) * len(wall_seen_y)
        if self.NR_STATES_WALL > 1:
            self._wall1 = np.add.outer(wall_y * len(wall_seen_x), wall_x)
            # Just to make self._wall1 print nicer set the outer edge to nr states
            # Makes no difference since you should never access the edge
            self._wall1 *= self._pit_flag
            self._wall1 += self._pit_empty * SnakesQ.TYPE_QSTATE(self.NR_STATES_WALL)
            # And avoid needing to do the apple multiplier:
            self._wall1 *= SnakesQ.NR_STATES_APPLE
            self._wall = self._wall1.reshape(self._wall1.size)
            if self._wall.base is not self._wall1:
                raise(AssertionError("wall is a copy instead of a view"))

        self.NR_STATES = SnakesQ.NR_STATES_APPLE * self.NR_STATES_NEIGHBOUR * self.NR_STATES_WALL
        if self.NR_STATES > SnakesQ.NR_STATES_MAX:
            raise(ValueError("Number of states too high for Q table index type"))
        self._q_table = np.empty((self.NR_STATES,
                                  Snakes.NR_DIRECTIONS),
                                 dtype=SnakesQ.TYPE_FLOAT)
        print("Q table has", self.NR_STATES, "states")
        print("Snake Vision:")
        print(vision.string(final_newline = False))
        sys.stdout.flush()

        # The previous test made sure this won't overflow
        self.neighbour_multiplier = np.expand_dims(1 << np.arange(0, nr_neighbours, 8, dtype=SnakesQ.TYPE_QSTATE), axis=1) * (SnakesQ.NR_STATES_APPLE * self.NR_STATES_WALL)
        if self.neighbour_multiplier.dtype != SnakesQ.TYPE_QSTATE:
            raise(AssertionError("Unexpected neighbour multiplier dtype"))

    def run_start_extra(self, display):
        self._q_table.fill(SnakesQ.REWARD0)

        # Prefill obvious collisions
        if self._accelerated:
            n = np.arange(self.NR_STATES_NEIGHBOUR, dtype=SnakesQ.TYPE_QSTATE)
            # Have a per neighbour view for fast neighbour selection
            q_table = self._q_table.reshape((self.NR_STATES_NEIGHBOUR, SnakesQ.NR_STATES_APPLE * self.NR_STATES_WALL, Snakes.NR_DIRECTIONS))
            if q_table.base is not self._q_table:
                raise(AssertionError("q_table is a copy instead of a view"))

            for d, (x, y) in enumerate(zip(np.nditer(Snakes.DIRECTIONS0_X), np.nditer(Snakes.DIRECTIONS0_Y))):
                pos = y+0, x+0
                if pos in self._vision_obj:
                    i = self._vision_obj[y+0, x+0]
                    bit = SnakesQ.TYPE_QSTATE(1 << i)
                    hit = (n & bit) == bit
                    q_table[hit,:,d] = SnakesQ.REWARD_CRASH

        self._eat_frame.fill(self.frame())
        #self._q_table = np.array(
        #    np.random.uniform(SnakesQ.REWARD0-SnakesQ.REWARD_RAND,
        #                      SnakesQ.REWARD0+SnakesQ.REWARD_RAND,
        #                      size=(SnakesQ.NR_STATES, Snakes.NR_DIRECTIONS)),
        #    dtype=SnakesQ.TYPE_FLOAT)

    def print_qtable(self, fh):
        nr_neighbours = len(self._vision_obj)
        bits = [0] * nr_neighbours
        state_max_len = len(str(self.NR_STATES_NEIGHBOUR-1))
        states_per_neighbour = self.NR_STATES_WALL * SnakesQ.NR_STATES_APPLE
        for neighbour_state in range(self.NR_STATES_NEIGHBOUR):
            min_i = neighbour_state       * states_per_neighbour
            max_i = (neighbour_state + 1) * states_per_neighbour
            if self._q_table[min_i:max_i].any():
                print("NState: %*d" % (state_max_len, neighbour_state), file=fh)
                print(self._vision_obj.string(final_newline = False,
                                              values = bits),
                      file = fh)
                for i in range(min_i, max_i, SnakesQ.NR_STATES_APPLE):
                    q_range = self._q_table[i:i+SnakesQ.NR_STATES_APPLE]
                    if q_range.any():
                        print(q_range, file=fh)
                    else:
                        print("[ ZERO ]", file=fh)

            # bit list + 1
            for i in range(nr_neighbours):
                bits[i] = 1 - bits[i]
                if bits[i]:
                    break

    def log_constants(self, fh):
        super().log_constants(fh)

        print("Wall left:%4d"  % self._wall_left,  file=fh)
        print("Wall right:%3d" % self._wall_right, file=fh)
        print("Wall up:%6d"    % self._wall_up,    file=fh)
        print("Wall down:%4d"  % self._wall_down,  file=fh)

        print("Vision: <<EOT", file=fh)
        print(self._vision_obj.string(final_newline = False), file=fh)
        print("EOT", file=fh)

        print("NrStates:%5d"   % self.NR_STATES,   file=fh)
        print("learning rate:", self._learning_rate * self.nr_snakes, file=fh)
        print("Discount:     ", self._discount,        file=fh)
        print("Epsilon:      ", 1/SnakesQ.EPSILON_INV, file=fh)
        print("Reward apple: ", SnakesQ.REWARD_APPLE,  file=fh)
        print("Reward crash: ", SnakesQ.REWARD_CRASH,  file=fh)
        print("Reward move:  ", SnakesQ.REWARD_MOVE,   file=fh)
        print("Reward rand:  ", SnakesQ.REWARD_RAND,   file=fh)
        print("Reward init:  ", SnakesQ.REWARD0,       file=fh)

    def dump_fh(self, fh):
        super().dump_fh(fh)

        print("Q table: <<EOT", file=fh)
        with np.printoptions(threshold=self._q_table.size):
            self.print_qtable(fh)
        print("EOT", file=fh)

    def state(self, neighbour, apple, wall=0):
        return apple + SnakesQ.NR_STATES_APPLE * (wall + self.NR_STATES_WALL * neighbour)

    def unstate(self, state):
        state_rest, apple = divmod(state,      SnakesQ.NR_STATES_APPLE)
        if self.NR_STATES_WALL > 1:
            neighbour,  wall  = divmod(state_rest, self.NR_STATES_WALL)
        else:
            # wall = 0
            wall = None
            neighbour = state_rest

        # Decode the apple direction
        apple = Snakes.DIRECTION8_Y[apple], Snakes.DIRECTION8_X[apple]

        # Decode the Neigbours
        nr_neighbours = len(self._vision_obj)
        nr_rows = math.ceil(nr_neighbours/8)
        scalar = not isinstance(state, collections.abc.Sized)
        size = 1 if scalar else len(state)
        rows = np_empty((nr_rows, size), dtype=np.uint8)
        for i in range(nr_rows-1):
            neighbour, row = divmod(neighbour, 256)
            rows[i] = row
        rows[-1] = neighbour
        # field will end up with padded extra zero rows up to a multiple of 8
        field = np.unpackbits(rows, axis=0, bitorder='little')
        if scalar:
            field = np.squeeze(field)

        # Todo: Decode the wall
        return apple, wall, field

    def vision_from_state(self, state):
        apple, field = self.unstate(state)

    def move_select(self, display, move_result):
        # debug = self.frame() % 100 == 0
        debug = self.debug

        if debug: print("=" * 40)
        # print(self._q_table)
        # if debug: print(self.snakes_string(display.rows, display.columns))
        if debug: print(self.snakes_string(1, 1))

        is_eat = move_result.is_eat
        head = self.head()

        # Determine the neighbourhood of the head
        # self.print_pos("Head", head)
        neighbours_pos = np.add.outer(self._vision_pos, head)
        neighbours = self._field[self._all_snakes, neighbours_pos]
        # Walls are automatically set unique, so no need to zero them
        # neighbours = np.logical_and(neighbours, self._pit_flag[neighbours_pos])
        neighbours_packed = np.packbits(neighbours, axis=0,
                                      bitorder="little")
        neighbour_state = neighbours_packed * self.neighbour_multiplier
        if len(neighbour_state > 1):
            neighbour_state = np.sum(neighbour_state,
                                     axis=0, dtype=SnakesQ.TYPE_QSTATE)
        else:
            neighbour_state = np.squeeze(neighbour_state, axis=0)
        # if debug: print("Neigbour state", neighbour_state / SnakesQ.NR_STATES_APPLE)

        # Determine the direction of the apple
        if self._xy_head:
            x = self.head_x()
            y = self.head_y()
        else:
            y, x = self.yx(head)

        if self._xy_apple:
            apple_x = self._apple_x
            apple_y = self._apple_y
        else:
            apple_y, apple_x = self.yx(self._apple)
        # print_xy("apple", apple_x, apple_y)

        dx = np.sign(apple_x - x)
        dy = np.sign(apple_y - y)
        # print_xy("Delta signs:", dx, dy)
        apple_state = Snakes.DIRECTION8_ID[dy, dx]
        # if debug: print("Apple state", apple_state)
        # Determine the wall state
        if self.NR_STATES_WALL > 1:
            wall_state = self._wall[head]
            state = neighbour_state + wall_state + apple_state
        else:
            state = neighbour_state + apple_state

        if debug:
            print("Apple state[%d]:" % self._debug_index,
                  apple_state[self._debug_index])
            print("Wall state[%d]:" % self._debug_index,
                  wall_state[self._debug_index] / SnakesQ.NR_STATES_APPLE if self.NR_STATES_WALL > 1 else None)
            print("Neigbour state[%d]:" % self._debug_index,
                  neighbour_state[self._debug_index] / (SnakesQ.NR_STATES_APPLE * self.NR_STATES_WALL))

        if debug:
            print(self._vision_obj.string(final_newline = False, values = neighbours[:,0]))
        # if debug: print("State", state)
        if debug:
            print("Old State[%d]" % self._debug_index,
                  None if is_eat is None else self._state[self._debug_index],
                  "New State[%d]" % self._debug_index, state[self._debug_index])

        # Evaluate the previous move
        if is_eat is not None:
            self._eat_frame[move_result.eaten] = self.frame()
            q_row = self._q_table[state]
            # rewards = self._rewards
            # rewards.fill(SnakesQ.REWARD_MOVE)
            rewards = np.random.uniform(-SnakesQ.REWARD_RAND/2,
                                        +SnakesQ.REWARD_RAND/2,
                                        size=self.nr_snakes)
            r = np.random.uniform(-SnakesQ.REWARD_RAND,
                                  +SnakesQ.REWARD_RAND)
            if debug:
                print("Qrow[%d] before:" % self._debug_index,
                      q_row[self._debug_index])
                #print("base r:", r)
                #print("rand r[%d]:" % self._debug_index, rewards[self._debug_index])
            rewards += SnakesQ.REWARD_MOVE + r
            rewards[move_result.eaten]    += SnakesQ.REWARD_APPLE
            # eaten at this point contains collided, so compensate by an apple
            rewards[move_result.collided] += SnakesQ.REWARD_CRASH - SnakesQ.REWARD_APPLE
            # Don't punish the snake for winning!
            if move_result.won.size:
                rewards[move_result.won] -= SnakesQ.REWARD_CRASH - SnakesQ.REWARD_APPLE
            # move_result.print()
            # print("Rewards", rewards)
            if debug:
                print("Rewards[%d]" % self._debug_index,
                      rewards[self._debug_index])
            advantage = np.amax(q_row, axis=-1) * self._discount
            advantage[move_result.collided] = 0
            advantage -= self._q_table[self._state, self._action]
            # print("Advantage", advantage)
            if debug:
                print("Advantage[%d]" % self._debug_index,
                      advantage[self._debug_index])
            rewards += advantage
            # print("Update", rewards)
            rewards *= self._learning_rate
            if debug:
                print("Q old", self._q_table[self._state[self._debug_index]])
                print("Update0[%d][%d]" % (self._state[self._debug_index],
                                           self._action[self._debug_index]),
                      rewards[self._debug_index])
            # Potentially need to multi-update the same position, so use ads.at
            np.add.at(self._q_table, (self._state, self._action), rewards)
            if debug:
                print("Q new", self._q_table[self._state[self._debug_index]])
            #if np.isnan(self._q_table).any():
            #    raise(AssertionError("Q table contains nan"))

        # Decide what to do
        q_row = self._q_table[state]
        if debug:
            print("Qrow[%d] after: " % self._debug_index,
                  q_row[self._debug_index])
        action = q_row.argmax(axis=-1)
        if debug:
            print("Old Action[%d]" % self._debug_index,
                  None if is_eat is None else self._action[self._debug_index],
                  "New action[%d]" % self._debug_index,
                  action[self._debug_index],
                  "Moves since apple[%d]" % self._debug_index,
                  self.frame() - self._eat_frame[self._debug_index])

        looping = self._eat_frame <= self.frame() - SnakesQ.LOOP_MAX * self.AREA
        looping = looping.nonzero()[0]
        if looping.size:
           if debug:
               print("Nr Looping=", looping.size,
                     "looping[0]:", looping[0])
           rand = np.random.randint(SnakesQ.LOOP_ESCAPE * self.AREA, size=looping.size)
           escaping = looping[rand == 0]
           if escaping.size:
               if debug:
                   print("Nr Escaping=", escaping.size,
                         "Escaping[0]", escaping[0])
               # We could also avoid crashing here
               action[escaping] = np.random.randint(Snakes.NR_DIRECTIONS,
                                                    size = escaping.size)

        accept = np.random.randint(SnakesQ.EPSILON_INV, size=self.nr_snakes)
        randomize = (accept == 0).nonzero()[0]
        if randomize.size:
            # print("Randomize", randomize)

            # different permutation for each randomization target
            direction_index = np.random.randint(
                Snakes.NR_DIRECTION_PERMUTATIONS, size=randomize.size)

            # different permutation of directions for each randomized target
            permutations = Snakes.DIRECTION_INDEX_PERMUTATIONS[direction_index]
            for i in range(Snakes.NR_DIRECTIONS):
                # print("Permutation", i)
                # print(permutations)
                directions = permutations[:, i]
                if directions.base is not permutations:
                    raise(AssertionError("directions is a copy instead of a view"))
                # print("D", directions)
                action[randomize] = directions
                hit = self._field[randomize, head[randomize] + self.DIRECTIONS[directions]]
                # print("Hit")
                # print(hit)
                hit = hit.nonzero()[0]
                # print("Hit index", hit)
                if hit.size == 0:
                    break
                randomize = randomize[hit]
                permutations = permutations[hit]

        # if debug:
        #    empty_state = self.state(apple = np.arange(SnakesQ.NR_STATES_APPLE),
        #                             wall = 0,
        #                             neighbour = 0)
        #    print(self._q_table[empty_state])
        if debug: sys.stdout.flush()
        # Take the selected action and remember where we came from
        self._state  = state
        self._action = action
        return head + self.DIRECTIONS[action]

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
            speed = max(speed, snakes.frame() * snakes.nr_snakes / snakes.elapsed())
    print("%.0f" % speed)
    sys.exit()

# +
columns    = int(arguments["--columns"])
rows       = int(arguments["--rows"])
nr_snakes  = int(arguments["--snakes"]) or rows*columns
block_size = int(arguments["--block"])
snake_kwargs = dict()
if arguments["--vision-file"] is not None:
    snake_kwargs["vision"] = VisionFile(arguments["--vision-file"])
wall       = int(arguments["--wall"])
learning_rate = float(arguments["--learning-rate"])
discount      = float(arguments["--discount"])
snakes = SnakesQ(nr_snakes = nr_snakes,
                 accelerated = arguments["--accelerated"],
                 debug       = arguments["--debug"],
                 width       = int(arguments["--width"]),
                 height      = int(arguments["--height"]),
                 frame_max   = int(arguments["--frames"]),
                 log_file    = arguments["--log-file"],
                 wall_left   = wall, wall_right = wall,
                 wall_up     = wall, wall_down = wall,
                 learning_rate = learning_rate, discount = discount,
                 **snake_kwargs)

# +
with Display(snakes,
             columns = columns,
             rows = rows,
             block_size = block_size,
             slow_updates = 0) as display:
    while snakes.draw_run(display,
                          fps=float(arguments["--fps"]),
                          stepping=arguments["--stepping"]):
        pass

print("Elapsed %.3f s (%.3fs used), Frames: %d, Frame Rate %.3f" %
      (snakes.elapsed(), snakes.elapsed_process(), snakes.frame(), snakes.frame_rate()))
print("Max Score: %d, Score/Game: %.3f, Max Moves: %d, Moves/Game: %.3f, Moves/Apple: %.3f" %
      (snakes.score_max(), snakes.score_per_game(), snakes.nr_moves_max(), snakes.nr_moves_per_game(), snakes.nr_moves_per_apple()))
print("Total Won Games: %d/%d, Played Game Max: %d" %
      (snakes.nr_games_won_total(), snakes.nr_games_total(), snakes.nr_games_max()))
