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
  --stepping            Start in paused mode, wait for the user to press SPACE
  --fps=<fps>           Frames per second (0 is no delays) [default: 40]
  --snakes=<snakes>     How many snakes to run at the same time [default: 0]
                        0 means use rows * colums or 1
  --block=<block_size>  Block size in pixels [default: 20]
  --width=<width>       Pit width  in blocks [default: 40]
  --height=<height>     Pit height in blocks [default: 40]
  --columns=<columns>   Columns of pits to display [default: 2]
  --rows=<rows>         Rows of pits to display [default: 1]
  -f <file>:            Used by jupyter, ignored

Display key actions:
  s:          enter pause mode after doing a single step
  r, SPACE:   toggle run/pause mode
  q, <close>: quit
  +:          More frames per second (wait time /= 2)
  -:          Less frames per second (wait time *= 2)
  =:          Go back to the original frames per second

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
    TextField("score_max",   "Max Score:", 5),
    TextField("moves_max",   "Max Moves:", 7),
    TextField("game_max",    "Max Game:",  5),
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
                 columns    = DEFAULTS["--columns"],
                 rows       = DEFAULTS["--rows"],
                 block_size = DEFAULTS["--block"],
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
        self._textrows.add(TextRow(self.TOP_TEXT_X,
                                   self.TOP_TEXT_Y,
                                   self.TOP_WIDTH, ROW_TOP), self.windows)

        self._textrows.add(TextRow(self.BOTTOM_TEXT_X,
                                   self.BOTTOM_TEXT_Y,
                                   self.BOTTOM_WIDTH, ROW_BOTTOM))

        Display._screen = pygame.display.set_mode((self.WINDOW_X * columns, self.WINDOW_Y * rows))
        rect = 0, 0, self.WINDOW_X * columns, self.WINDOW_Y * rows
        rect = pygame.draw.rect(Display._screen, Display.WALL, rect)
        self.draw_text(0, "step")
        self.draw_text(0, "time")
        self.draw_text(0, "score_max")
        self.draw_text(0, "moves_max")
        self.draw_text(0, "game_max")
        self.draw_text(0, "games")
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
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __del__(self):
        self.stop()

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
        events = pygame.event.get()
        if events:
            for event in events:
                if event.type == QUIT:
                    keys.append("q")
                elif event.type == KEYDOWN:
                    keys.append(event.unicode)
        return keys

    def draw_text(self, w, name, value=None,
                       fg_color=BACKGROUND, bg_color=WALL):
        text_data = self._textrows.lookup(name)

        if value is None:
            old_rect = None
            text = text_data.prefix
            x = text_data.prefix_x
        else:
            # Erase old text
            old_rect = text_data.old_rect[w]
            if old_rect:
                pygame.draw.rect(Display._screen, bg_color, old_rect)
            text = text_data.format % value
            x = text_data.format_x
        y = text_data.y

        # Draw new text
        rect = self._font.get_rect(text)
        rect.x += x
        if rect.x + rect.width > text_data.max_width:
            if value is not None and old_rect is not None:
                Display._updates.append(old_rect)
                text_data.old_rect[w] = None
            return None
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
            if is_collision[i]:
                body_rect = None
            else:
                # The current head becomes body
                # (For length 1 snakes the following tail erase will undo this)
                body_rect = self.draw_block(w, head_x_old[i], head_y_old[i], Display.BODY, update=False)
            if not eat[i]:
                # Drop the tail if we didn't eat an apple then
                self.draw_block(w, tail_x[i], tail_y[i], Display.BACKGROUND)
            if body_rect:
                head_rect = self.draw_block(w, head_x_new[i], head_y_new[i], Display.HEAD, update=False)
                Display._updates.append(head_rect.union(body_rect))
            else:
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

# TYPE_POS = np.uint8
TYPE_POS   = np.int8
TYPE_BOOL  = np.bool
TYPE_INDEX = np.intp
TYPE_FLAG  = np.uint8
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

    # Position of the first head of a game in the body array
    # The position before that will be used for one dummy tail set
    START_MOVE = 1

    # Possible directions for a random walk
    DIRECTIONS = [[1,0],[-1,0],[0,1],[0,-1]]
    DIRECTION_PERMUTATIONS = np.array(list(itertools.permutations(DIRECTIONS)), dtype=TYPE_POS)
    NR_DIRECTION_PERMUTATIONS = len(DIRECTION_PERMUTATIONS)
    DIRECTION_PERMUTATIONS_X=DIRECTION_PERMUTATIONS[:,:,0]
    DIRECTION_PERMUTATIONS_Y=DIRECTION_PERMUTATIONS[:,:,1]

    def __init__(self, nr_snakes=1,
                 width  = DEFAULTS["--width"],
                 height = DEFAULTS["--height"],
                 view_x=0, view_y=0):
        self.windows = None

        self._nr_snakes = nr_snakes
        self._all_snakes = np.arange(nr_snakes, dtype=TYPE_INDEX)

        self.VIEW_X = view_x or VIEW_X0
        self.VIEW_Y = view_y or VIEW_Y0
        self.WIDTH  = width
        self.HEIGHT = height
        self.AREA   = self.WIDTH * self.HEIGHT
        # First power of 2 greater or equal to AREA for fast modular arithmetic
        self.AREA2 = 1 << (self.AREA-1).bit_length()
        self.MASK  = self.AREA2 - 1

        width1  = self.HEIGHT+2*self.VIEW_Y
        height1 = self.WIDTH +2*self.VIEW_X
        # Pit is just the edges
        self._empty_pit = np.ones((width1, height1), dtype=TYPE_FLAG)
        self._empty_pit[self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH] = 0
        # self._field = np.ones((nr_snakes, width1, height1), dtype=TYPE_FLAG)

        # The playing field starts out as nr_snakes copies of the empty pit
        # Notice that we store in row major order, so use field[y,x]
        self._field = self._empty_pit.reshape(1,width1,height1).repeat(nr_snakes, axis=0)

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

    def nr_games_total(self):
        return self._nr_games_total

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

    # Sprinkle new apples in all pits where the snake ate them (todo)
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

    # Plot the shortest course to the apple completely ignoring any snake body
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

    # Pick a random direction that isn't blocked
    # Or just a random direction if all are blocked
    # But only for snakes with an index in collided
    def plan_random(self, collided):
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
    def move_collisions(self, display, x, y):
        is_collision = self._field[self._all_snakes, y, x]
        collided = is_collision.nonzero()[0]
        # print("Collided", collided)
        if collided.size:
            if self._score_max < 0:
                # This is the very first call. All pits need to be emptied
                self._score_max = 0
                self._moves_max = 0
                display.draw_text(0, "score_max", self.score_max())
                display.draw_text(0, "moves_max", self.nr_moves_max())
                display.draw_text(0, "game_max", self.nr_games_max())
            else:
                # Normal handling.
                self._nr_games[collided] += 1
                self._nr_games_total += collided.size
                nr_games_max = np.amax(self._nr_games[collided])
                if nr_games_max > self._nr_games_max:
                    self._nr_games_max = nr_games_max
                    display.draw_text(0, "game_max", self.nr_games_max())
                score_max = np.amax(self._body_length[collided])
                if score_max > self._score_max:
                    self._score_max = score_max
                    display.draw_text(0, "score_max", self.score_max())
                moves_max = self._cur_move - np.amin(self._nr_moves[collided])
                if moves_max > self._moves_max:
                    self._moves_max = moves_max
                    display.draw_text(0, "moves_max", self.nr_moves_max())
            display.draw_text(0, "games", self.nr_games_total())

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

    # Do whatever needs to be done after moving the snake
    # In this case we dislay the current state and wait for a bit
    # while handling events
    def move_finish(self, display):
        display.draw_text(0, "step", self.frame())
        elapsed = time.monotonic() - self._time_start;
        elapsed = int(elapsed+0.5)
        if elapsed != self._time_last:
            display.draw_text(0, "time", elapsed)
            self._time_last = elapsed
        display.update()
        return self.wait(display)

    # Setup initial variables for moving snakes
    def run_start(self, display,
                  fps      = DEFAULTS["--fps"],
                  stepping = DEFAULTS["--stepping"]):
        nr_windows = min(self._nr_snakes, display.windows)
        self._all_windows = np.arange(nr_windows-1, -1, -1, dtype=TYPE_INDEX)

        # Fake a collision for all snakes so all snakes will reset
        x = np.zeros(self._nr_snakes, TYPE_POS)
        y = np.zeros(self._nr_snakes, TYPE_POS)

        # Make sure we won't hit an apple left from a previous run
        # During collision handling all the x and y given above will get a
        # random position inside the pit (x, y >= 1), so they are guaranteed
        # not to hit an apple at (0,0)
        self._apple_x.fill(0)
        self._apple_y.fill(0)

        if fps > 0:
            self._poll_fast = 1 / fps
        elif fps == 0:
            self._poll_fast = 0
        else:
            raise(RuntimeError("fps must not be negative"))
        self._poll_fast0 = self._poll_fast

        self._nr_games.fill(0)
        self._nr_games_max = 0
        self._nr_games_total = 0
        self._score_max = -1
        self._moves_max = -1
        self._cur_move = Snakes.START_MOVE-1
        self._stepping = stepping
        self._paused = 0
        self._time_last = ""
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
        return x, y

    # We are done moving snake. Report some statistics and cleanup
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
                if key == "q":
                    self._time_process_end = time.process_time()
                    self._time_end  = now
                    if self._pause_time:
                        self._paused += now - self._pause_time
                        self._frames_skipped += self.frame() - self._pause_frame

                    return False
                elif key == " " or key == "r":
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

        x, y = self.run_start(display, fps=fps, stepping=stepping)
        while True:
            is_collision, collided = self.move_collisions(display, x, y)
            self.move_execute(display, x, y, is_collision, collided)
            if not self.move_finish(display):
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
             block_size = block_size,
             slow_updates=0) as display:
    while snakes.draw_run(display,
                          fps=float(arguments["--fps"]),
                          stepping=arguments["--stepping"]):
        pass

print("Elapsed %.3f s (%.3fs used), Frames: %d, Frame Rate %.3f" %
      (snakes.elapsed(), snakes.elapsed_process(), snakes.frame(), snakes.frame_rate()))
print("Max Score: %d, Max Moves: %d" %
      (snakes.score_max(), snakes.nr_moves_max()))
print("Total Lost Games: %d, Lost Game Max: %d" %
      (snakes.nr_games_total(), snakes.nr_games_max()))
# -
