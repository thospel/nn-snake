import numpy as np
import time
import os
from dataclasses import dataclass
from typing import List,Dict

from display import Display

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
    TextField(Display.TEXT_SCORE,    "Score:", 5),
    TextField(Display.TEXT_GAME,     "Game:",  5),
    TextField(Display.TEXT_MOVES,    "Moves:", 7),
    TextField(Display.TEXT_WON,      "Won:",   2),
    TextField(Display.TEXT_SNAKE_ID, "Id:",    3),
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

# +
# This must be signed because window offsets can be negative
TYPE_PIXELS = np.int32

class DisplayPygame(Display):
    KEY_INTERVAL = int(1000 / 20)  # twenty per second
    KEY_DELAY = 500                # Start repeating after half a second
    KEY_IGNORE = set((K_RSHIFT, K_LSHIFT, K_RCTRL, K_LCTRL, K_RALT, K_LALT,
                     K_RMETA, K_LMETA, K_LSUPER, K_RSUPER, K_MODE,
                     K_NUMLOCK, K_CAPSLOCK, K_SCROLLOCK))

    _updates_count = 0
    _updates_time  = 0

    COLORS = [
        (0, 0, 0),
        (255, 255, 255),
        (0, 255, 0),
        (200, 200, 0),
        (160, 160, 160),
        (255, 0, 0)
    ]

    # we test these at the start of some functions
    # Make sure they have a "nothing to see here" value in case __init__ fails
    _screen = None
    _updates = []

    # You can only have one pygame instance in one process,
    # so make display related variables into class variables
    def __init__(self, snakes,
                 slow_updates = 0,
                 **kwargs):
        super().__init__(snakes, **kwargs)

        if not self.windows:
            return

        self._slow_updates = slow_updates

        # coordinates relative to the upper left corner of the window
        self.TOP_TEXT_X  = self.BLOCK
        self.TOP_TEXT_Y  = self.DRAW_BLOCK

        # coordinates relative to the bottom left corner of the screen
        self.BOTTOM_TEXT_X  = self.BLOCK
        self.BOTTOM_TEXT_Y  = self.DRAW_BLOCK - self.BLOCK
        # self.BOTTOM_TEXT_Y  = -Display.EDGE

        # Fixup for window offset
        self.TOP_WIDTH = self.WINDOW_X-self.TOP_TEXT_X
        self.TOP_TEXT_X -= self.OFFSET_X
        self.TOP_TEXT_Y -= self.OFFSET_Y
        self.BOTTOM_WIDTH = self.WINDOW_X * self.columns -self.BOTTOM_TEXT_X
        self.BOTTOM_TEXT_X -= self.OFFSET_X
        self.BOTTOM_TEXT_Y -= self.OFFSET_Y
        self.BOTTOM_TEXT_Y += self.rows * self.WINDOW_Y

        self._window_x = np.tile  (np.arange(self.OFFSET_X, self.columns*self.WINDOW_X+self.OFFSET_X, self.WINDOW_X, dtype=TYPE_PIXELS), self.rows)
        self._window_y = np.repeat(np.arange(self.OFFSET_Y, self.rows   *self.WINDOW_Y+self.OFFSET_Y, self.WINDOW_Y, dtype=TYPE_PIXELS), self.columns)
        # print("window_x", self._window_x)
        # print("window_y", self._window_y)

        # self.last_collision_x = np.zeros(self.windows, dtype=TYPE_PIXELS)
        # self.last_collision_y = np.zeros(self.windows, dtype=TYPE_PIXELS)


    def start(self):
        super().start()
        if not self.windows:
            return

        # Avoid pygame.init() since the init of the mixer component leads to 100% CPU
        pygame.display.init()
        pygame.display.set_caption(self.caption)
        # pygame.mouse.set_visible(1)
        pygame.key.set_repeat(DisplayPygame.KEY_DELAY, DisplayPygame.KEY_INTERVAL)

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

        DisplayPygame._screen = pygame.display.set_mode((self.WINDOW_X * self.columns, self.WINDOW_Y * self.rows))
        rect = 0, 0, self.WINDOW_X * self.columns, self.WINDOW_Y * self.rows
        rect = pygame.draw.rect(DisplayPygame._screen,
                                DisplayPygame.COLORS[Display.WALL], rect)
        for w in range(self.windows):
            self.draw_text_row("top", w)
        self.draw_text_row("bottom", 0)
        DisplayPygame._updates = [rect]


    # Don't do a stop() in __del__ since object desctruction can be delayed
    # and a new display can have started
    def stop(self):
        if DisplayPygame._screen:
            DisplayPygame._screen  = None
            DisplayPygame._updates = []
            pygame.quit()
            if self._slow_updates:
                if DisplayPygame._updates_count:
                    print("Average disply update time: %.6f" %
                          (DisplayPygame._updates_time / DisplayPygame._updates_count))
                else:
                    print("No display updates")
        super().stop()


    def update(self):
        if DisplayPygame._updates:
            if self._slow_updates:
                self._time_monotone_start  = time.monotonic()
                pygame.display.update(DisplayPygame._updates)
                period = time.monotonic() - self._time_monotone_start
                if period > self._slow_updates:
                    print("Update took %.4f" % period)
                    for rect in DisplayPygame._updates:
                        print("    ", rect)
                DisplayPygame._updates_count +=1
                DisplayPygame._updates_time  += period
            else:
                pygame.display.update(DisplayPygame._updates)
            DisplayPygame._updates = []


    def updated(self, rect):
        DisplayPygame._updates.append(rect)


    def rect_union(self, rect1, rect2):
        return rect1.union(rect2)


    def events_key(self, now):
        keys = []
        if DisplayPygame._screen:
            events = pygame.event.get()
            if events:
                for event in events:
                    if event.type == QUIT:
                        keys.append("Q")
                    elif event.type == KEYDOWN and event.key not in DisplayPygame.KEY_IGNORE:
                        keys.append(event.unicode)
        return keys


    def draw_text_row(self, row_name, w):
        if not DisplayPygame._screen:
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
            self._font.render_to(DisplayPygame._screen, (x, y), None,
                                 DisplayPygame.COLORS[Display.BACKGROUND],
                                 DisplayPygame.COLORS[Display.WALL])
            # We could union all rects, but for now this is only called on
            # start() and start() already forces its own single rect
            DisplayPygame._updates.append(rect)


    def draw_text(self, w, name, value):
        if not DisplayPygame._screen:
            return

        text_data = self._textrows.lookup(name)

        text = text_data.format % value
        old_text = text_data.old_text[w]
        if old_text is not None and text == old_text:
            # Update does nothing
            return
        # Erase old text
        old_rect = text_data.old_rect[w]
        if old_rect:
            pygame.draw.rect(DisplayPygame._screen, DisplayPygame.COLORS[Display.WALL], old_rect)
        x = text_data.format_x
        y = text_data.y

        # Draw new text
        rect = self._font.get_rect(text)
        rect.x += x
        if rect.x + rect.width > text_data.max_width:
            if old_rect is not None:
                DisplayPygame._updates.append(old_rect)
                text_data.old_rect[w] = None
            return
        x += self._window_x[w]
        y += self._window_y[w]
        rect.x += self._window_x[w]
        rect.y = y - rect.y
        # print("Draw text", w, x, y, '"%s"' % text, x + self._window_x[w], y + self._window_y[w], rect, old_rect)
        self._font.render_to(DisplayPygame._screen, (x, y), None,
                             DisplayPygame.COLORS[Display.BACKGROUND],
                             DisplayPygame.COLORS[Display.WALL])
        if old_rect is None:
            DisplayPygame._updates.append(rect)
        else:
            DisplayPygame._updates.append(rect.union(old_rect))
        # Remember what we updated
        text_data.old_rect[w] = rect
        text_data.old_text[w] = text


    def draw_text_summary(self, name, value):
        self.draw_text(0, name, value)


    def draw_pit_empty(self, w):
        rect = (self._window_x[w] - self.OFFSET_X + self.BLOCK,
                self._window_y[w] - self.OFFSET_Y + self.BLOCK,
                self.WINDOW_X - 2 * self.BLOCK,
                self.WINDOW_Y - 2 * self.BLOCK)
        rect = pygame.draw.rect(DisplayPygame._screen,
                                DisplayPygame.COLORS[Display.BACKGROUND], rect)
        DisplayPygame._updates.append(rect)


    def draw_block(self, w, x, y, color, update=True):
        rect = (x * self.BLOCK + self._window_x[w] + Display.EDGE,
                y * self.BLOCK + self._window_y[w] + Display.EDGE,
                self.DRAW_BLOCK,
                self.DRAW_BLOCK)

        # print("Draw %d (%d,%d): %d,%d,%d: [%d %d %d %d]" % ((w, x, y)+color+(rect)))
        rect = pygame.draw.rect(DisplayPygame._screen, DisplayPygame.COLORS[color], rect)
        if update:
            DisplayPygame._updates.append(rect)
        return rect
