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
# This must be signed because window offsets can be negative
TYPE_PIXELS = np.int32

@dataclass
class Text:
    prefix:        str
    format:        str
    value:         str
    value_x:       int
    value_grabbed: int         = 0
    prefix_rect:   pygame.Rect = None
    # Notice that value_rect,x can differ from value_x!
    value_rect:    pygame.Rect = None
    value_left:	   bool        = True


@dataclass
class _TextRow:
    x:            int
    y:            int
    x_max:	  int
    font:         pygame.freetype.Font
    skip:         str = "  "


class TextRow(_TextRow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rect = self.font.get_rect(self.skip)
        self._skip_width = rect.width
        self._lookup = {}
        self._texts = []


    def add(self, text_field):
        if text_field.key in self._lookup:
            raise(AssertionError("Duplicate text key " + text_field.key))
        self._lookup[text_field.key] = len(self._texts)

        if self._texts:
            text_last = self._texts[-1]
            x_from = text_last.value_x + text_last.value_grabbed + self._skip_width
        else:
            x_from = self.x

        prefix = text_field.prefix + " "
        if x_from <= self.x_max:
            rect_prefix = self.font.get_rect(prefix)
            rect_prefix.x += x_from
            rect_prefix.y = self.y - rect_prefix.y
            x_prefix = x_from
            x_from = rect_prefix.x + rect_prefix.width

        if x_from > self.x_max:
            # Prefix doesn't fit. Evereything after this point is moot
            text = Text(
                prefix  = prefix,
                format  = text_field.format,
                value   = text_field.initial_value,
                # Exact value doesn't matter anymore as long as > self.x_max
                value_x = x_from,
            )
            self._texts.append(text)
            return

        self.font.render_to(DisplayPygame._screen, (x_prefix, self.y), None,
                            DisplayPygame.COLORS[Display.BACKGROUND],
                            DisplayPygame.COLORS[Display.WALL])
        # We could union all rects, but for now this is only called on
        # start() and start() already forces its own single rect
        DisplayPygame._updates.append(rect_prefix)

        # It seems the rect is in reality not always integer.
        # For example "0:" and "0:0:" don't "add" properly. This can cause the
        # value to be 1 pixel closer to the prefix than we want which looks ugly
        x_from += 1
        # By the way, we don't do the same thing for "value" and just let the
        # extra pixel be absorbed by skip_width

        rect_value  = self.font.get_rect(text_field.initial_value)
        grabbed = rect_value.x + rect_value.width

        if x_from + grabbed <= self.x_max:
            rect_value.x += x_from
            rect_value.y = self.y - rect_value.y
            self.font.render_to(DisplayPygame._screen, (x_from, self.y), None,
                                DisplayPygame.COLORS[Display.BACKGROUND],
                                DisplayPygame.COLORS[Display.WALL])
            DisplayPygame._updates.append(rect_value)
        else:
            rect_value = None

        text = Text(
            prefix        = prefix,
            prefix_rect   = rect_prefix,
            format        = text_field.format,
            value         = text_field.initial_value,
            value_x       = x_from,
            value_grabbed = grabbed,
            value_rect    = rect_value,
        )
        self._texts.append(text)


    def draw_text(self, key, value):
        i = self._lookup[key]
        text = self._texts[i]
        if text.value_x > self.x_max:
            return

        value = text.format % value
        if value == text.value:
            # Update does nothing
            return

        # Erase old text
        old_rect = text.value_rect

        # Draw new text
        text.value = value
        rect = self.font.get_rect(value)
        grab = rect.x + rect.width
        x = text.value_x
        y = self.y
        rect.x += x
        rect.y = y - rect.y
        if grab <= text.value_grabbed:
            if x + grab > self.x_max:
                if old_rect:
                    pygame.draw.rect(DisplayPygame._screen, DisplayPygame.COLORS[Display.WALL], old_rect)
                    DisplayPygame._updates.append(old_rect)
                text.value_rect = None
                return
            if text.value_left:
                offset = 0
            else:
                offset = text.value_grabbed - grab
                rect.x += offset
            text.value_rect = rect
            if old_rect:
                pygame.draw.rect(DisplayPygame._screen, DisplayPygame.COLORS[Display.WALL], old_rect)
                DisplayPygame._updates.append(rect.union(old_rect))
            else:
                DisplayPygame._updates.append(rect)
            self.font.render_to(DisplayPygame._screen, (x+offset, y), None,
                                DisplayPygame.COLORS[Display.BACKGROUND],
                                DisplayPygame.COLORS[Display.WALL])
            return

        # New value prints wider than the allocated space
        grow = grab - text.value_grabbed
        text.value_grabbed = grab
        if old_rect is None:
            # Text didn't fit even with the old smaller size. This means all
            # following text fields have also already been cut
            return

        # First erase the whole rest of the row
        for j in range(i+1, len(self._texts)):
            t = self._texts[j]
            t.value_x += grow
            # We need to union all texts, not just the last that is not None
            # not for the horizontal parts but for the vertical parts
            if t.prefix_rect:
                old_rect = old_rect.union(t.prefix_rect)
                t.prefix_rect = None
            if t.value_rect:
                old_rect = old_rect.union(t.value_rect)
                t.value_rect = None
        pygame.draw.rect(DisplayPygame._screen, DisplayPygame.COLORS[Display.WALL], old_rect)

        # Does new text even fit ?
        if x + grab > self.x_max:
            DisplayPygame._updates.append(old_rect)
            text.value_rect = None
            return

        text.value_rect = rect
        self.font.render_to(DisplayPygame._screen, (x, y), None,
                            DisplayPygame.COLORS[Display.BACKGROUND],
                            DisplayPygame.COLORS[Display.WALL])
        old_rect = old_rect.union(rect)
        x += grab

        for j in range(i+1, len(self._texts)):
            x += self._skip_width
            if x > self.x_max:
                break

            t = self._texts[j]

            rect = self.font.get_rect(t.prefix)
            rect.x += x
            rect.y = y - rect.y
            if rect.x + rect.width > self.x_max:
                break
            t.prefix_rect = rect
            self.font.render_to(DisplayPygame._screen, (x, y), None,
                                DisplayPygame.COLORS[Display.BACKGROUND],
                                DisplayPygame.COLORS[Display.WALL])
            old_rect = old_rect.union(rect)
            x = rect.x + rect.width
            x += 1

            # Sanity check
            assert x == t.value_x

            rect = self.font.get_rect(t.value)
            grab = rect.x + rect.width
            rect.x += x
            rect.y = y - rect.y
            if rect.x + rect.width > self.x_max:
                break

            assert grab <= t.value_grabbed
            if t.value_left:
                offset = 0
            else:
                offset = t.value_grabbed - grab
                rect.x += offset

            t.value_rect = rect
            self.font.render_to(DisplayPygame._screen, (x+offset, y), None,
                                DisplayPygame.COLORS[Display.BACKGROUND],
                                DisplayPygame.COLORS[Display.WALL])
            old_rect = old_rect.union(rect)
            x += t.value_grabbed

        # And finally queue all updates
        DisplayPygame._updates.append(old_rect)


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
        self.BOTTOM_WIDTH = self.WINDOW_X * self.columns
        self.BOTTOM_TEXT_Y += self.rows * self.WINDOW_Y

        self._window_x = np.tile  (np.arange(self.OFFSET_X, self.columns*self.WINDOW_X+self.OFFSET_X, self.WINDOW_X, dtype=TYPE_PIXELS), self.rows)
        self._window_y = np.repeat(np.arange(self.OFFSET_Y, self.rows   *self.WINDOW_Y+self.OFFSET_Y, self.WINDOW_Y, dtype=TYPE_PIXELS), self.columns)
        # print("window_x", self._window_x)
        # print("window_y", self._window_y)

        # self.last_collision_x = np.zeros(self.windows, dtype=TYPE_PIXELS)
        # self.last_collision_y = np.zeros(self.windows, dtype=TYPE_PIXELS)


    def start(self):
        rect = super().start()
        DisplayPygame._updates = [rect]


    def start_graphics(self):
        # Avoid pygame.init() since the init of the mixer component leads to 100% CPU
        pygame.display.init()
        pygame.display.set_caption(self.caption)
        # pygame.mouse.set_visible(1)
        pygame.key.set_repeat(DisplayPygame.KEY_DELAY, DisplayPygame.KEY_INTERVAL)

        pygame.freetype.init()
        self._font = pygame.freetype.Font(None, self.BLOCK)
        self._font.origin = True

        self._text_status = TextRow(x = self.BOTTOM_TEXT_X,
                                    y = self.BOTTOM_TEXT_Y,
                                    font = self._font,
                                    x_max = self.BOTTOM_WIDTH)
        self._text_status_pit = [
            TextRow(x = self.TOP_TEXT_X + self._window_x[w],
                    y = self.TOP_TEXT_Y + self._window_y[w],
                    font = self._font,
                    x_max = self.WINDOW_X + self._window_x[w]) for w in range(self.windows)]

        DisplayPygame._screen = pygame.display.set_mode((self.WINDOW_X * self.columns, self.WINDOW_Y * self.rows))
        rect = 0, 0, self.WINDOW_X * self.columns, self.WINDOW_Y * self.rows
        rect = pygame.draw.rect(DisplayPygame._screen,
                                DisplayPygame.COLORS[Display.WALL], rect)
        return rect


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


    def text_pit_register(self, text_field):
        for w in range(self.windows):
            self._text_status_pit[w].add(text_field)


    def text_register(self, text_field):
        self._text_status.add(text_field)


    def update(self):
        super().update()
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


    def draw_text(self, w, name, value):
        self._text_status_pit[w].draw_text(name, value)


    def draw_text_summary(self, name, value):
        self._text_status.draw_text(name, value)


    def draw_pit_empty(self, w):
        rect = (self._window_x[w] - self.OFFSET_X + self.BLOCK,
                self._window_y[w] - self.OFFSET_Y + self.BLOCK,
                self.WINDOW_X - 2 * self.BLOCK,
                self.WINDOW_Y - 2 * self.BLOCK)
        rect = pygame.draw.rect(DisplayPygame._screen,
                                DisplayPygame.COLORS[Display.BACKGROUND], rect)
        DisplayPygame._updates.append(rect)


    def draw_block(self, w, x, y, color, update = True, combine = None,
                   x_delta = None, y_delta = None):
        if x_delta:
            if x_delta > 0:
                rect = (x * self.BLOCK + self._window_x[w] + Display.EDGE,
                        y * self.BLOCK + self._window_y[w] + Display.EDGE,
                        self.BLOCK, self.DRAW_BLOCK)
            else:
                rect = (x * self.BLOCK + self._window_x[w] - Display.EDGE,
                        y * self.BLOCK + self._window_y[w] + Display.EDGE,
                        self.BLOCK, self.DRAW_BLOCK)
        elif y_delta:
            if y_delta > 0:
                rect = (x * self.BLOCK + self._window_x[w] + Display.EDGE,
                        y * self.BLOCK + self._window_y[w] + Display.EDGE,
                        self.DRAW_BLOCK, self.BLOCK)
            else:
                rect = (x * self.BLOCK + self._window_x[w] + Display.EDGE,
                        y * self.BLOCK + self._window_y[w] - Display.EDGE,
                        self.DRAW_BLOCK, self.BLOCK)
        elif color == Display.BACKGROUND:
            rect = (x * self.BLOCK + self._window_x[w],
                    y * self.BLOCK + self._window_y[w],
                    self.BLOCK, self.BLOCK)
        else:
            rect = (x * self.BLOCK + self._window_x[w] + Display.EDGE,
                    y * self.BLOCK + self._window_y[w] + Display.EDGE,
                    self.DRAW_BLOCK, self.DRAW_BLOCK)

        # print("Draw %d (%d,%d): %d,%d,%d: [%d %d %d %d]" % ((w, x, y)+color+(rect)))
        rect = pygame.draw.rect(DisplayPygame._screen, DisplayPygame.COLORS[color], rect)
        if combine:
            rect = rect.union(combine)
        if update:
            DisplayPygame._updates.append(rect)
        return rect
