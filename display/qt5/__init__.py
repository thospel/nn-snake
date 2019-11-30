import math

from display import Display
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout, QFrame, QVBoxLayout, QHBoxLayout, QShortcut
from PyQt5.QtGui  import QPixmap, QPainter, QColor, QBrush, QPen, QKeySequence
from PyQt5.QtCore import QTimer, Qt, QRect

# Some Painting notes (to collect some scattered information)
# - QPixmap is an offscreen buffer which can be used to restore obscured areas
#   is stored on the XServer when using X11 backend
# - QBitmap is just a convenient way of saying "QPixmap with depth 1"
# - QImage is an "array in memory" of the client program
# - QPicture remembers painter commands in a vectorial way
#   The list of painter commands is reset on each call to the QPainter::begin()


# Qpainter begin/end associate the painter with a QPaintdevive
# (the thing you paint on, e.g. a QPixmap, QWidget, QPicture etc)
# You can also pass the QPaintdevive on construction in which case begin/end
# are not needed
# Only one (active) Qpainter can be attached to a QPaintdevive at a time

# const QFont fixedFont = QFontDatabase::systemFont(QFontDatabase::FixedFont)

app = QApplication(["dummy", "-stylesheet", "snake.qss"])
# Replace this by a manual setStyleSheet() at some point

COLORS = [
    QColor(0, 0, 0),
    QColor(255, 255, 255),
    QColor(0, 255, 0),
    QColor(200, 200, 0),
    QColor(160, 160, 160),
    QColor(255, 0, 0)
]

EDGE_SHOWN_X = 1/2
EDGE_SHOWN_Y = 1/2
SCALE = 10
BLOCK = SCALE - 2 * Display.EDGE

TEXT_SPACING = 9

def matrix_from_transform(transform):
    return [[transform.m11(), transform.m12(), transform.m13()],
            [transform.m21(), transform.m22(), transform.m23()],
            [transform.m31(), transform.m32(), transform.m33()]]


# QLabel that never lets go of horizontal space it acquired
class QLabelGreedy(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._grabbed = 0


    def grabbed(self):
        return self._grabbed


    def grow(self, size):
        if size > self._grabbed:
            self._grabbed = size
        return self._grabbed


    def sizeHint(self):
        hint = super().sizeHint()
        if hint.width() > self._grabbed:
            self._grabbed = hint.width()
            hint.setWidth(self._grabbed)
            self.setMinimumSize(hint)
        return hint


class PitStatus(QFrame):
    def __init__(self, snake_nr):
        super().__init__()
        self._text = dict()

        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        # layout.addStretch()

        self.text_add(layout, Display.TEXT_SCORE,    "Score:")
        self.text_add(layout, Display.TEXT_GAME,     "Game:")
        self.text_add(layout, Display.TEXT_MOVES,    "Moves:")
        self.text_add(layout, Display.TEXT_WON,      "Won:")
        self.text_add(layout, Display.TEXT_SNAKE_ID, "Id:", str(snake_nr))

        layout.addStretch()
        self.setLayout(layout)
        self.setObjectName("pitStatus")


    def text_add(self, layout, key, name, format = "%d", initial_value = "0"):
        label_key   = QLabel(name)
        label_value = QLabelGreedy(initial_value)
        id = key.capitalize()
        label_key  .setObjectName("key"   + id)
        label_value.setObjectName("value" + id)

        # layout.addSpacing(TEXT_SPACING)
        layout.addWidget(QLabel(" "))
        layout.addWidget(label_key)
        layout.addWidget(label_value)
        # layout.addStretch()

        if key in self._text:
            raise(AssertionError("Duplicate text key " + key))
        self._text[key] = (label_value, format)


    def draw_text(self, name, value):
        (label, format) = self._text[name]
        label.setText(str(format % value))


class Pit(QWidget):
    def __init__(self, display, snake_nr):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        self._status = PitStatus(snake_nr)
        layout.addWidget(self._status)
        snakes = display.snakes()

        self._pos = 0
        label = QLabel()
        units_x = snakes.WIDTH  + EDGE_SHOWN_X * 2
        units_y = snakes.HEIGHT + EDGE_SHOWN_Y * 2
        pixmap = QPixmap(units_x * display.BLOCK, units_y * display.BLOCK)
        # Pixmap is created uninitialized
        pixmap.fill(COLORS[Display.WALL])
        label.setPixmap(pixmap)

        # Fetch the pixmap from the label which will be *different*
        # According to the official Qt documentation you are only allowed to
        # paint during a paint_event. But what do they know...
        painter = QPainter(label.pixmap())

        # Set up coordinates
        unit_x = pixmap.width()  / units_x
        unit_y = pixmap.height() / units_y
        if unit_x < unit_y:
            unit = unit_x
            side_y = units_y * unit
            x = EDGE_SHOWN_X * unit
            y = EDGE_SHOWN_Y * unit
            y += (pixmap.height() - side_y) / 2
        else:
            unit = unit_y
            side_x = units_x * unit
            x = EDGE_SHOWN_X * unit
            y = EDGE_SHOWN_Y * unit
            x += (pixmap.width() - side_x) / 2
        # print("Viewport", x, y, snakes.WIDTH * unit, snakes.HEIGHT * unit)
        painter.setViewport(x, y, snakes.WIDTH * unit, snakes.HEIGHT * unit)
        # print("Window", SCALE * snakes.VIEW_X, SCALE * snakes.VIEW_Y,
        #                 SCALE * snakes.WIDTH,  SCALE * snakes.HEIGHT)
        painter.setWindow(SCALE * snakes.VIEW_X, SCALE * snakes.VIEW_Y,
                          SCALE * snakes.WIDTH,  SCALE * snakes.HEIGHT)
        self._transform = painter.combinedTransform()

        self._painter = painter
        self._brush = QBrush(COLORS[Display.WALL], Qt.SolidPattern)
        self._label = label

        layout.addWidget(label)
        self.setLayout(layout)
        self.setObjectName("pit")


    def stop(self):
        self._painter.end()


    def draw_text(self, name, value):
        self._status.draw_text(name, value)


    def draw_block(self, x, y, color, update, combine):
        painter = self._painter

        brush = self._brush
        brush.setColor(COLORS[color])

        rect = QRect(SCALE * x + Display.EDGE, SCALE * y + Display.EDGE,
                     BLOCK, BLOCK)
        painter.fillRect(rect, brush)
        rect = painter.combinedTransform().mapRect(rect)
        if combine:
            rect |= combine
        if update:
            self._label.update(rect)
            return None
        else:
            return rect


    def draw_pit_empty(self, snakes):
        painter = self._painter

        brush = self._brush
        brush.setColor(COLORS[Display.BACKGROUND])

        rect = QRect(SCALE * snakes.VIEW_X, SCALE * snakes.VIEW_Y,
                     SCALE * snakes.WIDTH,  SCALE * snakes.HEIGHT)
        painter.fillRect(rect, brush)
        rect = painter.combinedTransform().mapRect(rect)
        self._label.update(rect)


class Screen(QWidget):
    def __init__(self, display):
        super().__init__()

        self._text = dict()
        self._display = display

        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        layout.addWidget(self.pit_box(display))
        layout.addWidget(self.status_bar())

        self.setLayout(layout)
        self.setObjectName("screen")

        QShortcut(QKeySequence('Shift+Q'), self, display.event_quit)
        QShortcut(QKeySequence('s'), self, display.event_single_step)
        QShortcut(QKeySequence(' '), self, display.event_toggle_run)
        QShortcut(QKeySequence('r'), self, display.event_toggle_run)
        QShortcut(QKeySequence('d'), self, display.event_debug)
        QShortcut(QKeySequence('Shift+D'), self, display.event_dump)
        QShortcut(QKeySequence('+'), self, display.event_speed_higher)
        QShortcut(QKeySequence('-'), self, display.event_speed_lower)
        QShortcut(QKeySequence('='), self, display.event_speed_normal)

        self.show()


    def closeEvent(self, event):
        self._display.event_quit()
        event.ignore()


    def stop(self):
        for pit in self._pits:
            pit.stop()
        self.close()


    def pit_box(self, display):
        frame = QFrame()

        layout = QGridLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        self._pits = []
        for r in range(display.rows):
            for c in range(display.columns):
                pit = Pit(display, len(self._pits))
                self._pits.append(pit)
                layout.addWidget(pit, r, c)
        frame.setLayout(layout)
        frame.setObjectName("pits")
        return frame


    def pit(self, i):
        return self._pits[i]


    def status_bar(self):
        frame = QFrame()

        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        layout.addStretch()

        self.text_add(layout, Display.TEXT_STEP,      "Step:"),
        self.text_add(layout, Display.TEXT_SCORE_MAX, "Max Score:"),
        self.text_add(layout, Display.TEXT_MOVES_MAX, "Max Moves:"),
        self.text_add(layout, Display.TEXT_GAME_MAX,  "Max Game:"),
        # self.text_add(layout, Display.TEXT_SCORE_PER_SNAKE, "Score/Snake:"
        #               format = "%.3f", initial_value = None),
        self.text_add(layout, Display.TEXT_SCORE_PER_GAME, "Score/Game:",
                      format = "%.3f", initial_value = None),
        self.text_add(layout, Display.TEXT_MOVES_PER_GAME, "Moves/Game:",
                      format = "%.3f", initial_value = None),
        self.text_add(layout, Display.TEXT_MOVES_PER_APPLE, "Moves/Apple:",
                      format = "%.3f", initial_value = None),
        self.text_add(layout, Display.TEXT_TIME,      "Time:"),
        self.text_add(layout, Display.TEXT_WINS,      "Won:"),
        self.text_add(layout, Display.TEXT_GAMES,     "Games:"),

        frame.setLayout(layout)
        frame.setObjectName("status")
        return frame


    def text_add(self, layout, key, name, format = "%d", initial_value = "0"):
        label_key   = QLabel(name)
        label_value = QLabelGreedy("---" if initial_value is None else initial_value)
        id = key.capitalize()
        label_key  .setObjectName("key"   + id)
        label_value.setObjectName("value" + id)

        # layout.addSpacing(TEXT_SPACING)
        layout.addWidget(QLabel(" "))
        layout.addWidget(label_key)
        layout.addWidget(label_value)
        layout.addStretch()

        if key in self._text:
            raise(AssertionError("Duplicate text key " + key))
        self._text[key] = (label_value, format)


    def draw_text(self, name, value):
        (label, format) = self._text[name]
        label.setText(str(format % value))


class DisplayQt5(Display):
    def __init__(self, snakes,
                 slow_updates = 0,
                 **kwargs):
        super().__init__(snakes, **kwargs)

        if not self.windows:
            return


    def start(self):
        super().start()
        if not self.windows:
            return

        screen = Screen(self)
        self._stepper = QTimer()
        self._screen = screen
        # self._stepper.timeout.connect(self.step)


    def stop(self):
        self._screen.stop()
        del self._stepper
        del self._screen

        # super().stop()


    def set_timer_step(self, to_sleep, callback, now_monotonic=None):
        self._stepper.singleShot(math.ceil(to_sleep * 1000), callback)


    def draw_pit_empty(self, w):
        self._screen.pit(w).draw_pit_empty(self.snakes())


    def loop(self):
        app.exec_()


    def draw_block(self, w, x, y, color, update=True, combine=None):
        return self._screen.pit(w).draw_block(x, y, color, update, combine)


    def draw_text(self, w, name, value):
        self._screen.pit(w).draw_text(name, value)


    def draw_text_summary(self, name, value):
        self._screen.draw_text(name, value)
