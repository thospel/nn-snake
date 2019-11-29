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

app = QApplication([])

COLORS = [
    QColor(0, 0, 0),
    QColor(255, 255, 255),
    QColor(0, 255, 0),
    QColor(200, 200, 0),
    QColor(160, 160, 160),
    QColor(255, 0, 0)
]

EDGE_SHOWN_X = 1
EDGE_SHOWN_Y = 1
SCALE = 10
BLOCK = SCALE - 2 * Display.EDGE

def matrix_from_transform(transform):
    return [[transform.m11(), transform.m12(), transform.m13()],
            [transform.m21(), transform.m22(), transform.m23()],
            [transform.m31(), transform.m32(), transform.m33()]]

class PitStatus(QWidget):
    def __init__(self, snake_nr):
        super().__init__()
        self._text = dict()

        layout = QHBoxLayout()

        layout.addWidget(QLabel("Score:"))
        label = QLabel("0")
        self._text[Display.TEXT_SCORE] = label
        layout.addWidget(label)

        layout.addWidget(QLabel("Game:"))
        label = QLabel("0")
        self._text[Display.TEXT_GAME] = label
        layout.addWidget(label)

        layout.addWidget(QLabel("Moves:"))
        label = QLabel("0")
        self._text[Display.TEXT_MOVES] = label
        layout.addWidget(label)

        layout.addWidget(QLabel("Won:"))
        label = QLabel("0")
        self._text[Display.TEXT_WON] = label
        layout.addWidget(label)

        layout.addWidget(QLabel("Id:"))
        label = QLabel(str(snake_nr))
        self._text[Display.TEXT_SNAKE_ID] = label
        layout.addWidget(label)

        self.setLayout(layout)


    def draw_text(self, name, value):
        self._text[name].setText(str(value))


class Pit(QWidget):
    def __init__(self, display, snake_nr):
        super().__init__()
        layout = QVBoxLayout()
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
        painter = QPainter(label.pixmap())
        painter.setPen(QPen(COLORS[Display.WALL], 1, Qt.SolidLine))
        painter.setBrush(QBrush(COLORS[Display.WALL], Qt.SolidPattern))

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

        self._label = label

        layout.addWidget(label)
        self.setLayout(layout)


    def draw_text(self, name, value):
        self._status.draw_text(name, value)


    def draw_block(self, x, y, color, update):
        print("Draw block", x, y, color, update)
        painter = self._painter

        pen = painter.pen()
        pen.setColor(COLORS[color])
        painter.setPen(pen)

        brush = painter.brush()
        brush.setColor(COLORS[color])
        painter.setBrush(brush)

        rect = QRect(SCALE * x + Display.EDGE, SCALE * y + Display.EDGE,
                     BLOCK, BLOCK)
        painter.drawRect(rect)
        print("Transform", matrix_from_transform(painter.combinedTransform()))
        new_rect = painter.combinedTransform().mapRect(rect)
        print("Rect", rect)
        print("New Rect", new_rect)
        self._label.update(new_rect)


    def draw_pit_empty(self, snakes):
        painter = self._painter

        pen = painter.pen()
        pen.setColor(COLORS[Display.BACKGROUND])
        painter.setPen(pen)

        brush = painter.brush()
        brush.setColor(COLORS[Display.BACKGROUND])
        painter.setBrush(brush)

        rect = QRect(SCALE * snakes.VIEW_X, SCALE * snakes.VIEW_Y,
                     SCALE * snakes.WIDTH,  SCALE * snakes.HEIGHT)
        painter.drawRect(rect)
        new_rect = painter.combinedTransform().mapRect(rect)
        self._label.update(new_rect)


class Screen(QWidget):
    def __init__(self, display):
        super().__init__()
        layout = QGridLayout()
        self._pits = []
        for r in range(display.rows):
            for c in range(display.columns):
                pit = Pit(display, len(self._pits))
                self._pits.append(pit)
                layout.addWidget(pit, r, c)
        self.setLayout(layout)
        QShortcut(QKeySequence('s'), self, display.event_single_step)
        QShortcut(QKeySequence(' '), self, display.event_toggle_run)
        QShortcut(QKeySequence('r'), self, display.event_toggle_run)
        QShortcut(QKeySequence('d'), self, display.event_debug)
        # QShortcut(QKeySequence('D'), self, display.event_dump)
        QShortcut(QKeySequence('+'), self, display.event_speed_higher)
        QShortcut(QKeySequence('-'), self, display.event_speed_lower)
        QShortcut(QKeySequence('='), self, display.event_speed_normal)
        self.show()

    def pit(self, i):
        return self._pits[i]

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
        super().stop()

        del self._stepper
        del self._screen


    def set_timer_step(self, to_sleep, callback, now_monotonic=None):
        self._stepper.singleShot(math.ceil(to_sleep * 1000), callback)


    def draw_pit_empty(self, w):
        self._screen.pit(w).draw_pit_empty(self.snakes())


    def draw_text_summary(self, *args):
        pass


    def loop(self):
        app.exec_()


    def draw_block(self, w, x, y, color, update=True):
        self._screen.pit(w).draw_block(x, y, color, update)


    def draw_text(self, w, name, value):
        self._screen.pit(w).draw_text(name, value)
