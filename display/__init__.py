import time
import math
import sys
import os

# +
from dataclasses import dataclass

@dataclass
class TextField:
    key:    str
    prefix: str
    format: str         = "%d"
    initial_value: str  = "0"

# +
TEXT_SCORE           = "score"
TEXT_GAME            = "game"
TEXT_MOVES           = "moves"
TEXT_WON             = "win"
TEXT_SNAKE_ID        = "snake"

TEXT_STEP            = "step"
TEXT_SCORE_MAX       = "score_max"
TEXT_MOVES_MAX       = "moves_max"
TEXT_GAME_MAX        = "game_max"
TEXT_SCORE_PER_GAME  = "score_per_game"
TEXT_MOVES_PER_GAME  = "moves_per_game"
TEXT_MOVES_PER_APPLE = "moves_per_apple"
TEXT_SCORE_PER_SNAKE = "score_per_snake"
TEXT_TIME            = "time"
TEXT_WINS            = "wins"
TEXT_GAMES           = "games"

TEXTS_STATUS_PIT = [
    TextField(TEXT_SCORE,    "Score:", format = "%3d"),
    # TextField(TEXT_GAME,     "Game:",  format = "%2d"),
    TextField(TEXT_MOVES,    "Moves:", format = "%4d"),
    TextField(TEXT_WON,      "Won:",   format = "%d / %d"),
    TextField(TEXT_SNAKE_ID, "Id:"),
]

TEXTS_STATUS = [
    TextField(TEXT_STEP,        "Step:", format = "%4d"),
    TextField(TEXT_SCORE_MAX,   "Max Score:", format = "%3d"),
    TextField(TEXT_MOVES_MAX,   "Max Moves:", format = "%4d"),
    TextField(TEXT_GAME_MAX,    "Max Game:",  format = "%3d"),
#   TextField(TEXT_SCORE_PER_SNAKE, "Score/Snake:"),
    TextField(TEXT_SCORE_PER_GAME,  "Score/Game:",
              format = "%7.3f", initial_value = "---"),
    TextField(TEXT_MOVES_PER_GAME,  "Moves/Game:",
              format = "%7.3f", initial_value = "---"),
    TextField(TEXT_MOVES_PER_APPLE, "Moves/Apple:",
              format = "%6.3f", initial_value = "---"),
    TextField(TEXT_TIME,        "Time:"),
    # Put games last. If you have a lot of snakes this can go up very fast
    TextField(TEXT_WINS,        "Won:", format = "%d / %d"),
    # TextField(TEXT_GAMES,       "Games:"),
]


def log_fh(fh, name, format, value):
    value = format % value
    if "\n" in value:
        # Should also check that value doesn't contain EOT...
        print(name + ": <<EOT", file=fh)
        print(value, file=fh)
        print("EOT", file=fh)
    else:
        print(name + ":" + value, file=fh)


class Display:
    EDGE = 1

    BACKGROUND = 0
    WALL       = 1
    APPLE      = 2
    HEAD       = 3
    BODY       = 4
    COLLISION  = 5
    TRACE      = 6

    # Event polling time in paused mode.
    # Avoid too much CPU waste or even a busy loop in case fps == 0
    POLL_SLOW = 1/25
    POLL_MAX = POLL_SLOW * 1.5
    WAIT_MIN = 1/1000

    STARTED = 0

    def __init__(self, snakes,
                 rows         = 1,
                 columns      = 2,
                 block_size   = 20,
                 log_period   = 1,
                 log_period_frames = False,
                 log_file     = "snakes.log.txt",
                 dump_file    = "snakes.dump.txt",
                 tensor_board = None,
                 stream_file  = "snakes.stream.%dx%d.rgb",
                 stepping     = False,
                 caption      = "Snakes"):
        self.rows    = rows
        self.columns = columns
        self.windows = rows*columns
        self._stepping = stepping

        self._log_period = log_period
        self._log_period_frames = log_period_frames
        self._log_file  = log_file
        self._log_fh = None
        self._dump_file = dump_file
        self._stream_file = stream_file

        if tensor_board:
            # The only reason I don't simply have this import at the top is that
            # tensorflow is *huge* and takes a long time to load and be default
            # it is not needed
            global tf
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)

            # Strictly speaking gmtime would be less ambiguous, but humans seem
            # to prefer local time for some reason. Go figure.
            when = time.strftime("%Y%m%d_%H%M%S")
            self._tb_dir = "%s/%dx%d/%s_%d" % (
                tensor_board, snakes.WIDTH, snakes.HEIGHT, when, os.getpid())
            self._tb_writer = tf.summary.create_file_writer(self._tb_dir)
            self._tb_writer.set_as_default()
            # tf.summary.trace_on(graph=True)
            self._log_tb_previous = None
        else:
            self._tb_writer = None

        if not self.windows:
            return

        self.caption = caption

        self.BLOCK = block_size
        self.DRAW_BLOCK = self.BLOCK-2 * Display.EDGE

        self.WINDOW_X = (snakes.WIDTH +2) * self.BLOCK
        self.WINDOW_Y = (snakes.HEIGHT+2) * self.BLOCK
        self.OFFSET_X = (1-snakes.VIEW_X) * self.BLOCK
        self.OFFSET_Y = (1-snakes.VIEW_Y) * self.BLOCK


    def snakes(self):
        return self._snakes


    def move(self):
        next(self._moves)


    def log_open(self):
        if self._log_file is not None and self._log_file != "":
            self._log_fh = open(self._log_file, "w")
            self._log_previous = None


    def log_close(self):
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None


    def log_start(self):
        if not self._log_fh:
            return

        fh = self._log_fh

        def log_action(name, format, value):
            log_fh(fh, name, format, value)
        def log_action_start(name, format, value):
            log_action(name + "_start", format, value)

        self._log_start(log_action_start)
        snakes = self.snakes()
        snakes.log_constants(log_action)
        log_action("Frame", "%d", snakes.frame())


    def _log_start(self, log_action):
        log_action("time", "%s",
                   time.strftime("%Y-%m-%d %H:%M:%S %z",
                                 time.localtime(self._time_start)))
        log_action("time_epoch",     "%.3f", self._time_start)
        log_action("time_monotonic", "%.3f", self._time_monotonic_start)
        log_action("time_process",   "%.3f", self._time_process_start)


    def log_stop(self):
        if not self._log_fh:
            return

        self.log_frame()
        fh = self._log_fh
        def log_action(name, format, value):
            log_fh(fh, name + "_end", format, value)
        self._log_stop()


    def _log_stop(self, log_action):
        log_action("time", "%s",
                   time.strftime("%Y-%m-%d %H:%M:%S %z",
                                 time.localtime(self._time_end)))
        log_action("time_epoch",     "%.3f", self._time_end)
        log_action("time_monotonic", "%.3f", self._time_monotonic_end)
        log_action("time_process",   "%.3f", self._time_process_end)


    def log_frame(self):
        if not self._log_fh:
            return

        fh = self._log_fh
        print("#----", file=fh)

        def log_action(name, format, value):
            log_fh(fh, name, format, value)
        self._log_previous = self._log_frame(log_action, self._log_previous)


    def dump(self):
        with open(self._dump_file, "w") as fh:
            self._dump_previous = None
            self.timestamp()
            self.dump_fh(fh)


    def dump_fh(self, fh):
        snakes = self.snakes()

        def log_action(name, format, value):
            log_fh(fh, name, format, value)

        snakes.log_constants(log_action)
        self._dump_previous = self._log_frame(log_action, self._dump_previous)
        snakes.dump_fh(fh)


    def log_tensor_board_start(self):
        if not self._tb_writer:
            return

        frame = self.snakes().frame()

        def log_action_start(name, format, value):
            name = "Start/%s" % name
            if format == " %s" or format == "%s":
                tf.summary.text(name,
                                "    " + value.replace("\n", "\n    "),
                                step=frame)
            else:
                tf.summary.scalar(name,
                                  value,
                                  step=frame)

        self._log_start(log_action_start)

        def log_action(name, format, value):
            if format == " %s" or format == "%s":
                tf.summary.text(name,
                                "    " + value.replace("\n", "\n    "),
                                step=frame)
            else:
                tf.summary.scalar("Constants/%s" % name,
                                  value,
                                  step=frame)
        snakes = self.snakes()
        snakes.log_constants(log_action)


    def log_tensor_board_stop(self):
        if not self._tb_writer:
            return

        self.log_tensor_board()

        frame = self.snakes().frame()
        def log_action_stop(name, format, value):
            name = "End/%s" % name
            if format == " %s" or format == "%s":
                tf.summary.text(name,
                                "    " + value.replace("\n", "\n    "),
                                step=frame)
            else:
                tf.summary.scalar(name,
                                  value,
                                  step=frame)
        self._log_stop(log_action_stop)


    def log_tensor_board(self):
        if not self._tb_writer:
            return

        frame = self.snakes().frame()
        def log_action(name, format, value):
            tf.summary.scalar(name, value, step=frame)
        self._log_tb_previous = self._log_frame(log_action, self._log_tb_previous)


    def log_graph(self, model, summary):
        if not self._tb_writer:
            return
        tf.summary.text("Model Summary",
                        "    " + summary.replace("\n", "\n    "),
                        step=0)
        tf.keras.utils.plot_model(model, "model.png",
                                  show_shapes = True,
                                  show_layer_names = True)
        #with open("model.png", "rb") as fh:
        #    image = tf.image.decode_png(fh.read(), channels=1)
        #tf.summary.image("Model Plot", tf.expand_dims(image, 0), step=0)
        # I was unable to find a way to get the actual graph for export
        # tf.summary.keras_model('keras', model,step=0)
        # tf.summary.graph(tf.get_default_graph())
        # tf.summary.trace_export("Graph", step=self.snakes().frame())
        # tf.summary.flush()


    def _log_frame(self, log_action, previous):
        snakes = self.snakes()

        current = {
            "elapsed": self.elapsed() - self.paused(),
            "used": self.elapsed_process(),
            "frame": snakes.frame(),
        }

        log_action("Frame",      "%12d"   , snakes.frame())
        log_action("Elapsed",    "%14.3f" , self.elapsed())
        log_action("Paused",     "%15.3f" , self.paused())
        log_action("Used",       "%17.3f" , self.elapsed_process())
        log_action("Frame rate", "%11.3f" , self.frame_rate(snakes))
        snakes.log_frame(log_action, current)

        if previous:
            delta = { k: v - previous[k] for k, v in current.items() }

            if delta["elapsed"]:
                log_action("L Frame rate",  "%9.3f" ,
                           delta["frame"] / delta["elapsed"])

            if delta["used"]:
                log_action("L Frame/Used",  "%9.3f" ,
                           delta["frame"] / delta["used"])
            snakes.log_delta(log_action, delta)

        return current


    def start(self):
        Display.STARTED +=1
        self._streaming = False

        if not self.windows:
            return []

        self._background = None

        self._stream_fh = None
        self._stream_warned = False

        rects = self.start_graphics()

        for text_field in TEXTS_STATUS:
            self.text_register(text_field),

        for text_field in TEXTS_STATUS_PIT:
            self.text_pit_register(text_field)

        return rects


    def stop(self):
        del self._streaming
        del self._snakes
        del self._moves

        if self.windows:
            del self._background
            del self._stream_warned
            if self._stream_fh:
                self._stream_fh.close()
            del self._stream_fh

        Display.STARTED -=1


    def start_graphics(self, key, prefix, **kwargs):
        raise(NotImplementedError("start_graphics not implemented for " +
                                   type(self).__name__))


    def text_pit_register(self, text_field):
        raise(NotImplementedError("text_pit_register not implemented for " +
                                   type(self).__name__))


    def text_register(key, text_field):
        raise(NotImplementedError("text_register not implemented for " +
                                   type(self).__name__))


    def image_save(self, w = 0):
        raise(NotImplementedError("image_save not implemented for " +
                                   type(self).__name__))


    def draw_line(self, w, x0, y0, x1, y1, color, update = True, combine = None):
        raise(NotImplementedError("draw_line not implemented for " +
                                   type(self).__name__))


    def draw_block(self, w, x, y, color, update=True):
        raise(NotImplementedError("draw_block not implemented for " +
                                   type(self).__name__))


    def draw_text(self, w, name, value):
        raise(NotImplementedError("draw_text not implemented for " +
                                   type(self).__name__))


    def draw_text_summary(self, *args):
        raise(NotImplementedError("draw_text_summary not implemented for " +
                                   type(self).__name__))


    def changed(self, w, rect):
        raise(NotImplementedError("changed not implemented for " +
                                   type(self).__name__))


    def wait_key(self, *args):
        raise(NotImplementedError("wait_key not implemented for " +
                                   type(self).__name__))


    def stream_geometry(self):
        return 0, 0


    def stream_image(self):
        if not self._stream_warned:
            print("stream_image not implemented for " + type(self).__name__)
            self._stream_warned = True


    def draw_apples(self, i_index, w_index, apple, score):
        apple_y, apple_x = apple
        # print("draw_apples", i_index, w_index, score)
        # print(np.stack((apple_x, apple_y), axis=-1))
        for i, w, x, y in zip(i_index, w_index, apple_x, apple_y):
            self.draw_block(w, x, y, Display.APPLE)
            self.draw_text(w, TEXT_SCORE, score[i])


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
                body_rect = self.draw_block(
                    w, w_head_x_old[w], w_head_y_old[w], Display.BODY,
                    update=False,
                    x_delta = w_head_x_new[w] - w_head_x_old[w],
                    y_delta = w_head_y_new[w] - w_head_y_old[w])
            if not is_eat[i]:
                # Drop the tail if we didn't eat an apple
                self.draw_block(w, w_tail_x[w], w_tail_y[w], Display.BACKGROUND)
            if body_rect:
                head_rect = self.draw_block(w, w_head_x_new[w], w_head_y_new[w], Display.HEAD, combine = body_rect)
            else:
                self.draw_block(w, w_head_x_new[w], w_head_y_new[w], Display.HEAD)
            self.draw_text(w, TEXT_MOVES, w_nr_moves[w])


    def draw_collisions(self, i_index, w_index, pos, nr_games, nr_games_won):
        y , x = pos
        for w, i, x, y in zip(w_index, i_index, pos[1], pos[0]):
            if False:
                # This test was for when the idea was to freeze some snakes
                # if snakes._nr_moves[w] > snakes.frame():
                self.draw_block(w, x, y, COLLISION)
            else:
                #self.draw_block(w,
                #                self.last_collision_x[w],
                #                self.last_collision_y[w],
                #                DisplayPygame.WALL)
                self.draw_text(w, TEXT_WON, (nr_games_won[i], nr_games[i]))
                # self.draw_text(w, TEXT_SNAKE_ID, i)
                self.draw_pit_empty(w)


    def draw_windows(self, window_iterator, w_head):
        snakes = self.snakes()
        w_head_y, w_head_x = w_head

        for i in window_iterator:
            w = window_iterator.iterindex

            self.draw_pit_empty(w)
            self.draw_block(w, w_head_x[w], w_head_y[w], Display.HEAD)

            self.draw_text(w, TEXT_MOVES, snakes.nr_moves(i))
            self.draw_text(w, TEXT_WON,   (snakes.nr_games_won(i), snakes.nr_games(i)))
            self.draw_text(w, TEXT_SNAKE_ID, i)


    # Dislay the current state
    def draw_result(self, initial = False):
        if not self.windows:
            return

        snakes = self.snakes()

        # draw_text is optimized to not draw any value that didn't change
        self.draw_text_summary(TEXT_STEP, snakes.frame())
        # self.draw_text_summary(TEXT_SCORE_PER_SNAKE, snakes.score_total_snakes() / snakes._nr_snakes)
        self.draw_text_summary(TEXT_SCORE_MAX, snakes.score_max())
        self.draw_text_summary(TEXT_MOVES_MAX, snakes.nr_moves_max())
        self.draw_text_summary(TEXT_GAME_MAX,  snakes.nr_games_max())
        self.draw_text_summary(TEXT_MOVES_MAX, snakes.nr_moves_max())
        self.draw_text_summary(TEXT_WINS,      (snakes.nr_games_won_total(), snakes.nr_games_total()))
        if snakes.nr_games_total():
            self.draw_text_summary(TEXT_SCORE_PER_GAME,  snakes.score_per_game())
            self.draw_text_summary(TEXT_MOVES_PER_GAME,  snakes.nr_moves_per_game())
        if snakes.score_total_games():
            self.draw_text_summary(TEXT_MOVES_PER_APPLE, snakes.nr_moves_per_apple())

        if initial:
            # Fake it (the real timers haven't started yet)
            self._elapsed_sec_draw = 0
            elapsed_sec = 0
        else:
            elapsed_sec = int(self.elapsed(time.monotonic()))
        if elapsed_sec != self._elapsed_sec_draw:
            self._elapsed_sec_draw = elapsed_sec
            self.draw_text_summary(TEXT_TIME, elapsed_sec)


    def timers_start(self):
        # print("Start at time 0, frame", snakes.frame())

        self._paused = 0
        self._elapsed_last  = -self._log_period

        self._time_start = time.time()
        self._time_process_start = time.process_time()
        self._time_monotonic_start  = time.monotonic()
        self._time_end           = self._time_start
        self._time_process_end   = self._time_process_start
        self._time_monotonic_end = self._time_monotonic_start
        self._time_target        = self._time_monotonic_start
        if self._stepping:
            self._pause_time = self._time_monotonic_start
            self._stepping = False
            # When stepping the first frame doesn't count
        else:
            self._pause_time = 0


    # Return elapsed monotonic seconds
    def timestamp(self, now_monotonic = None):
        self._time_end = time.time()
        self._time_process_end = time.process_time()
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        self._time_monotonic_end = now_monotonic
        if self._pause_time:
            self._paused += self._time_monotonic_end - self._pause_time
            self._pause_time = self._time_monotonic_end
            frame = self.snakes().frame()
        return now_monotonic - self._time_monotonic_start


    def elapsed(self, now_monotonic = None):
        if now_monotonic is None:
            return self._time_monotonic_end - self._time_monotonic_start
        else:
            return now_monotonic - self._time_monotonic_start


    def elapsed_process(self):
        return self._time_process_end - self._time_process_start


    # How long we didn't run
    def paused(self):
        return self._paused


    def frame_rate(self, snakes):
        elapsed = self.elapsed() - self.paused()
        frames  = snakes.frame()
        if elapsed == 0:
            # This properly handles positive, negative and 0 (+inf, -inf, nan)
            return math.inf * int(frames)
        return frames / elapsed


    def run(self, snakes, fps = 40, stepping = False,
            frame_max = -1, game_max = -1):
        self._stepping = stepping
        self._frame_max = frame_max
        self._game_max = game_max if game_max >= 0 else math.inf
        self._snakes = snakes
        if fps > 0:
            self._poll_fast = 1 / fps
        elif fps == 0:
            self._poll_fast = 0
        else:
            raise(ValueError("fps must not be negative"))
        self._poll_fast0 = self._poll_fast

        self.start()
        self.log_open()
        snakes.run_start(self)
        self._moves = snakes.move_generator(self)

        # This is not so much a move as initializing everything
        self.move()
        self.draw_result(initial=True)
        self.update()

        # Continue in run1()
        self._quit = False
        self.set_timer_step(0, self.run1)
        self.timers_start()


    # Continuation of run(), but called from loop()
    def run1(self):
        self.log_start()
        self.log_tensor_board_start()
        # Loop in step() until event_quit sets it to run2()
        self.set_timer_step(0, self.step)


    # Continuation of run1(), called from loop()
    def run2(self):
        self.timestamp()
        self.snakes().run_finish()
        self.log_tensor_board_stop()
        self.log_stop()
        self.log_close()
        self.stop()
        # And this finishes run()


    def loop(self):
        now_monotonic = time.monotonic()
        while Display.STARTED > 0:
            if self._to_sleep > 0:
                time.sleep(self._to_sleep)
                now_monotonic = time.monotonic()
            now_monotonic = self.events_process(now_monotonic)
            now_monotonic = self._callback()


    def set_timer_step(self, to_sleep=0, callback=None, now_monotonic=None):
        self._to_sleep = to_sleep
        self._callback = callback


    def step(self, now_monotonic = None):
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        if self._quit:
            self.set_timer_step(0, self.run2, now_monotonic)
            return now_monotonic

        if self._pause_time and not self._stepping:
            self._time_target = now_monotonic + Display.POLL_SLOW

        elif now_monotonic >= self._time_target - Display.WAIT_MIN or self._stepping:
            snakes = self.snakes()
            if (snakes.frame() == self._frame_max or
                snakes.nr_games_total() >= self._game_max):
                self.set_timer_step(0, self.run2, now_monotonic)
                return now_monotonic

            # Always do the timestamp
            elapsed = self.timestamp(now_monotonic)
            if self._log_period_frames:
                elapsed = snakes.frame()
            if elapsed - self._elapsed_last >= self._log_period:
                self._elapsed_last = elapsed
                self.log_frame()
                self.log_tensor_board()
                snakes.score_max_local_clear()

            self.move()
            self.draw_result()
            self.update()
            now_monotonic = time.monotonic()
            self._time_target += self._poll_fast

        if self._stepping:
            self._stepping = False
            to_sleep = 0
        else:
            to_sleep = self._time_target - now_monotonic
            # print("To_Sleep", to_sleep)
            if to_sleep > 0:
                # Don't become unresponsive
                if to_sleep > Display.POLL_MAX:
                    to_sleep = Display.POLL_SLOW
                if to_sleep < Display.WAIT_MIN:
                    to_sleep = 0
            else:
                to_sleep = 0
        self.set_timer_step(to_sleep, self.step, now_monotonic)
        return now_monotonic


    def update(self):
        if self.snakes().debug:
            sys.stdout.flush()
        if self._streaming:
            self.stream_image()


    def events_process(self, now_monotonic):
        for key in self.events_key(now_monotonic):
            self.event_key(key, now_monotonic)
        return now_monotonic


    def events_key(self, now_monotonic):
        return []


    def event_key(self, key, now_monotonic):
        if key == " " or key == "r":
            self.event_toggle_run(now_monotonic)
        elif key == "s":
            self.event_single_step(now_monotonic)
        elif key == "+":
            self.event_speed_higher(now_monotonic)
        elif key == "-":
            self.event_speed_lower(now_monotonic)
        elif key == "=":
            self.event_speed_normal(now_monotonic)
        elif key == "d":
            self.event_debug(now_monotonic)
        elif key == "D":
            self.event_dump(now_monotonic)
        elif key == "Q":
            self.event_quit(now_monotonic)
        elif key == "c":
            self.event_capture(now_monotonic)
        elif key == "C":
            self.event_toggle_stream(now_monotonic)
        elif self.snakes().debug:
            print("Unknown key <%s>" % key)


    def event_toggle_run(self, now_monotonic = None):
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        snakes = self.snakes()
        # Stop/start running
        if self._pause_time:
            self._time_target = now_monotonic
            self._paused += now_monotonic - self._pause_time
            # print("Start running at", now_monotonic - self._time_monotonic_start, "frame", self.frame())
            self._pause_time = 0
        else:
            # print("Stop running at", time-self._time_monotonic_start, "frame", snakes.frame())
            self._pause_time = now_monotonic


    def event_single_step(self, now_monotonic = None):
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        # Single step
        self._stepping = True
        if not self._pause_time:
            self._pause_time = now_monotonic


    def event_speed_higher(self, now_monotonic = None):
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        self._time_target -= self._poll_fast
        self._poll_fast /= 2
        self._time_target = max(now_monotonic, self._time_target + self._poll_fast)

    def event_speed_lower(self, now_monotonic = None):
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        self._time_target -= self._poll_fast
        self._poll_fast *= 2
        self._time_target = max(now_monotonic, self._time_target + self._poll_fast)

    def event_speed_normal(self, now_monotonic = None):
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        self._time_target -= self._poll_fast
        self._poll_fast = self._poll_fast0
        self._time_target = max(now_monotonic, self._time_target + self._poll_fast)

    def event_debug(self, now_monotonic = None):
        snakes = self.snakes()
        if snakes.debug:
            sys.stdout.flush()
            snakes.debug = False
        else:
            snakes.debug = True


    def event_dump(self, now_monotonic = None):
        self.dump()
        if self.snakes().debug:
            print("Dumped to", self._dump_file)


    def event_capture(self, now_monotonic = None):
        if self.snakes().debug:
            print("event_capture not implemented for " + type(self).__name__)


    def event_toggle_stream(self, now_monotonic = None):
        if self._streaming:
            # flush() not needed since we open without buffering
            # self._stream_fh.flush()
            self._stream_fh.close()
            self._stream_fh = 0
            self._streaming = False
            print("Stop streaming")
        else:
            file = self._stream_file % self.stream_geometry()
            if self._stream_fh is None:
                self._stream_fh = open(file, "wb", buffering = 0)
                print("Start streaming to", file)
            else:
                self._stream_fh = open(file, "ab", buffering = 0)
                print("Resume streaming to", file)
            self._streaming = True
        # self._elapsed_sec_draw


    def event_quit(self, now_monotonic = None):
        self._quit = True
