import time
import math

class Display:
    EDGE = 1

    BACKGROUND = 0
    WALL       = 1
    APPLE      = 2
    HEAD       = 3
    BODY       = 4
    COLLISION  = 5

    # Event polling time in paused mode.
    # Avoid too much CPU waste or even a busy loop in case fps == 0
    POLL_SLOW = 1/25
    POLL_MAX = POLL_SLOW * 1.5
    WAIT_MIN = 1/1000


    def __init__(self, snakes,
                 rows       = 1,
                 columns    = 2,
                 block_size = 20,
                 log_file   = "snakes.log.txt",
                 dump_file  = "snakes.dump.txt",
                 stepping   = False,
                 caption    = "Snakes"):
        self.rows    = rows
        self.columns = columns
        self.windows = rows*columns
        self._stepping = stepping

        self._log_file  = log_file
        self._log_fh = None
        self._dump_file = dump_file

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


    def log_close(self):
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None


    def log_start(self):
        if not self._log_fh:
            return

        fh = self._log_fh
        print(time.strftime("time_start: %Y-%m-%d %H:%M:%S %z",
                            time.localtime(self._time_start)),
              file=fh)
        print("time_start_epoch: %.3f" % self._time_start, file=fh)
        print("time_start_monotonic:", self._time_monotonic_start,
              file=fh)
        print("time_start_process:",  self._time_process_start,
              file=fh)

        snakes = self.snakes()
        snakes.log_constants(fh)

        print("Frame:", snakes.frame(), file=fh)


    def log_stop(self):
        if not self._log_fh:
            return

        self.log_frame()
        fh = self._log_fh
        print(time.strftime("time_end: %Y-%m-%d %H:%M:%S %z",
                            time.localtime(self._time_end)),
              file=fh)
        print("time_end_epoch: %.3f" % self._time_end, file=fh)
        print("time_end_monotonic:", self._time_monotonic_end, file=fh)
        print("time_end_process:",  self._time_process_end,  file=fh)


    def log_frame(self):
        if not self._log_fh:
            return

        fh = self._log_fh
        print("#----", file=fh)
        self._log_frame(fh)


    def dump(self):
        with open(self._dump_file, "w") as fh:
            self.timestamp()
            self.dump_fh(fh)


    def dump_fh(self, fh):
        snakes = self.snakes()
        snakes.log_constants(fh)
        self._log_frame(fh)
        snakes.dump_fh(fh)


    def _log_frame(self, fh):
        snakes = self.snakes()

        print("Frame:%12d"         % snakes.frame(), file=fh)
        print("Elapsed:%14.3f"     % self.elapsed(), file=fh)
        print("Paused:%15.3f"      % self.paused(),  file=fh)
        print("Used:%17.3f"        % self.elapsed_process(), file=fh)
        print("Frames skipped:%3d" % self.frames_skipped(), file=fh)
        print("Frame rate:%11.3f"  % self.frame_rate(snakes), file=fh)
        print("Games Total:%6d"    % snakes.nr_games_total(), file=fh)
        print("Games Won:%8d"      % snakes.nr_games_won_total(), file=fh)
        print("Score Max:%8d"      % snakes.score_max(), file=fh)
        print("Score Total:%6d"    % snakes.score_total_games(), file=fh)
        print("Moves Max:%8d"      % snakes.nr_moves_max(), file=fh)
        print("Moves total:%6d"    % snakes.nr_moves_total_games(), file=fh)
        print("Moves/Game:%11.3f"  % snakes.nr_moves_per_game(), file=fh)
        print("Moves/Apple:%10.3f" % snakes.nr_moves_per_apple(), file=fh)


    def start(self):
        pass


    def stop(self):
        del self._snakes
        del self._moves


    def draw_block(self, w, x, y, color, update=True):
        pass


    def draw_text(self, w, name, value = None):
        pass


    def draw_apples(self, i_index, w_index, apple, score):
        apple_y, apple_x = apple
        # print("draw_apples", i_index, w_index, score)
        # print(np.stack((apple_x, apple_y), axis=-1))
        for i, w, x, y in zip(i_index, w_index, apple_x, apple_y):
            self.draw_block(w, x, y, Display.APPLE)
            self.draw_text(w, "score", score[i])


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
                self.draw_block(w, w_head_x_old[w], w_head_y_old[w], Display.BODY, update=False)
            if not is_eat[i]:
                # Drop the tail if we didn't eat an apple
                self.draw_block(w, w_tail_x[w], w_tail_y[w], Display.BACKGROUND)
            self.draw_block(w, w_head_x_new[w], w_head_y_new[w], Display.HEAD)
            self.draw_text(w, "moves", w_nr_moves[w])


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
                self.draw_text(w, "game", nr_games[i])
                self.draw_text(w, "win",  nr_games_won[i])
                # self.draw_text(w, "snake", i)
                # self.draw_text(w, "x")
                # self.draw_text(w, "y")
                self.draw_pit_empty(w)


    def draw_windows(self, window_iterator, w_head):
        snakes = self.snakes()
        w_head_y, w_head_x = w_head

        for i in window_iterator:
            w = window_iterator.iterindex

            self.draw_pit_empty(w)
            self.draw_block(w, w_head_x[w], w_head_y[w], Display.HEAD)

            self.draw_text(w, "moves", snakes.nr_moves(i))
            self.draw_text(w, "game",  snakes.nr_games(i))
            self.draw_text(w, "win",   snakes.nr_games_won(i))
            self.draw_text(w, "snake", i)


    def draw_text_summary(self, *args):
        pass

    # Dislay the current state
    def draw_result(self, initial = False):
        snakes = self.snakes()

        # draw_text is optimized to not draw any value that didn't change
        self.draw_text_summary("step", snakes.frame())
        # self.draw_text_summary("score_per_snake", snakes.score_total_snakes() / snakes._nr_snakes)
        self.draw_text_summary("score_max", snakes.score_max())
        self.draw_text_summary("moves_max", snakes.nr_moves_max())
        self.draw_text_summary("game_max",  snakes.nr_games_max())
        self.draw_text_summary("moves_max", snakes.nr_moves_max())
        self.draw_text_summary("games",     snakes.nr_games_total())
        self.draw_text_summary("wins",      snakes.nr_games_won_total())
        if snakes.nr_games_total():
            self.draw_text_summary("score_per_game",  snakes.score_per_game())
            self.draw_text_summary("moves_per_game",  snakes.nr_moves_per_game())
        if snakes.score_total_games():
            self.draw_text_summary("moves_per_apple", snakes.nr_moves_per_apple())

        if initial:
            # Fake it (the real timers haven't started yet)
            self._elapsed_sec_draw = 0
            elapsed_sec = 0
        else:
            elapsed_sec = int(self.elapsed(time.monotonic()))
        if elapsed_sec != self._elapsed_sec_draw:
            self._elapsed_sec_draw = elapsed_sec
            self.draw_text_summary("time", elapsed_sec)


    def timers_start(self, fps):
        # print("Start at time 0, frame", snakes.frame())

        if fps > 0:
            self._poll_fast = 1 / fps
        elif fps == 0:
            self._poll_fast = 0
        else:
            raise(ValueError("fps must not be negative"))
        self._poll_fast0 = self._poll_fast
        self._paused = 0
        self._elapsed_sec_log  = 0

        self._time_start = time.time()
        self._time_process_start = time.process_time()
        self._time_monotonic_start  = time.monotonic()
        self._time_end           = self._time_start
        self._time_process_end   = self._time_process_start
        self._time_monotonic_end = self._time_monotonic_start
        self._time_target        = self._time_monotonic_start
        self._frames_skipped = 0
        if self._stepping:
            self._pause_time = self._time_monotonic_start
            self._pause_frame = self.snakes().frame()
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
            self._frames_skipped += frame - self._pause_frame
            self._pause_frame = frame
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


    # How many frames we manually single-stepped
    def frames_skipped(self):
        return self._frames_skipped


    def frame_rate(self, snakes):
        elapsed = self.elapsed() - self.paused()
        frames  = snakes.frame() - self.frames_skipped()
        if elapsed == 0:
            # This properly handles positive, negative and 0 (+inf, -inf, nan)
            return math.inf * int(frames)
        return frames / elapsed


    def run(self, snakes, fps = 40, stepping = False, frame_max = -1):
        self._stepping = stepping
        self._frame_max = frame_max
        self._snakes = snakes

        self.start()
        self.log_open()
        snakes.run_start(self)
        snakes.run_start_extra()
        self._moves = snakes.move_generator(self)
        # This is not so much a move as initializing everything
        self.move()
        # _unloop must be set here since draw_result can immediately set it True
        self._unloop = False
        self.draw_result(initial=True)
        self.update()
        self.timers_start(fps)
        self.log_start()
        self.set_timer_step(0)
        self.loop()
        self.timestamp()
        snakes.run_finish()
        self.log_stop()
        self.log_close()
        self.stop()


    def loop(self):
        now_monotonic = time.monotonic()
        while True:
            if self._to_sleep > 0:
                time.sleep(self._to_sleep)
                now_monotonic = time.monotonic()
            if self._unloop:
                break
            now_monotonic = self.events_process(now_monotonic)
            now_monotonic = self.step(now_monotonic)


    def set_timer_step(self, to_sleep, now_monotonic=None):
        self._to_sleep = to_sleep


    def step(self, now_monotonic = None):
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        if self._pause_time and not self._stepping:
            self._time_target = now_monotonic + Display.POLL_SLOW

        elif now_monotonic >= self._time_target - Display.WAIT_MIN or self._stepping:
            snakes = self.snakes()
            if snakes.frame() == self._frame_max:
                self._unloop = True
                self.set_timer_step(0, now_monotonic)
                return now_monotonic

            elapsed_sec = int(self.timestamp(now_monotonic))
            if elapsed_sec != self._elapsed_sec_log:
                self._elapsed_sec_log = elapsed_sec
                self.log_frame()

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
        self.set_timer_step(to_sleep, now_monotonic)
        return now_monotonic


    def update(self):
        pass


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
            self._frames_skipped += snakes.frame() - self._pause_frame
            # print("Start running at", now_monotonic - self._time_monotonic_start, "frame", self.frame())
            self._pause_time = 0
        else:
            # print("Stop running at", time-self._time_monotonic_start, "frame", snakes.frame())
            self._pause_time = now_monotonic
            self._pause_frame = snakes.frame()


    def event_single_step(self, now_monotonic = None):
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        # Single step
        self._stepping = True
        if not self._pause_time:
            self._pause_time = now_monotonic
            self._pause_frame = self.snakes().frame()


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
        snakes.debug = not snakes.debug


    def event_dump(self, now_monotonic = None):
        self.dump()
        if self.snakes().debug:
            print("Dumped to", self._dump_file, flush=True)


    def event_quit(self, now_monotonic = None):
        self._unloop = True
