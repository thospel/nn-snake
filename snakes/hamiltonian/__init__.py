from snakes import Snakes, np_empty, TYPE_BOOL, TYPE_POS, TYPE_UPOS
from display import Display
import numpy as np
import numpy.ma as ma
from dataclasses import dataclass

TYPE_FLOAT    = np.float32
RISK_MAX = TYPE_FLOAT(0.1)

@dataclass
class _Cycle():
    width:     int
    height:    int
    transpose: bool
    display:   any


class Cycle(_Cycle):
    def __init__(self, width, height, display = None):
        if height % 2 == 0:
            transpose = False
        elif width % 2 == 0:
            width, height = height, width
            transpose = True
        else:
            raise(AssertionError("We already checked that either width or height is even, and yet..."))
        super().__init__(width, height, transpose, display)

        # As usual we index with (y,x) so that a dump of the matrix is easy to
        # read from the screen
        self._horizontal = np.zeros((height, width-1), dtype=TYPE_BOOL)
        self._vertical   = np.zeros((height-1, width), dtype=TYPE_BOOL)
        for y in range(height-1):
            if y % 2 == 0:
                self.vertical(0, y)
                self.vertical(width-1, y)
        for y in range(height):
            for x in range(width-1):
                self.horizontal(x, y)

        self.pinch((self.width-1) * (self.height-1) * 4 * 4)
        self.cycle_find(display)
        self.cycle_join(display)
        self.cycle_follow(clockwise = np.random.randint(2))


    def horizontal(self, x, y, value = True):
        self._horizontal[y, x] = value
        return
        if self.display:
            color = Display.TRACE if value else Display.BACKGROUND
            snakes = self.display.snakes()
            x += 1/2
            y += 1/2
            if self.transpose:
                y += snakes.VIEW_X
                x += snakes.VIEW_Y
                self.display.draw_line(0, y, x, y, x+1, color)
            else:
                x += snakes.VIEW_X
                y += snakes.VIEW_Y
                self.display.draw_line(0, x, y, x+1, y, color)


    def vertical(self, x, y, value = True):
        self._vertical[y, x] = value
        return
        if self.display:
            color = Display.TRACE if value else Display.BACKGROUND
            snakes = self.display.snakes()
            x += 1/2
            y += 1/2
            if self.transpose:
                y += snakes.VIEW_X
                x += snakes.VIEW_Y
                self.display.draw_line(0, y, x, y+1, x, color)
            else:
                x += snakes.VIEW_X
                y += snakes.VIEW_Y
                self.display.draw_line(0, x, y, x, y+1, color)


    def pinch(self, n = 1):
        xn = np.random.randint(self.width-1,  size = n)
        yn = np.random.randint(self.height-1, size = n)
        iterator = np.nditer((xn, yn))
        for x, y in iterator:
            # print(x, y)
            x = x+0
            y = y+0
            if (self._vertical[y,x]
                and self._vertical[y,x+1]
                and not self._horizontal[y,x]
                and not self._horizontal[y+1,x]):
                self.vertical(x, y,   False)
                self.vertical(x+1, y, False)
                self.horizontal(x, y, True)
                self.horizontal(x, y+1, True)
                # print("Vertical to horizontal", iterator.iterindex)
            elif (not self._vertical[y,x]
                  and not self._vertical[y,x+1]
                  and self._horizontal[y,x]
                  and self._horizontal[y+1,x]):
                self.vertical(x, y,   True)
                self.vertical(x+1, y, True)
                self.horizontal(x, y, False)
                self.horizontal(x, y+1, False)
                # print("Horizontal to vertical", iterator.iterindex)
            # else: print("Failed", iterator.iterindex)


    # Fast union find root
    def root(self, pos):
        p = pos
        while True:
            p_next = self._parent[p]
            if p == p_next:
                break
            p = p_next
        r = p
        p = pos
        while p != r:
            p_next = self._parent[p]
            self._parent[p] = r
            p = p_next
        return r


    # Fast union find join
    def join(self, pos1, pos2):
        pos1 = self.root(pos1)
        pos2 = self.root(pos2)
        if pos1 == pos2:
            return
        self._sets -= 1
        if self._rank[pos1] < self._rank[pos2]:
            self._parent[pos1] = pos2
        elif self._rank[pos1] > self._rank[pos2]:
            self._parent[pos2] = pos1
        else:
            self._parent[pos2] = pos1
            self._rank[pos1] += 1


    # Give all cycles an id
    def cycle_find(self, display):
        # Uses fast union-find
        self._parent = np.arange(self.width * self.height, dtype = TYPE_UPOS)
        self._sets = self._parent.size
        # parent_yx = parent.reshape(self.height, self.width)
        # print(self._parent)
        self._rank = np.zeros(self.width * self.height, dtype = TYPE_UPOS)
        # self._horizontal = np.zeros((height, width-1), dtype=TYPE_BOOL)
        # self._vertical   = np.zeros((height-1, width), dtype=TYPE_BOOL)

        iterator = np.nditer(self._horizontal, flags=("multi_index",))
        for value in iterator:
            if value:
                y, x = iterator.multi_index
                pos = y * self.width + x
                self.join(pos, pos+1)

        iterator = np.nditer(self._vertical, flags=("multi_index",))
        for value in iterator:
            if value:
                y, x = iterator.multi_index
                pos = y * self.width + x
                self.join(pos, pos+self.width)

        # print(self._parent.reshape(self.height, self.width))


    # Join all cycles into one big hamiltonian cycle
    def cycle_join(self, display):
        x_range = np.arange(self.width-1, dtype = TYPE_UPOS)
        y_range = np.arange(0, (self.height-1) * self.width, self.width, dtype = TYPE_UPOS)
        indices = np.add.outer(y_range, x_range).reshape(-1)
        # print(indices)
        np.random.shuffle(indices)
        # display.wait_key()
        for p in indices:
            p1 = self.root(p)
            p2 = self.root(p + self.width + 1)
            if p1 == p2:
                # Loops are already joined
                continue
            (y, x) = divmod(p, self.width)
            # print("Consider", p, p1, p2, x, y)
            if (self._vertical[y,x]
                and self._vertical[y,x+1]
                and not self._horizontal[y,x]
                and not self._horizontal[y+1,x]):
                self.vertical(x, y,   False)
                self.vertical(x+1, y, False)
                self.horizontal(x, y, True)
                self.horizontal(x, y+1, True)
                # print("Vertical to horizontal", iterator.iterindex)
            elif (not self._vertical[y,x]
                  and not self._vertical[y,x+1]
                  and self._horizontal[y,x]
                  and self._horizontal[y+1,x]):
                self.vertical(x, y,   True)
                self.vertical(x+1, y, True)
                self.horizontal(x, y, False)
                self.horizontal(x, y+1, False)
            else:
                continue
            self.join(p1, p2)
            # print("Sets:", self._sets, x, y)
            # display.wait_key()
        # print("Done")
        assert self._sets == 1
        # print(self._parent.reshape(self.height, self.width))

        del self._rank
        del self._sets
        del self._parent


    def cycle_follow(self, clockwise = False):
        x0      = np_empty(self.width * self.height, dtype=TYPE_POS)
        y0      = np_empty(self.width * self.height, dtype=TYPE_POS)
        delta_x = np_empty(self.width * self.height, dtype=TYPE_POS)
        delta_y = np_empty(self.width * self.height, dtype=TYPE_POS)
        # Start at the top left and go right
        x = 0
        y = 0
        x0[0] = x
        y0[0] = y
        if clockwise:
            delta_x[0] = 1
            delta_y[0] = 0
        else:
            delta_x[0] = 0
            delta_y[0] = 1
        for i in range(1, x0.size):
            x += delta_x[i-1]
            y += delta_y[i-1]
            x0[i] = x
            y0[i] = y
            if x < self.width-1 and self._horizontal[y, x] and delta_x[i-1] != -1:
                delta_x[i] =  1
                delta_y[i] =  0
            elif x > 0 and self._horizontal[y, x-1] and delta_x[i-1] != 1:
                delta_x[i] = -1
                delta_y[i] =  0

            elif y < self.height-1 and self._vertical[y, x] and delta_y[i-1] != -1:
                delta_x[i] =  0
                delta_y[i] =  1
            elif y > 0 and self._vertical[y-1, x] and delta_y[i-1] != 1:
                delta_x[i] =  0
                delta_y[i] = -1
            else:
                raise(AssertionError("Cannot find cycle continuation"))

        # We must finish just below the top left going up so the cycle closes
        if clockwise:
            assert x0[-1] == 0
            assert y0[-1] == 1
            assert delta_x[-1] ==  0
            assert delta_y[-1] == -1
        else:
            assert x0[-1] == 1
            assert y0[-1] == 0
            assert delta_x[-1] == -1
            assert delta_y[-1] ==  0

        if self.transpose:
            self._x = y0
            self._y = x0
            self._delta_x = delta_y
            self._delta_y = delta_x
        else:
            self._x = x0
            self._y = y0
            self._delta_x = delta_x
            self._delta_y = delta_y


    # Give all cycles an id
    def draw(self, display, w=0):
        snakes = display.snakes()
        view_x = snakes.VIEW_X
        view_y = snakes.VIEW_Y
        rect = None

        iterator = np.nditer(self._horizontal, flags=("multi_index",))
        for value in iterator:
            if value:
                y, x = iterator.multi_index
                x += 1/2
                y += 1/2
                if self.transpose:
                    y += view_x
                    x += view_y
                    rect = display.draw_line(0, y, x, y, x+1, Display.TRACE,
                                             update = False, combine = rect)
                else:
                    x += view_x
                    y += view_y
                    rect = display.draw_line(0, x, y, x+1, y, Display.TRACE,
                                             update = False, combine = rect)

        iterator = np.nditer(self._vertical, flags=("multi_index",))
        for value in iterator:
            if value:
                y, x = iterator.multi_index
                x += 1/2
                y += 1/2
                if self.transpose:
                    y += view_x
                    x += view_y
                    rect = display.draw_line(0, y, x, y+1, x, Display.TRACE,
                                             update = False, combine = rect)
                else:
                    x += view_x
                    y += view_y
                    rect = display.draw_line(0, x, y, x, y+1, Display.TRACE,
                                             update = False, combine = rect)

        if rect:
            display.changed(0, rect)


    def fill_sequence(self, target):
        target[self._y, self._x] = np.arange(self._y.size, dtype = target.dtype)


    def fill_delta_x(self, target):
        target[self._y, self._x] = self._delta_x


    def fill_delta_y(self, target):
        target[self._y, self._x] = self._delta_y


class SnakesH(Snakes):
    def __init__(self, *args,
                 xy_head=False,
                 show_cycle = False,
                 **kwargs):
        self._show_cycle = show_cycle

        super().__init__(*args, xy_head=False, **kwargs)

        if self.WIDTH % 2 == 1 and self.HEIGHT % 2 == 1:
            raise(ValueError("No hamiltonian cycles exist on an odd x odd sized pit"))
        if self.WIDTH < 2:
            raise(ValueError("No hamiltonian cycles exist on width=1 pit"))
        if self.HEIGHT < 2:
            raise(ValueError("No hamiltonian cycles exist on height=1 pit"))

        # Subtract 1 for the head and 1 for the apple we plan to eat
        self.AREA_LEFT  = self.AREA-2
        self.risk_table()


    def risk_table(self):
        n = self.AREA_LEFT+1
        safe_first = np_empty(n, dtype=TYPE_POS)
        safe_last  = np_empty(n, dtype=TYPE_POS)
        # 0 cannot happen
        safe_first[0] = 1
        safe_last [0] = 0

        for left in range(1, n):
            # For given number of empty position calculate risk for each gap

            # Fraction of gap versus total area left gives the chance of an
            # apple being created between the planned route from head to tail
            # If this happen safe_apples times we are in trouble.
            # And safe_apples = gap -1 (because we move head before tail)

            # Only need to calculate half, the distribution is symmetric
            # The sequence is unimodal with risk(left, 1) = risk(left, left) = 1
            # and the minimum in the middle
            # For left odd there is 1 minimal value and 2 for left even
            steps = 1+left/2
            top    = np.arange(1, steps)
            bottom = np.arange(left+1,steps,-1)
            m = top/bottom
            # This will set the scale: risk(left, 1) == 1
            # And this gap==1 value will be at index 0
            m[0] *= left+1
            risks = np.multiply.accumulate(m)
            # print(risks)
            # We will take this risk left-1 times
            risks = 1 - (1 - risks) ** (left-1)
            # print(risks)
            risks = risks <= RISK_MAX
            # print(risks)
            if risks[-1]:
                safe = np.argmax(risks)
                # +1 because the first index is gap==1
                safe_first[left] = safe + 1
                safe_last[left]  = left - safe
                # print(left, safe_first[left], safe_last[left])
            else:
                safe_first[left] = 1
                safe_last[left]  = 0
        # Shift by 1 so we won't have to subtract 1 from gap when doing the
        # compares (gap -1 because we plan to eat the current apple which does
        # NOT get randomply placed, it's already here)
        # print(np.stack((safe_first, safe_last)))
        self._safe_first1 = safe_first + 1
        self._safe_last1  = safe_last  + 1


    def log_constants(self, fh):
        super().log_constants(fh)

        print("Sequence: <<EOT", file=fh)
        with np.printoptions(threshold=self._sequence.size, linewidth = 10 * self._sequence0[0].size):
            print(self._sequence0, file=fh)
        print("EOT", file=fh)

        print("RiskMax:%10.3f" %  RISK_MAX, file=fh)
        print("Safe: <<EOT", file=fh)
        safe = np.stack((self._safe_first1, self._safe_last1), axis=1)
        with np.printoptions(threshold=safe.size):
            print(safe, file=fh)
        print("EOT", file=fh)


    def run_start(self, display):
        # cycle = Cycle(self.WIDTH, self.HEIGHT, display)
        cycle = Cycle(self.WIDTH, self.HEIGHT)
        if self._show_cycle:
            display.draw_pit_empty(w=0)
            cycle.draw(display, w=0)
            display.image_save(w=0)

        self._sequence1 = np.zeros((self.HEIGHT1,self.WIDTH1), dtype=TYPE_POS)

        self._sequence = self._sequence1.reshape(-1)
        assert self._sequence.base is self._sequence1

        self._sequence0 = self._sequence1[self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH]
        assert self._sequence0.base is self._sequence1
        cycle.fill_sequence(self._sequence0)
        # print(self._sequence0)

        # display.wait_key()
        super().run_start(display)


    def move_select(self, move_result, display):
        debug = self.debug

        head = self.head()
        # self.print_pos("Head", head)

        tail_seq = self._sequence[self.tail()]
        neighbours_pos = np.add.outer(self.DIRECTIONS, head)
        walls = self._field[self._all_snakes, neighbours_pos]
        neighbours_seq = self._sequence[neighbours_pos]
        neighbours_pos = None
        # Suppose e.g. the tail is at seq 4. We are using this for planning.
        # Assume we will move the head to seq 0. This will move the tail to
        # seq 5, so the gap is 4
        gap = (tail_seq - neighbours_seq) % self.AREA

        # Starting from plan we must hit the tail before the head in sequence
        # If the head comes before the tail we may crash
        head_seq = self._sequence[self.head()]
        if debug:
            print("Tail seq", tail_seq[self._debug_index])
            print("Neighbours seq", neighbours_seq[:, self._debug_index])
            print("Head seq", head_seq[self._debug_index])
        head_steps = (head_seq - neighbours_seq) % self.AREA
        # head_seq = None
        head_before_tail = head_steps < gap
        head_steps = None

        # We plan to eat the current apple, after which the gap will be 3
        # So we can safely eat 2 more apples
        # (not 3 since we move the head before we move the tail out of the way)
        # So safe_apples = current gap-2
        # safe_first1 and safe_last1 are shifted by 1 so we don't need gap - 1
        left = self.AREA_LEFT - self.scores()
        unacceptable = (gap < self._safe_first1[left]) | (self._safe_last1[left] < gap)
        if debug:
            print("Left", left[self._debug_index])
            print("Gap: %d <= %s <= %d" % (
                self._safe_first1[left[self._debug_index]],
                str(gap[:, self._debug_index]),
                self._safe_last1[left[self._debug_index]]))
            print("Unacceptable risk:", unacceptable[:, self._debug_index])
        gap = None

        unacceptable |= head_before_tail
        if debug:
            print("Unaccptable head before tail:", unacceptable[:, self._debug_index])
        head_steps = None

        # Just following the sequence is always acceptable
        # Since the code above calculated that following the whole sequence is
        # guaranteed to run into all apples the risk is actually very high
        # (the calculation does not know about this concept called "winning")
        # and so this move had been pruned as being too risky.
        # Do this (almost) last so to guarantee there is at least one valid move
        # But do the wall test AFTER this since the edge is effectively random
        next_seq = (self._sequence[head] + 1) % self.AREA
        unacceptable &= neighbours_seq != next_seq
        if debug:
            print("Unacceptable just follow:", unacceptable[:, self._debug_index])

        # Don't run into walls!
        unacceptable |= walls
        if debug:
            print("Unacceptable crash:", unacceptable[:, self._debug_index])

        apple_seq = self._sequence[self.apple()]
        # print("Apple seq", apple_seq)
        apple_steps = (apple_seq - neighbours_seq) % self.AREA
        neighbours_seq = None
        if False:
            head_seq = None
            apple_seq = None
            tmp = ma.array(apple_steps, mask = unacceptable)
            assert tmp.data.base is apple_steps
            apple_steps = tmp
            if debug:
                print("Masked apple steps", apple_steps[self._debug_index])
            fastest = apple_steps.argmin(axis=0)
            apple_steps = None
        else:
            # Make sure the snakes moves towards the apple on the sequence
            # This is needed to avoid getting stuck in loops
            apple_steps0 =  (apple_seq - head_seq) % self.AREA
            unacceptable |= apple_steps > apple_steps0
            if debug:
                print("Apple steps", apple_steps[:, self._debug_index], "Head", apple_steps0[self._debug_index])
                print("Unacceptable toward apple:", unacceptable[:, self._debug_index])
            apple_steps0 = None
            apple_steps = None
            head_seq = None
            apple_seq = None
            # Determine the direction of the apple
            distance = self.apple_distance()[0]
            tmp = ma.array(distance, mask = unacceptable)
            assert tmp.data.base is distance
            distance = tmp
            if debug:
                print("Distance:", distance[:, self._debug_index])
            fastest = distance.argmax(axis=0)
            distance = None

        if debug:
            print("Fastest:", fastest[self._debug_index])
        plan = head + self.DIRECTIONS[fastest]
        return plan
