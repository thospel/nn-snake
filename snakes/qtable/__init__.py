from snakes import Snakes, Vision, np_empty, TYPE_MOVES, TYPE_POS, TYPE_UPOS, TYPE_ID
import numpy as np
import re
from dataclasses import dataclass


TYPE_FLOAT    = np.float32

@dataclass
class Rewards():
    apple: TYPE_FLOAT
    crash: TYPE_FLOAT
    move:  TYPE_FLOAT
    rand:  TYPE_FLOAT
    initial: TYPE_FLOAT


    @classmethod
    def parse_file(cls, file):
        with open(file, "r") as fh:
            return cls.parse(fh.read())


    @classmethod
    def parse(cls, str):
        rewards = {
            "apple":   None,
            "crash":   None,
            "move":    None,
            "rand":    None,
            "initial": None
        }

        for line in str.splitlines():
            line = re.sub(r"#.*", "", line).strip()
            if line == "":
                continue
            match = re.fullmatch(r"(\w+)\s*:\s*([+-]?[0-9]+(?:\.[0-9]*)?)", line)
            if not match:
                raise(ValueError("Could not parse: " + line))
            key, value = match.groups()
            if key not in rewards:
                raise(ValueError("Unknown key: " + key))
            if rewards[key] is not None:
                raise(ValueError("Multiple key: " + key))
            rewards[key] = TYPE_FLOAT(value)
        missing = [key for key in rewards if rewards[key] is None]
        if missing:
            raise(ValueError("Missing key: " + ", ".join(missing)))
        return cls(**rewards)


    @classmethod
    def default(cls):
        return cls.parse("""
		apple:   1
		crash: -10
		# A small penalty for taking too long to get to an apple
		move:   -0.001
		# Small random disturbance to escape from loops
		rand:    0.001
		# Prefill with 0 is enough to encourage some early exploration
		# since we have a negative reward for moving
		initial: 0
        """)


class SnakesQ(Snakes):
    TYPE_QSTATE   = np.uint32
    EPSILON_INV   = 10000
    # LOOP is in units of AREA
    LOOP_MAX      = 2
    LOOP_ESCAPE   = 1
    NR_STATES_MAX = np.iinfo(TYPE_QSTATE).max

    def __init__(self, *args,
                 view_x = None, view_y = None,
                 xy_head = False,
                 vision = None,
                 reward_file = None,
                 wall_left = 0, wall_right = 0, wall_up = 0, wall_down = 0,
                 single        = False,
                 learning_rate = 0.1,
                 discount      = 0.9,
                 accelerated   = False,
                 symmetry      = False,
                 **kwargs):

        if reward_file is None:
            self._rewards = Rewards.default()
        else:
            self._rewards = Rewards.parse_file(reward_file)
        self._accelerated = accelerated
        self._symmetry    = symmetry
        if vision is None:
            if symmetry:
                vision = Vision(Vision.VISION_DH3)
            else:
                vision = Vision(Vision.VISION4)
        print("Snake Vision:")
        print(vision.string(final_newline = False))
        self._vision_obj = vision

        if view_x is None:
            view_x = max(Snakes.VIEW_X0, -np.amin(vision.x), np.amax(vision.x))
        if view_y is None:
            view_y = max(Snakes.VIEW_Y0, -np.amin(vision.y), np.amax(vision.y))

        nr_neighbours = len(self._vision_obj)
        if nr_neighbours <= 0:
            raise(ValueError("Your snake is blind"))

        super().__init__(*args,
                         xy_head = xy_head,
                         view_x = view_x,
                         view_y = view_y,
                         **kwargs)

        self._single = single
        # self._rewards = np.empty(self._nr_snakes, dtype=TYPE_FLOAT)
        self._learning_rate = TYPE_FLOAT(learning_rate)
        if not self._single:
            self._learning_rate /= self.nr_snakes
        self._discount = TYPE_FLOAT(discount)
        self._eat_frame = np_empty(self.nr_snakes, TYPE_MOVES)
        self.init_vision()
        self.init_wall(wall_left, wall_right, wall_up, wall_down)
        self.init_q_table()


    def init_vision(self):
        vision = self._vision_obj

        if (vision.min_x < -self.VIEW_X or
            vision.max_x > +self.VIEW_X):
            raise(ValueError("X View not wide enough for vision"))
        if (vision.min_y < -self.VIEW_Y or
            vision.max_y > +self.VIEW_Y):
            raise(ValueError("Y View not high enough for vision"))
        if self._symmetry:
            if (0,-1) in vision:
                print(vision.string(final_newline = False))
                raise(ValueError("Vision has backwards move which is pointless for this symmetry"))
            self.NR_STATES_APPLE = 5
            self._vision_pos = np.tensordot(Snakes.ROTATIONS_DH @ np.stack((vision.x, vision.y),axis=0), np.array([1,self.WIDTH1], dtype=TYPE_POS), axes=(1,0))
            self._old_direction = np_empty(self.nr_snakes, TYPE_ID)
            # print(self._vision_pos)
        else:
            self.NR_STATES_APPLE = 8
            self._vision_pos = self.pos_from_xy(vision.x, vision.y)


    def init_wall(self, wall_left, wall_right, wall_up, wall_down):
        # Make sure the user entered sane values
        if wall_left  < 0: raise(ValueError("wall_left must not be negative"))
        if wall_right < 0: raise(ValueError("wall_right must not be negative"))
        if wall_up    < 0: raise(ValueError("wall_up must not be negative"))
        if wall_down  < 0: raise(ValueError("wall_down must not be negative"))

        # We could cleverly check if the pit is so narrow certain wall
        # combinations are impossible. But the code below checks that only
        # actually observed wall combinations lead to a wall state, so this is
        # not needed

        self._wall_left  = wall_left
        self._wall_right = wall_right
        self._wall_up    = wall_up
        self._wall_down  = wall_down

        # We do too much work for the non-symmetry case here
        # But it's only a startup cost and avoids code duplication

        # Distances to the right, up, left, down walls
        wall_distances = np.stack((
            np.tile(np.arange(self.WIDTH-1, -1, -1, dtype=TYPE_UPOS), (self.HEIGHT, 1)),
            np.repeat(np.arange(self.HEIGHT, dtype=TYPE_UPOS), self.WIDTH).reshape(-1, self.WIDTH),
            np.tile(np.arange(self.WIDTH, dtype=TYPE_UPOS), (self.HEIGHT, 1)),
            np.repeat(np.arange(self.HEIGHT-1, -1, -1, dtype=TYPE_UPOS), self.WIDTH).reshape(-1, self.WIDTH)
        ), axis=0)
        if self._symmetry:
            # Rotate to each of the 8 dihedral symmetries
            # Has shape (8 symmetries, 4 directions, HEIGHT, WIDTH)
            wall_distances = wall_distances[Snakes.DIRECTION4_ID_DH]
        else:
            wall_distances = np.expand_dims(wall_distances, axis=0)
        # Maximum wall distances. We will apply these AFTER rotation
        # So they are in the input vision orientation
        wall_max = np.array([wall_right, wall_up, wall_left, wall_down], dtype=TYPE_UPOS)
        # print("wall max:", wall_max)
        # Restrict to maximum
        wall_distances = np.minimum(wall_distances, wall_max.reshape(-1,1,1))
        # print(wall_distances)
        # Number of possible different values in each wall direction
        wall_values = np.amax(wall_distances, axis=(0,2,3)) + 1
        wall_values = wall_values.astype(np.uint64)
        # print("wall values:", wall_values)
        # We are going to consider the wall planes as mixed base digits
        # Here we construct the multiplier for each digit
        wall_base = np.roll(np.multiply.accumulate(wall_values), 1)
        # We rolled the highest possible value +1 to position 0
        nr_wall_states = wall_base[0]
        # print("Wall states upper bound:", nr_wall_states)
        # And the least significant digit has weight 1
        wall_base[0] = 1
        assert wall_base.dtype == np.uint64
        # Construct unique wall states
        # But the values sequence can have holes (impossible combinations)
        # tensordot makes the type float, but we want to stay int
        # wall_state = np.tensordot(wall_distances, wall_base, (1,0))
        wall_state = np.einsum("ijkl,j -> ikl", wall_distances, wall_base, dtype=np.uint64)
        assert wall_state.dtype == np.uint64
        # print(wall_state)
        # Mark which states actually occur
        wall_state_seen = np.zeros(nr_wall_states, dtype=SnakesQ.TYPE_QSTATE)
        wall_state_seen[wall_state] = 1
        # These states REALLY occur
        wall_state_used = wall_state_seen.nonzero()[0]
        self.NR_STATES_WALL = wall_state_used.size
        # print(wall_state_used)
        wall_state_seen[wall_state_used] = np.arange(
            0,
            wall_state_used.size * self.NR_STATES_APPLE,
            self.NR_STATES_APPLE)
        # print(wall_state_seen)
        # Convert wall state to a sequence without holes starting at 0
        wall_state = wall_state_seen[wall_state]
        # print(wall_state)
        assert wall_state.dtype == SnakesQ.TYPE_QSTATE

        # Convert used states back to mixed base
        wall_state_used, wall_state_used0 = np.divmod(wall_state_used, wall_values[0])
        wall_state_used, wall_state_used1 = np.divmod(wall_state_used, wall_values[1])
        wall_state_used, wall_state_used2 = np.divmod(wall_state_used, wall_values[2])
        assert np.all(wall_state_used < wall_values[3])
        wall_state_used = np.stack((wall_state_used0,
                                    wall_state_used1,
                                    wall_state_used2,
                                    wall_state_used),axis=-1)
        self._wall_state = wall_state_used
        # print(self._wall_state)
        wall_state_greater = np.core.defchararray.add("> ", wall_state_used.astype(str))
        wall_state_used = (wall_state_used + 1) % (wall_max+1)
        wall_state_used = np.where(
            wall_state_used == 0,
            wall_state_greater,
            np.core.defchararray.add("= ", wall_state_used.astype(str)))
        wall_state_used = tuple(map(tuple, wall_state_used))
        self._wall_state_tuple = wall_state_used
        # print(wall_state_used)

        if self.NR_STATES_WALL > 1:
            # Just to make self._wall1 print nicer set the outer edge to nr states
            if self._symmetry:
                self._wall1 = np.full(
                    (wall_state.shape[0], self.HEIGHT1, self.WIDTH1),
                    self.NR_STATES_WALL * self.NR_STATES_APPLE,
                    dtype=SnakesQ.TYPE_QSTATE)
                self._wall = self._wall1.reshape(wall_state.shape[0], -1)
                # self._wall0 = self._wall1[:, self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH]
                self._wall1[:, self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH] = wall_state
            else:
                self._wall1 = np.full(
                    (self.HEIGHT1, self.WIDTH1),
                    self.NR_STATES_WALL * self.NR_STATES_APPLE,
                    dtype=SnakesQ.TYPE_QSTATE)
                self._wall = self._wall1.reshape(-1)
                # self._wall0 = self._wall1[self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH]
                self._wall1[self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH] = wall_state
            assert self._wall.base  is self._wall1
            # assert self._wall0.base is self._wall1
            # print(self._wall1)


    def init_q_table(self):
        nr_neighbours = len(self._vision_obj)
        self.NR_STATES_NEIGHBOUR = 2 ** nr_neighbours

        self.NR_STATES = self.NR_STATES_APPLE * self.NR_STATES_NEIGHBOUR * self.NR_STATES_WALL
        if self.NR_STATES > SnakesQ.NR_STATES_MAX:
            raise(ValueError("Number of states too high for Q table index type"))
        # The previous test made sure this won't overflow
        self.neighbour_multiplier = np.expand_dims(1 << np.arange(0, nr_neighbours, 8, dtype=SnakesQ.TYPE_QSTATE), axis=1) * (self.NR_STATES_APPLE * self.NR_STATES_WALL)
        assert self.neighbour_multiplier.dtype == SnakesQ.TYPE_QSTATE

        if self._symmetry:
            # We never go backward
            nr_actions = Snakes.NR_DIRECTIONS - 1
        else:
            nr_actions = Snakes.NR_DIRECTIONS

        self._q_table = np.empty((self.NR_STATES, nr_actions),
                                 dtype=TYPE_FLOAT)
        print("Q table has %u states = %u wall * %u apple * %u neighbour" %
              (self.NR_STATES,
               self.NR_STATES_WALL,
               self.NR_STATES_APPLE,
               self.NR_STATES_NEIGHBOUR))


    def run_start(self, display):
        super().run_start(display)

        self._q_table.fill(self._rewards.initial)
        self._old_state  = [None] * self.HISTORY
        self._old_action = [None] * self.HISTORY

        # Prefill obvious collisions
        if self._accelerated:
            n = np.arange(self.NR_STATES_NEIGHBOUR, dtype=SnakesQ.TYPE_QSTATE)
            # Have a per neighbour view for fast neighbour selection
            q_table = self._q_table.reshape((self.NR_STATES_NEIGHBOUR, self.NR_STATES_APPLE * self.NR_STATES_WALL, -1))
            assert q_table.base is self._q_table

            if self._symmetry:
                # order: turn right, forward, turn left
                # These are relative to forward (x, y) = (1, 0)
                directions_x = np.array([0, 1, 0], dtype=TYPE_POS)
                directions_y = np.array([1, 0,-1], dtype=TYPE_POS)
            else:
                directions_x = Snakes.DIRECTIONS4_X
                directions_y = Snakes.DIRECTIONS4_Y
            iterator = np.nditer((directions_x, directions_y))
            for x, y in iterator:
                pos = y+0, x+0
                if pos in self._vision_obj:
                    i = self._vision_obj[y+0, x+0]
                    bit = SnakesQ.TYPE_QSTATE(1 << i)
                    hit = (n & bit) == bit
                    q_table[hit,:, iterator.iterindex] = self._rewards.crash

        self._eat_frame.fill(self.frame())


    def print_qtable(self, fh):
        nr_neighbours = len(self._vision_obj)
        bits = [0] * nr_neighbours
        state_max_len = len(str(self.NR_STATES_NEIGHBOUR-1))
        states_per_neighbour = self.NR_STATES_WALL * self.NR_STATES_APPLE
        for neighbour_state in range(self.NR_STATES_NEIGHBOUR):
            min_i = neighbour_state       * states_per_neighbour
            max_i = (neighbour_state + 1) * states_per_neighbour
            if self._q_table[min_i:max_i].any():
                print("NState: %*d" % (state_max_len, neighbour_state), file=fh)
                print(self._vision_obj.string(final_newline = False,
                                              values = bits),
                      file = fh)
                for i in range(min_i, max_i, self.NR_STATES_APPLE):
                    q_range = self._q_table[i:i+self.NR_STATES_APPLE]
                    if q_range.any():
                        print(q_range, file=fh)
                    else:
                        print("[ ZERO ]", file=fh)

            # bit list + 1
            for i in range(nr_neighbours):
                bits[i] = 1 - bits[i]
                if bits[i]:
                    break


    def log_constants(self, log_action):
        super().log_constants(log_action)

        lr = self._learning_rate
        if not self._single:
            lr *= self.nr_snakes

        log_action("Wall left", "%4d",  self._wall_left)
        log_action("Wall right", "%3d", self._wall_right)
        log_action("Wall up", "%6d",    self._wall_up)
        log_action("Wall down", "%4d",  self._wall_down)

        log_action("Vision", "%s", self._vision_obj.string(final_newline = False))

        log_action("NrStates",      " %8d",   self.NR_STATES)
        log_action("Learning Rate", "%8.3f",  lr)
        log_action("Discount",      "%13.3f", self._discount)
        log_action("Epsilon",       "%14.3f", 1/SnakesQ.EPSILON_INV)
        log_action("Reward apple",  "%9.3f",  self._rewards.apple)
        log_action("Reward crash",  "%9.3f",  self._rewards.crash)
        log_action("Reward move",   "%10.3f", self._rewards.move)
        log_action("Reward rand",   "%10.3f", self._rewards.rand)
        log_action("Reward init",   "%10.3f", self._rewards.initial)


    def dump_fh(self, fh):
        super().dump_fh(fh)

        print("Q table: <<EOT", file=fh)
        with np.printoptions(threshold=self._q_table.size):
            self.print_qtable(fh)
        print("EOT", file=fh)


    def state(self, neighbour, apple, wall=0):
        return apple + self.NR_STATES_APPLE * (wall + self.NR_STATES_WALL * neighbour)


    def unstate(self, state):
        state_rest, apple = divmod(state,      self.NR_STATES_APPLE)
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


    def move_select(self, move_result, display):
        # debug = self.frame() % 100 == 0
        debug = self.debug

        # print(self._q_table)

        h0 = self.frame_history_then()
        head = self.head()
        # self.print_pos("Head", head)

        # Determine the direction of the apple
        if self._xy_head:
            x = self.head_x()
            y = self.head_y()
        else:
            y, x = self.yx(head)

        if self._xy_apple:
            apple_x = self.apple_x()
            apple_y = self.apple_y()
        else:
            apple_y, apple_x = self.yx(self.apple())
        # print_xy("apple", apple_x, apple_y)

        dx = np.sign(apple_x - x)
        dy = np.sign(apple_y - y)
        # print_xy("Delta signs:", dx, dy)

        nr_collided = move_result.collided.size
        if self._symmetry:
            if nr_collided:
                # New games start with a random direction
                # Some of these will seem to come out of a wall. We don't care
                self._old_direction[move_result.collided] = np.random.randint(Snakes.NR_DIRECTIONS, size=nr_collided, dtype=TYPE_ID)
            symmetry_state = Snakes.SYMMETRY8_DH[self._old_direction, dy, dx]

            apple_state = symmetry_state[:, 1]
            assert apple_state.base is symmetry_state

            # Determine the neighbourhood of the head
            neighbours_pos = head + self._vision_pos[symmetry_state[:, 0]].transpose()
        else:
            symmetry_state = None
            apple_state = Snakes.DIRECTION8_ID[dy, dx]
            # if debug: print("Apple state", apple_state)

            # Determine the neighbourhood of the head
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
        # if debug: print("Neigbour state", neighbour_state / self.NR_STATES_APPLE)

        # Determine the wall state
        if self.NR_STATES_WALL > 1:
            if self._symmetry:
                wall_state = self._wall[symmetry_state[:, 0], head]
            else:
                wall_state = self._wall[head]
            state = neighbour_state + wall_state + apple_state
        else:
            state = neighbour_state + apple_state

        if debug:
            print(self._vision_obj.string(
                final_newline = True,
                values = neighbours[:,self._debug_index]))
            a_state = apple_state[self._debug_index]
            if self._symmetry:
                a_state = Snakes.LEFT8[a_state]
            if self.NR_STATES_WALL > 1:
                w_state = wall_state[self._debug_index] // self.NR_STATES_APPLE
                print("Wall: right %s, up %s, left %s, down %s" %
                      tuple(self._wall_state_tuple[w_state]))
            else:
                w_state = None
            print("Old State = %s, New State = %d (Apple = %d, Wall = %s, Neighbour = %u)" %
                  ("None" if h0 is None else str(self._old_state[h0][self._debug_index]),
                   state[self._debug_index], a_state, str(w_state),
                   neighbour_state[self._debug_index] // (self.NR_STATES_APPLE * self.NR_STATES_WALL)))

        # Eaten includes collided
        self._eat_frame[move_result.eaten] = self.frame()
        # Evaluate the historical move
        if h0 is not None:
            q_row = self._q_table[state]
            if debug:
                print("Qrow before:", q_row[self._debug_index])
                #print("base r:", r)
                #print("rand r:", rewards[self._debug_index])

            reward_moves = np.random.uniform(
                self._rewards.move - self._rewards.rand /2,
                self._rewards.move + self._rewards.rand /2)
            reward_moves = np.random.uniform(
                (reward_moves-self._rewards.rand/2) * self._history,
                (reward_moves+self._rewards.rand/2) * self._history,
                size=self.nr_snakes)
            rewards = self._history_gained * self._rewards.apple
            rewards += np.where(self._history_game0 == self._nr_games,
                                # Bootstrap from discounted best estimate
                                (np.amax(q_row, axis=-1)+reward_moves) * self._discount,
                                self._rewards.crash)
            if debug:
                print("Reward %f (Old Game = %d, New Game = %d)" %
                      (rewards[self._debug_index],
                       self._history_game0[self._debug_index],
                       self._nr_games[self._debug_index]))
            old_state  = self._old_state [h0]
            old_action = self._old_action[h0]
            rewards -= self._q_table[old_state, old_action]
            rewards *= self._learning_rate
            if debug:
                print("Q old", self._q_table[old_state[self._debug_index]])
                print("Update[%u][%u]" % (old_state [self._debug_index],
                                          old_action[self._debug_index]),
                      rewards[self._debug_index])
            if self._single:
                # Any state entry is only updated once. All other snakes trying
                # to update it at the same time are wasted.
                self._q_table[old_state, old_action] += rewards
            else:
                # Potentially need to multi-update the same state, so use ads.at
                np.add.at(self._q_table, (old_state, old_action), rewards)
            if debug:
                print("Q new", self._q_table[old_state[self._debug_index]])
            #if np.isnan(self._q_table).any():
            #    raise(AssertionError("Q table contains nan"))

        # Decide what to do
        q_row = self._q_table[state]
        if debug:
            print("Qrow after: ", q_row[self._debug_index])
        action = q_row.argmax(axis=-1).astype(TYPE_ID)
        if debug:
            print("Old Action = %s, New action = %u, Moves since apple = %u" %
                  ("None" if h0 is None else str(self._old_action[h0][self._debug_index]),
                   action[self._debug_index],
                   self.frame() - self._eat_frame[self._debug_index]))
            if self._symmetry:
                old_direction = self._old_direction[self._debug_index]
                mirror = symmetry_state[self._debug_index, 2]
                print("Old direction = %u, Mirror = %d, New direction = %u" %
                      (old_direction, mirror,
                       (old_direction + (action[self._debug_index]-1) * mirror) & 3))

        # Detect starving snakes. They are probably looping
        self.unloop(action)
        # Kick a small fraction of all snakes
        self.kick(action, symmetry_state)

        # if debug:
        #    empty_state = self.state(apple = np.arange(self.NR_STATES_APPLE),
        #                             wall = 0,
        #                             neighbour = 0)
        #    print(self._q_table[empty_state])
        # Take the selected action and remember where we came from
        h = self.frame_history_now()
        self._old_state[h]  = state
        self._old_action[h] = action
        if self._symmetry:
            self._old_direction += (action-1) * symmetry_state[:, 2]
            self._old_direction &= 3
            direction = self._old_direction
        else:
            direction = action
        return head + self.DIRECTIONS[direction]


    # Detect starving snakes. They are probably looping
    # Ever so often just give them a random direction even if that kills them
    def unloop(self, action):
        looping = self._eat_frame <= self.frame() - SnakesQ.LOOP_MAX * self.AREA
        looping = looping.nonzero()[0]
        if looping.size:
           if self.debug:
               print("Nr Looping=", looping.size,
                     "looping[0]:", looping[0])
           rand = np.random.randint(SnakesQ.LOOP_ESCAPE * self.AREA, size=looping.size)
           escaping = looping[rand == 0]
           if escaping.size:
               if self.debug:
                   print("Nr Escaping=", escaping.size,
                         "Escaping[0]", escaping[0])
                   # We have no easy way to check if snake self._debug_index
                   # was selected for escape, so fake it by detecting if the
                   # action changed.
                   # We miss the case where it's changed to same
                   old_action = action[self._debug_index]
               # We could also avoid crashing here
               action[escaping] = np.random.randint(self._q_table.shape[-1],
                                                    size = escaping.size)
               if self.debug and action[self._debug_index] != old_action:
                   print("Randomly set New action = %u (unloop)" %
                         action[self._debug_index])


    # Kick a small fraction of all snakes to a random non-blocked direction
    def kick(self, action, symmetry_state):
        accept = np.random.randint(SnakesQ.EPSILON_INV, size=self.nr_snakes)
        randomize = (accept == 0).nonzero()[0]
        if randomize.size:
            # print("Randomize", randomize)

            if self._symmetry:
                # different permutation of turns for each randomized target
                direction_id = np.random.randint(
                    Snakes.NR_DIRECTION3_PERMUTATIONS, size=randomize.size)
                permutations = Snakes.DIRECTION3_ID_PERMUTATIONS[direction_id]
            else:
                # different permutation of dirs for each randomization target
                direction_id = np.random.randint(
                    Snakes.NR_DIRECTION4_PERMUTATIONS, size=randomize.size)

                # different permutation of directions for each randomized target
                permutations = Snakes.DIRECTION4_ID_PERMUTATIONS[direction_id]
            for i in range(permutations[0].size):
                # print("Permutation", i, permutations)
                random_action = permutations[:, i]
                assert random_action.base is permutations
                # print("Random action:", random_action)
                action[randomize] = random_action
                if self._symmetry:
                    directions = ((random_action-1) * symmetry_state[randomize, 2] + self._old_direction[randomize]) & 3
                    # print("D", directions)
                else:
                    directions = random_action
                hit = self._field[randomize, self.head()[randomize] + self.DIRECTIONS[directions]]
                # print("Hit", hit)
                hit = hit.nonzero()[0]
                # print("Hit index", hit)
                if hit.size == 0:
                    break
                randomize = randomize[hit]
                permutations = permutations[hit]

            if self.debug and accept[self._debug_index] == 0:
                print("Randomly set New action = %u (kick)" %
                      action[self._debug_index])
