import itertools
import collections.abc
import time
import math
import re
import platform
import hashlib
import numpy as np
import numpy.ma as ma
from dataclasses import dataclass


# TYPE_POS needs to be signed since we also use it for vectors relative to (0,0)
TYPE_POS   = np.int16
TYPE_UPOS  = np.uint16
# TYPE_ID needs to be signed
# (Since we also use TYPE_ID to set mirror to -1 or 1 and direction ID to -1)
TYPE_ID    = np.int8
TYPE_BOOL  = np.bool
TYPE_INDEX = np.intp
# Using bool for TYPE_FLAG in field is about 5% faster than uint8
TYPE_FLAG  = np.bool
TYPE_SCORE = np.uint32
TYPE_MOVES = np.uint32
TYPE_GAMES = np.uint32
TYPE_SNAKE = np.uint32
TYPE_FLOAT = np.float32

CHANNEL_BODY  = 0
CHANNEL_APPLE = 1
CHANNEL_HEAD  = 2
CHANNELS = 3

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


# -

# (Intentionally) crashes if any length is 0
def shape_any(object):
    shape = []
    while isinstance(object, collections.abc.Sized) and not isinstance(object, str):
        shape.append(len(object))
        object = object[0]
    return shape


@dataclass
class Vision(dict):
    """
    Parse drawings like:
         #
       # O #
         #
    and returns a pair of coordinates arrays split into an x and y column:
     X  Y
     0 -1
    -1  0
     1  0
     0  1
    """

    VISION4       = """
      *
    * O *
      *
    """

    VISION_DH3    = """
      *
      O *
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

        iterator = np.nditer((vision_x, vision_y))
        for x,y in iterator:
            self[y+0, x+0] = iterator.iterindex


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
                            spaces = " "
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


@dataclass
class HistoryResult:
    collided:     np.ndarray = None
    eaten:        np.ndarray = None


class Snakes:
    VIEW_X0 = 1
    VIEW_Y0 = 1
    VIEW_X2 = VIEW_X0+2
    VIEW_Y2 = VIEW_Y0+2
    VIEW_WIDTH  = 2*VIEW_X0+1
    VIEW_HEIGHT = 2*VIEW_Y0+1

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
    # Next will set DIRECTIONS4 to all directions:
    # [[ 1  0]
    #  [ 0 -1]
    #  [-1  0]
    #  [ 0  1]]
    DIRECTIONS4 = np.matmul(ROTATIONS, np.array([1,0], dtype=TYPE_POS))
    DIRECTIONS4_X = DIRECTIONS4[:,0]
    DIRECTIONS4_Y = DIRECTIONS4[:,1]
    NR_DIRECTIONS = len(DIRECTIONS4)

    DIAGONALS4 = np.matmul(ROTATIONS, np.array([1,-1], dtype=TYPE_POS))
    DIAGONALS4_X = DIAGONALS4[:,0]
    DIAGONALS4_Y = DIAGONALS4[:,1]

    DIRECTION4_PERMUTATIONS   = np.array(list(itertools.permutations(DIRECTIONS4)), dtype=TYPE_POS)
    NR_DIRECTION4_PERMUTATIONS = len(DIRECTION4_PERMUTATIONS)
    DIRECTION4_PERMUTATIONS_X = DIRECTION4_PERMUTATIONS[:,:,0]
    DIRECTION4_PERMUTATIONS_Y = DIRECTION4_PERMUTATIONS[:,:,1]
    DIRECTION4_ID_PERMUTATIONS = np.array(list(itertools.permutations(range(NR_DIRECTIONS))), dtype=TYPE_ID)
    DIRECTION3_ID_PERMUTATIONS = np.array(list(itertools.permutations(range(3))), dtype=TYPE_ID)
    NR_DIRECTION3_PERMUTATIONS = len(DIRECTION3_ID_PERMUTATIONS)

    # Construct the left array with -1 based indexing, so really the right one
    # 5 1 4          8 0 2
    # 2 8 0    so    3 7 6
    # 6 3 7          1 4 5
    DIRECTION8_ID = np.empty((3,3), dtype=TYPE_ID)
    DIRECTION8_X = np.empty(DIRECTION8_ID.size, dtype=TYPE_POS)
    DIRECTION8_Y = np.empty(DIRECTION8_ID.size, dtype=TYPE_POS)
    DIRECTION8_X[0:4] = DIRECTIONS4_X
    DIRECTION8_Y[0:4] = DIRECTIONS4_Y
    DIRECTION8_X[4:8] = DIAGONALS4_X
    DIRECTION8_Y[4:8] = DIAGONALS4_Y
    DIRECTION8_X[8]   = 0
    DIRECTION8_Y[8]   = 0
    DIRECTION8_ID[DIRECTION8_Y, DIRECTION8_X] = np.arange(DIRECTION8_ID.size)

    # Does it need to mirror about the x-axis to get into quadrant 1 and 2
    # (-1, 0) is assigned to "mirror" for symmetry, negative y is "up"
    MIRROR8 = (DIRECTION8_Y > 0) | (DIRECTION8_Y == 0) & (DIRECTION8_X < 0)
    # Rotation lookup table:
    #     Rows are snake directions as defined by DIRECTIONS4
    #     Columns are (x,y) deltas for directions defined by DIRECTION8_ID
    DIRECTION8_ROTATIONS = ROTATIONS[-np.arange(NR_DIRECTIONS)] @ np.stack((DIRECTION8_X, DIRECTION8_Y), axis=0)
    # dihedral lookup table: Do we need to mirror to put the apple to the left ?
    #     Rows    are snake directions as defined by DIRECTIONS4
    #     Columns are apple directions as defined by DIRECTION8_ID
    MIRROR_DIRECTION8 = MIRROR8[DIRECTION8_ID[DIRECTION8_ROTATIONS[:,1],
                                              DIRECTION8_ROTATIONS[:,0]]]
    # Mirror lookup table: did_mirror = MIRROR8_DH[snake dir, dy, dx]
    MIRROR8_DH = MIRROR_DIRECTION8[:,DIRECTION8_ID]

    MIRROR_Y = np.diag(np.array([1,-1], dtype=TYPE_POS))
    # Symmetry group of the square (dihedral group).
    # First 4 successive left rotations (identity first),
    # then 4 mirrored versions
    ROTATIONS_DH = np.append(ROTATIONS, ROTATIONS @ MIRROR_Y, axis=0)
    # Inverse of ROTATIONS_DH
    ROTATIONS_DH_INV = np.append(ROTATIONS_INV, MIRROR_Y @ ROTATIONS_INV, axis=0)
    # For each symmetry: 4 columns for the 4 directions in DIRECTIONS4
    DIRECTIONS4_DH     = ROTATIONS_DH     @ DIRECTIONS4.transpose()
    DIRECTIONS4_DH_INV = ROTATIONS_DH_INV @ DIRECTIONS4.transpose()

    # after_id = DIRECTION4_ID_DH[symmetry, before_id]
    #  [[0 1 2 3]
    #   [1 2 3 0]
    #   [2 3 0 1]
    #   [3 0 1 2]
    #   [0 3 2 1]
    #   [1 0 3 2]
    #   [2 1 0 3]
    #   [3 2 1 0]]
    DIRECTION4_ID_DH = DIRECTION8_ID[DIRECTIONS4_DH[:,1,:],DIRECTIONS4_DH[:,0,:]]
    # before_id = DIRECTION4_ID_DH[symmetry, after_id]
    #  [[0 1 2 3]
    #   [3 0 1 2]
    #   [2 3 0 1]
    #   [1 2 3 0]
    #   [0 3 2 1]
    #   [1 0 3 2]
    #   [2 1 0 3]
    #   [3 2 1 0]]
    DIRECTION4_ID_DH_INV = DIRECTION8_ID[DIRECTIONS4_DH_INV[:,1,:],DIRECTIONS4_DH_INV[:,0,:]]
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
    ROTATIONS_DIRECTION8_ID = (np.arange(NR_DIRECTIONS).reshape(-1,1,1) + MIRROR_DIRECTION8[:,DIRECTION8_ID]*4).astype(TYPE_ID)

    # print(ROTATIONS_DH_INV)
    # print(ROTATIONS_DIRECTION8_ID)
    # print(ROTATIONS_DH_INV[ROTATIONS_DIRECTION8_ID])
    # ROTATED_DIRECTION8 maps apple directions from before to after rotation
    #     Axis 0: select x or y coordinate of output direction
    #     Axis 1: snake direction as defined by DIRECTIONS4
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
    LEFT8  = np.lexsort((ANGLE8, TAXI8, DIRECTION8_Y > 0, TAXI8 == 0)).astype(TYPE_ID)

    # LEFT8_INV is the inverted permutation of LEFT8,
    # Given a direction id it returns the index in LEFT
    LEFT8_INV = np.lexsort((LEFT8,)).astype(TYPE_ID)

    # ROTATED_LEFT8 maps apple directions to apple state id (index in LEFT8)
    #     Axis 0: snake direction as defined by DIRECTIONS4
    #     Axis 1: y of apple direction
    #     Axis 2: x of apple direction
    # Actual result:
    #   8 0 2     8 1 1     8 2 0     8 1 1
    #   1 3 4     2 4 4     1 4 3     0 3 3
    #   1 3 4     0 3 3     1 4 3     2 4 4
    ROTATED_LEFT8 = LEFT8_INV[DIRECTION8_ID[ROTATED_DIRECTION8[1],ROTATED_DIRECTION8[0]]]
    assert ROTATED_LEFT8.dtype == TYPE_ID
    # print(ROTATED_LEFT8)

    # The state id will be restricted to [0..4] if we did everything right
    # (3 directions to the left, forward and backward)
    ROTATED_LEFT8_MASK = np.zeros(ROTATED_LEFT8.shape, dtype=np.bool)
    ROTATED_LEFT8_MASK[:,0,0] = True
    # So any direction result strictly to the right is a bug
    # Except that index (0,0) is not a direction we should ever see, so mask it
    if (ma.masked_array(ROTATED_LEFT8, ROTATED_LEFT8_MASK) > 4).any():
        raise(AssertionError("dihedral symmetry left something right"))
    if False:
        # Check that all matrices work by displaying all symmetries
        # for manual inspection
        # VISION8 is good enough to show all symmetries, but then the
        # number of points is equal to the number of directions making it
        # much easier to mix up indices. So I use VISION12 instead
        V = Vision(Vision.VISION12, shuffle = True)

        # Make order different from the standard order in DIRECTION8
        # This debugs that we do the proper extra indexing
        if True:
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
            print(MIRROR8_DH[n,Dy,Dx])

    # Combined lookup:
    # (dh id, apple state, parity) = SYMMETRY8_DH[snake dir, apple dY, apple dX]
    SYMMETRY8_DH = np.stack((ROTATIONS_DIRECTION8_ID, ROTATED_LEFT8, np.array([1,-1], dtype=TYPE_ID)[MIRROR8_DH+0]), axis=-1)
    assert SYMMETRY8_DH.dtype == TYPE_ID
    # print(SYMMETRY8_DH)


    def pos_from_yx(self, y, x):
        return x + y * self.WIDTH1


    def pos_from_xy(self, x, y):
        return x + y * self.WIDTH1


    def __init__(self, nr_snakes=1,
                 width     = 40,
                 height    = 40,
                 channels  = 1,
                 history   = 0,
                 history_pit = False,
                 point_image = False,
                 win_bonus = 1,
                 view_x = None, view_y = None,
                 debug = False, xy_apple = True, xy_head = True):
        if nr_snakes <= 0:
            raise(ValueError("Number of snakes must be positive"))

        self.debug = debug
        self._win_bonus = win_bonus
        self._channels = channels
        # Do we keep a cache of apple coordinates ?
        # This helps if e.g. we need the coordinates on every move decission
        self._xy_apple = xy_apple
        # xy_head is currently a hack and not implemented for all planners
        # Check by turning debug on for a bit
        self._xy_head  = xy_head

        self.windows = None

        self._nr_snakes = nr_snakes
        self._all_snakes = np.arange(nr_snakes, dtype=TYPE_INDEX)

        if history:
            if history < 0:
                raise(ValueError("History cannot be negative"))
            # We need an extra history entry because when we look up the final
            # score of a game we want to possibly look up the current game but
            # also possibly the game <history> ago. These must therefore be
            # different. This also needs us to store the current game properties
            # before processing history. So we need 1 extra history slot
            self.HISTORY = history + 1
            self._history_pit = history_pit
        else:
            self.HISTORY = 0
            self._history_pit = False
        self._history = history

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
            raise(ValueError("No space to put both a snake and an apple"))
        # First power of 2 greater or equal to AREA+history for fast modular
        # arithmetic
        self.AREA2 = 1 << (self.AREA + history - 1).bit_length()
        self.MASK  = self.AREA2 - 1

        self.HEIGHT1 = self.HEIGHT+2*self.VIEW_Y
        self.WIDTH1  = self.WIDTH +2*self.VIEW_X

        # Set up offset based versions of DIRECTIONS and PERMUTATIONS
        self.DIRECTIONS = self.pos_from_xy(Snakes.DIRECTIONS4_X,
                                           Snakes.DIRECTIONS4_Y)
        self.DIRECTION_PERMUTATIONS = self.pos_from_xy(
            Snakes.DIRECTION4_PERMUTATIONS_X,
            Snakes.DIRECTION4_PERMUTATIONS_Y)

        # Table of all positions inside the pit
        x0 = np.arange(self.VIEW_X, self.VIEW_X+self.WIDTH,  dtype=TYPE_POS)
        y0 = np.arange(self.VIEW_Y, self.VIEW_Y+self.HEIGHT, dtype=TYPE_POS)
        all0 = self.pos_from_xy(x0, y0.reshape(-1, 1))
        # self.print_pos("All", all0)
        self._all0 = all0.flatten()
        # self.print_pos("All", self._all0)

        # Body length measures the snake *without* the head
        # This is therefore also the score (if we start with length 0 snakes)
        self._body_length = np_empty(nr_snakes, TYPE_INDEX)
        self._snake_body  = np_empty((nr_snakes, self.AREA2), TYPE_POS)

        # Don't need to pre-allocate _head.
        # run_start will implicitely create it
        self._apple = np_empty(nr_snakes, TYPE_POS)
        if self._xy_apple:
            self._apple_x  = np_empty(nr_snakes, TYPE_POS)
            self._apple_y  = np_empty(nr_snakes, TYPE_POS)
        self._nr_moves = np_empty(nr_snakes, TYPE_MOVES)
        self._nr_games_won = np_empty(nr_snakes, TYPE_GAMES)
        self._nr_games = np_empty(nr_snakes, TYPE_GAMES)

        # empty_pit is just the edges with a permanently empty playing field
        self._pit_empty = np.ones((self.HEIGHT1, self.WIDTH1), dtype=TYPE_FLAG)
        self._pit_empty[self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH] = 0
        # Easy way to check if a position is inside the pit
        self._pit_flag = np.logical_not(self._pit_empty)
        # self._field1 = np.ones((nr_snakes, self.HEIGHT1, self.WIDTH1), dtype=TYPE_FLAG)

        if channels == 1:
            # The playing field starts out as nr_snakes copies of the empty pit
            # Notice that we store in row major order, so use field[snake,y,x]
            # (This makes interpreting printouts a lot easier)
            base = self._pit_empty.reshape(1,self.HEIGHT1,self.WIDTH1).repeat(nr_snakes, axis=0)
            self._field1 = base
        else:
            # With edges as a rectangle
            base = np.zeros((nr_snakes, self.HEIGHT1, self.WIDTH1, channels), dtype=TYPE_FLOAT)
            self._deep_field1 = base

            # With edges, as a linear array
            self._deep_field = base.reshape(nr_snakes, self.HEIGHT1*self.WIDTH1, -1)
            assert self._deep_field.base is base

            # Without edges, a rectangle
            self._deep_field0 = base[:, self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH]
            assert self._deep_field0.base is base

            # Just the body (with edges, as a rectangle)
            self._field1 = base[:,:,:,CHANNEL_BODY]
            assert self._field1.base is base
            self._field1[:] = self._pit_empty.astype(self._field1.dtype)
            # print(self._deep_field)
        # The body (without edges, as a rectangle)
        self._field0 = self._field1[:, self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH]
        assert self._field0.base is base

        # The body (with edges, as a linear array)
        self._field = self._field1.reshape(nr_snakes, self.HEIGHT1*self.WIDTH1)
        assert self._field.base  is base

        if history:
            self.init_history()
        if point_image:
            self.init_point_image()


    def init_history(self):
        nr_snakes = self.nr_snakes
        history_shape = (self.HISTORY, nr_snakes)
        self._history_apple0      = np_empty(nr_snakes, dtype=TYPE_UPOS)
        self._history_score       = np_empty(nr_snakes, dtype=TYPE_UPOS)
        self._history_game0       = np_empty(nr_snakes, dtype=TYPE_GAMES)
        self._history_score_final = np_empty(history_shape, dtype=TYPE_UPOS)

        if self._history_pit:
            if self._channels == 1:
                # The body (with edges, as a rectangle)
                base = self._field1.copy()
                self._history_field1 = base
            else:
                # With edges as a rectangle
                base = self._deep_field1.copy()
                self._deep_history_field1 = base

                # With edges, as a linear array
                self._deep_history_field = base.reshape(nr_snakes, self.HEIGHT1*self.WIDTH1, -1)
                assert self._deep_history_field.base is base

                # Without edges, a rectangle
                self._deep_history_field0 = base[:, self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH]
                assert self._deep_history_field0.base is base

                # Just the body (with edges, as a rectangle)
                self._history_field1 = base[:,:,:,0]
                assert self._history_field1.base is base
            # The body (without edges, as a rectangle)
            self._history_field0 = self._history_field1[:, self.VIEW_Y:self.VIEW_Y+self.HEIGHT, self.VIEW_X:self.VIEW_X+self.WIDTH]
            assert self._history_field0.base is base

            # The body (with edges, as a linear array)
            self._history_field = self._history_field1.reshape(nr_snakes, self.HEIGHT1*self.WIDTH1)
            assert self._history_field.base is base


    def init_point_image(self):
        self._point_base = np.zeros(2*self.AREA-1, dtype=TYPE_BOOL)
        p = self.AREA-1
        self._point_base[p] = True
        size = self.pos_from_xy(self.WIDTH+self.VIEW_X-1,
                                self.HEIGHT+self.VIEW_Y-1)+1
        self._point_image = [None] * size
        for y in range(self.HEIGHT):
            pos = self.pos_from_yx(y+self.VIEW_Y, self.VIEW_X)
            for x in range(self.WIDTH):
                image = self._point_base[p:p+self.AREA].reshape(self.HEIGHT, self.WIDTH)
                assert image.base is self._point_base
                self._point_image[pos] = image
                pos += 1
                p -= 1


    def point_image(self, pos):
        return self._point_image[pos]


    def log_constants(self, log_action):
        log_action("Script",      " %s", script())
        log_action("Script Hash", " %s", file_object_hash())
        log_action("Type",        " %s", type(self).__name__)
        log_action("Hostname",    " %s", platform.node())
        log_action("Width",       "%8d", self.WIDTH)
        log_action("Height",      "%7d", self.HEIGHT)
        log_action("Snakes",      "%7d", self.nr_snakes)
        log_action("View X",      "%7d", self.VIEW_X)
        log_action("View_Y",      "%7d", self.VIEW_Y)
        log_action("History",     "%5d", self._history)


    def log_frame(self, log_action, current):
        current["score_total"]    = self.score_total_games()
        current["nr_games_won"]   = self.nr_games_won_total()
        current["nr_games_total"] = self.nr_games_total()
        current["nr_moves_total"] = self.nr_moves_total_games()

        log_action("Games Total", "%6d"    , current["nr_games_total"])
        log_action("Games Won",   "%8d"    , self.nr_games_won_total())
        log_action("Score Max",   "%8d"    , self.score_max())
        log_action("L Score Max", "%6d"    , self.score_max_local())
        log_action("Score Total", "%6d"    , current["score_total"])
        log_action("Moves Max",   "%8d"    , self.nr_moves_max())
        log_action("Moves total", "%6d"    , current["nr_moves_total"])
        log_action("Moves/Game", "%11.3f"  , self.nr_moves_per_game())
        log_action("Moves/Apple", "%10.3f" , self.nr_moves_per_apple())
        log_action("Score/Game", "%11.3f"  , self.score_per_game())


    def log_delta(self, log_action, delta):
        if delta["nr_games_total"]:
            log_action("L Score/Game", "%9.3f" ,
                       delta["score_total"] / delta["nr_games_total"])
            log_action("L Moves/Game", "%9.3f" ,
                       delta["nr_moves_total"] / delta["nr_games_total"])

        if delta["score_total"]:
            log_action("L Moves/Apple", "%8.3f" ,
                       delta["nr_moves_total"] / delta["score_total"])

        if delta["frame"]:
            log_action("L Games/Frame", "%8.3f",
                       delta["nr_games_total"] / delta["frame"])
            log_action("L Won/Frame", "%10.3f",
                       delta["nr_games_won"] / delta["frame"])
            log_action("L Score/Frame", "%8.3f",
                       delta["score_total"] / delta["frame"])

        if delta["nr_games_total"]:
            log_action("L Won/Game", "%11.3f",
                       delta["nr_games_won"] / delta["nr_games_total"])


    def dump_fh(self, fh):
        pass


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


    def score_max_local(self):
        return self._score_max_local


    def score_max_local_clear(self):
        self._score_max_local = 0


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


    def apple_x(self):
        return self._apple_x


    def apple_y(self):
        return self._apple_y


    def apple(self):
        return self._apple


    def head_x(self):
        return self._head_x


    def head_y(self):
        return self._head_y


    def head(self):
        return self._head


    def head_set(self, head_new):
        if self._channels > 1:
            self._deep_field[self._all_snakes, self._head, CHANNEL_HEAD] = 0
            self._deep_field[self._all_snakes, head_new,   CHANNEL_HEAD] = 1
        self._head = head_new
        offset = self._cur_move & self.MASK
        self._snake_body[self._all_snakes, offset] = head_new
        # print_xy("Head coordinates", self._head_x, self._head_y)
        # print(head_new)
        self._field[self._all_snakes, head_new] = 1


    def tail(self):
        # print("body length", self._body_length)

        # Bring potentially large cur_move into a reasonable range
        # so tail_offset will not use some large integer type
        offset = self._cur_move & self.MASK
        # print("Offset", offset)
        tail_offset = (offset - self._body_length) & self.MASK
        # print("tail offset", tail_offset)
        return self._snake_body[self._all_snakes, tail_offset]


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
        # self.print_pos("Tail pos", pos)
        self._field[self._all_snakes, pos] = values
        return pos


    def snake_string(self, shape):
        apple_y, apple_x = self.yx0(self.apple()[shape])
        head_y,  head_x  = self.yx0(self.head ()[shape])
        if self._history_pit:
            if self._history_result0 is not None:
                h_apple_y, h_apple_x = self.yx0(self._history_apple0[shape])
                h_head_y,  h_head_x  = self.yx0(self._history_head0 [shape])
            per_column = 2
        else:
            per_column = 1

        rows, columns = shape.shape
        horizontal = "+" + "-" * (2*self.WIDTH-1) + "+"
        horizontal = horizontal + (" " + horizontal) * (columns * per_column-1) + "\n"
        str = ""
        for r in range(rows):
            str += horizontal
            for y in range(self.HEIGHT):
                for c in range(columns):
                    for h in range(per_column):
                        if c != 0 or h != 0:
                            str += " "
                        str = str + "|"

                        i = shape[r,c]
                        if h == 0:
                            field = self._field0[i]
                            h_x = head_x
                            h_y = head_y
                            a_x = apple_x
                            a_y = apple_y
                        elif self._history_result0 is not None:
                            field = self._history_field0[i]
                            h_x = h_head_x
                            h_y = h_head_y
                            a_x = h_apple_x
                            a_y = h_apple_y
                        else:
                            str += "." * (2*self.WIDTH-1)
                            str = str + "|"
                            continue

                        for x in range(self.WIDTH):
                            if x != 0:
                                str += " "
                            if field[y][x]:
                                if y == h_y[r,c] and x == h_x[r,c]:
                                    str += "O"
                                else:
                                    str += "#"
                            elif y == a_y[r,c] and x == a_x[r,c]:
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
    # Does not work on negative numbers (is that so ? should work!)
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

        if self._channels > 1:
            self._deep_field[todo, self._apple[todo], CHANNEL_APPLE] = 0

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
        if self._channels > 1:
            self._deep_field[old_todo, self._apple[old_todo], CHANNEL_APPLE] = 1
        # self.print_pos("Placed apples", self._apple[old_todo])


    # If the apple is at position (x, y) relative to the head, then if you put
    # numbers around the head like this:
    #         y
    #      -x # x
    #        -y
    # and then select the largest value you will in fact move towards the apple
    # This function sets up such an array for each snake
    def apple_distance(self):
        if self._xy_head:
            head = None
            head_x = self.head_x()
            head_y = self.head_y()
        else:
            head = self.head()
            head_y, head_x = self.yx(head)

        if self._xy_apple:
            apple_x = self.apple_x()
            apple_y = self.apple_y()
        else:
            apple_y, apple_x = self.yx(self.apple())

        distance = np.empty((4, self.nr_snakes), dtype = TYPE_POS)
        # Signs of y are reversed from normal cartesian because y is *DOWN*
        distance[0] = apple_x - head_x
        distance[1] = head_y  - apple_y
        distance[2] = head_x  - apple_x
        distance[3] = apple_y - head_y

        return distance, head, head_x, head_y


    # Plot the shortest course to the apple completely ignoring any snake body
    def plan_greedy(self):
        if False:
            # The version below works but is notably slower
            distance, head, head_x, head_y = self.apple_distance()
            direction = distance.argmax(axis=0)

            if self._xy_head:
                # This updates self._head_x since head_x IS self._head_x.
                head_x += Snakes.DIRECTIONS4_X[direction]
                head_y += Snakes.DIRECTIONS4_Y[direction]
                return head_x + head_y * self.WIDTH1
            else:
                delta = self.DIRECTIONS[direction]
                return head+delta


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
        direction_index = np.random.randint(Snakes.NR_DIRECTION4_PERMUTATIONS,
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


    def frame_then(self):
        return self.frame() - self._history


    def frame_history_now(self):
        return self.frame() % self.HISTORY


    def frame_history_then(self):
        frame0 = self.frame_then()
        if frame0 < 0:
            return None
        return frame0 % self.HISTORY


    def move_remember(self, move_result):
        frame = self.frame()
        h = self.frame_history_now()

        collided = move_result.collided
        eaten    = move_result.eaten
        self._history_game[h]  = self.nr_games(collided)
        self._history_result[h] = HistoryResult(
            collided = collided,
            eaten    = eaten)
        if self._history_pit:
            self._history_apple[h] = self.apple()[eaten]
        if eaten.size:
            # print("Set [", self._history_game[h, eaten], ",", eaten, "] = ", self.score(eaten))
            self._history_score_final[self.nr_games(eaten)  % self.HISTORY, eaten] = self.score(eaten)
        if move_result.won.size:
            won = move_result.won
            # Add the apple that the snake never got a chance to eat
            self._history_score_final[(self.nr_games(won)-1) % self.HISTORY, won] += self._win_bonus

        frame0 = self.frame_then()
        if frame0 >= 0:
            h0 = self.frame_history_then()
            # frame  &= self.MASK
            frame0 &= self.MASK
            self._history_result0 = self._history_result[h0]
            eaten    = self._history_result0.eaten
            collided = self._history_result0.collided
            self._history_score[eaten] += 1
            if self._history_pit:
                if self._channels > 1:
                    self._deep_history_field[self._all_snakes, self._history_head0, CHANNEL_HEAD] = 0
                    self._deep_history_field[eaten, self._history_apple0[eaten], CHANNEL_APPLE] = 0
                self._history_apple0[eaten] = self._history_apple[h0]
                self._history_head0 = self._snake_body[self._all_snakes, frame0]
                if self._channels > 1:
                    self._deep_history_field[self._all_snakes, self._history_head0, CHANNEL_HEAD] = 1
                    self._deep_history_field[eaten, self._history_apple0[eaten], CHANNEL_APPLE] = 1

                # At frame0 == 0 this fetches nonsense tail values from
                # _snake_body. However the next line immediately replaces all
                # these values
                tail = self._snake_body[self._all_snakes, (frame0 -1 - self._history_score) & self.MASK]
                tail[eaten] = self._history_head0[eaten]
                self._history_field[self._all_snakes, tail] = False
                self._history_field0[collided] = 0
                self._history_field[self._all_snakes, self._history_head0] = True

            self._history_score[collided] = 0
            self._history_game0[collided] = self._history_game[h0]
            # print(self._history_score_final)
            self._history_gained = self._history_score_final[self._history_game0 % self.HISTORY, self._all_snakes] - self._history_score
            # print("Score", self._history_score)
            # print("Gained", self._history_gained)
            # print(self._history_score_final)


    def move_evaluate(self, pos):
        is_eat = pos == self.apple()
        is_collision = self._field[self._all_snakes, pos]
        # is_win really checks for "about to win" instead of "won"
        # So the pit is completely filled except for one spot which has
        # the apple. So we still need to check if the snake gets the apple
        # And in fact any non-collision move MUST get the apple
        is_win = self._body_length >= self.AREA-2
        won = is_win.nonzero()[0]
        if won.size:
            # Wins will be rare so it's no problem if this is a bit slow
            lost_index = is_collision[won].nonzero()[0]
            if lost_index.size:
                # You didn't win but crashed on the last move. Sooo close
                is_win[won[lost_index]] = False
                won = won[is_collision[won] == 0]

            if self.debug and is_win[self._debug_index]:
                print("Your debug snake ... Won!!!!")

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


    # In all pits where the snake won or lost we need to restart the game
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
            # So several counters need to be 1 higher than normal
            self._moves_total_games += won.size
            moves_max = np.amax(self.nr_moves(won))+1
            if moves_max > self._moves_max:
                self._moves_max = moves_max
            self._score_total_games += won.size
            score_max = np.amax(self._body_length[won])+1
            if score_max > self._score_max_local:
                self._score_max_local = score_max
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
        if score_max > self._score_max_local:
            self._score_max_local = score_max
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

        # self.print_pos("body", self._snake_body)
        # self.print_pos("Head", self.head())
        # self.print_pos("Apple", self._apple)
        # print("-------------------")


    def move_debug(self):
        print("=" * 20, self.frame(), "=" * 20)
        # print(self.snakes_string(rows=1, columns=1))
        print(self.snake_string(np.array([[self._debug_index]])))

        if self._xy_apple:
            y, x = self.yx(self._apple)
            assert np.array_equal(x, self._apple_x)
            assert np.array_equal(y, self._apple_y)

        if self._xy_head:
            y, x = self.yx(self._head)
            assert np.array_equal(x, self._head_x)
            assert np.array_equal(y, self._head_y)


    # Default move_select for derived classes that don't provide one
    def move_select(self, move_result, display):
        pos = self.plan_greedy_unblocked()
        # self.print_pos("Move", pos)
        return pos


    # Setup initial variables for moving snakes
    def run_start(self, display):
        nr_windows = min(self.nr_snakes, display.windows)
        # self._all_windows = np.arange(nr_windows-1, -1, -1, dtype=TYPE_INDEX)
        self._all_windows = np.arange(nr_windows, dtype=TYPE_INDEX)
        self._debug_index = self._all_windows[0] if nr_windows else 0

        self._nr_games_won_total = 0
        self._nr_games_max = 0
        self._nr_games_total = 0
        self._score_max = 0
        self._score_max_local = 0
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

        if self._channels > 1:
            pos0 = self.pos_from_xy(0, 0)
            self._apple.fill(pos0)
            # Sharing is good enough, the head_set() below will set a new value
            self._head = self._apple

        # print("Initial heads")
        if self._xy_head:
            self._head_x = self.rand_x(self.nr_snakes)
            self._head_y = self.rand_y(self.nr_snakes)
            head = self._head_x + self._head_y * self.WIDTH1
        else:
            head = self.rand(self.nr_snakes)
        self.head_set(head)
        self.new_apples(self._all_snakes)

        if self._history:
            self._history_result0 = None
            self._history_result  = [None] * self.HISTORY
            self._history_game    = [None] * self.HISTORY
            if self._history_pit:
                if self._channels == 1:
                    self._history_head0 = None
                else:
                    # Prefill history_apple0 so we can safely construct frame 0
                    # (at history later) when clearing old apple positions
                    pos0 = self.pos_from_xy(0, 0)
                    self._history_apple0.fill(pos0)
                    # Same with heads
                    self._history_head0 = self._history_apple0.copy()
                self._history_apple = [None] * self.HISTORY

        # Pass w_head instead of head so we do self.yx only where needed
        w_head = self.yx(head[self._all_windows])
        # print_xy("Initial head:", w_head)
        if self._all_windows.size:
            window_iterator = np.nditer(self._all_windows)
            display.draw_windows(window_iterator, w_head)
            display.draw_apples(self._all_windows, range(nr_windows),
                                self.yx(self._apple[self._all_windows]),
                                self._body_length)


    def move_result_start(self):
        # Initial values for move_result
        # This is for planners that want to know about what happened on
        # the previous move. But there was no previous move the first time...
        # Planners that care will have to test for <None> values or frame() or
        # override "run_start_results" with something that makes sense to them
        return MoveResult(collided = self._all_snakes,
                          eaten    = self._all_snakes,
                          won      = np.empty(0, dtype=TYPE_SNAKE))


    # We are done moving snakes. Report some statistics and cleanup
    def run_finish(self):
        score_max = np.amax(self._body_length)
        if score_max > self._score_max_local:
            self._score_max_local = score_max
            if score_max > self._score_max:
                self._score_max = score_max
        moves_max = self._cur_move - np.amin(self._nr_moves)
        if moves_max > self._moves_max:
            self._moves_max = moves_max
        # We could only measure nr_games_max here, but then
        # we wouldn't have a running update
        # self._nr_games_max = np.amax(self._nr_games)
        self._all_windows = None


    def move_generator(self, display):
        move_result = self.move_result_start()
        while True:
            if self._history:
                self.move_remember(move_result)

            if self.debug:
                self.move_debug()

            # Decide what we will do before showing the current state
            # This allows us to show the current plan.
            # It also makes debug messages during planning less confusing
            # since screen output represents the state during planning
            plan = self.move_select(move_result, display)
            # Forgetting to return is an easy bug leading to confusing errors
            assert plan is not None

            # From this point on until move_execute is finished x_head()/y_head
            # (if set) represent the plan while head() represents now
            yield

            # For each snake determine what happened (win, crash, eat, nothing)
            move_result = self.move_evaluate(plan)
            # Initial move_result has an "is_eat" but no "eaten" field

            # Start a new snake for all finished snakes (win, crash)
            self.move_collisions(display, plan, move_result)

            # Update snake state with the result of the move
            # Modifies "is_eat" with collided and sets "eaten in move_result
            self.move_execute(display, plan, move_result)


class SnakesRandom(Snakes):
    def __init__(self, *args, xy_head=False, xy_apple=False, **kwargs):
        super().__init__(*args, xy_head=False, xy_apple=False, **kwargs)

    def move_select(self, move_result, display):
        return self.plan_random()


class SnakesRandomUnblocked(Snakes):
    def move_select(self, move_result, display):
        return self.plan_random_unblocked(self._all_snakes)
