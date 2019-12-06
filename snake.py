#!/usr/bin/env python3
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
  snake.py greedy [--snakes=<snakes>] [--debug] [--stepping] [--fps=<fps>]
           [--width=<width>] [--height=<height>] [--frames=<frames>]
           [--columns=columns] [--rows=rows] [--block=<block_size>]
           [--wall=<wall>] [--pygame] [--dump-file=<file>] [--log-file=<log>]
  snake.py q-table [--snakes=<snakes>] [--debug] [--stepping] [--fps=<fps>]
           [--width=<width>] [--height=<height>] [--frames=<frames>]
           [--columns=columns] [--rows=rows] [--block=<block_size>]
           [--wall=<wall>] [--symmetry] [--single] [--pygame]
           [--vision-file=<file>] [--dump-file=<file>] [--log-file=<log>]
           [--learning-rate=<r>] [--discount <ratio>] [--accelerated]
  snake.py cycle [--snakes=<snakes>] [--debug] [--stepping] [--fps=<fps>]
           [--width=<width>] [--height=<height>] [--frames=<frames>]
           [--columns=columns] [--rows=rows] [--block=<block_size>]
           [--pygame] [--show-cycle] [--dump-file=<file>] [--log-file=<log>]
  snake.py benchmark
  snake.py -f <file>
  snake.py (-h | --help)
  snake.py --version

Options:
  -h --help               Show this screen
  --version               Show version
  --stepping              Start in paused mode, wait for the user to press SPACE
  --fps=<fps>             Frames per second (0 is no delays) [default: 40]
  --snakes=<snakes>       How many snakes to run at the same time [default: 0]
                          0 means use rows * colums or 1
  --block=<block_size>    Block size in pixels [default: 20]
  --width=<width>         Pit width  in blocks [default: 40]
  --height=<height>       Pit height in blocks [default: 40]
  --columns=<columns>     Columns of pits to display [default: 2]
  --rows=<rows>           Rows of pits to display [default: 1]
  --frames=<frames>       Stop automatically at this frames number [Default: -1]
  --wall=<wall>           Have state for distance from wall up to <wall>
                          [Default: 2]
  --symmetry              Apply dihedral symmetry
  --vision-file=<file>    Read snake vision from file
  --show-cycle            Show the Hamiltonian cycle on the background
  --pygame		  Use pygame for output (default is qt5)
  --log-file=<file>       Write to logfile. Use an empty string if you
                          explicitely don't want any logging
                          [Default: snakes.log.txt]
  --dump-file=<file>      Which file to dump to on keypress
                          [Default: snakes.dump.txt]
  --single                Any one state can be updated at most once per frame
                          Use this if all snakes tend to be in different states
                          since it allows you to use an undivided learning rate
                          (only reasonable if there are many states relative to
                          the number of snakes)
  -l --learning-rate=<r>  Learning rate
                          (will be divided by number of snakes if not --single)
                          [Default: 0.1]
  --discount <ratio>      state to state Discount [Default: 0.99]
  --debug                 Run debug code
  --accelerated           Prefill the Q table with walls
                          It will learn this by itself but takes a long time if
                          there are very many states
  -f <file>:              Used by jupyter, ignored

Key actions:
  s:          enter pause mode after doing a single step
  r, SPACE:   toggle run/pause mode
  Q, <close>: quit
  +:          More frames per second (wait time /= 2)
  -:          Less frames per second (wait time *= 2)
  =:          Restore the original frames per second
  d:          Toggle debug
  D:          Dump current state (without snake state)
  c:          Capture window (currently pygame only)
  C:          Toggle window stream (currently pygame only)
              Post processing examples:
               Make movie:
                 ffmpeg -y -f rawvideo -s 200x200 -pix_fmt rgb24 -r 40 -i snakes.stream.200x200.rgb -an -vcodec h264 snakes.mp4
               Make gif:
                 ffmpeg -y -f rawvideo -s 880x440 -pix_fmt rgb24 -r 10 -i snakes.stream.880x440.rgb -an -vf palettegen palette.png
                 ffmpeg -y -f rawvideo -s 880x440 -pix_fmt rgb24 -r 10 -i snakes.stream.880x440.rgb -i palette.png -an -lavfi paletteuse snakes.gif


"""
from docopt import docopt

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Snake 1.0')
    if arguments["--debug"]:
        print(arguments)

# For jupyter
arguments
# -

# %matplotlib inline
# from matplotlib import pyplot as plt

import numpy as np

np.set_printoptions(floatmode="fixed", precision=10, suppress=True)

import sys

if arguments["benchmark"]:
    from display import Display
elif arguments["--pygame"]:
    from display.pygame import DisplayPygame as Display
else:
    from display.qt5 import DisplayQt5 as Display


columns    = int(arguments["--columns"])
rows       = int(arguments["--rows"])
nr_snakes  = int(arguments["--snakes"]) or rows*columns
snake_kwargs = dict()
if arguments["greedy"] or arguments["benchmark"]:
    from snakes import Snakes

    snake_class = Snakes
elif arguments["q-table"]:
    from snakes import VisionFile
    from snakes.qtable import SnakesQ

    snake_class = SnakesQ
    if arguments["--vision-file"] is not None:
        snake_kwargs["vision"] = VisionFile(arguments["--vision-file"])
    wall = int(arguments["--wall"])
    snake_kwargs["wall_left"]  = wall
    snake_kwargs["wall_right"] = wall
    snake_kwargs["wall_up"]    = wall
    snake_kwargs["wall_down"]  = wall
    snake_kwargs["single"]        = arguments["--single"]
    snake_kwargs["learning_rate"] = float(arguments["--learning-rate"])
    snake_kwargs["discount"]      = float(arguments["--discount"])
    snake_kwargs["accelerated"]   = arguments["--accelerated"]
    snake_kwargs["symmetry"]      = arguments["--symmetry"]
elif arguments["cycle"]:
    from snakes.hamiltonian import SnakesH
    snake_class = SnakesH
    snake_kwargs["show_cycle"]    = arguments["--show-cycle"]
else:
    raise(AssertionError("Unspecified snake type"))

snakes = snake_class(nr_snakes = nr_snakes,
                     debug     = arguments["--debug"],
                     width     = int(arguments["--width"]),
                     height    = int(arguments["--height"]),
                     **snake_kwargs)

if arguments["benchmark"]:
    from snakes import Snakes

    speed = 0
    for i in range(1):
        np.random.seed(1)
        snakes = Snakes(nr_snakes = 100000,
                        width     = 40,
                        height    = 40)
        display = Display(snakes, rows=0, log_file = None)
        display.run(snakes, fps=0, stepping=False, frame_max = 1000)
        display.loop()
        speed = max(speed, snakes.frame() * snakes.nr_snakes / display.elapsed())
    print("%.0f" % speed)
    sys.exit()

# +
display = Display(
    snakes,
    columns    = columns,
    rows       = rows,
    block_size = int(arguments["--block"]),
    log_file   = arguments["--log-file"],
    dump_file  = arguments["--dump-file"]
)
display.run(snakes,
            frame_max  = int(arguments["--frames"]),
            fps        = float(arguments["--fps"]),
            stepping   = arguments["--stepping"]
)
display.loop()

print("Elapsed %.3f s (%.3fs used), Frames: %d, Frame Rate %.3f" %
      (display.elapsed(), display.elapsed_process(), snakes.frame(), display.frame_rate(snakes)))
print("Max Score: %d, Score/Game: %.3f, Max Moves: %d, Moves/Game: %.3f, Moves/Apple: %.3f" %
      (snakes.score_max(), snakes.score_per_game(), snakes.nr_moves_max(), snakes.nr_moves_per_game(), snakes.nr_moves_per_apple()))
print("Total Won Games: %d/%d, Played Game Max: %d" %
      (snakes.nr_games_won_total(), snakes.nr_games_total(), snakes.nr_games_max()))
