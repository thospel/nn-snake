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
           [--width=<width>] [--height=<height>]
           [--frames=<frames>] [--games=<games>] [--dump=<file>]
           [--columns=columns] [--rows=rows] [--block=<block_size>] [--pygame]
           [--log-period=<period>] [--log=<log>] [--tensor-board=<dir>]
  snake.py cycle [--snakes=<snakes>] [--debug] [--stepping] [--fps=<fps>]
           [--width=<width>] [--height=<height>]
           [--frames=<frames>] [--games=<games>] [--dump=<file>]
           [--columns=columns] [--rows=rows] [--block=<block_size>] [--pygame]
           [--log-period=<period>] [--log=<log>] [--tensor-board=<dir>]
           [--risk=<risk_max>] [--show-cycle]
  snake.py q-table [--snakes=<snakes>] [--debug] [--stepping] [--fps=<fps>]
           [--width=<width>] [--height=<height>]
           [--frames=<frames>] [--games=<games>] [--dump=<file>]
           [--columns=columns] [--rows=rows] [--block=<block_size>] [--pygame]
           [--log-period=<period>] [--log=<log>] [--tensor-board=<dir>]
           [--wall=<wall>] [--symmetry] [--single]
           [--vision-file=<file>] [--reward-file=<file>]
           [--learning-rate=<r>] [--discount <ratio>] [--accelerated]
           [--history=<history>] [--history-pit]
  snake.py a2c [--snakes=<snakes>] [--debug] [--stepping] [--fps=<fps>]
           [--width=<width>] [--height=<height>]
           [--frames=<frames>] [--games=<games>] [--dump=<file>]
           [--columns=columns] [--rows=rows] [--block=<block_size>] [--pygame]
           [--log-period=<period>] [--log=<log>] [--tensor-board=<dir>]
            [--reward-file=<file>] [--learning-rate=<r>] [--discount <ratio>]
           [--history=<history>] [--entropy-beta=<beta>]
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
  --games=<games>         Stop automatically once this game number is reached.
                          The actual number of games may be higher since extra
                          games can end during the final frame [Default: -1]
  --wall=<wall>           Have state for distance from wall up to <wall>
                          [Default: 2]
  --symmetry              Apply dihedral symmetry
  --vision-file=<file>    Read snake vision from file
  --reward-file=<file>    Read rewards from file
  --show-cycle            Show the Hamiltonian cycle on the background
  --risk=<risk_max>       Maximum risk to take when taking shortcuts in
                          Hamiltonian cycles [Default: 0.1]
  --pygame		  Use pygame for output (default is qt5)
  --log-period=<period>   How often to write a log entry. Can be given in
                          seconds (s) or frames (f) [Default: 1s]
  --log=<file>            Write to logfile <file>
  --dump=<file>           Which file to dump to on keypress
                          [Default: snakes.dump.txt]
  --tensor-board=<dir>    Write tensorbaord data to dir
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
                          It will learn this by itself but may take a long time
                          if there are very many states
  --history=<history>     Repeat everything <history> moves delayed. This
                          allows learning to evaluate a given old position by
                          how it will do by effectively looking <history> steps
                          into the future [Default: 1]
  --history-pit           Also restore the whole pit layout for history, taking
                          extra time and memory. This is not completely needed
                          in the given mode but allows for easier debugging
                          (the historic layout is shown during debug)
  --entropy-beta=<beta>   Fraction for entropy bonus to the loss function
                          [Default: 0.0001]
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
log_period = arguments["--log-period"]
log_period_frames = False
if log_period.endswith("s"):
    log_period = log_period[:-1]
elif log_period.endswith("f"):
    log_period = log_period[:-1]
    log_period_frames = True
if log_period_frames:
    log_period = int(log_period)
else:
    log_period = float(log_period)
if log_period <= 0:
    raise(ValueError("--log-period must be positive"))

snake_kwargs = dict()
if arguments["greedy"] or arguments["benchmark"]:
    from snakes import Snakes

    snake_class = Snakes
elif arguments["cycle"] or arguments["-f"]:
    from snakes.hamiltonian import SnakesH
    snake_class = SnakesH
    snake_kwargs["show_cycle"]    = arguments["--show-cycle"]
    snake_kwargs["risk_max"]      = float(arguments["--risk"])
elif arguments["q-table"]:
    from snakes import VisionFile
    from snakes.qtable import SnakesQ

    snake_class = SnakesQ
    if arguments["--vision-file"] is not None:
        snake_kwargs["vision"] = VisionFile(arguments["--vision-file"])
    snake_kwargs["reward_file"] = arguments["--reward-file"]
    wall = int(arguments["--wall"])
    snake_kwargs["wall_left"]     = wall
    snake_kwargs["wall_right"]    = wall
    snake_kwargs["wall_up"]       = wall
    snake_kwargs["wall_down"]     = wall
    snake_kwargs["single"]        = arguments["--single"]
    snake_kwargs["learning_rate"] = float(arguments["--learning-rate"])
    snake_kwargs["discount"]      = float(arguments["--discount"])
    snake_kwargs["accelerated"]   = arguments["--accelerated"]
    snake_kwargs["symmetry"]      = arguments["--symmetry"]
    snake_kwargs["history"]       = int(arguments["--history"])
    snake_kwargs["history_pit"]   = arguments["--history-pit"]
elif arguments["a2c"]:
    from snakes.actor_critic import SnakesA2C
    snake_class = SnakesA2C
    snake_kwargs["history"]       = int(arguments["--history"])
    snake_kwargs["reward_file"]   = arguments["--reward-file"]
    snake_kwargs["learning_rate"] = float(arguments["--learning-rate"])
    snake_kwargs["discount"]      = float(arguments["--discount"])
    snake_kwargs["entropy_beta"]  = float(arguments["--entropy-beta"])
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
    columns      = columns,
    rows         = rows,
    block_size   = int(arguments["--block"]),
    log_period   = log_period,
    log_period_frames = log_period_frames,
    log_file     = arguments["--log"],
    dump_file    = arguments["--dump"],
    tensor_board = arguments["--tensor-board"]
)
display.run(snakes,
            frame_max  = int(arguments["--frames"]),
            game_max   = int(arguments["--games"]),
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
