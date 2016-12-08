#!/usr/bin/env python

"""
generate.py

Generates one or more musical sequences using random seed data as configured
in config.yml. Specify files as separate command-line arguments. Do not use
"=" in filenames or they will be parsed as config arguments.
"""

import os

from lib.config import config as cf
from lib.sclog import sclog
import lib.utils as utils
from lib.net import Net

# Extract filenames
files = [e for e in cf.flags.keys()]

if (len(files) == 0):
  files.append(cf.output.default_file)

paths = [os.path.join(cf.output.output_dir, e) for e in files]

def main(files=files):
  """
  Generates music and saves to filenames set in command line args, or
  in config.output.default_file.

  Runs automatically from command line.

  Arguments:
    list:files -- List of files, taken from args when run on command line
  """

  # Build net
  print 'Loading net...'
  net = Net(Net.GEN)
  print 'Loading model...'
  net.load()

  # Generate
  for p in paths:
    y = net.gen()
    sclog('Generated "{0}."'.format(p))
    w = utils.freq2time(y)
    w = utils.fade(w)
    utils.write_wav(p, w)

if (__name__ == '__main__'):
  main()