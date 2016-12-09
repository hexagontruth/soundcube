#!/usr/bin/env python

"""
generate.py

Generates one or more musical sequences using random seed data as configured
in config.yml. Specify files as separate command-line arguments. Do not use
"=" in filenames or they will be parsed as config arguments.

Usage:

    generate.py [outputfile1,...] [key1=value1,...]

Example:

  generate.py fancy_song.wav net.model_name=simple state.model_base=fancymodel

This will load weights associated with models/fancymodel if they are available,
then generate and save "fancy_song.wav." If no weights for this model have
been saved, it will try other available weights, which may result in error).
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

filepaths = [os.path.join(cf.output.output_dir, e) for e in files]

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
  for filepath in filepaths:
    y = net.gen()
    print 'Finished generating "{0}."'.format(filepath)
    sclog('Generated "{0}."'.format(filepath))
    utils.write_output(filepath, y)
    if (cf.output.save_raw):
      utils.save_array(os.path.splitext(filepath)[0], y)

if (__name__ == '__main__'):
  main()