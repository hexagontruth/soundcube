#!/usr/bin/env python

"""
convert.py

Convert music from command line. Place music in music directory (or other
configured directory) and then run this script.
"""
import os

from lib.config import config as cf
import lib.utils as utils

clear = cf.flags.get('clear')

def main(clear=clear):
  """
  Converts source music to formatted wavs, and converts wavs to frequency
  timestep npy files.

  Runs utils.convert_source_dir() and utils.convert_wavs_to_freq() from
  command line.

  Arguments:
    bool:clear (cf.flags.clear) -- Remove existing files from data directory
  """
  if clear:
    for file in os.listdir(cf.data.target_dir):
      if file[0] == '.':
        continue
      utils.safe_remove(os.path.join(cf.data.target_dir, file))
  
  # Convert source dir
  utils.convert_source_dir()
  # Convert wavs to freq timesteps
  utils.convert_wavs_to_freq()

if (__name__ == '__main__'):
  main()