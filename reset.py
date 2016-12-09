#!/usr/bin/env python

"""
train.py

Moves model, weight, and history files to .bak files. This allows a new model
to be trained by default when train.py is run.

Usage:

  reset.py [clear] [logs] [data]

"clear" deletes model files instead of backing them up. Use with caution!
"logs" deletes log files.
"data" deletes converted music files in data directory.

(Note "logs" and "data" do not depend on "clear" --- they all "clear" in a
sense, for different directories.)
"""

import os

from lib.config import config as cf
import lib.utils as utils

clear = cf.flags.get('clear')
logs = cf.flags.get('logs')
data = cf.flags.get('data')

def main(clear=clear, logs=logs):
  """
  Reset state directory by moving existing files to .bak copies.

  If the clear argument is set to True, files are renamed instead of deleted.
  This is set to False unless a command line argument is passed.

  Arguments:
    bool:clear (cf.flags.clear) -- Delete files instead of renaming them
    bool:logs (cf.flags.logs) -- Delete log files
  """
  for filename in os.listdir(cf.state.model_dir):
    if filename[0] == '.':
      continue
    
    filepath = os.path.join(cf.state.model_dir, filename)

    # Delete all if clear option given
    if clear:
      utils.remove(filepath)
    # Otherwise rename existing files - Note this overwrites existing .baks.
    elif filename[-4:] != '.bak':
      utils.rename(filepath, filepath + '.bak')
  # Delete converted data
  if data:
    for filename in os.listdir(cf.data.target_dir):
      if filename[0] == '.':
        continue
      utils.remove(cf.data.target_dir, filename)

  # Delete logs
  if logs:
    for filename in os.listdir(cf.logging.log_dir):
      if filename[0] == '.':
        continue
      utils.remove(cf.logging.log_dir, filename)

if (__name__ == '__main__'):
  main()