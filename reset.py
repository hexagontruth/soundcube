#!/usr/bin/env python

import os

from lib.config import config as cf
import lib.utils as utils

clear = cf.flags.get('clear')
logs = cf.flags.get('logs')

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
      utils.safe_remove(filepath)
    # Otherwise rename existing files - Note this overwrites existing .baks.
    elif filename[-4:] != '.bak':
      utils.safe_rename(filepath, filepath + '.bak')
  # Delete logs
  if logs:
    for filename in os.listdir(cf.logging.log_dir):
      if filename[0] == '.':
        continue
      filepath = os.path.join(cf.logging.log_dir, filename)
      utils.safe_remove(filepath)

if (__name__ == '__main__'):
  main()