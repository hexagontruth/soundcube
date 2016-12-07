"""
sclog.py

Provides Soundcube logging functionality.
"""

import logging as _logging
import os

from config import config as cf

### --- LOGGING ---

if cf.logging.enabled:
  filename = os.path.join(cf.logging.log_dir, cf.logging.log_file)
  # TODO: Expand this
  if cf.logging.level == 'debug':
    level = _logging.DEBUG
  else:
    level = _logging.INFO

  _logging.basicConfig(
    filename=filename,
    level=level,
    format='%(asctime)s (%(levelname)s): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

def sclog(msg, level='info'):
  if cf.logging.enabled:
    if level == 'info':
      _logging.info(msg)
    elif level == 'debug':
      _logging.debug(msg)
    else:
      raise ScNotImplementedError('This logging level is not implemented.')