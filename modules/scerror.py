"""
exceptions.py

Provides Soundcube exception classes.
"""

class ScError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

class ScConfigError(ScError):
  pass

class ScIOError(ScError):
  pass

class ScNetError(ScError):
  pass

class ScNotImplementedError(ScError):
  pass

