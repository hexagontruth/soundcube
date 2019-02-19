"""
config.py

Populates and returns a recursive configuration object that provides
"JavaScript-style" configuration attribute access.

To override a given configuration value, provide a command line argument
of the form:

[key][.subkey]...[.subkey]=value

For instance:

data.target_dir=custom_dir

Arguments thus provided will in general be cast to the same type as the
original key value.

Any command line arguments passed without "=" are interpeted as boolean
arguments, and are set to True in the config.flags attribute of the base
configuration object.
"""

import os
import re
import sys
import yaml

from .scerror import *

CONFIG_PATH = 'config.yml'

class Config():
  """
  Recursive configuration object.

  Provides hierarchical, JS-style key-value pairs based on dict.
  """

  def __init__(self, opts, base=None):
    """
    Config constructor.

    Arguments:
      dict:opts -- dict or leaf key value
      Config:base (None) -- Base Config instance
    """

    # Set base config
    self.__base__ = base or self

    # This isn't really used at present
    self.__null__ = ConfigNull.instance

    # Recursively add key-value pairs
    for k, v in opts.items():
      if type(v) == dict:
        setattr(self, k, Config(v, self))
      else:
        setattr(self, k, v)

    # Add "flags" for unitary command-line args
    if self.__base__ == self:
      setattr(self, 'flags', Config({}, self.__base__))

  def get(self, k):
    """
    Get value of key

    Arguments:
      k -- key
    """
    return self.__dict__.get(k)

  def set(self, k, v):
    """
    Set value of key
    
    Arguments:
      k -- key
      v -- value
    """
    setattr(self, k, v)

  def dict(self):
    """
    Return sanitized dict of key-value pairs.
    """
    newdict = {}
    for k, v in self.__dict__.items():
      if k[0:2] != '__':
        newdict[k] = v
    return newdict

  def iter(self):
    """
    Returns iterator over key-value pairs in current instance.
    """
    self.dict().items()
    return


  def keys(self):
    """
    Return array of keys.
    """
    return self.dict().keys()

  def empty(self):
    """
    Returns True if no keys are assigned; False otherwise.
    """
    return len(self.keys()) == 0


  def set_args(self, args):
    """
    Set all arguments in iterable of command line argument strings.

    Iterates over args and passes them to set_arg().

    Arguments:
      iterable:args -- Iterable of strings of form "key.subkey=value"
    """
    for arg in args:
      self.set_arg(arg)

  def set_arg(self, arg):
    """
    Set argument based on command line argument string.

    Arguments:
      str:arg -- String of form "key.subkey=value"
    """
    if not '=' in arg:
      self.__base__.flags.set(arg, True)
      return

    fullkey, val = arg.split('=', 1)
    keys = [re.sub('\W', '', e) for e in fullkey.split('.')]
    obj = self
    key = keys[-1]
    for nodekey in keys[:-1]:
      if obj.get(nodekey):
        obj = obj.get(nodekey)
      else:
        return
    cur_val = obj.get(key)
    if type(cur_val) == int:
      obj.set(key, int(val))
    elif type(cur_val) == float:
      obj.set(key, float(val))
    elif type(cur_val) == bool:
      obj.set(key, True if (val.lower() == 'true' or val.lower() == 1) else False)
    else:
      obj.set(key, val)

class ConfigNull():
  instance = None
  def __init__(self):
    ConfigNull.instance = self
  def __nonzero__(self):
    return False
  def __eq__(self, other):
    return isinstance(other, ConfigNull)

# --- BUILD CONFIGS ---

try:
  file = open(CONFIG_PATH, 'r')
  config = Config(yaml.load(file))
  file.close()
  config.set_args(sys.argv[1:])
except IOError:
  raise ScConfigError('There was a problem loading the config file!')