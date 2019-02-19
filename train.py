#!/usr/bin/env python

"""
train.py

Trains new or existing model.

Usage:

  train.py [new] [epochs] [key1=value1,...]

Set "new" flag to ignore existing weights (useful for custom model names).

Example:

  train.py new 10 net.model_name=simple state.model_base=fancymodel

This will create a new model using "simple" model builder, and save it to
the models directory under the name "newmodel."
"""

import os

from lib.config import config as cf
from lib.sclog import sclog
import lib.utils as utils
from lib.net import Net

def main():
  # We set autosave to true even if it is False in config
  net = Net(Net.TRAIN, autosave=True)

  # Simplified parameters for setting epochs
  for k in cf.flags.keys():
    if utils.is_str_int(k):
      net.set(training_epochs=int(k))
      break

  # Load weights unless new flag provided
  if not cf.flags.get('new'):
    net.load()

  print('Completed {0} epochs. Training next {1}...'.format(
    net.epochs, net.training_epochs))

  net.train()
  sclog('Finished training {0} epochs.'.format(net.training_epochs))

if (__name__ == '__main__'):
  main()
