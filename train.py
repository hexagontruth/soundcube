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

new_flag = cf.flags.get('new')

# Simplified parameters for setting epochs
for k in cf.flags.keys():
  if utils.is_str_int(k):
    net.set(training_epochs=int(k))
    break

def main():
    net = Net(Net.TRAIN)
    if not new_flag:
      net.load()

    print 'Completed {0} epochs. Training next {1}...'.format(
      net.epochs, net.training_epochs)

    net.train()
    sclog('Finished training {0} epochs.'.format(net.training_epochs))

if (__name__ == '__main__'):
  main()
