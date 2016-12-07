#!/usr/bin/env python

import os

from modules.config import config as cf
from modules.sclog import sclog
import modules.utils as utils
from modules.net import Net

net = Net(Net.TRAIN)
net.load()

# Simplified parameters for setting epochs
if (not cf.flags.empty()):
  k = cf.flags.keys()[0]
  if utils.is_str_int(k):
    net.set(epochs_per_cycle=int(k))

print 'Completed {0} epochs. Training next {1}...'.format(
  net.epochs, net.epochs_per_cycle)

net.train()
sclog('Finished training {0} epochs.'.format(net.epochs_per_cycle))