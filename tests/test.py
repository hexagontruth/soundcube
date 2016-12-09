#!/usr/bin/env python

"""
Run this from the project root directory to perform tests.

There should be some music files in the music directory (source_dir) before
testing.

TODO: Everything.
"""

import os
import unittest
import shutil
import sys

import numpy as np

# I literally cannot believe this is how Python handles relative imports
sys.path.insert(0, './')

from lib.config import config as cf

# Set appropriate testing directories
test_target_dir = 'tests/data'
test_model_dir = 'tests/models'
test_output_dir = 'tests/output'

cf.data.target_dir = test_target_dir
cf.state.model_dir = test_model_dir
cf.output.output_dir = test_output_dir

# Now load other project modules - this is because we are setting config vars
# locally in, particularly, utils.py --- this should be fixed.

from lib.net import Net
import lib.utils as utils

# This all needs to be fleshed out a lot more; is a placeholder for now.

class ScBasicTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    test_dirs = [test_target_dir, test_model_dir, test_output_dir]
    for test_dir in test_dirs:
      test_files = os.listdir(test_dir)
      for file in test_files:
        if file[0] == '.':
          continue
        filepath = os.path.join(test_dir, file)
        os.remove(filepath)

  def test_01_convert_files(self):
    # Convert source to wavs
    utils.convert_source_dir()
    # Convert wavs to freq timesteps
    utils.convert_wavs_to_freqs()

  def test__02_data_load(self):
    z = utils.load_blocks()
    w = utils.load_wavs()

    testslice = max(len(z), 10)
    t = z[:testslice]

    u = utils.dechannel(t)
    u = utils.com2pol(u)
    # Test polar forms
    self.assertGreaterEqual(u.T[0].min(), 0)
    self.assertLessEqual(u.T[1].max(), 3.15)
    self.assertGreaterEqual(u.T[1].min(), -3.15)

    v = utils.pol2com(u)
    v = utils.enchannel(v)

    # Should be close to original
    diff = t - v
    self.assertLess(diff.max(), 1e-4)

  def test_03_model(self):
    n = Net(
      Net.ALL,
      training_epochs=25,
      epochs_per_save=5,
      epochs_per_archve=10,
      save_m=True,
      gen_steps=60,
      kwargs={'hidden_size':32})

    # Should not load - no files
    self.assertFalse(n.load())

    # Should train and gen output
    n.train()
    y = n.gen()
    filepath = os.path.join(test_output_dir, 'test1.wav')
    utils.write_output(filepath, y)

    # Build new model and load weights
    m = Net(
      Net.ALL,
      build=False,
      gen_steps=60,
      kwargs={'hidden_size':32})
    self.assertTrue(m.load())

    y = m.gen()
    filepath = os.path.join(test_output_dir, 'test2.wav')
    utils.write_output(filepath, y)

    # Let us confirm our files were saved
    files = filter(lambda e: e[0] != '.', os.listdir(test_output_dir))
    self.assertEqual(len(files), 2)

if __name__ == '__main__':
  unittest.main()