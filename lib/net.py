"""
net.py

Loads Net class and associated utility classes.

Required for all network training and generation functionality. Responsible for
storing, retrieving, and updating network state data from filesystem.
"""

import os
import sys

import numpy as np

import keras.models
from keras.callbacks import Callback

from .config import config as cf
from .sclog import sclog
from .scerror import *
from . import models
from . import utils

# I don't remember why this is here but I'm afraid to remove it.
sys.setrecursionlimit(50000)

class Net():
  NONE = 0
  TRAIN = 2
  GEN = 4
  ALL = 6

  """
  Stores basic configuration, training, and generation params for Keras model.
  """
  def __init__(self, mode=None, build=True, data=True, **kwargs):
    """
    Net constructor

    Additional keyword arguments can be arbitrary, for use by model building
    methods, etc.

    Keywork arguments:
      int:mode (Net.NONE) -- Whether to prebuild training, gen, or or both
        models. Accepted values: Net.NONE, Net.TRAIN, Net.GEN, Net.ALL.
      sequence/bool:data (False) -- If True, loads data using utils.load_data()
        method. If False, leaves data empty. Loads sequence if provided.
        Expecting sequence of form (train_x, train_y, val_x, val_y)
      **kwargs -- Keyword args to override default configuration.
    """

    # Set mode
    mode = mode or Net.NONE
    self.mode = mode
    self.tmodel = None
    self.gmodel = None

    self.set_defaults()
    self.set(**kwargs)

    # Set data
    is_data = utils.is_data()
    if data is True and is_data:
      self.data = utils.load_data(split=self.val_split)
    elif data is False or not is_data:
      self.data = [None, None, None, None]
    elif isinstance(data, tuple) or isinstance(data, list):
      self.data = data

    # Set models
    if build:
      self.build_models()

  def set_defaults(self):
    """
    Set default attribute values based on configuration.
    """
    self.hops_per_block = cf.data.hops_per_second * cf.data.seconds_per_block
    self.bins = cf.data.step_length // 2

    # Channels represent bin depth
    # Changing this could cause unforeseen failures
    self.channels = 2

    # Not to be confused with input/output channels, which refer to channels
    # sent to and from Keras mdoel itself
    # Again, change this at your own risk
    self.input_channels = 1
    self.output_channels = 1

    # Model settings
    self.model_name = cf.net.model_name
    self.batch_size = cf.net.batch_size
    self.training_epochs = cf.net.training_epochs
    self.val_split = cf.net.val_split
    self.loss = cf.net.loss
    self.opt = cf.net.opt
    self.metrics = []
    self.history = []
    self.kwargs = cf.net.kwargs.dict()

    # Save state configuration
    self.model_base = cf.state.model_base
    self.model_dir = cf.state.model_dir
    self.model_base_path = os.path.join(self.model_dir, self.model_base)
    self.autosave = cf.state.autosave
    self.save_m = cf.state.save_m
    self.epochs_per_save = cf.state.epochs_per_save
    self.epochs_per_archive = cf.state.epochs_per_archive
    self.epochs = cf.state.base_epochs
    self.base_epochs = cf.state.base_epochs # Override loaded epoch count

    # Defeault seed steps
    # Only affects default seeds - net will take any seed <= gen_steps
    self.seed_steps = cf.output.seed_steps
    # Default generation timesteps
    self.gen_steps = cf.output.gen_steps
    self.random_seed = cf.output.random_seed

    # Modify generated output before being used as input vector to net
    self.input_map = lambda x: x
    # Modify final generated output (filtering, etc.)
    self.output_map = lambda x: x

  def set(self, **kwargs):
    """
    Set attributes based on arbitrary keyword arguments. Used by constructor.

    Arguments:
      **kwargs -- Key-value pairs for fields
    """
    for k, v in kwargs.items():
      setattr(self, k, v)

    # Derived fields
    self.t = self.hops_per_block - 1

    if kwargs.get('model_base_path'):
      self.model_base = os.path.basename(self.model_base_path)
      self.model_dir = os.path.dirname(self.model_base_path)
    elif kwargs.get('model_dir') or kwargs.get('model_base'):
      self.model_base_path = os.path.join(self.model_dir, self.model_base)

  def is_training(self):
    """
    Determines of net instance is for training, based on mode settings.
    """
    return (self.mode >> 1) % 2 == 1

  def is_gen(self):
    """
    Determines of net instance is for training, based on mode settings.
    """
    return (self.mode >> 2) % 2 == 1

  def merge_kwargs(self, kwargs):
    """
    Used internally to merge kwargs with self.model_args.

    Arguments:
      dict:kwargs -- Kwargs, typically passed as **kwargs to calling method
    """
    return {k: v for d in [self.kwargs, kwargs] for k, v in d.items()}

  # --- MODELS ---

  def build_models(self, model_name=None, **kwargs):
    """
    Build models based on mode parameter set during initialization.

    This is used internally by the constructor. See build_model() to construct
    model manually.

    Arguments:
      str:model_name (self.model_name) -- Name of model builder method
      **kwargs -- Keyword arguments for model builder
    """
    if self.is_gen():
      self.build_model(True, model_name, **kwargs)
    if self.is_training():
      self.build_model(False, model_name, **kwargs)

  def build_check(self):
    """
    Builds needed models if they do not yet exist.

    Used for weight-loading.
    """
    if self.is_gen() and not self.gmodel:
      self.build_model(True)
    if self.is_training() and not self.tmodel:
      self.build_model(False)

  def build_model(self, stateful=True, model_name=None, **kwargs):
    """
    Build gen model using model builder method stored in models.py.

    Called by build_models() to build training or gen models based on value
    of self.mode.

    Keyword arguments:
      bool:training -- If True build training model; gen model otherwise
      str:model_name (self.model_name) -- Name of model builder method
      **kwargs -- Keyword arguments for model builder
    """
    model_name = model_name or self.model_name

    try:
      model_builder = getattr(models, model_name)
    except AttributeError:
      raise ScNetError('Invalid model builder provided!')
      return False

    # This system does not, admittedly, make much sense in retrospect
    # We are doing things this way such that the model builders can be used
    # as standalone functions, when convenient to do so.

    # Required kwargs
    kwargs = self.merge_kwargs(kwargs)
    kwargs['stateful'] = kwargs.get('stateful') or stateful
    kwargs['bins'] = kwargs.get('bins') or self.bins
    kwargs['channels'] = kwargs.get('channels') or self.channels
    kwargs['loss'] = kwargs.get('loss') or self.loss
    kwargs['opt'] = kwargs.get('opt') or self.opt

    if stateful:
      kwargs['t'] = kwargs.get('t') or 1
      self.gmodel = model_builder(**kwargs)
    else:
      kwargs['t'] = kwargs.get('t') or self.t
      self.tmodel = model_builder(**kwargs)

    return True

  # --- TRAINING ---

  def train(self, batch_size=None, epochs=None, save=True):
    """
    Train model.

    Keyword arguments:
      int:batch_size (cf.net.batch_size) -- Training batch size
      int:epochs (cf.net.training_epochs) -- Training epochs
      bool:save (True) -- Whether to save after completion of training (note
        that this is independent of save points specified in the configuration)
    """
    if not self.is_training():
      raise ScNetError('Not a training model!')

    batch_size = batch_size or self.batch_size
    epochs = epochs or self.training_epochs

    x, y, xx, yy = tuple(self.data)
    callback = NetCallback(self)

    h = self.tmodel.fit(
      x, y,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(xx,yy),
      callbacks=[callback])

    if self.autosave:
      if not self.is_save_epoch():
        self.save_history()
        self.save_weights()
      if self.save_m:
        self.save_model()

    if self.is_gen() and self.gmodel is not None:
      self.gmodel.set_weights(self.tmodel.get_weights())

  # --- HISTORY AND SAVING ---

  def is_save_epoch(self):
    """
    Returns true if current epoch is regular autosave point.
    """
    return \
      self.epochs_per_save != 0 and \
      self.epochs % self.epochs_per_save == 0


  def is_archive_epoch(self):
    """
    Returns true if current epoch is archive autosave point.
    """
    return \
      self.epochs_per_archive != 0 and \
      self.epochs % self.epochs_per_archive == 0

  def increment_history(self, logs={}):
    """
    Increment net attributes related to training history after each epoch.

    For internal use by training callbacks.
    """
    # Increment counters
    self.epochs += 1

    # Append logs
    vals = []
    vals.append(
      logs.get('mean_squared_error') or
      logs.get('mean_absolute_error') or
      logs.get('loss'))
    if logs.get('val_loss'):
      vals.append(logs['val_loss'])
    vals = [float(e) for e in vals]
    self.history.append(vals)

    # Save if necessary
    if self.autosave:
      if self.is_save_epoch():
        self.save_history()
        self.save_weights()
      if self.is_archive_epoch():
        self.save_weights(archive=True)

  def save_history(self,  filepath=None):
    """
    Save self.history to NumPy file.

    Arguments:
      str:filepath (self.model_base_path) -- Full path of save location
    """
    filepath = filepath or self.model_base_path

    try:
      np.save(filepath, self.history)
    except IOError:
      raise ScNetError('Error writing history file "{0}"'.format(filepath))

  def load_history(self, filepath=None):
    """
    Load .npy history file. Must be valid NumPy file, ideally in same format
    as dictated by current loss/metrics configuration.

    Arguments:
      str:filepath (self.model_base_path) -- Full path of file to load
    """
    filepath = filepath or self.model_base_path + '.npy'
    try:
      self.history = np.load(filepath).tolist()
      self.epochs = len(self.history) + self.base_epochs
    except IOError:
      raise ScNetError(
        'Error reading history file "{0}"'.format(self.model_base_path))

  def clear_history(self):
    """
    Clears history object and save-point trackers.
    """

    self.history = []
    self.epochs = 0

  def get_weights(self):
    """
    Get model weights.

    Returns weights from training model if available,
    otherwise gen model. Returns None if no models are built.
    """
    if self.tmodel:
      return self.tmodel.get_weights()
    elif self.gmodel:
      return self.gmodel.get_weights()
    else:
      return None

  def set_weights(self, weights):
    """
    Set model weights to existing weight array.

    list:weights -- Weights to set
    """
    if self.tmodel:
      self.tmodel.set_weights(weights)
    if self.gmodel:
      self.gmodel.set_weights(weights)

  def save_weights(self, archive=False):
    """
    Save model weights.

    Only saves training weights. If you need to save gen weights, do so
    manually with model.save_weights() method.

    Keyword arguments:
      bool:archive (False) -- Archive saves include cur_epoch value in filename,
      such that they are not overwritten
    """
    if not self.tmodel:
      raise ScNetError('No model to save!')

    try:
      if archive:
        filepath =  '{0}-{1}.w'.format(self.model_base_path, self.epochs)
      else:
        filepath = self.model_base_path + '.w'
      self.tmodel.save_weights(filepath)

      sclog('Saved weights to "{0}" at {1} epochs.'.format(filepath,
        self.epochs))
    except IOError:
      raise ScNetError('Error writing weights file. Possibly bad base path.')

  def load_weights(self, filepath=None):
    """
    Load weights.

    Keyword arguments:
      str:filepath (self.model_base_path) -- Full path of weights file
    """
    filepath = filepath or self.model_base_path + '.w'
    self.build_check()

    try:
      if self.is_training() and self.tmodel is not None:
        self.tmodel.load_weights(filepath)
        sclog('Loaded tmodel "{0}."'.format(filepath))
      if self.is_gen() and self.gmodel is not None:
        self.gmodel.load_weights(filepath)
        sclog('Loaded gmodel "{0}."'.format(filepath))
    except IOError:
      raise ScNetError('Error reading weights file "{0}"'.format(filepath))
    except Exception:
      raise ScNetError(
        'File "{0}" might contains wrong weights.'.format(filepath))

  def save_model(self):
    """
    Save full model (including optimizer state).

    This takes considerably more disk space than saving the weights alone,
    but allows one to resume training using the existing optimizer state.
    Among other things, this will prevent "blips" in the loss numbers.

    Saving full model state does not reset epochs_per_save, which is only
    used in relation to the slightly more baroque weight-saving regime;
    by default full model saves only occur after a complete training cycle.

    Only applies to training model.
    """
    if not self.tmodel:
      raise ScNetError('No model to save!')

    try:
      filepath = self.model_base_path + '.m'
      self.tmodel.save(filepath)
      sclog('Saved model to "{0}" at {1} epochs.'.format(filepath, self.epochs))
    except IOError:
      raise ScNetError('Error writing model file. Possibly bad base path.')

  def load_model(self, filepath=None):
    """
    Load full model.

    Only applies to training model.

    Keyword arguments:
      str:filepath (self.model_base_path) -- Path of model file
    """
    filepath = filepath or self.model_base_path + '.m'
    try:
      if self.is_training():
        self.tmodel = keras.models.load_model(filepath)
        sclog('Loaded model "{0}."'.format(filepath))
    except IOError:
      raise ScNetError('Error reading model file "{0}"'.format(filepath))

  def load(self):
    """
    Load full state automagically based on contents of model folder.

    Uses paths specified in self.model_dir, self.model_base, and
    self.model_base_path.

    See load_history(), load_weights(), and load_model() for finer-grained
    control over state loading, with specified filenames, etc.
    """

    # Load history
    if os.path.isfile(self.model_base_path + '.npy'):
      self.load_history()
    # Load full model file if available
    if self.is_training() and os.path.isfile(self.model_base_path + '.m'):
      self.load_model()
    # Elif load weights file
    elif os.path.isfile(self.model_base_path + '.w'):
      self.load_weights()
    # Else let's  see what other weight files are lying around
    else:
      mfiles = os.listdir(self.model_dir)
      mfiles = [e for e in mfiles if e[-2:] == '.w']
      mfiles = [e for e in mfiles if e[0] != '.']
      mfiles = [os.path.join(self.model_dir, e) for e in mfiles]
      mfiles.sort()

      # TODO: Fix this for names that start the same
      files = [e for e in mfiles if utils.index(e, self.model_base) == 0]
      if len(files) > 0:
        # Last file should have highest training numbers
        self.load_weights(files[-1])
      # This does not seem like a good idea at present.
      # elif len(mfiles) > 0: # Load whatever the last model file is
      #   self.load_weights(mfiles[-1])
      else: # Nothing to load!
        self.clear_history()
        return False
    return True

  # --- GENERATION ---

  def quickseed(self, steps=None):
    """
    Generate seed data for gen method.

    If self.random_seed == true, generates seed with normally distributed
    random values. Otherwise, selects random subsequence of validation data.

    For finer-grained control over seed data, gen() can be executed from the
    interactive shell.

    Arguments:
      int:steps (None) -- Length of seed sequence in timesteps
    """
    steps = steps if steps else self.seed_steps

    vdata = self.data[2]

    if not (self.random_seed or vdata is None):
      # Use random vdata if available, and random_seed == false
      r0 = np.random.randint(vdata.shape[0])
      r1 = np.random.randint(vdata.shape[1] - steps)
      seed = vdata[r0, r1:r1+steps]
      sclog('Generated vdata seed at [{0},{1}].'.format(r0, r1))
    else:
      # Fall back if vdata data not available, or if random_seed == true
      seed = np.random.rand(steps, self.bins, 2)
      seed = seed * 6 - 3
      win = np.hamming(int(self.bins * 1.5))[-self.bins:]
      win = win.reshape(1, len(win), 1)
      seed = seed * win
    return seed

  def gen(self, steps=None, seed=None, reset=True):
    """
    Generates output sequence.

    Arguments:
      int:steps (self.gen_steps) -- Steps to generate
      ndarray:seed (None) -- Specific seed to use, if any - otherwise calls
        quickseed() method
      bool:reset (True) -- Whether to reset recurrent model state before
        generating - set to false when generating longer music incrementally

    Returns generated sequence.
    """
    seed = seed if type(seed) == np.ndarray else self.quickseed()
    steps = steps or self.gen_steps

    return Net.generate(
      self.gmodel,
      seed,
      steps=steps,
      input_map=self.input_map,
      output_map=self.output_map,
      input_channels=self.input_channels,
      output_channels=self.output_channels
      )

  # --- STATIC METHODS ---

  @staticmethod
  def generate(
      model,
      seed,
      steps=cf.output.gen_steps,
      input_map=lambda x: x,
      output_map=lambda x: x,
      input_channels=1,
      output_channels=1,
      reset=True):
    """
    Static method to generate output sequence.

    Much of this is written to accommodate multi-channel input and output,
    though this is not widely supported by the current generation processes.
    It may still be useful for multi-channel models when used via the
    interactive shell, though the seed data must be provided manually in
    this case.

    This is a static method that can be run on a standalone model, without
    a Net instance.

    Keyword attributes:
      ndarray:seed (None) -- Specific seed to use, if any - otherwise calls
        quickseed() method
      bool:reset (True) -- Whether to reset recurrent model state before
        generating - set to false when generating longer music incrementally
    """

    if input_channels == 1 and len(seed) != 1:
      seed = [seed]
    seed_length = len(seed[0])

    if reset:
      model.reset_states()
    g = [[] for i in range(output_channels)]

    for i in range(0, seed_length):
      s = [e[i:i+1] for e in seed]
      s = s[0] if input_channels == 1 else s
      p = model.predict(s, batch_size=1)
      p = [p] if output_channels == 1 else p
      for i, e in enumerate(p):
        g[i].append(e[0])
    for i in range(seed_length, steps):
      s = input_map(np.array(g)[:,-1:])
      s = s[0] if input_channels == 1 else s
      p = model.predict(s, batch_size=1)
      p = [p] if output_channels == 1 else p
      for i, e in enumerate(p):
        g[i].append(output_map(e[0]))

    return np.array(g[0]) if output_channels == 1 \
      else tuple([np.array(e) for e in g if e])

# --- SUPPORT CLASSES ---

class NetCallback(Callback):
  """
  Callback to retrieve training metrics, update save attributes when training.
  """
  def __init__(self, net):
    self.net = net
    super(NetCallback, self).__init__()

  def on_epoch_end(self, epoch, logs={}):
    # Increment history
    self.net.increment_history(logs)