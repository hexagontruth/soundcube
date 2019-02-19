import numpy as np

"""
models.py

Consists of model building functions that can be used by Net class, or as
standalone Keras model-builders.

Add functions to this file in the build custom models. A model-building
function should take the following form:

def my_model_builder(bool:stateful, int:t, int:bins, int:channels, **kwargs)

and should return the compiled model. Note t ought to depend on statefulness,
so should probably default to None, as seen below.
"""

import keras

import keras.backend as K

from keras.layers import *
from keras.layers.convolutional import *
from keras.layers.core import *
from keras.layers.local import *
from keras.layers.pooling import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.models import *
from keras.regularizers import *

from .config import config as cf
from . import utils as utils

# Defaults

default_t = cf.data.hops_per_second * cf.data.seconds_per_block - 1
default_bins = cf.data.step_length // 2
default_loss = cf.net.loss
default_opt = cf.net.opt
default_metrics = cf.net.metrics

default_hidden_size = cf.net.kwargs.get('hidden_size') or 2048
default_dropout = cf.net.kwargs.get('dropout') or 0.5

# --- MODELS ---

def default(stateful=True, t=None, bins=default_bins, channels=2, 
    loss=default_loss, opt=default_opt, metrics=default_metrics,
    noise = 0.1, normalize=True, **kwargs):
  """
  This is the default Soundcube model.

  Consists of a fully-connected feature layer followed by an LSTM layer,
  followed by two distinct LSTM layers, each with a fully-connected output
  layer corresponding to the final output size.

  One of these output layers is given a relu activation, producing output
  values >= 0, while the other is given a tanh activation, producting output
  in the range (-1, 1). The final output is the element-wise product
  of these two output vectors.

  Arguments:
    bool:stateful (True) -- Use stateful networks for training
    int:t (None) -- Timestep size; can be inferred from cf.data.hops_per_block
    int:bins (cf.net.bins) -- Bins
    int:channels (2) -- Channels per bin; change at your own risk

  Keyword Arguments:
    int:hidden_size -- Size for all hidden layers; inferred from config
    sequence:sizes -- Exactly 3 integer sizes for hidden layers; overrides
      "hidden_size"
    float:dropout -- Dropout to w_dropout params; inferred from config
    sequence:w_dropout -- Exactly 3 dropous. Again overrides "dropout"
    float:noise (0.1) -- Gaussian noise factor for training
    bool:normalize (True) -- Whether to preform timestep-wise normalization
  """

  # Set t
  t = t or (1 if stateful else default_t)

  # Model-specific inferred args
  if kwargs.get('sizes'):
    sizes = kwargs['sizes']
  elif kwargs.get('hidden_size'):
    sizes = [kwargs['hidden_size']] * 3
  else:
    sizes = [default_hidden_size] * 3

  if kwargs.get('w_dropout'):
    w_dropout = kwargs.get('w_dropout')
  elif kwargs.get('dropout'):
    w_dropout = [kwargs['dropout']] * 2 + [0]
  else:
    w_dropout = [default_dropout] * 3

  # Build model

  main_in = m = __input__(stateful, t, bins, channels)

  if normalize:
    m = Lambda(lambda x: x / (K.cast(K.std(x, -2, True), 'float32') + 1e-8))(m)
    m = Lambda(lambda x: x - K.mean(x, -2, True))(m)
  if noise:
    m = GaussianNoise(noise)(m)

  m = Reshape((t, bins * 2))(m)

  if sizes[0]:
    m = TimeDistributed(Dense(sizes[0], kernel_initializer='he_uniform'))(m)
    m = Dropout(w_dropout[0])(m)

  m = LSTM(sizes[1], stateful=stateful, return_sequences=True)(m)

  b0 = b1 = Dropout(w_dropout[1])(m)
  b0 = LSTM(sizes[2], stateful=stateful, return_sequences=True)(b0)
  b0 = TimeDistributed(Dense(bins * 2, kernel_initializer='he_uniform', activation='relu'))(b0)
  b0 = Dropout(w_dropout[2])(b0)

  b1 = LSTM(sizes[2], stateful=stateful, return_sequences=True)(b1)
  b1 = TimeDistributed(Dense(bins * 2, kernel_initializer='he_uniform', activation='tanh'))(b1)
  b1 = Dropout(w_dropout[2])(b1)

  m = Multiply()([b0, b1])

  if stateful:
    main_out = Reshape((bins, 2))(m)
  else:
    main_out = Reshape((t, bins, 2))(m)

  model = Model(inputs=main_in, outputs=main_out)
  model.compile(loss=loss, optimizer=opt, metrics=metrics)

  return model

def simple(stateful=True, t=None, bins=default_bins, channels=2, 
    loss=default_loss, opt=default_opt, metrics=default_metrics,
    layers=1, noise=0.1, normalize=True, dropout=default_dropout, **kwargs):
  """
  Simple model.

  This is a more "traditional" model featuring only a the initial fully-
  connected layer, the LSTM layer, and the fully-connected output layer.

  Arguments:
    bool:stateful (True) -- Use stateful networks for training
    int:t (None) -- Timestep size; can be inferred from cf.data.hops_per_block
    int:bins (cf.net.bins) -- Bins
    int:channels (2) -- Channels per bin; change at your own risk

  Keyword Arguments:
    int:hidden_size -- Size for both hidden layers; inferred from config
    sequence:sizes -- Exactly 2 integer sizes for hidden layers; overrides
      "hidden_size"
    float:dropout -- Dropout to w_dropout params; inferred from config
    float:noise (0.1) -- Gaussian noise factor for training
    bool:normalize (True) -- Whether to preform timestep-wise normalization
  """

  # General params
  t = t or 1 if stateful else default_t

  # Model-specific params
  if kwargs.get('sizes'):
    sizes = kwargs['sizes']
  elif kwargs.get('hidden_size'):
    sizes = [kwargs['hidden_size']] * 2
  else:
    sizes = [default_hidden_size] * 2

  main_in = m = __input__(stateful, t, bins, channels)

  if normalize:
    m = Lambda(lambda x: x / (K.cast(K.std(x, -2, True), 'float32') + 1e-8))(m)
    m = Lambda(lambda x: x - K.mean(x, -2, True))(m)
  if noise > 0:
    m = GaussianNoise(noise)(m)

  m = Reshape((t, bins * 2))(m)

  if sizes[0]:
    m = TimeDistributed(Dense(sizes[0], kernel_initializer='he_uniform'))(m)
  for i in range(layers):
    m = Dropout(dropout)(m)
    m = LSTM(sizes[1], stateful=stateful, return_sequences=True)(m)
  m = TimeDistributed(Dense(bins * 2))(m)

  if stateful:
    main_out = Reshape((bins, 2))(m)
  else:
    main_out = Reshape((t, bins, 2))(m)

  model = Model(input=main_in, output=main_out)
  model.compile(loss=loss, optimizer=opt, metrics=metrics)

  return model


# --- UTILITY FUNCTIONS ---

def __input__(stateful, t, bins, channels=2):
  """
  Build input shape based on provided parameters.

  Training and gen models use different input shapes.

  Arguments:
    bool:stateful -- Determines 
    int:t -- Timesteps per block; hops_per_block - 1 for training, 1 for gen
    int:bins -- Nonflattened number of bins
    int:channels (2) -- Channels per bin; 2 for complex/polar format

  """
  shape = [t, bins, channels]

  if stateful:
    main_in = Input(batch_shape=shape)
  else:
    main_in = Input(shape=shape)
  return main_in