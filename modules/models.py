import numpy as np

"""
models.py

Consists of model building functions that can be used by Net class, or as
standalone Keras model-builders.

Add functions to this file in the build custom models.
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

from config import config as cf
import utils as utils

# --- MODELS ---

def default(stateful=True, t=None, bins=cf.net.bins, channels=2, **kwargs):
  """
  This is the default Soundcube model.

  Consists of a fully-connected feature layer followed by an LSTM layer,
  followed by two distinct LSTM layers, each with a fully-connected output
  layer corresponding to the final output size.

  One of these output layers is given a relu activation, producing output
  values >= 0, while the other is given a tanh activation, producting output
  in the range (-1, 1). The final output is the element-wise product
  of these two output vectors.

  It is not entirely clear why, but this seems to produce decidedly better
  output than similarly-sized "vanilla" LSTM networks without this final
  product between two differently-activated branches.
  """

  # General params
  t = t or 1 if stateful else cf.data.hops_per_block - 1
  loss = kwargs.get('loss') or cf.net.loss
  opt = kwargs.get('opt') or cf.net.opt
  metrics = kwargs.get('metrics') or []
  # Model-specific params
  sizes = kwargs.get('sizes') or [cf.net.default_hidden_size] * 3
  noise = kwargs.get('noise') or 0.1
  normalize = kwargs.get('normalize') or True
  w_dropout = kwargs.get('w_dropout') or (0.5, 0.5, 0)

  main_in = m = __input__(stateful, t, bins, channels)

  if normalize:
    m = Lambda(lambda x: x / (K.cast(K.std(x, -2, True), 'float32') + 1e-8))(m)
    m = Lambda(lambda x: x - K.mean(x, -2, True))(m)
  if noise:
    m = GaussianNoise(noise)(m)

  m = Reshape((t, bins * 2))(m)

  if sizes[0]:
    m = TimeDistributed(Dense(sizes[0], init='he_uniform'))(m)
    m = Dropout(w_dropout[0])(m)

  m = LSTM(sizes[1], stateful=stateful, return_sequences=True)(m)

  b0 = b1 = Dropout(w_dropout[1])(m)
  b0 = LSTM(sizes[2], stateful=stateful, return_sequences=True)(b0)
  b0 = TimeDistributed(Dense(bins * 2, init='he_uniform', activation='relu'))(b0)
  b0 = Dropout(w_dropout[2])(b0)

  b1 = LSTM(sizes[2], stateful=stateful, return_sequences=True)(b1)
  b1 = TimeDistributed(Dense(bins * 2, init='he_uniform', activation='tanh'))(b1)
  b1 = Dropout(w_dropout[2])(b1)

  m = merge([b0, b1], mode='mul')

  if stateful:
    main_out = Reshape((bins, 2))(m)
  else:
    main_out = Reshape((t, bins, 2))(m)

  model = Model(input=main_in, output=main_out)
  model.compile(loss=loss, optimizer=opt, metrics=metrics)

  return model

def simple(stateful=True, t=None, bins=cf.net.bins, channels=2, **kwargs):
  """
  Simple model.

  This is a more "traditional" model featuring only a the initial fully-
  connected layer, the LSTM layer, and the fully-connected output layer.
  """

  # General params
  t = t or 1 if stateful else cf.data.hops_per_block - 1
  loss = kwargs.get('loss') or cf.net.loss
  opt = kwargs.get('opt') or cf.net.opt
  metrics = kwargs.get('metrics') or []
  # Model-specific params
  size = kwargs.get('size') or (hsize, hsize)
  layers = kwargs.get('layers') or 1
  noise = kwargs.get('noise') or 0.1
  dropout = kwargs.get('dropout') or 0.5

  main_in = m = __input__(stateful, t, bins, channels)

  m = Lambda(lambda x: x / (K.cast(K.std(x, -2, True), 'float32') + 1e-8))(m)
  m = Lambda(lambda x: x - K.mean(x, -2, True))(m)
  m = GaussianNoise(noise)(m)

  m = Reshape((t, bins * 2))(m)

  if size[0]:
    m = TimeDistributed(Dense(size[0], init='he_uniform'))(m)
  for i in range(layers):
    m = Dropout(dropout)(m)
    m = LSTM(size[1], stateful=stateful, return_sequences=True)(m)
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