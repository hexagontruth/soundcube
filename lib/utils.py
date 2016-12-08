"""
utils.py

Various & sundry utility methods, mostly related to dataset conversion and
loading, signal processing, etc. The rationale for providing these functions
here rather than as part of the Net class is to facilitate the convenient
manipulation of data in the interactive shell, particularly when using
NN models different from the default, that may require different and varied
training data formats, etc.

Many of these methods are useful only when training or generating through the
interactive shell, and are thus not used directly by the CLI scripts or core
classes.
"""
import os
import re
import sys

import numpy as np
import scipy.io.wavfile as wav

from config import config as cf
from scerror import *

# There are a number of presently-unused functions that I have left here in case
# they may be of value to those wishing to expand the functionality of the
# program. The unused functions have *not been thoroughly tested*, and certainly
# have not been tested with the current training and generation apparatus. Use
# at your own risk.

# Some convenient local configuration variables
source_dir = cf.data.source_dir
target_dir = cf.data.target_dir
source_formats = cf.data.source_formats
sr = cf.data.sample_rate
bit_depth = cf.data.bit_depth
signed = cf.data.signed
hop_length = cf.data.hop_length
step_length = cf.data.step_length
hops_per_second = cf.data.hops_per_second
seconds_per_block = cf.data.seconds_per_block
hops_per_block = hops_per_second * seconds_per_block
float_type = cf.data.float_type
float_storage = cf.data.float_storage
complex_type = cf.data.complex_type

convert_cmd = cf.data.convert_cmd

# --- DIRECTORY CONVERSION ---

def convert_source_dir(source_dir=source_dir, target_dir=target_dir,
    base_dir=None, sr=sr):
  """
  Convert audio in source dir to formatted wavs.

  Arguments:
    str:source_dir (cf.data.source_dir) -- Source dir
    str:target_dir (cf.data.target_dir) -- Target dir
    str:base_dir (None) -- Base dir for recursion; defaults to source dir
    int:sr (cf.data.sample_rate) -- Conversion sample rate
  """
  base_dir = base_dir or source_dir
  for filename in os.listdir(source_dir):
    match = re.search("\.(\w+)$", filename)
    if (match and match.group(1) in source_formats):
      file2wav(source_dir + '/' + filename, base_dir, target_dir, sr)
    elif (os.path.isdir(source_dir + '/' + filename)):
      convert_source_dir(source_dir + '/' + filename, target_dir, base_dir, sr)

def convert_wavs_to_freq(target_dir=target_dir, step_length=step_length,
    hop_length=None):
  """
  Converts wavs in data dir to .npy files of frequency timesteps

  Arguments:
    str:target_dir (cf.data.target_dir) -- Target dir
    int:step_length (cf.data.step_length) -- Window (timestep) length
    int:hop_length (None) -- Not implemented; defaults to step_length
  """
  hop_length = hop_length or step_length
  for filename in os.listdir(target_dir):
    if (filename[-4:] == '.wav'):
      filepath = target_dir + '/' + filename
      array = read_wav(filepath)
      if hop_length == step_length:
        diff = len(array) % step_length
        if diff != 0:
          array = array[:-(len(array) % step_length)]
        array = array.reshape(-1, step_length)
        ft = np.fft.rfft(array)[:,:-1]
      else:
        raise ScNotImplementedError('Not yet implemented.')
      ft = enchannel(ft)
      #ft = ft / pol_std
      np.save(filepath[:-4], ft.astype(float_type))

def file2wav(infile, base_dir=source_dir, target_dir=target_dir, sr=sr):
  """
  Internal method for converting files to configured wav format.

  This function will save the converted file in the given location using
  FFmpeg, or other conversion tool as specified in config file.

  Arguments:
    str:infile -- Full path of file to convert
    str:base_dir (cf.data.source_dir) -- Base source dir (for file renaming)
    str:target_dir (cf.data.target_dir) -- Target dir
    int:sr (cf.data.sample_rate) -- Sample rate
  """
  command = convert_cmd
  outfile = infile.replace(base_dir, '')[1:].replace('/', '_')
  outfile = re.sub('\s+', '_', outfile)
  outfile = re.sub('[\_\-]+', '_', outfile)
  outfile = re.sub('\.\w+$', '', outfile)
  outfile = re.sub('[\W]+', '', outfile) + '.wav'
  outfile = target_dir + '/' + outfile
  outfile = outfile.lower()
  values = {
    'infile': infile,
    'outfile': outfile,
    'rate': str(sr),
    'depth': str(bit_depth),
    'signed': 's' if True else 'u'
  }
  command_vars = re.findall("\%(\w+)\%", command)
  for var in command_vars:
    if (var in values):
      command = command.replace('%{0}%'.format(var), values[var])
  os.system(command)

# --- DATA IO & CONVERSION ---

def load_array(path = None):
  """
  Used for specifically loading loss arrays, but can be used for any .npy
  arrays.

  Defaults to loading model loss file, but can load any array.

  Arguments:
    str:path -- Full path if given; defaults to model base path + '.npy'
  """
  path = path or os.path.join(cf.state.model_dir, cf.sate.model_base, '.npy')
  try:
    array = np.load(path).astype(float_type)
  except IOError:
    raise ScIOError('File "{0}" not found!'.format(path))
  return array

def read_wav(infile):
  """
  Read in a formatted wav file and convert it to float ndarray
  """
  rate, data = wav.read(infile)
  data = data.astype(float_type) / (2 ** bit_depth / 2 - 1)
  return data

def write_wav(filename, s, sr=sr):
  """
  Wrapper for scipy.io.wavfile.write using configured sample rate

  Arguments:
    str:filename -- Filename
    ndarray:s -- Audio time series
    sr:int (cf.audio.sample_rate) -- Sample rate
  """
  if not is_int(s):
    # TODO: Fix this to use config dtype
    s = (s * (2 ** bit_depth / 2 - 1)).astype('int16')
  wav.write(filename, sr, s.flatten())

def wav2freq(infile):
  """
  Convenience method for reading in a wav and converting it to blocks.

  See read_wav() and time2freq() for details.

  Returns timestep series
  """
  return time2freq(read_wav(infile))

def freq2wav(filename, array):
  """
  Convenience method for writing wav from timestep series.

  See freq2time() and write_wav().
  """
  w = freq2time(array)
  write_wav(filename, w)

def freq2time(s):
  """
  Converts timesteps to time domain waveform.

  Arguments:
    ndarray:s -- Timestep series as ndarray
  """
  s = dechannel(s)
  s = add_bin(s)
  s = np.fft.irfft(s).astype(float_type)
  return s.flatten()

def time2freq(array):
  """
  Converts time domain waveform to frequency domain timesteps series.

  Returns timestep series
  """
  trim = len(array) % step_length
  array = array[:-trim]
  array = array.reshape(-1, step_length)
  fd = np.fft.rfft(array)[:,:-1]
  fd = enchannel(fd)
  return fd.astype(float_type)

# --- DATA LOADING ---

def tvsplit(z, split=10):
  """
  Split data into training and validation components.

  Takes timestep series returned by load_blocks() or similar function, and
  splits into training and validation data. the split parameter specifies
  the frequency of array slicing to compose the validation data --- i.e., the
  default value of 10 will take one out of every ten blocks, starting with
  the first one. The remaining blocks are kept as training data.

  Arguments:
    ndarray:z -- Timestep series
    int:split (10) -- Split ratio

  Returns tuple of training, validation data
  """
  v = z[::split]
  t = np.delete(z, np.s_[::split], 0)
  return t, v

def tvbatch(t, v):
  """
  Convenience method for paring x and y data for training and validation.

  Returns tuples pairing, e.g., the offsetting each block by one timestep
  with respect to x and y versions, such that at each timestep t,
  y[t] = x[t + 1].

  Note that here and elsewhere it has been my custom to identify training data
  with the variables x and y, and validation data with xx and yy. This seems
  marginally more parsimonious to me than something like "training_x," etc.

  Returns:
    x, y, xx, yy
  """
  x, y   = t[:,:-1], t[:,1:]
  xx, yy = v[:,:-1], v[:,1:]
  return x, y, xx, yy

def load_all(target_dir=target_dir, split=10):
  """
  Returns 7-member tuple of data.

  See load_blocks(), normalize(), tvsplit() for details.

  Arguments:
    str:target_dir (cf.data.target_dir) -- Target directory
    int:split (10) -- Split ratio

  Returns:
    z, t, v, x, y, xx, yy
  """
  z = load_blocks()
  z = normalize(z)
  t,v = tvsplit(z,split)
  x, y, xx, yy = tvbatch(t,v)
  return z, t, v, x, y, xx, yy

def load_data(target_dir=target_dir, split = 10):
  """
  Convenience method for only loading the training/validation tetrad.

  See load_all() for details.

  Arguments:
    str:target_dir (cf.data.target_dir) -- Target directory
    int:sp

  Returns:
    x, y, xx, yy
  """
  return load_all(target_dir, split)[-4:]

def normalize(s, axis=None):
  """
  Normalize along axis, or by timestep if no axis profided.

  Arguments:
    ndarray:s -- Array of training blocks, or arbitrary ndarray
    int:axis (None) -- Axis to normalize over
  """
  float_upcast = float_type if float_type == 'float64' else 'float32'

  if axis:
    shape = list(s.shape)
    shape[axis] = 1
    s = s / np.std(s.astype(float_upcast) + 1e-8, axis=axis, keepdims=True)
  else:
    shape = list(s.shape)
    s = s.reshape(shape[0], -1)
    s = s / np.std(s.astype(float_upcast) + 1e-8, axis=1, keepdims=True)
    s = s.reshape(shape)
  return s.astype(float_type)

def load_wavs(target_dir=target_dir):
  """
  Load all wavs from data directory into time series training blocks.

  Not used in current implementation, but would be useful for time domain
  networks.

  Arguments:
    str:target_dir (cf.data.target_dir) -- Target dir

  Returns array of wavs arranged into training blocks
  """
  samples_per_block = sr * seconds_per_block
  wavfiles = []
  files = os.listdir(target_dir)
  files.sort()
  for filename in files:
    if filename[-4:] == '.wav':
      wavfile = read_wav(target_dir + '/' + filename)
      for i in range(0, len(wavfile), samples_per_block):
        block = wavfile[i:i+samples_per_block]
        if len(block) == samples_per_block:
          wavfiles.append(block)
  x = np.array(wavfiles)
  return x

def load_blocks(target_dir=target_dir):
  """
  Load frequency timestep data from data directory into training blocks.

  Use this to load training data without partitioning it into training and
  validation sets, etc.

  Arguments:
    str:target_dir (cf.data.target_dir) -- Target dir

  Returns array of frequency timesteps arranged into training blocks
  """
  blockfiles = []
  files = os.listdir(target_dir)
  files.sort()
  for filename in files:
    if filename[-4:] == '.npy':
      blockfile = np.load(target_dir + '/' + filename)
      for i in range(0, len(blockfile), hops_per_block):
        block = blockfile[i:i+hops_per_block]
        if len(block) == hops_per_block:
          blockfiles.append(block)
  x = np.array(blockfiles).astype(float_type)
  return x

# --- SIGNAL PROCESSING ---

def fade(s, fade_in_arg=cf.output.fade_in, fade_out_arg=cf.output.fade_out):
  """
  Conenience function for applying fade-in and fade-out.

  See fade_in() and fade_out().

  Arguments:
    ndarray:s -- Audio time series
    float:fade_in (cf.output.fade_in) -- Fade-in length in seconds
    float:fade_out (cf.output.fade_out) -- Fade-out length in seconds
  """
  s = fade_in(s, fade_in_arg)
  s = fade_out(s, fade_out_arg)
  return s

def fade_in(s, fade=cf.output.fade_in):
  """
  Apply fade-in to waveform time signal.

  Arguments:
    ndarray:s -- Audio time series
    float:fade (cf.output.fade_in) -- Fade-in length in seconds
  """
  length = int(fade * sr)
  shape = [1] * len(s.shape)
  shape[0] = length
  win = np.hanning(length * 2)[:length]
  win = win.reshape(shape)
  if length < len(s):
    s[:length] = s[:length] * win
  return s

def fade_out(s, fade=cf.output.fade_out):
  """
  Apply fade-out to waveform time signal.

  Arguments:
    ndarray:s -- Audio time series
    float:fade (cf.output.fade_out) -- Fade-out length in seconds
  """
  length = int(fade * sr)
  shape = [1] * len(s.shape)
  shape[0] = length
  win = np.hanning(length * 2)[length:]
  win = win.reshape(shape)
  if length < len(s):
    s[-length:] = s[-length:] * win
  return s

# --- SUNDRY TRANSFORMATIONS ---

def com2pol(*args):
  """
  Convert complex to polar form.

  Arguments:
    *args -- ndarrays to convert to polar form

  Returns args in polar form.
  """
  y = []
  if type(args[0]) == tuple:
    args = list(args[0])
  for e in args:
    y.append(np.stack((np.abs(e), np.angle(e)), -1))
  return tuple(y) if len(y) > 1 else y[0]

def pol2com(*args):
  """
  Convert polar to complex form.

  Arguments:
    *args -- ndarrays to convert to standard complex form

  Returns args in complex form.
  """
  y = []
  if type(args[0]) == tuple:
    args = list(args[0])
  for e in args:
    e = e.transpose()
    mag, angle = e[0].transpose(), e[1].transpose()
    y.append(mag * np.cos(angle) + 1j * mag * np.sin(angle))
  return tuple(y) if len(y) > 1 else y[0]

def enchannel(*args):
  """
  Split complex arrays into distinct real and imaginary channels.

  Arguments:
    *args -- ndarrays to split
  """
  y = []
  if type(args[0]) == tuple:
    args = list(args[0])
  for e in args:
    y.append(np.stack((e.real, e.imag), -1))
  return tuple(y) if len(y) > 1 else y[0]

def dechannel(s):
  """
  Merge real and imaginary channels on last axis into complex form.

  Arguments:
    ndarray:s -- ndarray to merge

  """
  shape = list(s.shape)
  y = np.zeros(shape[:-1], dtype=float_type).T
  y = s.T[0] + 1j*s.T[1]
  return y.T

def pairchannels(arg):
  """
  Convenience method for performing certain transformations related to
  multi-channel input and output.

  Not used in current implementation.
  """
  if type(arg) == tuple:
    r = []
    for e in arg:
      r.append(pairchannels(e))
    return tuple(r)
  else:
    return [arg.T[0].T, arg.T[1].T]

def split_sign(x):
  """
  Return tuple of absolute value and sign.

  Arguments:
    ndarray:x -- ndarray to split

  Returns:
    abs(x), sign
  """
  sign = (np.sign(x) * -1 + 1) / 2
  return np.abs(x), sign

def merge_sign(m, s):
  """
  Merges magnitude and sign.

  Inverse of split_sign()

  Arguments:
    ndarray:m -- Magnitide array
    ndarray:s -- Sign array

  Returns merged array
  """
  m, s = tuple(arg)
  s[s<0.5] = -1
  s[s>=0.5] = 1
  return m * -s

def add_bin(s, axis=-1, first=False):
  """
  Convenience method for re-adding stripped FFT bin to final output.

  Additional bin is populated with zeroes. Necessary for reconstituting time
  domain signal.

  Arguments:
    ndarray:s -- Timestep array
    int:axis (-1) -- Axis to add bin
    bool:first (False) -- Add bin to beginning instead of end

  Returns expanded array
  """
  shape = list(s.shape)
  shape[axis] = 1
  concat = [s, np.zeros((shape))]
  if first:
    concat.reverse()
  return np.concatenate(concat, axis=axis)

def to_categorical(s, n, dtype='byte'):
  """
  Expands array of categorical values into n-dimensional feature space.

  This is a slight improvement on a similar method provided with Keras. Not
  used in present implementation.

  Arguments:
    ndarray:s -- Categorical array
    int:n -- Numbr of categorical dimensions
    str:dtype ('byte') -- dtype of returned array

  Returns array with additional n-length final axis
  """
  shape = list(s.shape) + [n]
  s = s.flatten().astype(dtype)
  cat = np.zeros(shape, dtype=dtype).reshape(-1, n)
  for i in range(cat.shape[0]):
    cat[i,s[i]] = 1
  return cat.reshape(shape)

def zip_loss(h, keys=None):
  """
  Convenience method for converting Keras history object into list.

  This is done internally by Net class.

  Arguments:
    keras.callbacks.History:h -- History object
    dict:keys (None) -- Keys to extract
  """
  vals = []
  if keys:
    for e in keys:
      vals.append(h.history[e])
  else:
    if h.history.get('mean_squared_error'):
      vals.append(h.history['mean_squared_error'])
    else:
      vals.append(h.history['loss'])
    if h.history.get('val_loss'):
      vals.append(h.history['val_loss'])
  return zip(*vals)

# --- OTHER UTILITIES ---

def rms(s):
  """
  Returns root mean square of given array.

  Arguments:
    ndarray:s -- Array
  """
  return np.sqrt(np.mean(np.square(s)))

def is_int(n):
  """
  Checks if NumPy array is int or float type.

  Arguments:
    ndarray:n -- Array to check
  """
  floats = ['float16', 'float32', 'float64']
  dtype = str(n.dtype)
  return False if dtype in floats else True

def base_dir():
  """
  Convenience method for providing project base path.

  Wrapper for somewhat unreasonably long Python command.

  This is for file safety functions, and is thus not based on CWD.
  """

  #TODO: Improve this
  return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def is_subdir(path):
  """
  Check if given path is in project folder.

  Used internally as sanity-check before deleting files.

  Arguments:
    str:path -- Path
  """
  base = base_dir()
  path = os.path.realpath(path)
  try:
    val = True if (path != base and path.index(base) == 0) else False
  except ValueError:
    val = False
  return val

def safe_remove(filepath):
  """
  Safely remove file from subdirectory of path.

  To prevent accidental deletion of files outside project subdirectories.

  Arguments:
    str:filepath -- Full path of file to be removed

  Returns True if success, False otherwise
  """
  if is_subdir(os.path.dirname(filepath)):
    try:
      os.remove(filepath)
    except OSError:
      raise ScError('Cannot safely remove "{0}."'.format(filepath))
    return True
  else:
    return False

def safe_rename(filepath, newpath):
  """
  Safely rename file in subdirectory path.

  To prevent accidental renaming of files outside project subdirectories.

  Arguments:
    str:filepath -- Full path of file to be removed
    str:newpath -- New path name

  Returns True if success, False otherwise
  """
  if (is_subdir(os.path.dirname(filepath)) and
    is_subdir(os.path.dirname(newpath))):
    try:
      os.rename(filepath, newpath)
    except OSError:
      raise ScError('Cannot safely rename "{0}."'.format(filepath))
    return True
  else:
    return False

def index(string, substr):
  """
  Convenience method for getting substring index without exception.

  Returns index if substring is found, -1 otherwise
  """
  try:
    n = string.index(substr)
  except ValueError:
    n = -1
  return n

def is_str_int(string):
  """
  Checks if string is int.
  """
  try:
    int(string)
    return True
  except ValueError:
    return False

def is_str_num(string):
  """
  Checks if string is numeric.
  """
  try:
    float(string)
    return True
  except ValueError:
    return False