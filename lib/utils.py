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
import subprocess

import numpy as np
import scipy.io.wavfile as wav

from .config import config as cf
from .sclog import sclog
from .scerror import *

# There are a number of presently-unused functions that I have left here in case
# they may be of value to those wishing to expand the functionality of the
# program. The unused functions have *not been thoroughly tested*, and certainly
# have not been tested with the current training and generation apparatus. Use
# at your own risk.

# Some convenient local configuration variables
source_dir = cf.data.source_dir
target_dir = cf.data.target_dir

bit_depth = cf.data.bit_depth
sr = cf.data.sample_rate
hop_length = cf.data.hop_length
step_length = cf.data.step_length
hops_per_second = cf.data.hops_per_second
seconds_per_block = cf.data.seconds_per_block
hops_per_block = hops_per_second * seconds_per_block

float_type = cf.data.float_type
float_storage = cf.data.float_storage
complex_type = cf.data.complex_type

source_formats = cf.data.source_formats
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
  
  Returns number of files converted (excluding directories).
  """
  k = 0

  base_dir = base_dir or source_dir
  for filename in os.listdir(source_dir):
    match = re.search("\.(\w+)$", filename)
    if (match and match.group(1) in source_formats):
      file2wav(source_dir + '/' + filename, base_dir, target_dir, sr)
      k += 1
    elif (os.path.isdir(source_dir + '/' + filename)):
      k += convert_source_dir(
        source_dir + '/' + filename, target_dir, base_dir, sr)
  return k

def convert_wavs_to_freqs(target_dir=target_dir, step_length=step_length,
    hop_length=None):
  """
  Converts wavs in data dir to .npy files of frequency timesteps

  Arguments:
    str:target_dir (cf.data.target_dir) -- Target dir
    int:step_length (cf.data.step_length) -- Window (timestep) length
    int:hop_length (None) -- Not implemented; defaults to step_length

  Returns number of files converted.  
  """
  k = 0

  hop_length = hop_length or step_length
  for filename in os.listdir(target_dir):
    if (filename[-4:] == '.wav'):
      filepath = os.path.join(target_dir, filename)
      if hop_length == step_length:
        freq = wav2freq(filepath)
        k += 1
      else:
        raise ScNotImplementedError('Change hop_length to match step_length.')
      save_array(filepath[:-4], freq)
  return k

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
  with open(os.devnull, 'w') as devnull:
    subprocess.call(command, shell=True, stdout=devnull, stderr=devnull)

# --- DATA IO & CONVERSION ---

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
    s = (s * (2 ** bit_depth / 2 - 1)).astype(get_int_type())
  wav.write(filename, sr, s.flatten())

def write_output(filename, array, process=True):
  """
  Convenience method for writing wav from timestep series.

  See freq2time(), postprocess(), and write_wav().

  Arguments:
    str:filename -- Filename to save
    ndarray:array -- Frequency timestep array to convert and save
    bool:postprocess (True) -- Whether to apply postprocessing before saving
  """
  w = freq2time(array)
  if (postprocess):
    w = postprocess(w)
  write_wav(filename, w)

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

def save_array(path, array):
  """
  Save arbitrary ndarray to configured dtype.

  Arguments:
    str:path -- Path
    ndarray:array -- Array
  """
  if np.iscomplexobj(array):
    np.save(path, array.astype(complex_type))
  else:
    np.save(path, array.astype(float_storage))

def wav2freq(filename):
  """
  Convenience method for reading in a wav and converting it to blocks.

  See read_wav() and time2freq() for details.

  Arguments:
    str:filename -- Wav file to load

  Returns timestep series
  """
  return time2freq(read_wav(filename))

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
  Converts time domain waveform to frequency domain timestep series.

  Returns timestep series
  """
  trim = len(array) % step_length
  array = array[:-trim]
  array = array.reshape(-1, step_length)
  freq = np.fft.rfft(array)[:,:-1]
  freq = enchannel(freq).astype(float_type)
  return freq

# --- DATA LOADING ---

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

def load_data(target_dir=target_dir, split=10):
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

def is_data(target_dir=target_dir):
  """
  Check if data directory contains any timestep series files.

  Arguments:
    str:target_dir (cf.data.target_dir) -- Target directory
  """
  files = os.listdir(target_dir)
  files = [e for e in files if e[-4:] == '.npy']
  return len(files) > 0

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

# --- SIGNAL PROCESSING ---

def postprocess(s):
  """
  Apply all configured postprocessing of audio signal.

  Presently this is just a wrapper for fade(), but it will be expanded in the
  future.

  Arguments:
    str:s -- Audio time series to process

  Returns processed waveform.
  """
  y = fade(s)
  return y

def fade(s, fade_in_arg=cf.output.fade_in, fade_out_arg=cf.output.fade_out):
  """
  Conenience function for applying fade-in and fade-out.

  See fade_in() and fade_out().

  Arguments:
    ndarray:s -- Audio time series
    float:fade_in (cf.output.fade_in) -- Fade-in length in seconds
    float:fade_out (cf.output.fade_out) -- Fade-out length in seconds

  Returns faded waveform.
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

  Returns faded waveform. 
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

  Returns faded waveform. 
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

def dechannel(*args):
  """
  Merge real and imaginary channels on last axis into complex form.

  Arguments:
    *args -- ndarrays to merge
  """
  y = []
  if type(args[0]) == tuple:
    args = list(args[0])
  for e in args:
    shape = list(e.shape)
    temp = np.zeros(shape[:-1], dtype=float_type).T
    temp = e.T[0] + 1j*e.T[1]
    y.append(temp.T)
  return tuple(y) if len(y) > 1 else y[0]

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

def to_categorical(array, n, dtype='byte'):
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
  shape = list(array.shape) + [n]
  array = array.flatten().astype(dtype)
  cat = np.zeros(shape, dtype=dtype).reshape(-1, n)
  for i in range(cat.shape[0]):
    cat[i,array[i]] = 1
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

# --- FILE DELETION AND RENAMING ---


def remove(*paths):
  """
  A method for removing arbrary files.

  Originally meant to protect files outside of project directory. This
  functionality is no longer operative, but this is still used as a placeholder
  for arbitrary file operations, and may be expanded in the future.

  Arguments:
    *paths -- Path components of file, merged by os.path.join

  Raises True if success; raises ScError if failure.
  """
  filepath = os.path.realpath(os.path.join(*paths))
  try:
    os.remove(filepath)
    return True
  except OSError:
    raise ScError('Cannot remove "{0}."'.format(filepath))

def rename(filepath, newpath):
  """
  Safely rename file in subdirectory path.

  Again, originally to protect files outside of project directory. This
  functionality is no longer operative, but this is still used as a placeholder
  for file renaming operations, and may be expanded in the future.

  Arguments:
    str:filepath -- Full path of file to be removed
    str:newpath -- New path name

  Raises True if success; raises ScError if failure.
  """
  try:
    os.rename(filepath, newpath)
    return True
  except OSError:
    raise ScError('Cannot rename "{0}."'.format(filepath))
  return True

# --- OTHER UTILITIES ---

def rms(array):
  """
  Returns root mean square of given array.

  Arguments:
    ndarray:s -- Array
  """
  return np.sqrt(np.mean(np.square(array)))

def get_int_type(bit_depth=None, signed=None):
  """
  Returns dtype string based on provided bit depth and signed flag.

  Arguments:
    int:bit_depth (cf.data.bit_depth) -- Bit depth
    bool:signed (cf.data.signed) -- Signed if True; unsigned if False
  """
  bit_depth = bit_depth or cf.data.bit_depth
  signed = signed if signed is not None else cf.data.signed

  schar = '' if signed else 'u'
  return schar + 'int' + str(bit_depth)

def is_int(array):
  """
  Checks if NumPy array is int or float type.

  Arguments:
    ndarray:n -- Array to check
  """
  floats = ['float16', 'float32', 'float64']
  dtype = str(array.dtype)
  return False if dtype in floats else True

def base_dir():
  """
  Convenience method for providing project base path.

  Wrapper for somewhat unreasonably long Python command. This was for file
  safety functions, and is thus not based on CWD. It doesn't really do anything
  at the moment though.

  """

  #TODO: Improve this
  return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def is_subpath(path):
  """
  Check if given path is in project folder.

  Used internally as sanity-check before deleting files.

  Arguments:
    str:path -- Path
  """
  base = os.path.realpath(base_dir())
  path = os.path.realpath(path)
  try:
    val = True if (path != base and path.index(base) == 0) else False
  except ValueError:
    val = False
  return val

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