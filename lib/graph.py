"""
graph.py

Loads Soundcube graphing tools.

Requires matplotlib.

This is a work in progress, mainly provided for debugging purposes, etc., and
should not yet be considered part of the core program functionality.

The main purpose for this is to perform rough, cursory analysis of generated
output by examining the general pattern of frequency bin magnitudes over time.
These graphs are not particularly expressive of meaningful musical features,
etc. --- they are just a technique to see, at a glance, whether your output
even makes sense.
"""

from scerror import *
from sclog import sclog
import utils

try:
  import matplotlib
  import matplotlib.pyplot as pp
  from matplotlib import patches
except ImportError:
  raise ScError('Unable to import matplotlib. Try installing it.')

try:
  from matplotlib.backends.backend_pdf import PdfPages as pdf
except ImportError:
  sclog('Unable to import PdfPages.')
  pdf = False

plot = pp.plot
show = pp.show
clf = pp.clf
cla = pp.cla

# These are all fairly arbitrary and need to be cleaned up quite a bit.

def pfreq(s, pol=False, figsize=(8,6), ymax=None):
  """
  Plot real part of a generation sample for each timestep.

  Arguments:
    ndarray:s -- Frequency timestep series
    bool:pol (False) -- Graph polar magnitude instead
    tuple:figsize (8,6) -- PyPlot figsize
    int:ymax (None) -- Y range of plot
  """
  pp.close()
  pp.figure(figsize=figsize)
  ymax = ymax or max(s.max(), abs(s.min()))

  ymax = ymax or get_ymax(s)

  pp.xlabel('Timesteps')
  pp.grid(True)
  if pol:
    pp.axis([0, len(s), 0, ymax])
    polar = utils.com2pol(utils.dechannel(a))
    _= pp.plot(polar[:,:,0])
  else:
    pp.axis([0, len(s), -ymax, ymax])
    _=pp.plot(s[:,:,0])
  pp.tight_layout()

def pfreq_grid(slist, cols=1, figsize=(8,8), ymax=None):
  """
  Plot real part of multiple generation samples in grid layout

  Arguments:
    sequence:slist -- List of ndarray timestep series
    int:cols (1) -- Columns in grid
    tuple:figsize (8,8) -- PyPlot figsize
    int:ymax (None) -- Y range of plot
  """
  fig = pp.figure(figsize=figsize, frameon=False)
  num = len(slist)
  lng = len(slist[0])

  ymax = ymax or get_ymax(slist)

  for i in range(num):
    fig.add_subplot(num/cols, cols, i+1)
    pp.grid(True)
    pp.axis([0, lng, -ymax, ymax])
    pp.gca().axes.get_yaxis().set_visible(False)
    _= pp.plot(slist[i][:,:,0])
  pp.tight_layout()

def get_ymax(arg):
  """
  Returns upwards-rounded ymax.
  """
  slist = [arg] if len(arg) == 1 else arg

  ymax = 0;
  for s in slist:
    ymax = max(abs(s).max(), ymax)

  if ymax < 10:
    ymax = npp.ceil(ymax)
  elif ymax < 50:
    ymax = ymax + 5 - (ymax % 5)
  elif ymax < 200:
    ymax = ymax + 10 - ((ymax + 5) % 10)
  else:
    ymax = ymax + 30 - ((ymax + 5) % 25)
  return ymax

# --- SAVE ---

def save(path):
  """
  Wrapper for saving plot using PdfPages.

  Returns True if successful, false otherwise.
  """
  if not pdf:
    return False
  page = pdf(path)
  page.savefig()
  page.close()
  pp.close()
  return True