from __future__ import print_function, division, absolute_import
import os
import sys
import shutil
import base64
import pickle
import itertools
from zipfile import ZipFile, ZIP_DEFLATED
from collections import OrderedDict, Mapping, defaultdict

import numpy as np
from scipy import signal
import soundfile as sf

# ===========================================================================
# Data loader
# ===========================================================================
class DIGITLoader(object):
  """ DIGITLoader """

  def __init__(self, path):
    super(DIGITLoader, self).__init__()
    self._path = path
    all_sr = []
    self._data_map = {}
    self._spk_map = defaultdict(list)
    self._num_map = defaultdict(list)
    for name in os.listdir(path):
      if '.wav' not in name:
        continue
      with open(os.path.join(path, name), 'rb') as f:
        # loading the audio using Soundfile
        # return raw array (ndarray), and sample rate (int)
        y, sr = sf.read(f)
        if y.ndim > 1:
          y = y[:, 0]
        assert y.ndim == 1, name
        all_sr.append(sr)
        name = name.replace('.wav', '')
        num, spk, ident = name.split('_')
        self._spk_map[spk].append(name)
        self._num_map[num].append(name)
        self._data_map[name] = y
    assert len(set(all_sr)) == 1
    self._sr = all_sr[0]

  def __len__(self):
    return len(self._data_map)

  @property
  def speakers(self):
    return list(self._spk_map.keys())

  @property
  def numbers(self):
    return sorted([int(i) for i in self._num_map.keys()])

  @property
  def sr(self):
    return self._sr

  @property
  def path(self):
    return self._path

  def keys(self):
    return list(self._data_map.keys())

  def values(self):
    return list(self._data_map.values())

  def items(self):
    return list(self._data_map.items())

  def spk_items(self):
    return list(self._spk_map.items())

  def num_items(self):
    return list(self._num_map.items())

  def __iter__(self):
    return self._data_map.items()

  def __getitem__(self, key):
    if key in self._data_map:
      return self._data_map[key]
    if key in self._spk_map:
      return self._spk_map[key]
    if key in self._num_map:
      return self._num_map[key]
    raise KeyError(key)

# ===========================================================================
# download, unzip and create loader
# ===========================================================================
def _get_script_path():
  """Return the path of the script that calling this methods"""
  path = os.path.dirname(sys.argv[0])
  path = os.path.join('.', path)
  return os.path.abspath(path)


def _check_is_folder(path):
  if not os.path.exists(path):
    raise ValueError("'%s' does not exists!" % path)
  if not os.path.isdir(path):
    raise ValueError("'%s' is not a folder!" % path)

def _unzip(inpath, outpath):
  zf = ZipFile(inpath, mode='r', compression=ZIP_DEFLATED)
  zf.extractall(path=outpath)
  zf.close()

def load_digit_data():
  dat_path = os.path.join(_get_script_path(),
                          'free-spoken-digit-dataset',
                          'recordings')
  if not os.path.exists(dat_path):
    raise RuntimeError("Cannot find data folder at path: %s" % dat_path)
  return DIGITLoader(path=dat_path)

# ===========================================================================
# Visualization
# ===========================================================================
def plot_confusion_matrix(cm, labels, axis=None, fontsize=13, colorbar=False,
                          title=None):
  from matplotlib import pyplot as plt
  cmap = plt.cm.Blues
  # calculate F1
  N_row = np.sum(cm, axis=-1)
  N_col = np.sum(cm, axis=0)
  TP = np.diagonal(cm)
  FP = N_col - TP
  FN = N_row - TP
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  F1 = 2 / (1 / precision + 1 / recall)
  F1[np.isnan(F1)] = 0.
  F1_mean = np.mean(F1)
  # column normalize
  nb_classes = cm.shape[0]
  cm = cm.astype('float32') / np.sum(cm, axis=1, keepdims=True)
  if axis is None:
    axis = plt.gca()

  im = axis.imshow(cm, interpolation='nearest', cmap=cmap)
  # axis.get_figure().colorbar(im)
  tick_marks = np.arange(len(labels))
  axis.set_xticks(tick_marks)
  axis.set_yticks(tick_marks)
  axis.set_xticklabels(labels, rotation=-57,
                       fontsize=fontsize)
  axis.set_yticklabels(labels, fontsize=fontsize)
  axis.set_ylabel('True label', fontsize=fontsize)
  axis.set_xlabel('Predicted label', fontsize=fontsize)
  # center text for value of each grid
  worst_index = {i: np.argmax([val if j != i else -1
                               for j, val in enumerate(row)])
                 for i, row in enumerate(cm)}
  for i, j in itertools.product(range(nb_classes),
                                range(nb_classes)):
    color = 'black'
    weight = 'normal'
    fs = fontsize
    text = '%.2f' % cm[i, j]
    if i == j: # diagonal
      color = "darkgreen" if cm[i, j] <= 0.8 else 'forestgreen'
      weight = 'bold'
      fs = fontsize
      text = '%.2f\nF1:%.2f' % (cm[i, j], F1[i])
    elif j == worst_index[i]: # worst mis-classified
      color = 'red'
      weight = 'semibold'
      fs = fontsize
    plt.text(j, i, text,
             weight=weight, color=color, fontsize=fs,
             verticalalignment="center",
             horizontalalignment="center")
  # Turns off grid on the left Axis.
  axis.grid(False)
  # ====== colorbar ====== #
  if colorbar == 'all':
    fig = axis.get_figure()
    axes = fig.get_axes()
    fig.colorbar(im, ax=axes)
  elif colorbar:
    plt.colorbar(im, ax=axis)
  # ====== set title ====== #
  if title is None:
    title = ''
  title += ' (F1: %.3f)' % F1_mean
  axis.set_title(title, fontsize=fontsize + 2, weight='semibold')
  # axis.tight_layout()
  return axis

def plot_spectrogram(x, vad=None, ax=None, colorbar=False,
                     linewidth=0.5, title=None):
  '''
  Parameters
  ----------
  x : np.ndarray
      2D array
  vad : np.ndarray, list
      1D array, a red line will be draw at vad=1.
  ax : matplotlib.Axis
      create by fig.add_subplot, or plt.subplots
  colorbar : bool, 'all'
      whether adding colorbar to plot, if colorbar='all', call this
      methods after you add all subplots will create big colorbar
      for all your plots
  path : str
      if path is specified, save png image to given path

  Notes
  -----
  Make sure nrow and ncol in add_subplot is int or this error will show up
   - ValueError: The truth value of an array with more than one element is
      ambiguous. Use a.any() or a.all()

  Example
  -------
  >>> x = np.random.rand(2000, 1000)
  >>> fig = plt.figure()
  >>> ax = fig.add_subplot(2, 2, 1)
  >>> dnntoolkit.visual.plot_weights(x, ax)
  >>> ax = fig.add_subplot(2, 2, 2)
  >>> dnntoolkit.visual.plot_weights(x, ax)
  >>> ax = fig.add_subplot(2, 2, 3)
  >>> dnntoolkit.visual.plot_weights(x, ax)
  >>> ax = fig.add_subplot(2, 2, 4)
  >>> dnntoolkit.visual.plot_weights(x, ax, path='/Users/trungnt13/tmp/shit.png')
  >>> plt.show()
  '''
  from matplotlib import pyplot as plt

  # colormap = _cmap(x)
  # colormap = 'spectral'
  colormap = 'nipy_spectral'

  if x.ndim > 2:
    raise ValueError('No support for > 2D')
  elif x.ndim == 1:
    x = x[:, None]

  if vad is not None:
    vad = np.asarray(vad).ravel()
    if len(vad) != x.shape[1]:
      raise ValueError('Length of VAD must equal to signal length, but '
                       'length[vad]={} != length[signal]={}'.format(
                           len(vad), x.shape[1]))
    # normalize vad
    vad = np.cast[np.bool](vad)

  ax = ax if ax is not None else plt.gca()
  ax.set_aspect('equal', 'box')
  # ax.tick_params(axis='both', which='major', labelsize=6)
  ax.set_xticks([])
  ax.set_yticks([])
  # ax.axis('off')
  if title is not None:
    ax.set_ylabel(str(title) + '-' + str(x.shape), fontsize=6)
  img = ax.pcolorfast(x, cmap=colormap, alpha=0.9)
  # ====== draw vad vertical line ====== #
  if vad is not None:
    for i, j in enumerate(vad):
      if j: ax.axvline(x=i, ymin=0, ymax=1, color='r', linewidth=linewidth,
                       alpha=0.3)
  # plt.grid(True)

  if colorbar == 'all':
    fig = ax.get_figure()
    axes = fig.get_axes()
    fig.colorbar(img, ax=axes)
  elif colorbar:
    plt.colorbar(img, ax=ax)

  return ax

def plot_multiple_features(features, order=None, title=None, fig_width=4,
                  sharex=False):
  """ Plot a series of 1D and 2D in the same scale for comparison

  Parameters
  ----------
  features: Mapping
      pytho Mapping from name (string type) to feature matrix (`numpy.ndarray`)
  order: None or list of string
      if None, the order is keys of `features` sorted in alphabet order,
      else, plot the features or subset of features based on the name
      specified in `order`
  title: None or string
      title for the figure

  Note
  ----
  delta or delta delta features should have suffix: '_d1' and '_d2'
  """
  known_order = [
      # For audio processing
      'raw',
      'energy', 'energy_d1', 'energy_d2',
      'vad',
      'sad',
      'sap', 'sap_d1', 'sap_d2',
      'pitch', 'pitch_d1', 'pitch_d2',
      'loudness', 'loudness_d1', 'loudness_d2',
      'f0', 'f0_d1', 'f0_d2',
      'spec', 'spec_d1', 'spec_d2',
      'pspec',
      'mspec', 'mspec_d1', 'mspec_d2',
      'mfcc', 'mfcc_d1', 'mfcc_d2',
      'sdc',
      'qspec', 'qspec_d1', 'qspec_d2',
      'qmspec', 'qmspec_d1', 'qmspec_d2',
      'qmfcc', 'qmfcc_d1', 'qmfcc_d2',
      'bnf', 'bnf_d1', 'bnf_d2',
      'ivec', 'ivec_d1', 'ivec_d2',
      # For image processing
      # For video processing
  ]

  from matplotlib import pyplot as plt
  if isinstance(features, (tuple, list)):
    features = OrderedDict(features)
  if not isinstance(features, Mapping):
    raise ValueError("`features` must be mapping from name -> feature_matrix.")
  # ====== check order or create default order ====== #
  if order is not None:
    order = [str(o) for o in order]
  else:
    if isinstance(features, OrderedDict):
      order = features.keys()
    else:
      keys = sorted(features.keys() if isinstance(features, Mapping) else
                    [k for k, v in features])
      order = []
      for name in known_order:
        if name in keys:
          order.append(name)
      # add the remain keys
      for name in keys:
        if name not in order:
          order.append(name)
  # ====== get all numpy array ====== #
  features = [(name, features[name]) for name in order
              if name in features and
              isinstance(features[name], np.ndarray) and
              features[name].ndim <= 4]
  plt.figure(figsize=(int(fig_width), len(features)))
  for i, (name, X) in enumerate(features):
    X = X.astype('float32')
    plt.subplot(len(features), 1, i + 1)
    # flatten 2D features with one dim equal to 1
    if X.ndim == 2 and any(s == 1 for s in X.shape):
      X = X.ravel()
    # check valid dimension and select appropriate plot
    if X.ndim == 1:
      plt.plot(X)
      plt.xlim(0, len(X))
      plt.ylabel(name, fontsize=6)
    elif X.ndim == 2: # transpose to frequency x time
      plot_spectrogram(X.T, title=name)
    elif X.ndim == 3:
      plt.imshow(X)
      plt.xticks(())
      plt.yticks(())
      plt.ylabel(name, fontsize=6)
    else:
      raise RuntimeError("No support for >= 3D features.")
    # auto, equal
    plt.gca().set_aspect(aspect='auto')
    # plt.axis('off')
    plt.xticks(())
    # plt.yticks(())
    plt.tick_params(axis='y', size=6, labelsize=4, color='r', pad=0,
                    length=2)
    # add title to the first subplot
    if i == 0 and title is not None:
      plt.title(str(title), fontsize=8)
    if sharex:
      plt.subplots_adjust(hspace=0)

def plot_save(path='/tmp/tmp.pdf', figs=None, dpi=180,
              tight_plot=False, clear_all=True, log=True):
  """
  Parameters
  ----------
  clear_all: bool
      if True, remove all saved figures from current figure list
      in matplotlib
  """
  try:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    if tight_plot:
      plt.tight_layout()
    if os.path.exists(path) and os.path.isfile(path):
      os.remove(path)
    pp = PdfPages(path)
    if figs is None:
      figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
      fig.savefig(pp, format='pdf', bbox_inches="tight")
    pp.close()
    if log:
      sys.stderr.write('Saved pdf figures to:%s \n' % str(path))
    if clear_all:
      plt.close('all')
  except Exception as e:
    sys.stderr.write('Cannot save figures to pdf, error:%s \n' % str(e))

# ===========================================================================
# Additional helper for speech processing
# ===========================================================================
def get_energy(y, win_length, hop_length, log=True):
  """ Calculate frame-wise energy
  Parameters
  ----------
  frames: ndarray
      framed signal with shape (nb_frames x window_length)
  log: bool
      if True, return log energy of each frames

  Return
  ------
  E : ndarray [shape=(nb_frames,), dtype=float32]
  """
  # ====== extract frames ====== #
  shape = y.shape[:-1] + (y.shape[-1] - win_length + 1, win_length)
  strides = y.strides + (y.strides[-1],)
  frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
  if frames.ndim > 2:
    frames = np.rollaxis(frames, 1)
  frames = frames[::hop_length] # [n, frame_length]
  # ====== windowing ====== #
  fft_window = signal.get_window('hann', win_length, fftbins=True).reshape(1, -1)
  frames = fft_window * frames
  # ====== energy ====== #
  log_energy = (frames**2).sum(axis=1)
  log_energy = np.where(log_energy == 0., np.finfo(np.float32).eps,
                        log_energy)
  if log:
    log_energy = np.log(log_energy)
  return np.expand_dims(log_energy.astype('float32'), -1)

# ===========================================================================
# Shape processing
# ===========================================================================
def one_hot(y, nb_classes=None, dtype='float32'):
  '''Convert class vector (integers from 0 to nb_classes)
  to binary class matrix, for use with categorical_crossentropy

  Note
  ----
  if any class index in y is smaller than 0, then all of its one-hot
  values is 0.
  '''
  if 'int' not in str(y.dtype):
    y = y.astype('int32')
  if nb_classes is None:
    nb_classes = np.max(y) + 1
  else:
    nb_classes = int(nb_classes)
  return np.eye(nb_classes, dtype=dtype)[y]

def segment_axis(a, frame_length=2048, step_length=512, axis=0,
                 end='cut', pad_value=0, pad_mode='post'):
  """Generate a new array that chops the given array along the given axis
  into overlapping frames.

  This method has been implemented by Anne Archibald,
  as part of the talk box toolkit
  example::

      segment_axis(arange(10), 4, 2)
      array([[0, 1, 2, 3],
         ( [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

  Parameters
  ----------
  a: numpy.ndarray
      the array to segment
  frame_length: int
      the length of each frame
  step_length: int
      the number of array elements by which the frames should overlap
  axis: int, None
      the axis to operate on; if None, act on the flattened array
  end: 'cut', 'wrap', 'pad'
      what to do with the last frame, if the array is not evenly
          divisible into pieces. Options are:
          - 'cut'   Simply discard the extra values
          - 'wrap'  Copy values from the beginning of the array
          - 'pad'   Pad with a constant value
  pad_value: int
      the value to use for end='pad'
  pad_mode: 'pre', 'post'
      if "pre", padding or wrapping at the beginning of the array.
      if "post", padding or wrapping at the ending of the array.

  Return
  ------
  a ndarray

  The array is not copied unless necessary (either because it is unevenly
  strided and being flattened or because end is set to 'pad' or 'wrap').

  Note
  ----
  Modified work and error fixing Copyright (c) TrungNT

  """
  if axis is None:
    a = np.ravel(a) # may copy
    axis = 0

  length = a.shape[axis]
  overlap = frame_length - step_length

  if overlap >= frame_length:
    raise ValueError("frames cannot overlap by more than 100%")
  if overlap < 0 or frame_length <= 0:
    raise ValueError("overlap must be nonnegative and length must" +
                     "be positive")

  if length < frame_length or (length - frame_length) % (frame_length - overlap):
    if length > frame_length:
      roundup = frame_length + (1 + (length - frame_length) // (frame_length - overlap)) * (frame_length - overlap)
      rounddown = frame_length + ((length - frame_length) // (frame_length - overlap)) * (frame_length - overlap)
    else:
      roundup = frame_length
      rounddown = 0
    assert rounddown < length < roundup
    assert roundup == rounddown + (frame_length - overlap) \
    or (roundup == frame_length and rounddown == 0)
    a = a.swapaxes(-1, axis)

    if end == 'cut':
      a = a[..., :rounddown]
    elif end in ['pad', 'wrap']: # copying will be necessary
      s = list(a.shape)
      s[-1] = roundup
      b = np.empty(s, dtype=a.dtype)
      # pre-padding
      if pad_mode == 'post':
        b[..., :length] = a
        if end == 'pad':
          b[..., length:] = pad_value
        elif end == 'wrap':
          b[..., length:] = a[..., :roundup - length]
      # post-padding
      elif pad_mode == 'pre':
        b[..., -length:] = a
        if end == 'pad':
          b[..., :(roundup - length)] = pad_value
        elif end == 'wrap':
          b[..., :(roundup - length)] = a[..., :roundup - length]
      a = b
    a = a.swapaxes(-1, axis)
    length = a.shape[0] # update length

  if length == 0:
    raise ValueError("Not enough data points to segment array " +
            "in 'cut' mode; try 'pad' or 'wrap'")
  assert length >= frame_length
  assert (length - frame_length) % (frame_length - overlap) == 0
  n = 1 + (length - frame_length) // (frame_length - overlap)
  s = a.strides[axis]
  newshape = a.shape[:axis] + (n, frame_length) + a.shape[axis + 1:]
  newstrides = a.strides[:axis] + ((frame_length - overlap) * s, s) + a.strides[axis + 1:]

  try:
    return np.ndarray.__new__(np.ndarray, strides=newstrides,
                              shape=newshape, buffer=a, dtype=a.dtype)
  except TypeError:
    a = a.copy()
    # Shape doesn't change but strides does
    newstrides = a.strides[:axis] + ((frame_length - overlap) * s, s) \
    + a.strides[axis + 1:]
    return np.ndarray.__new__(np.ndarray, strides=newstrides,
                              shape=newshape, buffer=a, dtype=a.dtype)
