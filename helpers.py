from __future__ import print_function, division, absolute_import
import os
import sys
import shutil
import base64
import pickle
from collections import OrderedDict, Mapping, defaultdict
from zipfile import ZipFile, ZIP_DEFLATED
from six.moves.urllib.request import urlretrieve

import numpy as np
import soundfile as sf

# ===========================================================================
# Data loader
# ===========================================================================
class IMDBLoader(object):
  """ IMDBLoader """

  def __init__(self, path):
    super(IMDBLoader, self).__init__()
    self._path = path
    files = ['index_word',
             'x_test', 'x_train',
             'y_test', 'y_train']
    self.files_map = {}
    for name in files:
      with open(os.path.join(path, name), 'rb') as f:
        self.files_map[name] = pickle.load(f)

  @property
  def path(self):
    return self._path

  @property
  def x_train(self):
    return self.files_map['x_train']

  @property
  def y_train(self):
    return self.files_map['y_train']

  @property
  def x_test(self):
    return self.files_map['x_test']

  @property
  def y_test(self):
    return self.files_map['y_test']

  @property
  def index_word(self):
    return self.files_map.index_word

  def decode(self, sentence):
    pass

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
    return list(self._num_map.keys())

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

def load_imdb_data(path, override=False):
  """
  Return
  """
  _check_is_folder(path)
  dat_path = os.path.join(path, 'IMDB')
  zip_path = os.path.join(path, '__tmp_imdb__.zip')
  if override and os.path.exists(dat_path):
    shutil.rmtree(dat_path)
  # ====== download the zip file ====== #
  if not os.path.exists(dat_path):
    if os.path.exists(zip_path):
      os.remove(zip_path)
    url = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL0lNREJfMjAwMDBfcHJlcHJvY2Vz\nc2VkLnppcA==\n'
    url = str(base64.decodebytes(url), 'utf-8')

    def dl_progress(count, block_size, total_size):
      if count % 180 == 0:
        print("Downloaded: %.2f/%.2f(kB)" % (count * block_size / 1024,
                                             total_size / 1024))
    try:
      urlretrieve(url, zip_path, dl_progress)
    except Exception as e:
      os.remove(zip_path)
      raise e
    # ====== unzip the file ====== #
    _unzip(zip_path, dat_path)
  # ====== clean up ====== #
  if os.path.exists(zip_path):
    os.remove(zip_path)
  # ====== load the data ====== #
  return IMDBLoader(path=dat_path)

# ===========================================================================
# Visualization
# ===========================================================================
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
