from __future__ import print_function, division, absolute_import
import os
import sys
import shutil
import base64
import pickle
from six.moves.urllib.request import urlretrieve
from zipfile import ZipFile, ZIP_DEFLATED

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
    for name in os.listdir(path):
      if '.wav' not in name:
        continue
      with open(os.path.join(path, name), 'rb') as f:
        y, sr = sf.read(f)
        all_sr.append(sr)
        self._data_map[name] = y
    assert len(set(all_sr)) == 1
    self._sr = all_sr[0]

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

  def __iter__(self):
    return self._data_map.items()

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

def load_digit_data(path, override=False):
  _check_is_folder(path)
  dat_path = os.path.join(path, 'DIGIT')
  zip_path = os.path.join(_get_script_path(), 'data/digits_audio.zip')
  if not os.path.exists(zip_path):
    raise ValueError("Cannot find zipped dataset at: %s" % zip_path)
  if override and os.path.exists(dat_path):
    shutil.rmtree(dat_path)
  if not os.path.exists(dat_path):
    _unzip(inpath=zip_path, outpath=dat_path)
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
