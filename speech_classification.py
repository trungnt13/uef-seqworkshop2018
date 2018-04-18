from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

from helpers import load_digit_data, plot_multiple_features, plot_save, plot_spectrogram
from librosa.core import stft
from librosa.feature import melspectrogram, mfcc

# seed for reproducibility
np.random.seed(5218)
# ===========================================================================
# Loading data
# ===========================================================================
digit = load_digit_data()
sr = digit.sr
print("#Files:", len(digit))
print("SampleRate:", sr)
print("Speakers:", digit.speakers)
print("Numbers:", digit.numbers)
# ===========================================================================
# Const
# ===========================================================================
NFFT = 1024
HOP_LENGTH = int(0.010 * sr)
WIN_LENGTH = int(0.025 * sr)
# ===========================================================================
# Initial visualization of the dataset
# ===========================================================================
def visualize_same_number(files):
  data = [(name, digit[name]) for name in files]
  plt.figure()
  for idx, (name, y) in enumerate(data):
    # raw signal
    plt.subplot(3, 1, idx + 1)
    plt.plot(y)
    plt.title(name)
  plt.tight_layout()
  plt.figure()
  for idx, (name, y) in enumerate(data):
    idx += 1
    s = stft(y=y, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    # magnitude and phase
    plt.subplot(1, 3, idx)
    plot_spectrogram(np.abs(s), title=name)
visualize_same_number(files=('2_jackson_8',
                             '2_theo_8',
                             '2_nicolas_8'))
visualize_same_number(files=('8_jackson_8',
                             '8_theo_8',
                             '8_nicolas_8'))
# ===========================================================================
# Extracting acoustic features
# ===========================================================================
# name, y = digit.items()[0]
# s = stft(y=y, n_fft=1024, hop_length=int(0.01 * sr), win_length=int(0.025 * sr))
# spec = np.abs(s)
# phase = np.angle(s)
# plot_multiple_features(features={'raw': y, 'spec': spec, 'phase': phase})
plot_save()
# spec = _spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1)
