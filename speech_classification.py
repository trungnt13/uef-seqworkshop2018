from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

from helpers import load_digit_data, plot_multiple_features
from librosa.core import stft
from librosa.feature import melspectrogram, mfcc, _spectrogram

# ===========================================================================
# Loading data
# ===========================================================================
digit = load_digit_data('/tmp', override=False)
sr = digit.sr
print("#Files:", len(digit))
print("SampleRate:", sr)
# ===========================================================================
# Extracting acoustic features
# ===========================================================================
name, y = digit.items()[0]
s = stft(y=y, n_fft=1024, hop_length=int(0.01 * sr), win_length=int(0.025 * sr))
spec = np.abs(s)
phase = np.angle(s)
plot_multiple_features
# spec = _spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1)
