from __future__ import print_function, division, absolute_import
import matplotlib
# comment this if you want to show plot directly on your personal computer
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

import tqdm

from helpers import (load_digit_data, get_energy, segment_axis, one_hot,
                     plot_multiple_features, plot_save, plot_spectrogram,
                     plot_confusion_matrix)
from librosa.core import stft, power_to_db
from librosa.feature import melspectrogram, mfcc

from keras.utils import plot_model
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.pooling import MaxPool1D, AvgPool1D, MaxPool2D, AvgPool2D
from keras.layers import (Dense, Conv1D, Conv2D,
                          LSTM, BatchNormalization,
                          Dropout, Flatten, Reshape, Activation,
                          Embedding)

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

# seed for reproducibility
np.random.seed(5218)
# ===========================================================================
# Loading data
# ===========================================================================
digit = load_digit_data()
sr = digit.sr
nb_classes = len(digit.numbers)
print("#Files:", len(digit))
print("SampleRate:", sr)
print("Speakers:", digit.speakers)
print("Numbers:", digit.numbers)
# ===========================================================================
# Const
# ===========================================================================
# ====== features configuration ====== #
NFFT = 512
NMELS = 40
NMFCC = 20
HOP_LENGTH = int(0.010 * sr)
WIN_LENGTH = int(0.025 * sr)
# ====== preprocessing configuration ====== #
VALID_PERCENT = 0.2
TRAINING_FEATURE = 'mspec' # 'spec', 'mfcc', 'energy'
PAD_MODE = 'pre' # post
# ====== training configuration ====== #
BATCH_SIZE = 32
NUM_EPOCH = 4
OPTIMIZER = 'adadelta' # https://keras.io/optimizers/
# ====== other ====== #
FIGURE_SAVE_PATH = '/tmp/tmp.pdf'
MODEL_SAVE_PATH = '/tmp/model.png'
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
# ====== you can play here with different configuration ====== #
visualize_same_number(files=('2_jackson_8',
                             '2_theo_8',
                             '2_nicolas_8'))
visualize_same_number(files=('8_jackson_8',
                             '8_theo_8',
                             '8_nicolas_8'))
# ===========================================================================
# Extracting acoustic features
# ===========================================================================
def extracting_acoustic_features(y):
  power = 2
  energy = get_energy(y, win_length=NFFT, hop_length=HOP_LENGTH)
  s = stft(y=np.ascontiguousarray(y), n_fft=NFFT,
           hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
           center=False)
  assert energy.shape[0] == s.shape[1]
  # magnitude spectrogram
  spec = np.abs(s)
  # phase spectrogram
  phase = np.angle(s)
  # spectrum (all the arrays are: [freq, time])
  power_spec = spec**power # power-spectrogram
  # power mel-filter banks spectrogram
  power_mel_spec = power_to_db(
      melspectrogram(sr=sr, S=power_spec,
                     n_fft=NFFT, hop_length=HOP_LENGTH, n_mels=NMELS))
  # MFCCs coefficiences
  mfcc_spec = mfcc(sr=sr, S=power_mel_spec, n_mfcc=NMFCC)
  return {'energy': energy, 'spec': spec.T, 'pspec': power_spec.T,
          'mspec': power_mel_spec.T, 'mfcc': mfcc_spec.T, 'phase': phase.T}

features = {}
for name, y in tqdm.tqdm(digit.items(), desc='Extracting acoustic features'):
  features[name] = extracting_acoustic_features(y)
# plt.figure()
# tmp = features['2_jackson_8']
# tmp['raw'] = digit['2_jackson_8']
# plot_multiple_features(tmp)
# plot_save()
# ===========================================================================
# Spliting train, valid, test data
# ===========================================================================
# 2 speakers for training and validating, 1 speakers for testing
train_spk = np.random.choice(a=digit.speakers, size=2, replace=False)
test_spk = [i
            for i in digit.speakers
            if i not in train_spk]

train_utt = []
for spk in train_spk:
  train_utt += digit[spk]
valid_utt = np.random.choice(a=train_utt,
                             size=int(0.2 * len(train_utt)),
                             replace=False)
train_utt = [i
             for i in train_utt
             if i not in valid_utt]

test_utt = []
for spk in test_spk:
  test_utt += digit[spk]

print("Train Speakers:", train_spk, "#Train Utterances:", len(train_utt))
print("Test Speakers:", test_spk, "#Test Utterances:", len(test_utt))

longest_utt = max(len(x['energy']) for x in features.values())
print("Longest Utterance:", longest_utt)
# ===========================================================================
# Create training data
# ===========================================================================
def generate_mini_batch(flist):
  flist = np.array(flist)
  np.random.shuffle(flist)
  X = []
  y = []
  for f in flist:
    feat = segment_axis(features[f][TRAINING_FEATURE],
                        frame_length=longest_utt, step_length=1,
                        end='pad', pad_value=0, pad_mode=PAD_MODE)
    label = int(f.split('_')[0])
    X.append(feat)
    y.append(label)
  X = np.concatenate(X, axis=0)
  y = np.array(y)
  indices = np.random.permutation(X.shape[0])
  return X[indices], one_hot(y[indices], nb_classes=nb_classes)

X_train, y_train = generate_mini_batch(train_utt)
X_valid, y_valid = generate_mini_batch(valid_utt)
X_test, y_test = generate_mini_batch(test_utt)
input_ndim = X_train.shape[-1]
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_valid:', X_valid.shape)
print('y_valid:', y_valid.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)
# ====== Plot some features ====== #
plt.figure()
for i in range(8):
  plt.subplot(8, 1, i + 1)
  plot_spectrogram(X_train[i].T)
  plt.title(str(np.argmax(y_train[i])))
# ===========================================================================
# Create the network
# For more information:
# https://github.com/keras-team/keras/tree/master/examples
# ===========================================================================
method = 3
INPUT_SHAPE = (longest_utt, input_ndim)
# ====== Fully connected feedforward network ====== #
if method == 1:
  model = Sequential()
  model.add(Flatten(input_shape=INPUT_SHAPE))
  model.add(Dropout(rate=0.3))

  model.add(Dense(units=2048, activation=None, use_bias=False))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation(activation='relu'))

  model.add(Dense(units=1024, activation=None, use_bias=False))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation(activation='relu'))

  model.add(Dense(units=512, activation=None, use_bias=False))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation(activation='relu'))

  model.add(Dense(units=nb_classes, activation='softmax'))
# ====== Mixed of convolutional network and feedforward network ====== #
elif method == 2:
  model = Sequential()
  model.add(Dropout(rate=0.3, input_shape=INPUT_SHAPE))

  model.add(Conv1D(filters=32, kernel_size=7,
                   use_bias=False, activation='linear',
                   padding='valid', strides=1))
  model.add(BatchNormalization())
  model.add(Activation(activation='relu'))

  model.add(Conv1D(filters=64, kernel_size=5,
                   use_bias=False, activation='linear',
                   padding='valid', strides=1))
  model.add(BatchNormalization())
  model.add(Activation(activation='relu'))

  model.add(Flatten())

  model.add(Dense(units=512, activation=None, use_bias=False))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation(activation='relu'))

  model.add(Dense(nb_classes, activation='softmax'))
# ====== CNN + LSTM + FNN ====== #
elif method == 3:
  model = Sequential()
  model.add(Dropout(rate=0.3, input_shape=INPUT_SHAPE))

  model.add(Conv1D(filters=32, kernel_size=7,
                   use_bias=False, activation='linear',
                   padding='valid', strides=1))
  model.add(BatchNormalization())
  model.add(Activation(activation='relu'))

  model.add(Conv1D(filters=64, kernel_size=5,
                   use_bias=False, activation='linear',
                   padding='valid', strides=1))
  model.add(BatchNormalization())
  model.add(Activation(activation='relu'))

  model.add(LSTM(128, return_sequences=True))

  model.add(Flatten())

  model.add(Dense(units=512, activation=None, use_bias=False))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation(activation='relu'))

  model.add(Dense(nb_classes, activation='softmax'))
# ====== bonus: 2D-CNN + LSTM + FNN ====== #
elif method == 4:
  model = Sequential()
  model.add(Dropout(rate=0.3, input_shape=INPUT_SHAPE))
  model.add(Reshape(target_shape=(longest_utt, input_ndim, 1)))

  model.add(Conv2D(filters=32, kernel_size=(7, 9),
                   use_bias=False, activation='linear',
                   padding='valid', strides=(1, 1)))
  model.add(BatchNormalization())
  model.add(Activation(activation='relu'))

  model.add(Conv2D(filters=64, kernel_size=(5, 7),
                   use_bias=False, activation='linear',
                   padding='valid', strides=(1, 1)))
  model.add(BatchNormalization())
  model.add(Activation(activation='relu'))

  last_shape = model.output_shape
  model.add(Reshape(target_shape=(last_shape[1], last_shape[2] * last_shape[3])))
  model.add(LSTM(128, return_sequences=True))

  model.add(Flatten())
  model.add(Dense(units=512, activation=None, use_bias=False))
  model.add(BatchNormalization(axis=-1))
  model.add(Activation(activation='relu'))

  model.add(Dense(nb_classes, activation='softmax'))
# ====== plot model to file ====== #
print(model.summary())
# plot_model(model, to_file=MODEL_SAVE_PATH, show_shapes=True, rankdir='LR')
# ===========================================================================
# Training the networks
# ===========================================================================
# 2 classes: 'binary_crossentropy'
# mean squared error
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE, epochs=NUM_EPOCH,
          validation_data=(X_valid, y_valid))
# ====== evaluation ====== #
y_test_pred_proba = model.predict_proba(X_test, batch_size=BATCH_SIZE * 2)
y_test = np.argmax(y_test, axis=-1)
y_test_pred = np.argmax(y_test_pred_proba, axis=-1)
print("Accuracy:", accuracy_score(y_true=y_test, y_pred=y_test_pred))
print("F1 score:", f1_score(y_true=y_test, y_pred=y_test_pred,
                            average='micro', labels=digit.numbers))
print(classification_report(y_true=y_test, y_pred=y_test_pred, labels=digit.numbers))
# ====== plot confusion matrix ====== #
plt.figure()
cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred, labels=digit.numbers)
plot_confusion_matrix(cm=cm, labels=digit.numbers, colorbar=True, fontsize=6)
# ===========================================================================
# Cleaning
# ===========================================================================
plot_save(path=FIGURE_SAVE_PATH)
