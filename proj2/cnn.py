import pandas as pd
import io
import requests
import numpy as np
import collections
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.callbacks import ModelCheckpointß

dt = pd.read_csv('CSC215_P2_Stock_Price.csv')
dt['Close_y'] = dt['Close']
split = int(0.7 * len(dt))
df_train = dt[:split]
df_test = dt[split:len(dt)]

print("Training set has {} records.".format(len(df_train)))
print("Test set has {} records.".format(len(df_test)))

close_train = df_train['Close_y']
close_test = df_test['Close_y']

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd

normal_list = ['Open', 'High', 'Low', 'Volume', 'Close'];

for element in normal_list:
    encode_numeric_zscore(df_train, element)
    encode_numeric_zscore(df_test, element)



params_train = df_train[['Open', 'High', 'Low', 'Volume', 'Close']].values.tolist()
params_test = df_test[['Open', 'High', 'Low', 'Volume', 'Close',]].values.tolist()


def to_sequences(seq_size, data):
    x = []
    y = []

    for i in range(len(data)-SEQUENCE_SIZE-1):
        #print(i)
        window = data[i:(i+SEQUENCE_SIZE)]
        after_window = data[i+SEQUENCE_SIZE]
        window = [x for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)

    return np.array(x),np.array(y)

SEQUENCE_SIZE = 7
x_train,y_train = to_sequences(SEQUENCE_SIZE,params_train)
obs_train = close_train[SEQUENCE_SIZE:len(close_train)].values.tolist()
obs_train.pop()
obs_train = np.asarray(obs_train)

x_test,y_test = to_sequences(SEQUENCE_SIZE,params_test)
obs_test = close_test[SEQUENCE_SIZE:len(close_test)].values.tolist()
obs_test.pop()
obs_test = np.asarray(obs_test)

print("Shape of x_train: {}".format(x_train.shape))
print("Shape of x_test: {}".format(x_test.shape))
print("Shape of y_train: {}".format(obs_train.shape))
print("Shape of y_test: {}".format(obs_test.shape))

print("PREPROCESSING DONE")

print("RESHAPING DATA TO IMAGE...")

sample = 1
digit = x_train[sample]
#print(type(digit))
#print(digit.shape)


# plt.imshow(digit, cmap='gray')
# print("Image (#{}): Which is digit '{}'".format(sample,y_train[sample]))
# plt.show()
# print(pd.DataFrame(digit))

batch_size = 128
img_rows, img_cols = 7, 1
channels = 5

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

print(x_train.shape)
print(x_test.shape)

print(x_train.dtype)

# convert to float32 for normalization
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# normalize the data values to the range [0, 1]
# x_train /= 255
# x_test /= 255

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print("Training samples: {}".format(x_train.shape[0]))
print("Test samples: {}".format(x_test.shape[0]))

print(obs_train.shape)
print(obs_train[:10])

obs_all = np.concatenate((obs_train, obs_test), axis=0)

obs_all = np.round(obs_all, 2)

print(obs_train.shape)
print(obs_test.shape)
print(obs_train)
print(obs_test)
print(obs_all.shape)


# obs_train2 = obs_all[:3066]
# obs_test2 = obs_all[3066:]
#
# print(obs_train2.shape)
# print(obs_test2.shape)
# print(obs_train2)
# print(obs_test2)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto', sparse=False)

obs_all = np.reshape(obs_all, (-1, 1))

print(obs_all)

output = encoder.fit_transform(obs_all)

print(output.shape)

obs_train = output[:3066]
obs_test = output[3066:]

print(obs_train.shape)
print(obs_test.shape)


# CNN Architecture

model = Sequential()

input_shape = (img_rows, img_cols, channels)

# two conv layers

model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                 activation='relu',
                 input_shape=input_shape))    #  in this case, input_shape = (img_rows, img_cols, 1)

model.add(Conv2D(64, (1, 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 1), strides=None))
model.add(Dropout(0.25))

# dense layer

num_classes = 2567 # from num cols one hot encoding

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

# COMPILE MODEL
# -------------------

from tensorflow.keras.optimizers import Adam

# show not only log loss but also accuracy for each epoch using metrics=['accuracy']

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])

print(x_train.shape)


# TRAINING/FITTING CNN
# -----------------------

import time

start_time = time.time()

# 1% of dataset

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
model.fit(x_train, obs_train,
          batch_size=batch_size,
          epochs=100,
          verbose=2,
          validation_data=(x_test, obs_test),
          callbacks=[monitor])

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(hms_string(elapsed_time)))
