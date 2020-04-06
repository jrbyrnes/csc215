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

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, LSTM, Conv2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


####################
##DATA IMPORTING && PREPROCESSING
####################
dt = pd.read_csv('CSC215_P2_Stock_Price.csv')
dt['Close_y'] = dt['Close']
split = int(0.7 * len(dt))
df_train = dt[:split]
df_test = dt[split:len(dt)]

print("Training set has {} records.".format(len(df_train)))
print("Test set has {} records.".format(len(df_test)))

close_train = df_train['Close_y']
close_test = df_test['Close_y']

def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd

normal_list = ['Open', 'High', 'Low', 'Volume'];

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

def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred.flatten(), 'y' : y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

####################
##Convolutional NN
####################

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

x_train_CNN = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test_CNN = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

print('x_train shape:', x_train_CNN.shape)
print('x_test shape:', x_test_CNN.shape)
print("Training samples: {}".format(x_train_CNN.shape[0]))
print("Test samples: {}".format(x_test_CNN.shape[0]))

myDict4 = dict()
activationType = ['relu', 'sigmoid', 'tanh']
optimizerType = ['adam', 'sgd']
iteration = 0
input_shape = (img_rows, img_cols, channels)

for act in activationType:
    for opt in optimizerType:
        checkpointer4 = ModelCheckpoint(filepath="best_weights4.hdf5", verbose=0, save_best_only=True) # save best model

        for i in range(2):
            print(i)
            model_CNN = Sequential()
            model_CNN.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=act, input_shape=input_shape))
            model_CNN.add(MaxPooling2D(pool_size=(1, 1), strides=None)) # (1, 1) doesnt really do anything
            model_CNN.add(Conv2D(64, (1, 1), activation=act))
            model_CNN.add(MaxPooling2D(pool_size=(1, 1), strides=None)) # (1, 1) doesnt really do anything
            model_CNN.add(Dropout(0.25))

            model_CNN.add(Flatten())
            model_CNN.add(Dense(64, activation=act))
            model_CNN.add(Dropout(0.25))
            model_CNN.add(Dense(32, activation=act))
            model_CNN.add(Dense(1))
            print(model_CNN.summary())
            model_CNN.compile(loss='mean_squared_error', optimizer=opt)

            monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
            model_CNN.fit(x_train_CNN, obs_train, epochs=100, verbose=2, validation_data=(x_test_CNN, obs_test), callbacks=[monitor, checkpointer4])

        print('Training finished...Loading the best model')
        print()
        model_CNN.load_weights("best_weights4.hdf5") # load weights from best model
        myDict4.update({iteration : (act, opt, model_CNN)})
        iteration += 1

for ele in myDict4.values():
    print('Analyzing model with activation {} and optimizer {}'.format(ele[0], ele[1]))
    model_CNN_b = ele[2]
    pred_CNN = model_CNN_b.predict(x_test_CNN)

    score_CNN = np.sqrt(metrics.mean_squared_error(pred_CNN, obs_test))
    chart_regression(pred_CNN, obs_test)
    print("Score (RMSE): {}".format(score_CNN))
