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
from tensorflow.keras.layers import Dense, Activation, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


####################
##DATA IMPORTING && PREPROCESSING
####################
dt = pd.read_csv('C:/Users/Owner/Documents/Sac State/CSC215_P2_Stock_Price.csv')
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
myDict = dict()
##activationType = ['relu', 'sigmoid', 'tanh']
optimizerType = ['adam', 'sgd']
iteration = 0

for opt in optimizerType:
    ##checkpointer = ModelCheckpoint(filepath="best_weights3.hdf5", verbose=0, save_best_only=True) # save best model        
    ##for i in range(2):
        ##print(i)        
        # Build network
        model = Sequential()
        model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, input_shape=(SEQUENCE_SIZE, 5)))
        model.add(Dense(32))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=opt)        
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')        
        model.fit(x_train,obs_train,validation_data=(x_test,obs_test), callbacks=[monitor],verbose=2, epochs=100)  


    ##print('Training finished...Loading the best model')  
    ##print()
    ##model.load_weights("best_weights3.hdf5") # load weights from best model
        myDict.update({iteration : (opt, model)})
        iteration += 1

def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred.flatten(), 'y' : y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()


for ele in myDict.values():
    print('Analyzing model with optimizer {}'.format(ele[0]))
    model = ele[1]
    pred = model.predict(x_test)
    obs_test = np.asarray(obs_test)
    score = np.sqrt(metrics.mean_squared_error(pred,obs_test))
    chart_regression(pred,obs_test)
    print("Score (RMSE): {}".format(score))



###Fully Connected NN
test_NN = []
for ele in x_test:
    flat_list = []
    for sublist in ele:
        for item in sublist:
            flat_list.append(item)
    test_NN.append(flat_list)

train_NN = []
for ele in x_train:
    flat_list = []
    for sublist in ele:
        for item in sublist:
            flat_list.append(item)
    train_NN.append(flat_list)


train_NN = pd.DataFrame(train_NN)
train_NN['out'] = obs_train
test_NN = pd.DataFrame(test_NN)
test_NN['out'] = obs_test


def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column. 
    target_type = df[target].dtypes
    target_type = target_type[0] if isinstance(target_type, collections.Sequence) else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    else:
        # Regression
        return df[result].values.astype(np.float32), df[target].values.astype(np.float32)


train_NN,y_train2 = to_xy(train_NN, 'out')
test_NN, y_test2 = to_xy(test_NN, 'out')


myDict2 = dict()
activationType = ['relu', 'sigmoid', 'tanh']
optimizerType = ['adam', 'sgd']
iteration = 0

for act in activationType:
    for opt in optimizerType:
        checkpointer2 = ModelCheckpoint(filepath="C:/Users/Owner/Documents/Sac State/csc215/proj2/best_weights2.hdf5", verbose=0, save_best_only=True) # save best model
        
        for i in range(5):
            print(i)
            model_NN = Sequential()
            model_NN.add(Dense(20, input_dim=train_NN.shape[1], activation='relu'))
            model_NN.add(Dense(10, input_dim=train_NN.shape[1], activation='relu'))
            model_NN.add(Dense(1))
            model_NN.compile(loss='mean_squared_error', optimizer='adam')        
            monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')        
            model_NN.fit(train_NN,y_train2,validation_data=(test_NN,y_test2),callbacks=[monitor, checkpointer2],verbose=2,epochs=100)

        print('Training finished...Loading the best model')  
        print()
        model_NN.load_weights("C:/Users/Owner/Documents/Sac State/csc215/proj2/best_weights2.hdf5") # load weights from best model
        myDict2.update({iteration : (act, opt, model)})
        iteration += 1


for ele in myDict2.values():
    print('Analyzing model with activation {} and optimizer {}'.format(ele[0], ele[1]))
    model_NN_b = ele[2]
    pred_NN = model_NN_b.predict(test_NN)
    pred_NN = np.argmax(pred_NN,axis=1)
    
    y_true_NN = np.argmax(obs_test,axis=1)
    score_NN = np.sqrt(metrics.mean_squared_error(pred_NN,obs_test))
    chart_regression(pred_NN,obs_test)
    print("Score (RMSE): {}".format(score_NN))
    
