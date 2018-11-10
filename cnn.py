#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:04:09 2018

@author: dslasdoce
"""
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import operator

train = pd.read_csv('training_set.csv')
train_meta = pd.read_csv('training_set_metadata.csv')
obj_ids = train['object_id'].unique()
targets = train_meta.set_index('object_id', drop=True).loc[obj_ids, 'target']
target_map = {6: 0, 15:1, 16:2, 42:3, 52: 4, 53: 5, 62: 6, 64: 7,
              65: 8, 67: 9, 88: 10, 90: 11, 92: 12, 95: 13, 99: 14}
target_map = dict(sorted(target_map.items(), key=operator.itemgetter(1)))

targets_id =  train_meta['target'].map(target_map)
y_categorical = to_categorical(targets_id)
conv_dataset = np.zeros((obj_ids.shape[0], 60, 3, 6))
z = train.loc[(train.object_id==615)&(train.passband==0)].sort_values('mjd').values
z = train.set_index(['object_id', 'passband'])
seq_len = 60
#z.loc[615]
for i, oid in enumerate(obj_ids):
    for pb in range(6):
        tmp = train.loc[(train.object_id==oid)&(train.passband==pb)].sort_values('mjd')
        tmp = tmp[['mjd', 'flux', 'flux_err']].values
        if tmp.shape[0] < seq_len:
            diff = 60 - len(tmp)
            tmp = np.pad(tmp, ((0,diff),(0,0)), mode='constant', constant_values=0)
        elif tmp.shape[0] > 0:
            tmp = tmp[:seq_len]
        conv_dataset[i, :, :, pb] = tmp
#a = pd.MultiIndex(labels=[0,1,2,3,4,5], levels=[0,1,2,3,4,5])
        
a = np.array([[1,2,3], [4,5,6]])
np.pad(a, ((0,2),(0,0)), mode='constant', constant_values=(0,0))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
#from keras.utils import np_utils
#from matplotlib import pyplot as plt
def createCNNModel(num_classes):
    """ Adapted from: # http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
# """
    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(60, 3, 6), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 3  # >>> should be 25+
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model

from sklearn.model_selection import StratifiedKFold
def getFolds(ser_target=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
#    idx = np.arange(df.shape[0])
    fold_idx = []
    for train_idx, val_idx in folds.split(X=ser_target, y=ser_target):
        fold_idx.append([train_idx, val_idx])

    return fold_idx
folds = getFolds(targets)

oof_preds_nn = np.zeros((y_categorical.shape[0], 15))
epochs = 5
for i, (train_idx, cv_idx) in enumerate(folds):
    X_train = conv_dataset[train_idx]
    Y_train = y_categorical[train_idx]#train_full['target_id'].iloc[train_idx]
    X_cv = conv_dataset[cv_idx]
    Y_cv = y_categorical[cv_idx]#train_full['target_id'].iloc[cv_idx]
    cnn = createCNNModel(14)
    
    cnn.fit(X_train, Y_train,
              validation_data=(X_cv, Y_cv),
              nb_epoch=epochs, batch_size=64)
    