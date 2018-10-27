#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 08:27:51 2018

@author: dslasdoce
"""
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Dense, Dropout, LSTM, concatenate
from keras.models import Model, Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
from keras import regularizers
from sklearn.model_selection import StratifiedKFold
import dataproc as dproc
import numpy as np
import pandas as pd


train = pd.read_csv('training_set.csv')
target_map, label_features, all_classes, all_class_weights \
    = dproc.getDataParameters()
    
train_meta, test_meta_data = dproc.getMetaData()
train_full = pd.read_csv('training_set.csv')
train_full, train_meta_set, train_features_ts, train_features_meta\
    = dproc.getFullDataTS(ts_data=train_full, meta_data=train_meta, label=True)
all_clmap_vals = np.array(list(target_map.values()))

dropout = 0.3
#model_ts = Sequential()
#model_ts.add(LSTM(32, return_sequences=True,
#                  dropout=dropout,
#                  input_shape=(None, len(train_features_ts)),
#                  name='ts_lstm_hid1'))
#model_ts.add(LSTM(16, return_sequences=False,
#                  dropout=dropout,
#                  input_shape=(None, len(train_features_ts)),
#                  name='ts_lstm_hid2'))
#model_ts.add(Dense(14, activation='softmax', name='ts_lstm_out'))
input_ts = Input(shape=(None,len(train_features_ts)), name='ts_input')
input_meta = Input(shape=(len(train_features_meta),), name='meta_input')
ts_layers = LSTM(32, return_sequences=True,
                  dropout=dropout,
                  name='ts_lstm_hid1')(input_ts)
ts_layers = LSTM(16, return_sequences=False,
                  dropout=dropout,
                  name='ts_lstm_hid2')(ts_layers)
#ts_layers = Dense(14, activation='softmax', name='ts_lstm_out')(ts_layers)

meta_layers = Dense(32, activation='relu', name='ts_lstm_out')(input_meta)
merged_output = concatenate([ts_layers, meta_layers], name ="multi_merge")
merged_output = Dense(14, activation='softmax',
                      name='merged_output')(merged_output)
model_ts = Model(inputs=[input_ts, input_meta], outputs=merged_output)
plot_model(model_ts, show_shapes=True, to_file='model_multi.png')
    
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(handle_unknown='ignore')
onehot.fit(np.arange(14).reshape(-1,1))
#y_true = onehot.transform(target_set['target_id']).toarray()
#
from keras.optimizers import Adam
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
           amsgrad=False)
dau_list = []
def getFolds(ser_target=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get folds
    folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=13)
#    idx = np.arange(df.shape[0])
    fold_idx = []
    for train_idx, val_idx in folds.split(X=ser_target, y=ser_target):
        fold_idx.append([train_idx, val_idx])

    return fold_idx

folds = getFolds(train_meta_set['target_id'])

def multiWeightedLoss(target_class, pred_class, no_class99=False):
    #remove class 99
    classes = all_classes.copy()
    cl_vals = all_clmap_vals.copy()
    cl_weights = all_class_weights.copy()
    if no_class99 is True:
        classes = classes[:-1]
        cl_vals = cl_vals[:-1]
        del cl_weights['class_99']
#        
#    #make dataframe of weights so the operations are broadcasted by columns
    cl_weights = pd.DataFrame(cl_weights, index=[0])
#    
    tmp_labels = ['class_' + str(cl) for cl in classes]
#    enc = OneHotEncoder(handle_unknown='ignore')
#    enc.fit(cl_vals.reshape(-1,1))
#    y_truth_encoded = enc.transform(target_class.values.reshape(-1,1)).toarray()
    print(target_class)
    y_truth_encoded = pd.DataFrame(data=target_class, columns=tmp_labels)
        
    eps = 1e-15
    #make sum of probability distribution = 1
    pred_class = pred_class/pred_class.sum(axis=1).reshape(-1,1)
    y_prediction = np.clip(pred_class, eps, 1 - eps)
    sum_loss_per_class = (y_truth_encoded * np.log(y_prediction)).sum(axis=0)
    object_per_class = np.clip(y_truth_encoded.sum(axis=0), 1, float('inf'))
    weighted_loss_class = (cl_weights * sum_loss_per_class)/object_per_class
    loss = np.sum(-weighted_loss_class.sum(axis=1)/cl_weights.sum(axis=1))
    return loss

#import dataproc as dproc
oof_pred = np.zeros((train_full.shape[0], 14))
for i, (train_idx, cv_idx) in enumerate(folds):
    X_train_ts = train_full.iloc[train_idx]
    X_train_meta = train_meta_set[train_features_meta].iloc[train_idx]
    Y_train = train_meta_set['target_id'].iloc[train_idx]
    X_cv_ts = train_full.iloc[cv_idx]
    X_cv_meta = train_meta_set[train_features_meta].iloc[cv_idx]
    Y_cv = train_meta_set['target_id'].iloc[cv_idx]
    
#    break
    #for classification
    Y_train = onehot.transform(Y_train.values.reshape(-1, 1)).toarray()
    Y_cv = onehot.transform(Y_cv.values.reshape(-1, 1)).toarray()
    traingen = dproc.dataGenerator(X_train_ts, X_train_meta, Y_train)
    cvgen = dproc.dataGenerator(X_cv_ts, X_cv_meta, Y_cv)
    # this model maps an input to its reconstruction
    model_ts = Model(inputs=[input_ts, input_meta], outputs=merged_output)
    model_ts.compile(optimizer=opt, loss='mse')
    modelfile = "ts_model-{0}.h5".format(i)
    model_checkpoint = ModelCheckpoint(modelfile, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(patience=20)
    model_ts.fit_generator(generator=traingen,  
                    epochs=200,
                    callbacks=[model_checkpoint, early_stop],
                    verbose=True,
                    validation_data=cvgen)
    try:
        predgen = dproc.dataGenerator(X_cv_ts, X_cv_meta, Y_cv, pred=True)
        oof_pred[cv_idx] = model_ts.predict_generator(predgen)
    except:
        pass
#    break
#    model_ts = load_model(modelfile)
#    dau_list.append(model_ts)
#    oof_pred[cv_idx] = model_ts.predict(X_cv)
#
        
for df in X_train_ts:
    if df.isnull().values.any() is True:
        print("ERRR")
    
    
#import seaborn as sns
##import matplotlib.pyplot as plt   
#import pandas as pd 
#train_data = train_full[train_features]
#dist = np.log10(np.linalg.norm(train_data-oof_reconstructed, axis=1) + 1)
#sns.distplot(dist)
#desc = pd.DataFrame(columns=['error', 'z-score'])
#desc['error'] = dist
#desc['z-score'] =  (dist - dist.mean())/dist.std()
#
#import gc
#import time
#train_mean = train_full.mean(axis=0)
#del train_full, train_idx
#gc.collect()
#start = time.time()
#chunks = 5000000
#chunk_last = pd.DataFrame() 
#test_row_num = 453653104 
#total_steps = int(np.ceil(test_row_num/chunks))
#
##del train_full
#for i_c, df in enumerate(pd.read_csv('/media/dslasdoce/Data/Astro/test_set.csv',
#                                     chunksize=chunks, iterator=True)):
#    #make sure object_ids do not get separated
#    print("*"*20 + "chunk: " + str(i_c) + "*"*20)
#    df = pd.concat([chunk_last, df], ignore_index=True)
#    if i_c+1<total_steps:
#        #get the last object id
#        id_last = df['object_id'].values[-1] 
#        #get boolean indeces of rows with object_id == id_last
#        mask_last = (df['object_id']==id_last).values 
#        #get the rows with last object_id
#        chunk_last = df[mask_last] 
#        #remove the rows of the last object_id from the dataset
#        df = df[~mask_last]
#    
#    # Group by object id
#    agg_test = df.groupby('object_id').agg(aggs)
#    agg_test.columns = new_columns
#    agg_test['mjd_length'] = agg_test['mjd_max'] - agg_test['mjd_min']
##    agg_test = agg_test.drop(['mjd_max', 'mjd_min'], axis=1)
##    agg_test['mjd_diff'] = agg_test['mjd_max'] - agg_test['mjd_min']
#    del agg_test['mjd_max'], agg_test['mjd_min']
#    del df
#    gc.collect()
#    
#    # Merge with meta data
#    full_test = agg_test.reset_index().merge(
#        right=test_meta_data,
#        how='left',
#        on='object_id'
#    )
#    full_test = full_test.fillna(train_mean)
#    
#    # Make predictions
#    preds = None
#    for dau in dau_list:
#        if preds is None:
#            preds = dau.predict(full_test[train_features]) / len(folds)
#        else:
#            preds += dau.predict(full_test[train_features]) / len(folds)
#    preds = np.log10(np.linalg.norm(full_test[train_features] - preds, axis=1) + 1)
#    
#    preds = pd.DataFrame(preds, columns=['L2DistLog'])
#    preds['object_id'] = full_test['object_id']
#    if i_c == 0:
#        preds.to_csv('predictions_dau.csv', index=False)
#    else: 
#        preds.to_csv('predictions_dau.csv',
#                        header=False, mode='a', index=False)
#    
#    if (i_c + 1) % 10 == 0:
#        print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
##load all autoencoder distance predictions
#preds_dau = pd.read_csv('predictions_dau.csv')
##calculae z-score
#preds_dau['z-score'] \
#    =  (preds_dau['L2DistLog'] - (preds_dau['L2DistLog'].mean()))\
#       /preds_dau['L2DistLog'].std()
#sns.distplot(preds_dau['L2DistLog'])
#
##set histogram outliers to 1 / class_99
#preds_all = pd.read_csv('predictions_comb_naive99.csv')
#preds_all.loc[preds_all['object_id']\
#              .isin(preds_dau.loc[preds_dau['z-score']>2.5, 'object_id']),
#              'class_99'] = 1
#preds_all.to_csv('predictions_comb_naive_dau.csv', index=False)