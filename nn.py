#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 08:55:59 2018

@author: dslasdoce
"""

import numpy as np
import pandas as pd
import gc
import dataproc as dproc
from sklearn.preprocessing import OneHotEncoder

do_prediction = False
loaded_test = False
########################### Data and Parameters Import ##########################
target_map, label_features, all_classes, all_class_weights \
    = dproc.getDataParameters()

train_meta, test_meta_data = dproc.getMetaData()
train = pd.read_csv('input/training_set.csv')
train_full, train_features = dproc.getFullData(train, train_meta, perpb=False)
del train
gc.collect()
#train_full[['object_id', 'period']].to_csv("train_periods.csv", index=False)
#target_id list: will be used in one hot encoding of labels
all_clmap_vals = np.array(list(target_map.values()))
print("Train Feats: {0}".format(train_features))
print("Train Data All COLS: {0}".format(train_full.columns))

calc_feats = ['period', 'power', 'Eta_e']
label_lc_feats = ['object_id']
for f in calc_feats:
    for i in range(6):
        label_lc_feats.append(f + '--' + str(i))
############################# LOSS FUNCTION ####################################
def multiWeightedLoss(target_class, pred_class, no_class99=True):
    #remove class 99
    classes = all_classes.copy()
    cl_vals = all_clmap_vals.copy()
    cl_weights = all_class_weights.copy()
    if no_class99 is True:
        classes = classes[:-1]
        cl_vals = cl_vals[:-1]
        del cl_weights['class_99']
        
    #make dataframe of weights so the operations are broadcasted by columns
    cl_weights = pd.DataFrame(cl_weights, index=[0])
    
    tmp_labels = ['class_' + str(cl) for cl in classes]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(cl_vals.reshape(-1,1))
    y_truth_encoded = enc.transform(target_class.values.reshape(-1,1)).toarray()
    y_truth_encoded = pd.DataFrame(data=y_truth_encoded, columns=tmp_labels)
        
    eps = 1e-15
    pred_class = pred_class/pred_class.sum(axis=1).reshape(-1,1) #normalize to max 1
    y_prediction = np.clip(pred_class, eps, 1 - eps) #limit values
    sum_loss_per_class = (y_truth_encoded * np.log(y_prediction)).sum(axis=0)
    object_per_class = np.clip(y_truth_encoded.sum(axis=0), 1, float('inf'))
    weighted_loss_class = (cl_weights * sum_loss_per_class)/object_per_class
    loss = np.sum(-weighted_loss_class.sum(axis=1)/cl_weights.sum(axis=1))
    return loss

#loss_weigths = [all_class_weights[cl] for cl in label_features ]
#loss_weigths = np.array(loss_weigths[:-1], dtype=np.float32)
#def mywloss(y_true,y_pred):  
#    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
#    sum_loss_per_class = tf.reduce_sum(y_true*tf.log(yc),axis=0)
#    object_per_class = tf.clip_by_value(tf.reduce_sum(y_true, axis=0), 1, float('inf'))
#    loss=-(tf.reduce_mean(loss_weigths*sum_loss_per_class/object_per_class))/tf.reduce_sum(loss_weigths)
#    return loss
################################## NN ##########################################
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
#import keras
#from keras import regularizers
from collections import Counter
#from sklearn.metrics import confusion_matrix
K.clear_session()
def build_model(dropout_rate=0.25,activation='relu'):
    start_neurons = 512
    # create model
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=train_full[train_features].shape[1], activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//2,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//4,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//8,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate/2))
    
    model.add(Dense(len(all_classes), activation='softmax'))
    return model

unique_y = all_classes#np.unique(y)
class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i
        
y_map = train_full['target_id']#np.zeros((y.shape[0],))
#y_map = np.array([class_map[val] for val in y])
ohc = OneHotEncoder(categories=[range(15)])
ohc.fit(all_clmap_vals.reshape(-1, 1))
y_categorical = ohc.transform(y_map.values.reshape(-1,1)).toarray()#to_categorical(y_map)

y_count = Counter(y_map)
wtable = np.zeros((len(unique_y),))
for i in range(len(unique_y)):
    wtable[i] = y_count[i]/y_map.shape[0]
wtable[-1] = 1   
def mywloss(y_true, y_pred):  
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss

################################ Folding #######################################
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

folds = getFolds(train_full['target_id'])

#fill na
train_mean = train_full.mean(axis=0)
train_full.fillna(train_mean, inplace=True)
#scale
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_full[train_features] = train_full[train_features].clip(upper=10000000)
train_full[train_features] = ss.fit_transform(train_full[train_features])


import matplotlib.pyplot as plt
def plot_loss_acc(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'][1:], c='b')
    ax.plot(history.history['val_loss'][1:], c='r')
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show(block=False)
    
    ax.plot(history.history['acc'][1:], c='b')
    ax.plot(history.history['val_acc'][1:], c='r')
    plt.title('model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show(block=False)
    
################### Training ########################   
nn_list = []
oof_preds_nn = np.zeros((train_full.shape[0], 15))
epochs = 600
batch_size = 100

early_stopping = EarlyStopping(patience=50, verbose=1)
for i, (train_idx, cv_idx) in enumerate(folds):
    X_train = train_full[train_features].iloc[train_idx]
    Y_train = y_categorical[train_idx]#train_full['target_id'].iloc[train_idx]
    X_cv = train_full[train_features].iloc[cv_idx]
    Y_cv = y_categorical[cv_idx]#train_full['target_id'].iloc[cv_idx]
    print ("\n\n" + "-"*20 + "Fold " + str(i+1) + "-"*20)
    clf_nn = build_model(dropout_rate=0.5,activation='tanh')
    clf_nn.compile(loss=mywloss, optimizer='adam', metrics=['accuracy'])
    checkPoint = ModelCheckpoint("./keras.model",monitor='val_loss',
                             mode = 'min', save_best_only=True, verbose=0)
    history = clf_nn.fit(X_train, Y_train,
                    validation_data=[X_cv, Y_cv], 
                    epochs=epochs,
                    batch_size=batch_size,shuffle=True,verbose=0,
                    callbacks=[checkPoint, early_stopping])
    plot_loss_acc(history)
    
    print('Loading Best Model')
    clf_nn.load_weights('./keras.model')
    # since there is no class 15 in train set,
    # the lightgbm will only predict 14 classes
    oof_preds_nn[cv_idx] = clf_nn.predict_proba(X_cv, batch_size=batch_size)
#    print(multi_weighted_logloss(y_valid, model.predict_proba(x_valid,batch_size=batch_size)))

    print(multiWeightedLoss(train_full['target_id'].iloc[cv_idx],
                            clf_nn.predict_proba(X_cv,batch_size=batch_size)[:, :14]))
    nn_list.append(clf_nn)

print("NN: {0}".format(multiWeightedLoss(train_full['target_id'],
                                         oof_preds_nn[:, :14])))
for preds, name in zip([oof_preds_nn], ['nn']):
    df = pd.DataFrame(data=preds, columns=label_features)
    df['object_id'] = train_full['object_id']
    df['target_id'] = train_full['target_id']
    df['target'] = train_full['target']
    df.to_csv('output/oof_{0}.csv'.format(name), index=False)
    
    
######################### #create submission file #############################
temp_label_features = label_features.copy()
temp_label_features.remove("class_99")
if loaded_test is True:
    chunks = 200000
    for i_c, full_test in enumerate(pd.read_csv('/media/dslasdoce/Data/Astro/full_test_saved.csv',
                                         chunksize=chunks, iterator=True)):
        print("*"*20 + "chunk: " + str(i_c) + "*"*20)

        full_test = full_test.fillna(train_mean)
        full_test[train_features] = full_test[train_features].clip(upper=10000000)
        full_test[train_features] = ss.transform(full_test[train_features])
        full_test = full_test.reset_index(drop=True)
        
        # Make predictions
        print("Predicting...")
        preds_nn = None
        for clf_nn in nn_list:
            if preds_nn is None:
                preds_nn = clf_nn.predict_proba(full_test[train_features]) / len(folds)
            else:
                preds_nn \
                    += clf_nn.predict_proba(full_test[train_features]) / len(folds)
        # Store predictions
        print("Saving predictions...")
         #create prediction file for each model
#        for _pred, name in zip([preds_lgb, preds_xgb, preds_comb], ['lgb', 'xgb', 'comb']):
        for _pred, name in zip([preds_nn], ['nn']):
            preds_df = pd.DataFrame(_pred, columns=[label_features])
            preds_df['object_id'] = full_test['object_id']
            preds_df['class_99'] = 0.1
            
            if i_c == 0:
                preds_df.to_csv('output/ld-nn_predictions_{0}.csv'.format(name), index=False)
            else: 
                preds_df.to_csv('output/ld-nn_predictions_{0}.csv'.format(name),
                                header=False, mode='a', index=False)
            del preds_df
            gc.collect()
            
    do_prediction = False
    
if do_prediction is True:
    import time
    del train_full, train_idx
    gc.collect()
    start = time.time()
    chunks = 6000000
    chunk_last = pd.DataFrame() 
    test_row_num = 453653104 
    total_steps = int(np.ceil(test_row_num/chunks))
    #temporarily remove class 99 since it will predicted separately
    #del train_full
    for i_c, df in enumerate(pd.read_csv('/media/dslasdoce/Data/Astro/test_set.csv',
                                         chunksize=chunks, iterator=True)):
        #make sure object_ids do not get separated
        print("*"*20 + "chunk: " + str(i_c) + "*"*20)
        df = pd.concat([chunk_last, df], ignore_index=True)
        if i_c+1<total_steps:
            #get the last object id
            id_last = df['object_id'].values[-1] 
            #get boolean indeces of rows with object_id == id_last
            mask_last = (df['object_id']==id_last).values 
            #get the rows with last object_id
            chunk_last = df[mask_last] 
            #remove the rows of the last object_id from the dataset
            df = df[~mask_last]
        
        full_test, train_feats = dproc.getFullData(ts_data=df,
                                                      meta_data=test_meta_data)
        del df
        gc.collect()        
        
        if i_c == 0:
            full_test.to_csv("/media/dslasdoce/Data/Astro/full_test_saved.csv", index=False)
        else: 
            full_test.to_csv("/media/dslasdoce/Data/Astro/full_test_saved.csv",
                             index=False, header=False, mode='a')
            
        full_test = full_test.fillna(train_mean)
        full_test[train_features] = full_test[train_features].clip(upper=10000000)
        full_test[train_features] = ss.transform(full_test[train_features])
        
        # Make predictions
        print("Predicting...")
        
        preds_nn = None
        for clf_nn in nn_list:
            if preds_nn is None:
                preds_nn = clf_nn.predict_proba(full_test[train_features]) / len(folds)
            else:
                preds_nn \
                    += clf_nn.predict_proba(full_test[train_features]) / len(folds)
        # Store predictions
        print("Saving predictions...")
         #create prediction file for each model
#        for _pred, name in zip([preds_lgb, preds_xgb, preds_comb], ['lgb', 'xgb', 'comb']):
        for _pred, name in zip([preds_nn], ['nn']):
            preds_df = pd.DataFrame(_pred, columns=label_features)
            preds_df['object_id'] = full_test['object_id']
            preds_df['class_99'] = 0.1
            
            if i_c == 0:
                preds_df.to_csv('output/nn_predictions_{0}.csv'.format(name), index=False)
            else: 
                preds_df.to_csv('output/nn_predictions_{0}.csv'.format(name),
                                header=False, mode='a', index=False)
            del preds_df
            gc.collect()
            
        del preds_nn
        gc.collect()
        
        if (i_c + 1) % 10 == 0:
            print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

if do_prediction is True or loaded_test is True:
    model = 'output/ld-nn_predictions_nn'
    z = pd.read_csv(model + '.csv')
    
    preds_99 = np.ones(z.shape[0])
    no_99 = label_features.copy()
    no_99.remove('class_99')
    for i in range(z[no_99].values.shape[1]):
        preds_99 *= (1 - z[no_99].values[:, i])
    z['class_99'] = 0.18 * preds_99 / np.mean(preds_99)

    cols = list(z.columns)
    cols.remove('object_id')
#    z['class_99'] = 1 - z[cols].max(axis=1)
    #z = z[['object_id'] + label_features]
#    tech ='_99sc'
#    z.to_csv(model + tech + '.csv', index=False)
#    label_features_exgal
    gal_classes = train_meta['target'].loc[train_meta['hostgal_specz'] == 0].unique()
    gal_classes = all_classes[np.isin(all_classes, gal_classes)]
    exgal_classes = train_meta['target'].loc[train_meta['hostgal_specz'] != 0].unique()
    exgal_classes = all_classes[np.isin(all_classes, exgal_classes)]
    label_features_gal = ['class_' + str(cl) for cl in gal_classes]
    label_features_exgal = ['class_' + str(cl) for cl in exgal_classes]
    gal_objs = test_meta_data\
                .loc[test_meta_data['hostgal_photoz']==0, 'object_id']
    exgal_objs = test_meta_data\
            .loc[test_meta_data['hostgal_photoz']>0, 'object_id']
    z.loc[z['object_id'].isin(gal_objs), label_features_exgal] = 0
    z.loc[z['object_id'].isin(exgal_objs), label_features_gal] = 0
    
    tech ='_scgal'
    z.to_csv(model + tech + '.csv', index=False)
    
#    z = pd.read_csv('full_test_saved.csv')
