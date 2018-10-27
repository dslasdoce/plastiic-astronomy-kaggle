#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 08:55:59 2018

@author: dslasdoce
"""

import numpy as np
import pandas as pd
from time import sleep
train = pd.read_csv('training_set_metadata.csv')
train_meta_data = pd.read_csv('training_set_metadata.csv')
test_meta_data = pd.read_csv('test_set_metadata.csv')

# create list of all unique classes and add class 99 (undefiend class) to the list
classes = np.unique(train_meta_data['target'])
classes_all = np.hstack([classes, [99]])
classes_all

# create a dictionary {class : index} to map class number with the index 
# (index will be used for submission columns like 0, 1, 2 ... 14)

target_map = {j:i for i, j in enumerate(classes_all)}
#target_map

# create 'target_id' column to map with 'target' classes
# target_id is the index defined in previous step: see dictionary target_map
# this column will be used later as index for the columns in the final submission
#target_ids = [target_map[i] for i in train_meta_data['target']]
train_meta_data['target_id'] = train_meta_data['target'].map(target_map)
#train_meta_data.head()

# Build the flat probability arrays for both the galactic and extragalactic groups

# Extract galactic and Extragalactic classes
galactic_cut = train_meta_data['hostgal_specz'] == 0
galactic_data = train_meta_data[galactic_cut]
extragalactic_data = train_meta_data[~galactic_cut]

galactic_classes = np.unique(galactic_data['target_id'])
extragalactic_classes = np.unique(extragalactic_data['target_id'])
#print('Galactic classes:', galactic_classes)
#print('Extragalactic classes:', extragalactic_classes)

# Add class 99 (id=14) to both groups (Galactic and Extragalactic classes).
galactic_classes = np.append(galactic_classes, 14)
extragalactic_classes = np.append(extragalactic_classes, 14)
#print('Galactic classes:', galactic_classes)
#print('Extragalactic classes:', extragalactic_classes)

# create a 15 zeros array 'galactic_probabilities'
galactic_probabilities = np.zeros(15)
#print('Zeros for Galactic probabilities:', galactic_probabilities)

# suppose that probability for the Galactic object to have certain
# Galactic class is evenly distributed create an array of probabilities 
# for the galactic object to belong to a certain Galactic class
galactic_probabilities[galactic_classes] = 1 / len(galactic_classes)
#print('Galactic flat probabilities: ',galactic_probabilities)

# create a 15 zeros array 'extragalactic_probabilities'
extragalactic_probabilities = np.zeros(15)
#print('Zeros for Extragalactic probabilities:', extragalactic_probabilities)

# suppose that probability for the Extragalactic object to have certain
# Extragalactic class is evenly distributed create an array of probabilities
#for the galactic object to belong to a certain Extragalactic class
extragalactic_probabilities[extragalactic_classes] \
    = 1 / len(extragalactic_classes)
#print('Extragalactic flat probabilities: ', extragalactic_probabilities)

#galactic_probabilities = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
#galactic_probabilities[[0,7,14]] = 2
#galactic_probabilities = galactic_probabilities/18
#galactic_probabilities.astype(np.float16)
# Apply this prediction to a table

############################# LOSS FUNCTION ####################################
def multiWeightedLoss(target_class, pred_class, no_class99=False):
    target_map = {6: 0, 15:1, 16:2, 42:3, 52: 4, 53: 5, 62: 6, 64: 7,
                  65: 8, 67: 9, 88: 10, 90: 11, 92: 12, 95: 13, 99: 14}
    classes_all = np.array(list(target_map.keys()))
    class_map_values = np.array(list(target_map.values()))
    class_weights = {'class_' + str(cl):1 for cl in classes_all}
    class_weights['class_99'] = 2
    class_weights['class_64'] = 2
    class_weights['class_15'] = 2
    if no_class99 is True:
        classes_all = classes_all[:-1]
        class_map_values = class_map_values[:-1]
        del class_weights['class_99']
    class_weights = pd.DataFrame(class_weights, index=[0])
    
    label_features = ['class_' + str(cl) for cl in classes_all]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(class_map_values.reshape(-1,1))
    y_truth_encoded = enc.transform(target_class.values.reshape(-1,1)).toarray()
    y_truth_encoded = pd.DataFrame(data=y_truth_encoded, columns=label_features)
    
    for i, cl in enumerate(label_features):
        train_meta_data[cl] = y_truth_encoded.loc[:, cl]
    
    eps = 1e-15
    y_prediction = np.clip(pred_class, eps, 1 - eps)
    sum_loss_per_class = (y_truth_encoded * np.log(y_prediction)).sum(axis=0)
    object_per_class = np.clip(y_truth_encoded.sum(axis=0), 1, float('inf'))
    weighted_loss_class = (class_weights * sum_loss_per_class)/object_per_class
    loss = np.sum(-weighted_loss_class.sum(axis=1)/class_weights.sum(axis=1))
    return loss

def lgbMultiWeightedLoss(target_class, pred_class):
    target_map = {6: 0, 15:1, 16:2, 42:3, 52: 4, 53: 5, 62: 6, 64: 7,
                  65: 8, 67: 9, 88: 10, 90: 11, 92: 12, 95: 13, 99: 14}
    classes_all = np.array(list(target_map.keys()))
    class_map_values = np.array(list(target_map.values()))
    class_weights = {'class_' + str(cl):1 for cl in classes_all}
    class_weights['class_99'] = 2
    class_weights['class_64'] = 2
    class_weights['class_15'] = 2
    if len(np.unique(target_class)) < 15:
        classes_all = classes_all[:-1]
        class_map_values = class_map_values[:-1]
        del class_weights['class_99']
    class_weights = pd.DataFrame(class_weights, index=[0])
    pred_class = pred_class.reshape(target_class.shape[0], len(classes), order='F')
    
    label_features = ['class_' + str(cl) for cl in classes_all]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(class_map_values.reshape(-1,1))
    y_truth_encoded = enc.transform(target_class.reshape(-1,1)).toarray()
    y_truth_encoded = pd.DataFrame(data=y_truth_encoded, columns=label_features)
    
#    for i, cl in enumerate(label_features):
#        train_meta_data[cl] = y_truth_encoded.loc[:, cl]
    
    eps = 1e-15
    y_prediction = np.clip(pred_class, eps, 1 - eps)
    sum_loss_per_class = (y_truth_encoded * np.log(y_prediction)).sum(axis=0)
    object_per_class = np.clip(y_truth_encoded.sum(axis=0), 1, float('inf'))
    weighted_loss_class = (class_weights * sum_loss_per_class)/object_per_class
    loss = -np.sum(weighted_loss_class.sum(axis=1)/class_weights.sum(axis=1))
    return 'wloss', loss, False

# import progress bar package
import tqdm

def do_prediction(table, model=None):
    probs = []
    probs_model = []
    for index, row in tqdm.tqdm(table.iterrows(), total=len(table)):
        
        # we use 'hostgal_photoz' (photometric redshift) here instead of 
        #'hostgal_specz' (spectral redshift) it is the same redshift measure
        #but made faster on a larger area and thus less accurate,
        #but we have this measure for all of the objects 

        # if object is in the Milky Way Galaxy
        if row['hostgal_photoz'] == 0:
            prob = galactic_probabilities
            
        # if object is out of the Milky Way Galaxy
        else:
            prob = extragalactic_probabilities
#        prob = galactic_probabilities
        
        probs.append(prob)
#        print(row)
        if model:
            probs_model.append(model.predict(row.values.reshape(1, -1)))
            
    return np.array(probs), np.array(probs_model)


excluded_features = ['target', 'target_id', 'y', 'object_id']
train_features = [f for f in train_meta_data.columns if f not in excluded_features]

################################ Folding #######################################
from sklearn.model_selection import KFold

def getFolds(df_indeces=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get folds
    folds = KFold(n_splits=5, shuffle=False, random_state=15)
#    idx = np.arange(df.shape[0])
    fold_idx = []
    for train_idx, val_idx in folds.split(X=df_indeces, y=df_indeces):
        fold_idx.append([train_idx, val_idx])

    return fold_idx
folds = getFolds(train_meta_data.index)

################### LightGBM ########################
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 15,
    'metric': ['multi_error', multiWeightedLoss],
    "learning_rate": 0.05,
     "num_leaves": 60,
     "max_depth": 9,
     "feature_fraction": 0.45,
     "bagging_fraction": 0.3,
     "reg_alpha": 0.15,
     "reg_lambda": 0.15,
#      "min_split_gain": 0,
      "min_child_weight": 0
      }

from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgbm

def predictBatch(dataset, model, batchsize=1000):
    iter_size = int(dataset.shape[0]/batchsize)
    remaining = dataset.shape[0]%batchsize
    predictions = np.zeros((dataset.shape[0], 15))
    for i in tqdm.tqdm(range(iter_size)):
#        print(iter_size)
        predictions[i*batchsize:i*batchsize + batchsize, :] \
            = model.predict(dataset[i*batchsize:i*batchsize + batchsize])
    
    if remaining > 0:
        predictions[-remaining:, :] = model.predict(dataset[-remaining:])
        
    return predictions
        
    
trainset_prediction = np.zeros((train_meta_data.shape[0], 15))
test_prediction = np.zeros((test_meta_data.shape[0], 15))
for i, (train_idx, cv_idx) in enumerate(folds):
    X_train = train_meta_data[train_features].iloc[train_idx]
    Y_train = train_meta_data['target_id'].iloc[train_idx]
    X_cv = train_meta_data[train_features].iloc[cv_idx]
    Y_cv = train_meta_data['target_id'].iloc[cv_idx]
    print ("-"*20 + "Fold " + str(i+1) + "-"*20)

    lgtrain = lgbm.Dataset(X_train, Y_train, categorical_feature="auto")
    lgtvalid = lgbm.Dataset(X_cv, Y_cv, categorical_feature="auto")
    
    bst = lgbm.train(lgbm_params, lgtrain, num_boost_round=5000,
                     valid_sets=[lgtvalid], verbose_eval=50,
                     early_stopping_rounds=100)
    trainset_prediction[cv_idx] = bst.predict(X_cv, verbose_eval=100)
    
    sleep(0.5)
    print("Predict CV")
    trainset_prediction[cv_idx] = predictBatch(X_cv, bst)
    sleep(0.5)
#    print("Predict Test")
#    _preds = predictBatch(test_meta_data[train_features], bst)
#    test_prediction += _preds/ len(folds)

class_weights = {'class_' + str(cl):1 for cl in classes_all}
class_weights['class_99'] = 2
class_weights['class_64'] = 2
class_weights['class_15'] = 2
class_weights = pd.DataFrame(class_weights, index=[0])


#np.apply_along_axis(lambda a: a/a.sum(), arr=test_pred, axis=1)

############################## one class SVM ################################### 
#train_meta_data['galactic'] = train_meta_data['hostgal_specz'].map(lambda a: 0 if a==0 else 1)
#test_meta_data['galactic'] = test_meta_data['hostgal_specz'].map(lambda a: 0 if a==0 else 1)
#
#test_meta_data = test_meta_data.fillna(0)
#train_meta_data = train_meta_data.fillna(0)
#
#
#import sklearn.svm as svm
#oneclass = svm.OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05, verbose=True)
#oneclass.fit(train_meta_data[train_features])
#test_pred_base, test_pred_model = do_prediction(test_meta_data[train_features], oneclass)
#test_pred = test_prediction.copy()
##set detected class 99 to 1, 1 = within the training set class, -1 = anomaly(class 99 )
#test_pred[np.array(test_pred_model!=1).reshape(-1), 14] = 1

########################## #create submission file #############################
#test_df = pd.DataFrame(index=test_meta_data['object_id'], data=test_pred,
#                       columns=label_features)
#test_df.to_csv('./naive_benchmark.csv')


#import pandas as pd
#pd.read_csv('sample_submission.csv', index_col='object_id',
#            converters={'class_15': lambda p:2,'class_64': lambda p:2,'class_99': lambda p:2})\
#            .replace(0,1).div(18, axis=0).to_csv('sub_weighted.csv', float_format='%.5f')

sleep(0.5)
print("\ntrain weighted score: ", end='')
print(multiWeightedLoss(train_meta_data['target_id'], trainset_prediction))