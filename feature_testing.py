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
import lightgbm as lgbm
import xgboost as xgb

do_prediction = False

########################### Data and Parameters Import ##########################
target_map, label_features, all_classes, all_class_weights \
    = dproc.getDataParameters()

train_meta, test_meta_data = dproc.getMetaData()
train = pd.read_csv('training_set.csv')
train_full, train_features = dproc.getFullData(train, train_meta)
del train
gc.collect()
#train_full[['object_id', 'period']].to_csv("train_periods.csv", index=False)
#target_id list: will be used in one hot encoding of labels
all_clmap_vals = np.array(list(target_map.values()))
print("Train Feats: {0}".format(train_features))
print("Train Data All COLS: {0}".format(train_full.columns))
############################# LOSS FUNCTION ####################################
def multiWeightedLoss(target_class, pred_class, no_class99=False):
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
    pred_class = pred_class/pred_class.sum(axis=1).reshape(-1,1)
    y_prediction = np.clip(pred_class, eps, 1 - eps)
    sum_loss_per_class = (y_truth_encoded * np.log(y_prediction)).sum(axis=0)
    object_per_class = np.clip(y_truth_encoded.sum(axis=0), 1, float('inf'))
    weighted_loss_class = (cl_weights * sum_loss_per_class)/object_per_class
    loss = np.sum(-weighted_loss_class.sum(axis=1)/cl_weights.sum(axis=1))
    return loss

def lgbMultiWeightedLoss(target_class, pred_class):
    classes = all_classes.copy()
    cl_vals = all_clmap_vals.copy()
    cl_weights = all_class_weights.copy()
    if len(np.unique(target_class)) < 15:
        classes = classes[:-1]
        cl_vals = cl_vals[:-1]
        del cl_weights['class_99']
    cl_weights = pd.DataFrame(cl_weights, index=[0])
    pred_class = pred_class.reshape(target_class.shape[0],
                                    len(classes), order='F')
    
    label_features = ['class_' + str(cl) for cl in classes]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(cl_vals.reshape(-1,1))
    y_truth_encoded = enc.transform(target_class.reshape(-1,1)).toarray()
    y_truth_encoded = pd.DataFrame(data=y_truth_encoded, columns=label_features)
    
#    for i, cl in enumerate(label_features):
#        train_full[cl] = y_truth_encoded.loc[:, cl]
    
    eps = 1e-15
    pred_class = pred_class/pred_class.sum(axis=1).reshape(-1,1)
    y_prediction = np.clip(pred_class, eps, 1 - eps)
    sum_loss_per_class = (y_truth_encoded * np.log(y_prediction)).sum(axis=0)
    object_per_class = np.clip(y_truth_encoded.sum(axis=0), 1, float('inf'))
    weighted_loss_class = (cl_weights * sum_loss_per_class)/object_per_class
    loss = -np.sum(weighted_loss_class.sum(axis=1)/cl_weights.sum(axis=1))
    return 'wloss', loss, False

def xgbMultiWeightedLoss(pred_class, target_class_dmatrix):
    classes = all_classes.copy()
    cl_vals = all_clmap_vals.copy()
    cl_weights = all_class_weights.copy()
    target_class = target_class_dmatrix.get_label()
    if len(np.unique(target_class)) < 15:
        classes = classes[:-1]
        cl_vals = cl_vals[:-1]
        del cl_weights['class_99']
    cl_weights = pd.DataFrame(cl_weights, index=[0])
#    pred_class = pred_class.reshape(target_class.shape[0], len(classes), order='F')
    
    label_features = ['class_' + str(cl) for cl in classes]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(cl_vals.reshape(-1,1))
    y_truth_encoded = enc.transform(target_class.reshape(-1,1)).toarray()
    y_truth_encoded = pd.DataFrame(data=y_truth_encoded, columns=label_features)
    
#    for i, cl in enumerate(label_features):
#        train_full[cl] = y_truth_encoded.loc[:, cl]
    
    eps = 1e-15
    pred_class = pred_class/pred_class.sum(axis=1).reshape(-1,1)
    y_prediction = np.clip(pred_class, eps, 1 - eps)
    sum_loss_per_class = (y_truth_encoded * np.log(y_prediction)).sum(axis=0)
    object_per_class = np.clip(y_truth_encoded.sum(axis=0), 1, float('inf'))
    weighted_loss_class = (cl_weights * sum_loss_per_class)/object_per_class
    loss = -np.sum(weighted_loss_class.sum(axis=1)/cl_weights.sum(axis=1))
    return 'wloss', loss

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

lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': ['multi_error'],
    "learning_rate": 0.02,
     "num_leaves": 30,
     "max_depth": 6,
     "feature_fraction": 0.45,
     "bagging_fraction": 0.3,
     "reg_alpha": 0.3,
     "reg_lambda": 0.3,
      "min_split_gain": 0.01,
      "min_child_weight": 0,
      "n_estimators": 2000
      }    

def train(train_features):
    lgbm_list = []
    oof_preds_lgbm = np.zeros((train_full.shape[0], 15))
    for i, (train_idx, cv_idx) in enumerate(folds):
        X_train = train_full[train_features].iloc[train_idx]
        Y_train = train_full['target_id'].iloc[train_idx]
        X_cv = train_full[train_features].iloc[cv_idx]
        Y_cv = train_full['target_id'].iloc[cv_idx]
        print ("\n\n" + "-"*20 + "Fold " + str(i+1) + "-"*20)
        
        print ("\n" + "*"*10 + "LightGBM" + "*"*10)
        clf_lgbm = lgbm.LGBMClassifier(**lgbm_params)
        
        clf_lgbm.fit(
            X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
            verbose=100,
            eval_metric=lgbMultiWeightedLoss,
            early_stopping_rounds=50,
        )
        
        # since there is no class 15 in train set,
        # the lightgbm will only predict 14 classes
        oof_preds_lgbm[cv_idx, :14] \
            = clf_lgbm.predict_proba(X_cv, num_iteration=clf_lgbm.best_iteration_)
    
        lgbm_list.append(clf_lgbm)
    
    
    print("LightGBM: {0}".format(multiWeightedLoss(train_full['target_id'],
                                                   oof_preds_lgbm)))
    return multiWeightedLoss(train_full['target_id'], oof_preds_lgbm)



excluded_features_orig = ['target', 'target_id', 'y', 'object_id', 'passband',
                     'hostgal_specz', 'distmod']
excluded_features_holder = ['mjd_size']
excluded_features_holder += ['flux_min', 'flux_max', 'flux_mean', 'flux_median',
                      'flux_std', 'flux_size', 'flux_skew']
excluded_features_holder += ['flux_err_min', 'flux_err_max', 'flux_err_mean',
                      'flux_err_median', 'flux_err_std', 'flux_err_skew']
excluded_features_holder += ['detected_mean']
excluded_features_holder += ['flux_ratio_sq_sum', 'flux_ratio_sq_skew']
excluded_features_holder += ['flux_by_flux_ratio_sq_sum', 'flux_by_flux_ratio_sq_skew']
excluded_features_holder += ['flux_standard_err']
excluded_features = excluded_features_orig.copy()
excluded_features += ['flux_by_flux_ratio_sq_skew' + '--' + str(i) for i in range(6)]

score_df = pd.DataFrame(columns=['removed_feat', 'score_pb_rem', 'score_add_agg'])
for i, f in enumerate(excluded_features_holder):
    print("="*30 + f + "="*30)
    excluded_features = excluded_features_orig.copy()
    excluded_features += excluded_features_holder
    excluded_features += [f + '--' + str(i) for i in range(6)]
    train_features = [f for f in train_full.columns if f not in excluded_features]
    score_df.loc[i, 'removed_feat'] = f
    score_df.loc[i, 'score_pb_rem'] = train(train_features)
    excluded_features.remove(f)
    train_features = [f for f in train_full.columns if f not in excluded_features]
    score_df.loc[i, 'score_add_agg'] = train(train_features)
    


