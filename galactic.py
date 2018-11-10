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
gal_classes = train_meta['target'].loc[train_meta['hostgal_photoz'] == 0].unique()
gal_classes = all_classes[np.isin(all_classes, gal_classes)]
exgal_classes = train_meta['target'].loc[train_meta['hostgal_photoz'] != 0].unique()
exgal_classes = all_classes[np.isin(all_classes, exgal_classes)]
mode = None

label_features_gal = ['class_' + str(cl) for cl in gal_classes]
label_features_exgal = ['class_' + str(cl) for cl in exgal_classes]
del train
gc.collect()
#train_full[['object_id', 'period']].to_csv("train_periods.csv", index=False)
#target_id list: will be used in one hot encoding of labels
#all_clmap_vals = np.array(list(target_map.values()))
print("Train Feats: {0}".format(train_features))
print("Train Data All COLS: {0}".format(train_full.columns))
############################# LOSS FUNCTION ####################################
def multiWeightedLoss(target_class, pred_class, no_class99=False):
    #remove class 99
    classes = all_classes.copy()
#    cl_vals = all_clmap_vals.copy()
    cl_weights = all_class_weights.copy()
    if no_class99 is True:
        classes = classes[:-1]
#        cl_vals = cl_vals[:-1]
        del cl_weights['class_99']
        
    #make dataframe of weights so the operations are broadcasted by columns
    cl_weights = pd.DataFrame(cl_weights, index=[0])
    
    tmp_labels = ['class_' + str(cl) for cl in classes]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(classes.reshape(-1,1))
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
    cl_weights = all_class_weights.copy()
    if mode == 'galactic':
        label_features = ['class_' + str(cl) for cl in gal_classes]
    elif mode == 'extra_galactic':
        label_features = ['class_' + str(cl) for cl in exgal_classes]
    else:
         label_features = ['class_' + str(cl) for cl in all_classes]
         label_features.remove('class_99')
    cl_weights = {key:val for key, val in cl_weights.items()\
                  if key in label_features}
    cl_weights = pd.DataFrame(cl_weights, index=[0])
        
    pred_class = pred_class.reshape(target_class.shape[0],
                                    len(label_features), order='F')
    
    enc = OneHotEncoder(handle_unknown='ignore')    
    y_truth_encoded = enc.fit_transform(target_class.reshape(-1,1)).toarray()
    y_truth_encoded = pd.DataFrame(data=y_truth_encoded, columns=label_features)
    
    eps = 1e-15
    pred_class = pred_class/pred_class.sum(axis=1).reshape(-1,1)
    y_prediction = np.clip(pred_class, eps, 1 - eps)
    sum_loss_per_class = (y_truth_encoded * np.log(y_prediction)).sum(axis=0)
    object_per_class = np.clip(y_truth_encoded.sum(axis=0), 1, float('inf'))
    weighted_loss_class = (cl_weights * sum_loss_per_class)/object_per_class
    loss = -np.sum(weighted_loss_class.sum(axis=1)/cl_weights.sum(axis=1))
    return 'wloss', loss, False

def xgbMultiWeightedLoss(pred_class, target_class_dmatrix):
    cl_weights = all_class_weights.copy()
    target_class = target_class_dmatrix.get_label()

    if mode == 'galactic':
        label_features = ['class_' + str(cl) for cl in gal_classes]
    elif mode == 'extra_galactic':
        label_features = ['class_' + str(cl) for cl in exgal_classes]
    else:
         label_features = ['class_' + str(cl) for cl in all_classes]
         label_features.remove('class_99')
    cl_weights = {key:val for key, val in cl_weights.items()\
                  if key in label_features}
    cl_weights = pd.DataFrame(cl_weights, index=[0])
    
    enc = OneHotEncoder(handle_unknown='ignore')
    y_truth_encoded = enc.fit_transform(target_class.reshape(-1,1)).toarray()
    y_truth_encoded = pd.DataFrame(data=y_truth_encoded, columns=label_features)
    
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

################### LightGBM ########################
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': ['multi_error'],
    "learning_rate": 0.02,
     "num_leaves": 60,
     "max_depth": 6,
     "feature_fraction": 0.45,
     "bagging_fraction": 0.3,
     "reg_alpha": 0.15,
     "reg_lambda": 0.15,
#      "min_split_gain": 0,
      "min_child_weight": 0,
      "n_estimators": 2000
      }

xgb_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 15,
    'metric': ['multi_error'],
    "learning_rate": 0.05,
     "num_leaves": 60,
     "max_depth": 9,
     "feature_fraction": 0.45,
     "bagging_fraction": 0.3,
     "reg_alpha": 0.15,
     "reg_lambda": 0.15,
#      "min_split_gain": 0,
      "min_child_weight": 0,
      "n_estimators": 1000
      }

############################### Galactic #######################################
train_full_gal = train_full.loc[train_meta['hostgal_photoz'] == 0]\
                 .reset_index(drop=True)
folds = getFolds(train_full_gal['target'])
lgbm_list = []
xgb_list = []
oof_preds_lgbm = np.zeros((train_full_gal.shape[0], 15))
oof_preds_lgbm = pd.DataFrame(data=oof_preds_lgbm, columns=label_features)
oof_preds_lgbm['target'] = np.nan
oof_preds_xgb = np.zeros((train_full_gal.shape[0], 15))
oof_preds_xgb = pd.DataFrame(data=oof_preds_xgb, columns=label_features)
oof_preds_xgb['target'] = np.nan
imp_lgb = pd.DataFrame()
imp_xgb = pd.DataFrame()
mode = 'galactic'
for i, (train_idx, cv_idx) in enumerate(folds):
    X_train = train_full_gal[train_features].iloc[train_idx]
    Y_train = train_full_gal['target'].iloc[train_idx]
    X_cv = train_full_gal[train_features].iloc[cv_idx]
    Y_cv = train_full_gal['target'].iloc[cv_idx]
    print ("\n\n" + "-"*20 + "Fold " + str(i+1) + "-"*20)
    
    print("*"*10 + "XGBoost" + "*"*10)
    clf_xgb = xgb.XGBClassifier(**xgb_params)
    clf_xgb.fit(
        X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
        verbose=100,
        eval_metric=xgbMultiWeightedLoss,
        early_stopping_rounds=50,
    )
#    imp_df = pd.DataFrame()
#    imp_df['feature'] = train_features
#    imp_df['gain'] = clf_xgb.feature_importances_(importance_type='gain')
#    imp_df['fold'] = i + 1
#    imp_xgb = pd.concat([imp_xgb, imp_df], axis=0, sort=False)
    
    print ("\n" + "*"*10 + "LightGBM" + "*"*10)
    clf_lgbm = lgbm.LGBMClassifier(**lgbm_params)
    
    clf_lgbm.fit(
        X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
        verbose=100,
        eval_metric=lgbMultiWeightedLoss,
        early_stopping_rounds=50,
    )
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = clf_lgbm.booster_.feature_importance(importance_type='gain')
    imp_df['fold'] = i + 1
    imp_lgb = pd.concat([imp_lgb, imp_df], axis=0, sort=False)
    
    # since there is no class 15 in train set,
    # the lightgbm will only predict 14 classes
    oof_preds_lgbm.loc[cv_idx, label_features_gal] \
        = clf_lgbm.predict_proba(X_cv, num_iteration=clf_lgbm.best_iteration_)
    oof_preds_xgb.loc[cv_idx, label_features_gal] \
        = clf_xgb.predict_proba(X_cv)
    oof_preds_lgbm.loc[cv_idx, 'target'] = Y_cv
    oof_preds_xgb.loc[cv_idx, 'target'] = Y_cv
    
    lgbm_list.append(clf_lgbm)
    xgb_list.append(clf_xgb)
    
oof_preds_comb_gal = 0.7*oof_preds_lgbm[label_features] + 0.3*oof_preds_xgb[label_features]
oof_preds_comb_gal['target'] = train_full_gal['target'].astype(int)
oof_preds_comb_gal['object_id'] = train_full_gal['object_id']

print("\n" + "-"*20)
print("LightGBM: {0}"\
      .format(multiWeightedLoss(oof_preds_comb_gal['target'],
                                oof_preds_lgbm[label_features].values)))
print("XGBoost: {0}"\
      .format(multiWeightedLoss(oof_preds_comb_gal['target'],
                                oof_preds_xgb[label_features].values)))
print("Combined: {0}"\
      .format(multiWeightedLoss(oof_preds_comb_gal['target'],
                                oof_preds_comb_gal[label_features].values)))

xgb_gals = xgb_list
lgb_gals = lgbm_list

############################### Ex Galactic ####################################
train_full_exgal = train_full.loc[train_meta['hostgal_photoz'] != 0]\
                 .reset_index(drop=True)
folds = getFolds(train_full_exgal['target'])
lgbm_list = []
xgb_list = []
oof_preds_lgbm = np.zeros((train_full_exgal.shape[0], 15))
oof_preds_lgbm = pd.DataFrame(data=oof_preds_lgbm, columns=label_features)
oof_preds_lgbm['target'] = np.nan
oof_preds_xgb = np.zeros((train_full_exgal.shape[0], 15))
oof_preds_xgb = pd.DataFrame(data=oof_preds_xgb, columns=label_features)
oof_preds_xgb['target'] = np.nan
imp_lgb = pd.DataFrame()
imp_xgb = pd.DataFrame()
mode = 'extra_galactic'
for i, (train_idx, cv_idx) in enumerate(folds):
    X_train = train_full_exgal[train_features].iloc[train_idx]
    Y_train = train_full_exgal['target'].iloc[train_idx]
    X_cv = train_full_exgal[train_features].iloc[cv_idx]
    Y_cv = train_full_exgal['target'].iloc[cv_idx]
    print ("\n\n" + "-"*20 + "Fold " + str(i+1) + "-"*20)
    
    print("*"*10 + "XGBoost" + "*"*10)
    clf_xgb = xgb.XGBClassifier(**xgb_params)
    clf_xgb.fit(
        X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
        verbose=100,
        eval_metric=xgbMultiWeightedLoss,
        early_stopping_rounds=50,
    )
#    imp_df = pd.DataFrame()
#    imp_df['feature'] = train_features
#    imp_df['gain'] = clf_xgb.feature_importances_(importance_type='gain')
#    imp_df['fold'] = i + 1
#    imp_xgb = pd.concat([imp_xgb, imp_df], axis=0, sort=False)
    
    print ("\n" + "*"*10 + "LightGBM" + "*"*10)
    clf_lgbm = lgbm.LGBMClassifier(**lgbm_params)
    
    clf_lgbm.fit(
        X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
        verbose=100,
        eval_metric=lgbMultiWeightedLoss,
        early_stopping_rounds=50,
    )
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = clf_lgbm.booster_.feature_importance(importance_type='gain')
    imp_df['fold'] = i + 1
    imp_lgb = pd.concat([imp_lgb, imp_df], axis=0, sort=False)
    
    # since there is no class 15 in train set,
    # the lightgbm will only predict 14 classes
    oof_preds_lgbm.loc[cv_idx, label_features_exgal] \
        = clf_lgbm.predict_proba(X_cv, num_iteration=clf_lgbm.best_iteration_)
    oof_preds_xgb.loc[cv_idx, label_features_exgal] \
        = clf_xgb.predict_proba(X_cv)

    oof_preds_lgbm.loc[cv_idx, 'target'] = Y_cv
    oof_preds_xgb.loc[cv_idx, 'target'] = Y_cv
    
    lgbm_list.append(clf_lgbm)
    xgb_list.append(clf_xgb)
    

oof_preds_comb_exgal = 0.7*oof_preds_lgbm[label_features] + 0.3*oof_preds_xgb[label_features]
oof_preds_comb_exgal['target'] = train_full_exgal['target'].astype(int)
oof_preds_comb_exgal['object_id'] = train_full_exgal['object_id']
print("\n" + "-"*20)
print("LightGBM: {0}"\
      .format(multiWeightedLoss(oof_preds_comb_exgal['target'],
                                oof_preds_lgbm[label_features].values)))
print("XGBoost: {0}"\
      .format(multiWeightedLoss(oof_preds_comb_exgal['target'],
                                oof_preds_xgb[label_features].values)))
print("Combined: {0}"\
      .format(multiWeightedLoss(oof_preds_comb_exgal['target'],
                                oof_preds_comb_exgal[label_features].values)))
xgb_exgals = xgb_list
lgb_exgals = lgbm_list

########################### Evaluate Stacked Prediction ##########
mode = None
lgbm_list = lgb_gals
xgb_list = lgb_gals
for clf_lgb, clf_xgb in zip(lgbm_list, xgb_list):
    preds_lgb_gal = clf_lgb.predict_proba(train_full[train_features],
                              num_iteration=clf_lgb.best_iteration_)\
                / len(folds)
    preds_xgb_gal = clf_xgb.predict_proba(train_full[train_features])\
                / len(folds)
preds_comb_gal = 0.7*preds_lgb_gal + 0.3*preds_xgb_gal
preds_comb_gal=pd.DataFrame(data=preds_comb_gal, columns=label_features_gal)

lgbm_list = lgb_exgals
xgb_list = lgb_exgals
for clf_lgb, clf_xgb in zip(lgbm_list, xgb_list):
    preds_lgb_exgal = clf_lgb.predict_proba(train_full[train_features],
                              num_iteration=clf_lgb.best_iteration_)\
                / len(folds)
    preds_xgb_exgal = clf_xgb.predict_proba(train_full[train_features])\
                / len(folds)
preds_comb_exgal = 0.7*preds_lgb_exgal + 0.3*preds_xgb_exgal
preds_comb_exgal = pd.DataFrame(data=preds_comb_exgal, columns=label_features_exgal)

preds_comb_stack = pd.concat([preds_comb_exgal, preds_comb_gal], axis=1)
preds_comb_stack['class_99'] = 0
preds_comb_stack = preds_comb_stack[label_features]
print("Stacked: {0}"\
      .format(multiWeightedLoss(train_full['target'],
                                preds_comb_stack[label_features].values)))

preds_comb_stack = preds_comb_stack[label_features].copy()
preds_comb_stack.loc[train_full['hostgal_photoz']==0, label_features_exgal] = 0
preds_comb_stack.loc[train_full['hostgal_photoz']!=0, label_features_gal] = 0
print("Manual Hostgal: {0}".format(multiWeightedLoss(train_full['target'],
                                               preds_comb_stack.values)))
#import sys
#sys.exit(0)
##################################### Stacked Model #############################
#oof_preds_full = pd.concat([oof_preds_comb_gal, oof_preds_comb_exgal], axis=0,
#                           ignore_index=True)
#stacked_features = oof_preds_full.columns.tolist()
#stacked_features.remove('target')
#stacked_features.remove('object_id')
#stacked_features.remove('class_99')
#oof_preds_full = oof_preds_full.merge(train_full, how='inner',
#                                      on=['object_id', 'target'])
#stacked_features = stacked_features + train_features
#del oof_preds_full['class_99']
#stacked_labels = label_features.copy()
#stacked_labels.remove('class_99')
#folds = getFolds(oof_preds_full['target'])
#lgbm_list = []
#xgb_list = []
#oof_preds_lgbm = np.zeros((oof_preds_full.shape[0], 15))
#oof_preds_lgbm = pd.DataFrame(data=oof_preds_lgbm, columns=label_features)
#oof_preds_lgbm['target'] = np.nan
#oof_preds_xgb = np.zeros((oof_preds_full.shape[0], 15))
#oof_preds_xgb = pd.DataFrame(data=oof_preds_xgb, columns=label_features)
#oof_preds_xgb['target'] = np.nan
#oof_preds_xgb['target'] = np.nan
#mode = None
#for i, (train_idx, cv_idx) in enumerate(folds):
#    X_train = oof_preds_full[stacked_features].iloc[train_idx]
#    Y_train = oof_preds_full['target'].iloc[train_idx]
#    X_cv = oof_preds_full[stacked_features].iloc[cv_idx]
#    Y_cv = oof_preds_full['target'].iloc[cv_idx]
#    print ("\n\n" + "-"*20 + "Fold " + str(i+1) + "-"*20)
#    
#    print("*"*10 + "XGBoost" + "*"*10)
#    clf_xgb = xgb.XGBClassifier(**xgb_params)
#    clf_xgb.fit(
#        X_train, Y_train,
#        eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
#        verbose=100,
#        eval_metric=xgbMultiWeightedLoss,
#        early_stopping_rounds=50,
#    )
#    
#    print ("\n" + "*"*10 + "LightGBM" + "*"*10)
#    clf_lgbm = lgbm.LGBMClassifier(**lgbm_params)
#    
#    clf_lgbm.fit(
#        X_train, Y_train,
#        eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
#        verbose=100,
#        eval_metric=lgbMultiWeightedLoss,
#        early_stopping_rounds=50,
#    )
#    
#    # since there is no class 15 in train set,
#    # the lightgbm will only predict 14 classes
#    oof_preds_lgbm.loc[cv_idx, stacked_labels] \
#        = clf_lgbm.predict_proba(X_cv, num_iteration=clf_lgbm.best_iteration_)
#    oof_preds_xgb.loc[cv_idx, stacked_labels] \
#        = clf_xgb.predict_proba(X_cv)
#
#    oof_preds_lgbm.loc[cv_idx, 'target'] = Y_cv
#    oof_preds_xgb.loc[cv_idx, 'target'] = Y_cv
#    
#    lgbm_list.append(clf_lgbm)
#    xgb_list.append(clf_xgb)
#
#oof_preds_comb_full = 0.7*oof_preds_lgbm[label_features] + 0.3*oof_preds_xgb[label_features]
#oof_preds_comb_full['target'] = oof_preds_full['target'].astype(int)
#print("\n" + "-"*20)
#print("LightGBM: {0}"\
#      .format(multiWeightedLoss(oof_preds_comb_full['target'],
#                                oof_preds_lgbm[label_features].values)))
#print("XGBoost: {0}"\
#      .format(multiWeightedLoss(oof_preds_comb_full['target'],
#                                oof_preds_xgb[label_features].values)))
#print("Combined: {0}"\
#      .format(multiWeightedLoss(oof_preds_comb_full['target'],
#                                oof_preds_comb_full[label_features].values)))
#
#stacked_xgb_list = xgb_list
#stacked_lgb_list = xgb_list
#import seaborn as sns
#import matplotlib.pyplot as plt
##mean barplot of importances
##imp_xgb_mean = np.log1p(imp_xgb[['gain', 'feature']]\
##                       .groupby('feature').mean())
##imp_xgb_mean = imp_xgb_mean.reset_index()
##fig = plt.figure(figsize=(8, 25))
##sns.barplot(x='gain', y='feature',
##            data=imp_xgb_mean.sort_values('gain', ascending=False))
##fig.suptitle('XGB Mean Feature Importance', fontsize=16)
##fig.tight_layout()
#
##imp_lgb_mean = np.log1p(imp_lgb[['gain', 'feature']])
##imp_lgb_mean = imp_lgb_mean.reset_index()
#fig = plt.figure(figsize=(8, 25))
#sns.barplot(x='gain', y='feature',
#            data=imp_lgb.sort_values('gain', ascending=False))
#fig.suptitle('LGB Mean Feature Importance', fontsize=16)
#fig.tight_layout()

import time
train_mean = train_full.mean(axis=0)
del train_full, train_idx
######################### #create submission file #############################
if do_prediction is True:
    gc.collect()
    start = time.time()
    chunks = 6000000
    chunk_last = pd.DataFrame() 
    test_row_num = 453653104 
    total_steps = int(np.ceil(test_row_num/chunks))
    #temporarily remove class 99 since it will predicted separately
#    temp_label_features = label_features.copy()
#    temp_label_features.remove("class_99")
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
        
        full_test, train_features = dproc.getFullData(ts_data=df,
                                                      meta_data=test_meta_data)
        del df
        gc.collect()
        full_test = full_test.fillna(train_mean)
        
        ####################### Make predictions gal ###########################
        print("Predicting galactic...")
        xgb_list = xgb_gals
        lgbm_list = lgb_gals
        preds_lgb = None
        preds_xgb = None
        for clf_lgb, clf_xgb in zip(lgbm_list, xgb_list):
            if preds_lgb is None:
                preds_lgb = clf_lgb.predict_proba(full_test[train_features],
                                          num_iteration=clf_lgb.best_iteration_)\
                            / len(folds)
                preds_xgb = clf_xgb.predict_proba(full_test[train_features])\
                            / len(folds)
            else:
                preds_lgb \
                    += clf_lgb.predict_proba(full_test[train_features],
                                             num_iteration=clf_lgb.best_iteration_)\
                        / len(folds)
                preds_xgb \
                    += clf_xgb.predict_proba(full_test[train_features]) / len(folds)
        preds_comb_gal = 0.5*preds_lgb + 0.5*preds_xgb
        preds_comb_gal = pd.DataFrame(data=preds_comb_gal,
                                      columns=label_features_gal)
        preds_comb_gal['object_id'] = full_test['object_id']
        
        ####################### Make predictions exgal #########################
        print("Predicting extra galactic...")
        xgb_list = xgb_exgals
        lgbm_list = lgb_exgals
        preds_lgb = None
        preds_xgb = None
        for clf_lgb, clf_xgb in zip(lgbm_list, xgb_list):
            if preds_lgb is None:
                preds_lgb = clf_lgb.predict_proba(full_test[train_features],
                                          num_iteration=clf_lgb.best_iteration_)\
                            / len(folds)
                preds_xgb = clf_xgb.predict_proba(full_test[train_features])\
                            / len(folds)
            else:
                preds_lgb \
                    += clf_lgb.predict_proba(full_test[train_features],
                                             num_iteration=clf_lgb.best_iteration_)\
                        / len(folds)
                preds_xgb \
                    += clf_xgb.predict_proba(full_test[train_features]) / len(folds)
        preds_comb_exgal = 0.5*preds_lgb + 0.5*preds_xgb
        preds_comb_exgal = pd.DataFrame(data=preds_comb_exgal,
                                      columns=label_features_exgal)
        preds_comb_exgal['object_id'] = full_test['object_id']
        
        print("Saving predictions...")
        name = 'comb'
        
#        preds_comb_stack = pd.concat([preds_comb_exgal, preds_comb_gal], axis=1)
        preds_comb_stack = preds_comb_exgal.merge(preds_comb_gal,
                                                  on='object_id',
                                                  how='inner')
        preds_99 = np.ones(preds_comb_stack.shape[0])
        no_99 = label_features.copy()
        no_99.remove('class_99')
        for i in range(preds_comb_stack[no_99].values.shape[1]):
            preds_99 *= (1 - preds_comb_stack[no_99].values[:, i])
        preds_comb_stack['class_99'] = 0.14 * preds_99 / np.mean(preds_99)
        preds_comb_stack = preds_comb_stack[['object_id'] + label_features]
        preds_comb_stack.loc[full_test['hostgal_photoz']==0, label_features_exgal] = 0
        preds_comb_stack.loc[full_test['hostgal_photoz']>0, label_features_gal] = 0
        
        if i_c == 0:
            preds_comb_stack.to_csv('sfd_predictions_{0}.csv'.format(name), index=False)
        else: 
            preds_comb_stack.to_csv('sfd_predictions_{0}.csv'.format(name),
                            header=False, mode='a', index=False)
            
        del preds_lgb, preds_xgb, preds_comb_gal, preds_comb_exgal, preds_comb_stack
        gc.collect()
        
        if (i_c + 1) % 10 == 0:
            print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
    #
    model = 'sfd_predictions_comb'
    z = pd.read_csv(model + '.csv')
    cols = list(z.columns)
    cols.remove('object_id')
    z['class_99'] = 1 - z[cols].max(axis=1)
    #z = z[['object_id'] + label_features]
    tech ='_1-Pmax'
    z.to_csv(model + tech + '.csv', index=False)
