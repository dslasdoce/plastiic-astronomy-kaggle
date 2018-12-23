#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:48:26 2018

@author: dslasdoce
"""
import numpy as np
import pandas as pd
import dataproc as dproc
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgbm
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

target_map, label_features, all_classes, all_class_weights \
    = dproc.getDataParameters()
all_clmap_vals = np.array(list(target_map.values()))


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

def trainLGBXGB(train_full, train_features, folds, lgbm_params, xgb_params, weights):
    lgbm_list = []
    xgb_list = []
    oof_preds_lgbm = np.zeros((train_full.shape[0], 15))
    oof_preds_xgb = np.zeros((train_full.shape[0], 15))
    imp_lgb = pd.DataFrame()
    imp_xgb = pd.DataFrame()
    for i, (train_idx, cv_idx) in enumerate(folds):
        X_train = train_full[train_features].iloc[train_idx]
        Y_train = train_full['target_id'].iloc[train_idx]
        X_cv = train_full[train_features].iloc[cv_idx]
        Y_cv = train_full['target_id'].iloc[cv_idx]
        print ("\n\n" + "-"*20 + "Fold " + str(i+1) + "-"*20)
        
        print("*"*10 + "XGBoost" + "*"*10)
        clf_xgb = xgb.XGBClassifier(**xgb_params)
        clf_xgb.fit(
            X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
            verbose=100,
            eval_metric=xgbMultiWeightedLoss,
            early_stopping_rounds=50,
            sample_weight=Y_train.map(weights)
        )
        imp_df = pd.DataFrame()
        imp_df['feature'] = train_features
        imp_df['gain'] = clf_xgb.feature_importances_
        imp_df['fold'] = i + 1
        imp_xgb = pd.concat([imp_xgb, imp_df], axis=0, sort=False)
        
        print ("\n" + "*"*10 + "LightGBM" + "*"*10)
        clf_lgbm = lgbm.LGBMClassifier(**lgbm_params)
        
        clf_lgbm.fit(
            X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
            verbose=100,
            eval_metric=lgbMultiWeightedLoss,
            early_stopping_rounds=50,
            sample_weight=Y_train.map(weights)
        )
        
        imp_df = pd.DataFrame()
        imp_df['feature'] = train_features
        imp_df['gain'] = clf_lgbm.booster_.feature_importance(importance_type='gain')
        imp_df['fold'] = i + 1
        imp_lgb = pd.concat([imp_lgb, imp_df], axis=0, sort=False)
        
        # since there is no class 15 in train set,
        # the lightgbm will only predict 14 classes
        oof_preds_lgbm[cv_idx, :14] \
            = clf_lgbm.predict_proba(X_cv, num_iteration=clf_lgbm.best_iteration_)
        oof_preds_xgb[cv_idx, :14] \
            = clf_xgb.predict_proba(X_cv)
    
        lgbm_list.append(clf_lgbm)
        xgb_list.append(clf_xgb)
    
    print("LightGBM: {0}".format(multiWeightedLoss(train_full['target_id'],
                                                   oof_preds_lgbm[:, :14])))
    print("XGBoost: {0}".format(multiWeightedLoss(train_full['target_id'],
                                                  oof_preds_xgb[:, :14])))
    
    return lgbm_list, oof_preds_lgbm, xgb_list, oof_preds_xgb

def trainLGB(train_full, train_features, folds, lgbm_params, weights):
    lgbm_list = []
    oof_preds_lgbm = np.zeros((train_full.shape[0], 15))
    imp_lgb = pd.DataFrame()
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
            sample_weight=Y_train.map(weights)
        )
        
        imp_df = pd.DataFrame()
        imp_df['feature'] = train_features
        imp_df['gain'] = clf_lgbm.booster_.feature_importance(importance_type='gain')
        imp_df['fold'] = i + 1
        imp_lgb = pd.concat([imp_lgb, imp_df], axis=0, sort=False)
        
        # since there is no class 15 in train set,
        # the lightgbm will only predict 14 classes
        oof_preds_lgbm[cv_idx, :14] \
            = clf_lgbm.predict_proba(X_cv, num_iteration=clf_lgbm.best_iteration_)
    
        lgbm_list.append(clf_lgbm)
    score = multiWeightedLoss(train_full['target_id'],
                              oof_preds_lgbm[:, :14], no_class99=True)
    print("LightGBM: {0}".format(score))
    return lgbm_list, oof_preds_lgbm, score