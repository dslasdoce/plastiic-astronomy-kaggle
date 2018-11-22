#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 17:17:24 2018

@author: dslasdoce
"""
import pandas as pd
import dataproc as dproc
import numpy as np
from sklearn.preprocessing import OneHotEncoder

target_map, label_features, all_classes, all_class_weights \
    = dproc.getDataParameters()
all_clmap_vals = np.array(list(target_map.values()))
    
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
    
train_meta, test_meta_data = dproc.getMetaData()

lgb_weight = 0.5

oof_preds_nn = pd.read_csv('output/oof_nn.csv')
oof_preds_lgbm = pd.read_csv('output/oof_lgbm.csv')
oof_preds_xgb = pd.read_csv('output/oof_xgb.csv')
oof_preds_blend = train_meta[['object_id', 'target_id', 'target']].copy()
oof_preds_blend[label_features] = (1- lgb_weight)*oof_preds_xgb[label_features] \
                                  +  lgb_weight*oof_preds_lgbm[label_features]
                                                      
                                                        
print("LightGBM: {0}".format(multiWeightedLoss(train_meta['target_id'],
                                               oof_preds_lgbm[label_features].values[:, :14],
                                               no_class99=True)))
print("XGBoost: {0}".format(multiWeightedLoss(train_meta['target_id'],
                                              oof_preds_xgb[label_features].values[:, :14],
                                               no_class99=True)))
print("NN: {0}".format(multiWeightedLoss(train_meta['target_id'],
                                              oof_preds_nn[label_features].values[:, :14],
                                               no_class99=True)))
print("Combined: {0}".format(multiWeightedLoss(train_meta['target_id'],
                                               oof_preds_blend[label_features].values[:, :14],
                                               no_class99=True)))

oof_preds_blend[label_features] = 0.6*oof_preds_blend[label_features]\
                                  +  0.4*oof_preds_nn[label_features]
print("Blend: {0}".format(multiWeightedLoss(train_meta['target_id'],
                                               oof_preds_blend[label_features].values[:, :14],
                                               no_class99=True)))
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

import xgboost as xgb
xgb_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 15,
    'metric': ['multi_error'],
    "learning_rate": 0.05,
     "num_leaves": 30,
     "max_depth": 6,
     "feature_fraction": 0.45,
     "bagging_fraction": 0.3,
     "reg_alpha": 0.3,
     "reg_lambda": 0.3,
      "min_split_gain": 0.01,
      "min_child_weight": 0,
      "n_estimators": 1000
      }   

def train(train_full, train_features, folds):
    clf_xgb_list = []
    oof_stack = np.zeros((train_full.shape[0], 15))
    for i, (train_idx, cv_idx) in enumerate(folds):
        X_train = train_full[train_features].iloc[train_idx]
        Y_train = train_full['target_id'].iloc[train_idx]
        X_cv = train_full[train_features].iloc[cv_idx]
        Y_cv = train_full['target_id'].iloc[cv_idx]
        print ("\n\n" + "-"*20 + "Fold " + str(i+1) + "-"*20)
        
#        print("*"*10 + "XGBoost" + "*"*10)
        clf_xgb = xgb.XGBClassifier(**xgb_params)
        clf_xgb.fit(
            X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
            verbose=False,
            eval_metric=xgbMultiWeightedLoss,
            early_stopping_rounds=50,
        )
        oof_stack[cv_idx, :14] = clf_xgb.predict_proba(X_cv)
        clf_xgb_list.append(clf_xgb)
        
    print("Stacked: {0}".format(multiWeightedLoss(train_full['target_id'],
                                              oof_stack)))
        
    return clf_xgb_list

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

folds = getFolds(train_meta['target_id'])
        
excluded_feats = ['object_id', 'target', 'target_id']
#oof_preds_blend.to_csv('blend.csv', index=False)
#print("CL1")
#oof_preds_blend = oof_preds_nn.merge(oof_preds_lgbm, on=excluded_feats, how='inner')
#train_features = [f for f in oof_preds_blend if f not in excluded_feats]
#clf_lst1 = train(oof_preds_blend, train_features, folds)
#
#print("CL2")
#oof_preds_blend = oof_preds_nn.merge(oof_preds_xgb, on=excluded_feats, how='inner')
#train_features = [f for f in oof_preds_blend if f not in excluded_feats]
#clf_lst2 = train(oof_preds_blend, train_features, folds)
#
#print("CL3")
#oof_preds_blend = oof_preds_blend.merge(oof_preds_lgbm, on=excluded_feats, how='inner')
#train_features = [f for f in oof_preds_blend if f not in excluded_feats]
#clf_lst3 = train(oof_preds_blend, train_features, folds)

#print("CL4")
oof_comb = oof_preds_lgbm.copy()
oof_comb[label_features] = (1- lgb_weight)*oof_preds_xgb[label_features] \
                                  +  lgb_weight*oof_preds_lgbm[label_features]
                                  
oof_preds_blend = oof_preds_nn.merge(oof_comb, on=excluded_feats, how='inner')
train_features = [f for f in oof_preds_blend if f not in excluded_feats]
clf_lst4 = train(oof_preds_blend, train_features, folds)

do_prediction = True
if do_prediction is True:
    ################# Load ##################
    preds_nn = pd.read_csv('output/nn_predictions_nn.csv')
    preds_gb = pd.read_csv('output/gb_predictions_comb.csv')
    preds_merge = preds_nn.merge(preds_gb, on='object_id', how='inner')
    ################# Prediction ##############
    temp_label_features = label_features.copy()
    temp_label_features.remove("class_99")
    train_features = [f for f in preds_merge.columns if f not in excluded_feats]
    preds_blend = None
    for clf in clf_lst4:
        if preds_blend is None:
            preds_blend = clf.predict_proba(preds_merge[train_features])/len(folds)
        else:
            preds_blend += clf.predict_proba(preds_merge[train_features])/len(folds)
            
    preds_blend = pd.DataFrame(preds_blend, columns=temp_label_features)
    preds_blend['object_id'] = preds_merge['object_id']
    preds_blend.to_csv('output/blend.csv', index=False)
    ################# Class 99 ##############
    preds_99 = np.ones(preds_blend.shape[0])
    no_99 = label_features.copy()
    no_99.remove('class_99')
    for i in range(preds_blend[no_99].values.shape[1]):
        preds_99 *= (1 - preds_blend[no_99].values[:, i])
    preds_blend['class_99'] = 0.18 * preds_99 / np.mean(preds_99)
    ############# Galactic Separation ###############
    gal_classes = train_meta['target'].loc[train_meta['hostgal_specz'] == 0].unique()
    gal_classes = all_classes[np.isin(all_classes, gal_classes)]
    exgal_classes = train_meta['target'].loc[train_meta['hostgal_specz'] != 0].unique()
    exgal_classes = all_classes[np.isin(all_classes, exgal_classes)]
    label_features_gal = ['class_' + str(cl) for cl in gal_classes]
    label_features_exgal = ['class_' + str(cl) for cl in exgal_classes]
    
    cols = list(preds_blend.columns)
    cols.remove('object_id')
    
    gal_objs = test_meta_data\
                .loc[test_meta_data['hostgal_photoz']==0, 'object_id']
    exgal_objs = test_meta_data\
            .loc[test_meta_data['hostgal_photoz']>0, 'object_id']
    preds_blend.loc[preds_blend['object_id'].isin(gal_objs), label_features_exgal] = 0
    preds_blend.loc[preds_blend['object_id'].isin(exgal_objs), label_features_gal] = 0
    
    tech ='_scgal'
    preds_blend.to_csv('output/blend' + tech + '.csv', index=False)
    
    
    
    
    
    
    
    
    
    
