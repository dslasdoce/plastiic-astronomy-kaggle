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
from model_train import trainLGB, trainLGBXGB

do_prediction = True
loaded_test = False
########################### Data and Parameters Import ##########################
target_map, label_features, all_classes, all_class_weights \
    = dproc.getDataParameters()

train_meta, test_meta_data = dproc.getMetaData()
train = pd.read_csv('input/training_set.csv')
train_full, train_features = dproc.getFullData(train, train_meta)
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

################################ Folding #######################################
from sklearn.model_selection import StratifiedKFold

def getFolds(ser_target=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
#    idx = np.arange(df.shape[0])
    fold_idx = []
    for train_idx, val_idx in folds.split(X=ser_target, y=ser_target):
        fold_idx.append([train_idx, val_idx])

    return fold_idx

folds = getFolds(train_full['target_id'])

################### LightGBM ########################
lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': 'multi_logloss',
    'learning_rate': 0.03,
    'subsample': .9,
    'colsample_bytree': 0.5,
    'reg_alpha': .1,
    'reg_lambda': .1,
    'min_split_gain': 0.01,
    'min_child_weight': 10,
    'n_estimators': 1000,
    'silent': -1,
    'verbose': -1,
    'max_depth': 3
}

xgb_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': ['multi_logloss'],
    "learning_rate": 0.05,
     "num_leaves": 30,
     "max_depth": 3,
     "feature_fraction": 0.45,
     "bagging_fraction": 0.3,
     "reg_alpha": 0.1,
     "reg_lambda": 0.1,
      "min_split_gain": 0.01,
      "min_child_weight": 10,
      "n_estimators": 1000
      }     

xgb_params = {    
        'objective': 'multi:softprob', 
        'eval_metric': 'mlogloss', 
        'silent': True, 
        'num_class':14,
        'booster': 'gbtree',
        'n_jobs': 4,
        'n_estimators': 1000,
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'base_score': 0.25,
        'max_depth': 7,
        'max_delta_step': 2,  #default=0
        'learning_rate': 0.03,
        'max_leaves': 11,
        'min_child_weight': 64,
        'gamma': 0.1, # default=
        'subsample': 0.7,
        'colsample_bytree': 0.68,
        'reg_alpha': 0.01, # default=0
        'reg_lambda': 10., # default=1
        'seed': 538
    } 

lgbm_list = []
xgb_list = []
oof_preds_lgbm = np.zeros((train_full.shape[0], 15))
oof_preds_xgb = np.zeros((train_full.shape[0], 15))
imp_lgb = pd.DataFrame()
imp_xgb = pd.DataFrame()
#oof_preds_both = np.zeros((train_full.shape[0], 15))
#fill na
train_mean = train_full.mean(axis=0)
train_full.fillna(train_mean, inplace=True)
w = train_full['target_id'].value_counts()
weights = {i : np.sum(w) / w[i] for i in w.index}
#train
lgbm_list, oof_preds_lgbm, xgb_list, oof_preds_xgb =\
    trainLGBXGB(train_full, train_features, folds, lgbm_params, xgb_params, weights)
lgb_weight = 0.5
oof_preds_comb = lgb_weight*oof_preds_lgbm + (1-lgb_weight)*oof_preds_xgb
print("Combined: {0}".format(multiWeightedLoss(train_full['target_id'],
                                               oof_preds_comb[:, :14])))

#save oof
for preds, name in zip([oof_preds_lgbm, oof_preds_xgb], ['lgbm', 'xgb']):
    df = pd.DataFrame(data=preds, columns=label_features)
    df['object_id'] = train_full['object_id']
    df['target_id'] = train_full['target_id']
    df['target'] = train_full['target']
    df.to_csv('output/oof_{0}.csv'.format(name), index=False)

oof_preds_comb = pd.DataFrame(data=oof_preds_comb, columns=label_features)
z = oof_preds_comb.copy()
gal_classes = train_meta['target'].loc[train_meta['hostgal_specz'] == 0].unique()
gal_classes = all_classes[np.isin(all_classes, gal_classes)]
exgal_classes = train_meta['target'].loc[train_meta['hostgal_specz'] != 0].unique()
exgal_classes = all_classes[np.isin(all_classes, exgal_classes)]
label_features_gal = ['class_' + str(cl) for cl in gal_classes]
label_features_exgal = ['class_' + str(cl) for cl in exgal_classes]
z.loc[train_full['hostgal_specz']==0, label_features_exgal] = 0
z.loc[train_full['hostgal_specz']!=0, label_features_gal] = 0
#print("Manual Hostgal: {0}".format(multiWeightedLoss(train_full['target_id'],
#                                               z.values)))
import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()
#mean barplot of importances
#imp_xgb_mean = np.log1p(imp_xgb[['gain', 'feature']]\
#                       .groupby('feature').mean())
#imp_xgb_mean = imp_xgb_mean.reset_index()
fig, ax = plt.subplots(figsize=(8, 14))
sns.barplot(x='gain', y='feature', ax=ax,
            data=imp_xgb.sort_values('gain', ascending=False).head(200))
#fig.suptitle('XGB Mean Feature Importance', fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
plt.tight_layout()
plt.show(block=False)

#imp_lgb_mean = np.log1p(imp_lgb[['gain', 'feature']])
#imp_lgb_mean = imp_lgb_mean.reset_index()
fig, ax = plt.subplots(figsize=(8, 14))
sns.barplot(x='gain', y='feature',
            data=imp_lgb.sort_values('gain', ascending=False).head(200))
fig.suptitle('LGB Mean Feature Importance', fontsize=16)
fig.tight_layout()
plt.tight_layout()
plt.show(block=False)

######################### #create submission file #############################
if loaded_test is True:
    chunks = 300000
    temp_label_features = label_features.copy()
    temp_label_features.remove("class_99")
    for i_c, full_test in \
    enumerate(pd.read_csv('/media/dslasdoce/Data/Astro/full_test_saved.csv',
                          chunksize=chunks, iterator=True)):
        print("*"*20 + "chunk: " + str(i_c) + "*"*20)
        
        # Make predictions
        print("Predicting...")
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
        preds_comb = lgb_weight*preds_lgb + (1-lgb_weight)*preds_xgb
        # Store predictions
        print("Saving predictions...")
         #create prediction file for each model
#        for _pred, name in zip([preds_lgb, preds_xgb, preds_comb], ['lgb', 'xgb', 'comb']):
        for _pred, name in zip([preds_comb], ['comb']):
            preds_df = pd.DataFrame(_pred, columns=[temp_label_features])
            preds_df['object_id'] = full_test['object_id'].values
            preds_df['class_99'] = 0.1
#            print(preds_df.iloc[0])
            if i_c == 0:
                preds_df.to_csv('output/gb_predictions_{0}.csv'.format(name), index=False)
            else: 
                preds_df.to_csv('output/gb_predictions_{0}.csv'.format(name),
                                header=False, mode='a', index=False)
            del preds_df
            gc.collect()
            
        del preds_lgb, preds_xgb, preds_comb
        gc.collect()
        
if do_prediction is True:
    import time
    del train_full, train_idx, oof_preds_comb, oof_preds_lgbm, oof_preds_xgb, train_meta
    del X_cv, X_train, Y_cv, Y_train, z, preds
    gc.collect()
    start = time.time()
    chunks = 3000000
    chunk_last = pd.DataFrame()
    test_row_num = 453653104 
    total_steps = int(np.ceil(test_row_num/chunks))
    #temporarily remove class 99 since it will predicted separately
    temp_label_features = label_features.copy()
    temp_label_features.remove("class_99")
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
        
        
        if i_c <124:
            del mask_last, df
            gc.collect()
            continue
        gc.collect()
        full_test, train_features_test = dproc.getFullData(ts_data=df,
                                                      meta_data=test_meta_data)
        try:
            del mask_last
        except NameError:
            pass
        del df
        
        gc.collect()
        
        if i_c == 0:
            full_test.to_csv("/media/dslasdoce/Data/Astro/full_test_saved.csv", index=False)
        else: 
            full_test.to_csv("/media/dslasdoce/Data/Astro/full_test_saved.csv",
                             index=False, header=False, mode='a')
        
        full_test = full_test.fillna(train_mean)
        # Make predictions
        print("Predicting...")
        
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
        preds_comb = lgb_weight*preds_lgb + (1-lgb_weight)*preds_xgb

        # Store predictions
        print("Saving predictions...")
         #create prediction file for each model
#        for _pred, name in zip([preds_lgb, preds_xgb, preds_comb], ['lgb', 'xgb', 'comb']):
        for _pred, name in zip([preds_comb, preds_lgb, preds_xgb], ['comb', 'lgb', 'xgb']):
            preds_df = pd.DataFrame(_pred, columns=[temp_label_features])
            preds_df['object_id'] = full_test['object_id']
            preds_df['class_99'] = 0.1
            
            if i_c == 0:
                preds_df.to_csv('output/gb_predictions_{0}.csv'.format(name), index=False)
            else: 
                preds_df.to_csv('output/gb_predictions_{0}.csv'.format(name),
                                header=False, mode='a', index=False)
            del preds_df
            gc.collect()
            
        del preds_lgb, preds_xgb, preds_comb, full_test
        gc.collect()
        
        if (i_c + 1) % 10 == 0:
            print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

if do_prediction is True or loaded_test is True:
    model = 'output/gb_predictions_comb'
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
    #tech ='_99sc'
    #.to_csv(model + tech + '.csv', index=False)
    #label_features_exgal
    gal_objs = test_meta_data\
                .loc[test_meta_data['hostgal_photoz']==0, 'object_id']
    exgal_objs = test_meta_data\
            .loc[test_meta_data['hostgal_photoz']>0, 'object_id']
    z.loc[z['object_id'].isin(gal_objs), label_features_exgal] = 0
    z.loc[z['object_id'].isin(exgal_objs), label_features_gal] = 0
    
    tech ='_scgal'
    z.to_csv(model + tech + '.csv', index=False)
        
    #    z = pd.read_csv('full_test_saved.csv')
