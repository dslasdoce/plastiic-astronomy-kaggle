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
train_full, train_features = dproc.getFullDataFromSaved(train, train_meta)
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

###################################### Tuning ##################################
#from skopt.space import Real, Integer
#dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform',name='learning_rate')
#dim_estimators = Integer(low=50, high=2000,name='n_estimators')
#dim_max_depth = Integer(low=1, high=6,name='max_depth')
#
#dimensions = [dim_learning_rate,
#              dim_estimators,
#              dim_max_depth]
#
#default_parameters = [0.03,1000,3]
#
#def createModel(learning_rate,n_estimators,max_depth):
#    oof_preds_lgbm = np.zeros((train_full.shape[0], 15))     
#    for i, (train_idx, cv_idx) in enumerate(folds):
#        X_train = train_full[train_features].iloc[train_idx]
#        Y_train = train_full['target_id'].iloc[train_idx]
#        X_cv = train_full[train_features].iloc[cv_idx]
#        Y_cv = train_full['target_id'].iloc[cv_idx]
#        print ("\n\n" + "-"*20 + "Fold " + str(i+1) + "-"*20)
#        print ("\n" + "*"*10 + "LightGBM" + "*"*10)
#        
#        clf_lgbm = lgbm.LGBMClassifier(**lgb_params,learning_rate=learning_rate,
#                                n_estimators=n_estimators,max_depth=max_depth)
#        clf_lgbm.fit(
#            X_train, Y_train,
#            eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
#            verbose=False,
#            eval_metric=lgbMultiWeightedLoss,
#            early_stopping_rounds=50,
#        )
#        
#        oof_preds_lgbm[cv_idx, :14] \
#            = clf_lgbm.predict_proba(X_cv, num_iteration=clf_lgbm.best_iteration_)
#    
##        lgbm_list.append(clf_lgbm)
#    
#    loss = multiWeightedLoss(train_full['target_id'], oof_preds_lgbm)
#    print('MULTI WEIGHTED LOG LOSS : %.5f ' % loss)
#    
#    return loss
#
#from skopt.utils import use_named_args
#@use_named_args(dimensions=dimensions)
#def fitness(learning_rate,n_estimators,max_depth):
#    """
#    Hyper-parameters:
#    learning_rate:     Learning-rate for the optimizer.
#    n_estimators:      Number of estimators.
#    max_depth:         Maximum Depth of tree.
#    """
#
#    # Print the hyper-parameters.
#    print('learning rate: {0:.2e}'.format(learning_rate))
#    print('estimators:', n_estimators)
#    print('max depth:', max_depth)
#    
#    lv= createModel(learning_rate=learning_rate,
#                    n_estimators=n_estimators,
#                    max_depth = max_depth)
#    return lv
#          
#lgb_params = {
#    'boosting_type': 'gbdt',
#    'objective': 'multiclass',
#    'num_class': 14,
#    'metric': 'multi_logloss',
#    'subsample': .9,
#    'colsample_bytree': .7,
#    'reg_alpha': .01,
#    'reg_lambda': .01,
#    'min_split_gain': 0.01,
#    'min_child_weight': 10,
#    'silent':True,
#    'verbosity':-1,
#    'nthread':-1
#}
#from skopt import gp_minimize
#
#search_result = gp_minimize(func=fitness,
#                            dimensions=dimensions,
#                            acq_func='EI', # Expected Improvement.
#                            n_calls=20,
#                           x0=default_parameters)
#import sys
#sys.exit()
#learning_rate = search_result.x[0]
#n_estimators = search_result.x[1]
#max_depth = search_result.x[2]
################### LightGBM ########################
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

lgbm_list = []
oof_preds_lgbm = np.zeros((train_full.shape[0], 15))
imp_lgb = pd.DataFrame()

#oof_preds_both = np.zeros((train_full.shape[0], 15))
#test_prediction = np.zeros((test_meta_data.shape[0], 15))
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


print("LightGBM: {0}".format(multiWeightedLoss(train_full['target_id'],
                                               oof_preds_lgbm)))

gal_classes = train_meta['target'].loc[train_meta['hostgal_specz'] == 0].unique()
gal_classes = all_classes[np.isin(all_classes, gal_classes)]
exgal_classes = train_meta['target'].loc[train_meta['hostgal_specz'] != 0].unique()
exgal_classes = all_classes[np.isin(all_classes, exgal_classes)]
label_features_gal = ['class_' + str(cl) for cl in gal_classes]
label_features_exgal = ['class_' + str(cl) for cl in exgal_classes]
import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()

#mean barplot of importances
#imp_lgb_mean = np.log1p(imp_lgb[['gain', 'feature']])
#imp_lgb_mean = imp_lgb_mean.reset_index()
fig, ax= plt.subplots(figsize=(8, 25))
sns.barplot(x='gain', y='feature',
            data=imp_lgb.sort_values('gain', ascending=False).head(200))
fig.suptitle('LGB Mean Feature Importance', fontsize=16)
fig.tight_layout()
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
plt.tight_layout()
plt.show(block=False)


######################### #create submission file #############################
if do_prediction is True:
    import time
    train_mean = train_full.mean(axis=0)
    del train_full, train_idx
    gc.collect()
    start = time.time()
    chunks = 6000000
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
        
        full_test, train_features = dproc.getFullData(ts_data=df,
                                                      meta_data=test_meta_data)
        del df
        gc.collect()
        full_test = full_test.fillna(train_mean)
    
        # Make predictions
        print("Predicting...")
    #    pred_99 = oneclass.predict(full_test[train_features])
    #    pred_99[pred_99==1] = 0
    #    pred_99[pred_99==-1] = 1
        
        preds_lgb = None
        preds_xgb = None
        for clf_lgb in lgbm_list:
            if preds_lgb is None:
                preds_lgb = clf_lgb.predict_proba(full_test[train_features],
                                          num_iteration=clf_lgb.best_iteration_)\
                            / len(folds)
            else:
                preds_lgb \
                    += clf_lgb.predict_proba(full_test[train_features],
                                             num_iteration=clf_lgb.best_iteration_)\
                        / len(folds)
        # preds_99 = 0.1 gives 1.769
    #    preds_99 = np.zeros(preds_lgb.shape[0])
    #    for i in range(preds.shape[1]):
    #        preds_99 *= (1 - preds[:, i])
        
        #save periods
        if i_c == 0:
            full_test[['object_id', 'period']]\
            .to_csv("test_periods.csv", index=False)
        else: 
            full_test[['object_id', 'period']].to_csv("test_periods.csv",
                         index=False, header=False, mode='a')
        # Store predictions
        print("Saving predictions...")
         #create prediction file for each model
#        for _pred, name in zip([preds_lgb, preds_xgb, preds_comb], ['lgb', 'xgb', 'comb']):
        for _pred, name in zip([preds_lgb], ['lgbm']):
            preds_df = pd.DataFrame(_pred, columns=[temp_label_features])
            preds_df['object_id'] = full_test['object_id']
            preds_df['class_99'] = 0.1
            
            if i_c == 0:
                preds_df.to_csv('sfd_predictions_{0}.csv'.format(name), index=False)
            else: 
                preds_df.to_csv('sfd_predictions_{0}.csv'.format(name),
                                header=False, mode='a', index=False)
            del preds_df
            gc.collect()
            
        del preds_lgb, preds_xgb, preds_comb
        gc.collect()
        
        if (i_c + 1) % 10 == 0:
            print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
    #
    model = 'sfd_preds'
    z = pd.read_csv(model + '.csv')
    
#    preds_99 = np.ones(z.shape[0])
#    no_99 = label_features.copy()
#    no_99.remove('class_99')
#    for i in range(z[no_99].values.shape[1]):
#        preds_99 *= (1 - z[no_99].values[:, i])
#    z['class_99'] = 0.14 * preds_99 / np.mean(preds_99)

    cols = list(z.columns)
    cols.remove('object_id')
    z['class_99'] = 1 - z[cols].max(axis=1)
    #z = z[['object_id'] + label_features]
    tech ='_99sc'
    z.to_csv(model + tech + '.csv', index=False)
