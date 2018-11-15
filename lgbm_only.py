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
import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()
do_prediction = False

########################### Data and Parameters Import ##########################
target_map, label_features, all_classes, all_class_weights \
    = dproc.getDataParameters()

train_meta, test_meta_data = dproc.getMetaData()
train = pd.read_csv('input/training_set.csv')
train_full, train_features = dproc.getFullData(train, train_meta)
#del train
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
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
#    idx = np.arange(df.shape[0])
    fold_idx = []
    for train_idx, val_idx in folds.split(X=ser_target, y=ser_target):
        fold_idx.append([train_idx, val_idx])

    return fold_idx

folds = getFolds(train_full['target_id'])

#########################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#    print(cm)
    fmt = '.2f' if normalize else 'd'
    
    fig, ax = plt.subplots(figsize=(14,9))
    sns.heatmap(cm, ax=ax, cmap='Blues', annot=True, fmt=fmt)
    ax.set_xticklabels(label_features, rotation=45)
    ax.set_yticklabels(label_features, rotation=0)
    fig.tight_layout()

    
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
################### LightGBM ########################
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': ['multi_logloss'],
    "learning_rate": 0.02,
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
#https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data
lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 14,
        'metric': 'multi_logloss',
        'learning_rate': 0.03,
        'subsample': .9,
        'colsample_bytree': .7,
        'reg_alpha': .01,
        'reg_lambda': .01,
        'min_split_gain': 0.01,
        'min_child_weight': 10,
        'n_estimators': 1000,
        'silent': -1,
        'verbose': -1,
        'max_depth': 3
    }
#https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135
lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': 'multi_logloss',
    'learning_rate': 0.03,
    'subsample': .9,
    'colsample_bytree': 0.5,
    'reg_alpha': .01,
    'reg_lambda': .01,
    'min_split_gain': 0.01,
    'min_child_weight': 10,
    'n_estimators': 1000,
    'silent': -1,
    'verbose': -1,
    'max_depth': 3
}

lgbm_list = []
oof_preds_lgbm = np.zeros((train_full.shape[0], 15))
imp_lgb = pd.DataFrame()

#oof_preds_both = np.zeros((train_full.shape[0], 15))
#test_prediction = np.zeros((test_meta_data.shape[0], 15))
train_mean = train_full.mean(axis=0)
train_full.fillna(train_mean, inplace=True)
#train_full.fillna(train_mean, inplace=True)
#fts = ['15decay_std', '15decay_min', '15decay_max', '15decay_mean']
#for f in fts:
#    try:
#        train_features.remove(f)
#    except:
#        pass
#train_features += ['15decay_mean']
#train_full['cheat'] = 0
#train_full.loc[train_full['target']==52, 'cheat'] = 1 
#train_features += ['rd_skew'] #rd_std
#train_features.remove('rd_std')
w = train_full['target_id'].value_counts()
weights = {i : np.sum(w) / w[i] for i in w.index}

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

print("LightGBM: {0}".format(multiWeightedLoss(train_full['target_id'],
                                               oof_preds_lgbm[:, :14],
                                               no_class99=True)))
############################
from sklearn.metrics import confusion_matrix
cnf_matrix_lgb = confusion_matrix(train_meta['target_id'],
                                     np.argmax(oof_preds_lgbm,
                                               axis=-1))
plot_confusion_matrix(cnf_matrix_lgb, classes=label_features,normalize=True,
                  title='Confusion matrix')
plt.show(block=False)

###########################
df = pd.DataFrame(data=oof_preds_lgbm, columns=label_features)
df['object_id'] = train_full['object_id']
df['target_id'] = train_full['target_id']
df['target'] = train_full['target']
df.to_csv('output/oof_{0}.csv'.format('lgbm'), index=False)

gal_classes = train_meta['target'].loc[train_meta['hostgal_specz'] == 0].unique()
gal_classes = all_classes[np.isin(all_classes, gal_classes)]
exgal_classes = train_meta['target'].loc[train_meta['hostgal_specz'] != 0].unique()
exgal_classes = all_classes[np.isin(all_classes, exgal_classes)]
label_features_gal = ['class_' + str(cl) for cl in gal_classes]
label_features_exgal = ['class_' + str(cl) for cl in exgal_classes]

#sns.heatmap(train_full[train_features].corr())
#plt.matshow(train_full[train_features].corr())

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
            .to_csv("input/test_periods.csv", index=False)
        else: 
            full_test[['object_id', 'period']].to_csv("input/test_periods.csv",
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
                preds_df.to_csv('output/sfd_predictions_{0}.csv'.format(name), index=False)
            else: 
                preds_df.to_csv('output/sfd_predictions_{0}.csv'.format(name),
                                header=False, mode='a', index=False)
            del preds_df
            gc.collect()
            
        del preds_lgb, preds_xgb, preds_comb
        gc.collect()
        
        if (i_c + 1) % 10 == 0:
            print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
    #
    model = 'output/sfd_preds'
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
