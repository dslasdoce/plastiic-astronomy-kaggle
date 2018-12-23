
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
from model_train import trainLGB
plt.ioff()
do_prediction = True

########################### Data and Parameters Import ##########################
target_map, label_features, all_classes, all_class_weights \
    = dproc.getDataParameters()

train_meta, test_meta_data = dproc.getMetaData()
train = pd.read_csv('input/training_set.csv')
train_full, train_features = dproc.getFullData(train, train_meta)
train_feats_orig = train_features.copy()
#t = train_full[['target', 'q_count']]
#del train
gc.collect()
#train_full[['object_id', 'period']].to_csv("train_periods.csv", index=False)
#target_id list: will be used in one hot encoding of labels
all_clmap_vals = np.array(list(target_map.values()))
print("Train Feats: {0}".format(train_features))
print("Train Data All COLS: {0}".format(train_full.columns))
############################# LOSS FUNCTION ####################################

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
cl_weights_tmp = all_class_weights.copy() 
del cl_weights_tmp['class_99'] 
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
    'max_depth': 3,
    'class_weight': {np.where(all_classes==int(cl.split('_')[1]))[0][0]: i\
                     for (cl, i) in cl_weights_tmp.items()}
}

lgbm_params = {
    'boosting_type': 'dart',
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
    'n_estimators': 2000,
    'silent': -1,
    'verbose': -1,
    'max_depth': 3,
    'class_weight': {np.where(all_classes==int(cl.split('_')[1]))[0][0]: i\
                     for (cl, i) in cl_weights_tmp.items()},
    'min_data_in_leaf': 50
}

#oof_preds_both = np.zeros((train_full.shape[0], 15))
#test_prediction = np.zeros((test_meta_data.shape[0], 15))
train_mean = train_full.mean(axis=0)
train_full.fillna(train_mean, inplace=True)

w = train_full['target_id'].value_counts()
weights = {i : np.sum(w) / w[i] for i in w.index}

lgbm_list, oof_preds_lgbm, score =\
    trainLGB(train_full, train_features, folds, lgbm_params, weights)
rep = pd.DataFrame(columns=['F', 'Score'])
rep.loc[0, :] = ['Orig', score]

#for i, f in enumerate(train_feats_orig):
#    train_features = train_feats_orig.copy()
#    train_features.remove(f)
#    lgbm_list, oof_preds_lgbm, score =\
#        trainLGB(train_full, train_features, folds, lgbm_params, weights)
#    rep.loc[i+1, :] = [f, score]
#    break
    
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
    chunks = 300000
    temp_label_features = label_features.copy()
    temp_label_features.remove("class_99")
    for i_c, full_test in enumerate(pd.read_csv('input/full_test_saved_gb.csv', chunksize=chunks, iterator=True)):
        print("*"*20 + "chunk: " + str(i_c) + "*"*20)
        
        # Make predictions
        print("Predicting...")
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
        # Store predictions
        print("Saving predictions...")
         #create prediction file for each model
    #        for _pred, name in zip([preds_lgb, preds_xgb, preds_comb], ['lgb', 'xgb', 'comb']):
        for _pred, name in zip([preds_lgb], ['final']):
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
            
        del preds_lgb
        gc.collect()
if do_prediction is True:
    model = 'output/gb_predictions_final'
    z = pd.read_csv(model + '.csv')
    
    no_99 = label_features.copy()
    no_99.remove('class_99')

    preds_99 = np.ones(z.shape[0])    
    for i in range(z[no_99].values.shape[1]):
        preds_99 *= (1 - z[no_99].values[:, i])
    z['class_99'] = 0.18 * preds_99 / np.mean(preds_99)
        
#    y = pd.DataFrame()
#    model = 'output/gb_predictions_lgb'
#    z = pd.read_csv(model + '.csv')
#    y['mymean'] = z[no_99].mean(axis=1)
#    y['mymedian'] = z[no_99].median(axis=1)
#    y['mymax'] = z[no_99].max(axis=1)
#    z['class_99'] = GenUnknown(y)
    
    cols = list(z.columns)
    cols.remove('object_id')
    gal_objs = test_meta_data\
                .loc[test_meta_data['hostgal_photoz']==0, 'object_id']
    exgal_objs = test_meta_data\
            .loc[test_meta_data['hostgal_photoz']>0, 'object_id']
    z.loc[z['object_id'].isin(gal_objs), label_features_exgal] = 0
    z.loc[z['object_id'].isin(exgal_objs), label_features_gal] = 0
    
    tech ='_gb'
    z.to_csv(model + tech + '.csv', index=False)
        
    preds_blend = pd.read_csv('output/nn_predictions_nn_scgal.csv')
    preds_blend[label_features] = 0.4*preds_blend[label_features] + 0.6*z[label_features]
    preds_blend.to_csv('output/blend.csv', index=False)

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
from model_train import trainLGB
plt.ioff()
do_prediction = True

########################### Data and Parameters Import ##########################
target_map, label_features, all_classes, all_class_weights \
    = dproc.getDataParameters()

train_meta, test_meta_data = dproc.getMetaData()
train = pd.read_csv('input/training_set.csv')
train_full, train_features = dproc.getFullData(train, train_meta)
train_feats_orig = train_features.copy()
#t = train_full[['target', 'q_count']]
#del train
gc.collect()
#train_full[['object_id', 'period']].to_csv("train_periods.csv", index=False)
#target_id list: will be used in one hot encoding of labels
all_clmap_vals = np.array(list(target_map.values()))
print("Train Feats: {0}".format(train_features))
print("Train Data All COLS: {0}".format(train_full.columns))
############################# LOSS FUNCTION ####################################

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
cl_weights_tmp = all_class_weights.copy() 
del cl_weights_tmp['class_99'] 
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
    'max_depth': 3,
    'class_weight': {np.where(all_classes==int(cl.split('_')[1]))[0][0]: i\
                     for (cl, i) in cl_weights_tmp.items()}
}

lgbm_params = {
    'boosting_type': 'dart',
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
    'n_estimators': 2000,
    'silent': -1,
    'verbose': -1,
    'max_depth': 3,
    'class_weight': {np.where(all_classes==int(cl.split('_')[1]))[0][0]: i\
                     for (cl, i) in cl_weights_tmp.items()},
    'min_data_in_leaf': 50
}

#oof_preds_both = np.zeros((train_full.shape[0], 15))
#test_prediction = np.zeros((test_meta_data.shape[0], 15))
train_mean = train_full.mean(axis=0)
train_full.fillna(train_mean, inplace=True)

w = train_full['target_id'].value_counts()
weights = {i : np.sum(w) / w[i] for i in w.index}

lgbm_list, oof_preds_lgbm, score =\
    trainLGB(train_full, train_features, folds, lgbm_params, weights)
rep = pd.DataFrame(columns=['F', 'Score'])
rep.loc[0, :] = ['Orig', score]

#for i, f in enumerate(train_feats_orig):
#    train_features = train_feats_orig.copy()
#    train_features.remove(f)
#    lgbm_list, oof_preds_lgbm, score =\
#        trainLGB(train_full, train_features, folds, lgbm_params, weights)
#    rep.loc[i+1, :] = [f, score]
#    break
    
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
    chunks = 300000
    temp_label_features = label_features.copy()
    temp_label_features.remove("class_99")
    for i_c, full_test in enumerate(pd.read_csv('input/full_test_saved_gb.csv', chunksize=chunks, iterator=True)):
        print("*"*20 + "chunk: " + str(i_c) + "*"*20)
        
        # Make predictions
        print("Predicting...")
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
        # Store predictions
        print("Saving predictions...")
         #create prediction file for each model
    #        for _pred, name in zip([preds_lgb, preds_xgb, preds_comb], ['lgb', 'xgb', 'comb']):
        for _pred, name in zip([preds_lgb], ['final']):
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
            
        del preds_lgb
        gc.collect()
if do_prediction is True:
    model = 'output/gb_predictions_final'
    z = pd.read_csv(model + '.csv')
    
    no_99 = label_features.copy()
    no_99.remove('class_99')

    preds_99 = np.ones(z.shape[0])    
    for i in range(z[no_99].values.shape[1]):
        preds_99 *= (1 - z[no_99].values[:, i])
    z['class_99'] = 0.18 * preds_99 / np.mean(preds_99)
        
#    y = pd.DataFrame()
#    model = 'output/gb_predictions_lgb'
#    z = pd.read_csv(model + '.csv')
#    y['mymean'] = z[no_99].mean(axis=1)
#    y['mymedian'] = z[no_99].median(axis=1)
#    y['mymax'] = z[no_99].max(axis=1)
#    z['class_99'] = GenUnknown(y)
    
    cols = list(z.columns)
    cols.remove('object_id')
    gal_objs = test_meta_data\
                .loc[test_meta_data['hostgal_photoz']==0, 'object_id']
    exgal_objs = test_meta_data\
            .loc[test_meta_data['hostgal_photoz']>0, 'object_id']
    z.loc[z['object_id'].isin(gal_objs), label_features_exgal] = 0
    z.loc[z['object_id'].isin(exgal_objs), label_features_gal] = 0
    
    tech ='_gb'
    z.to_csv(model + tech + '.csv', index=False)
        
    preds_blend = pd.read_csv('output/nn_predictions_nn_scgal.csv')
    preds_blend[label_features] = 0.4*preds_blend[label_features] + 0.6*z[label_features]
    preds_blend.to_csv('output/blend.csv', index=False)
    preds_blend = pd.read_csv('output/blend_best.csv')
    preds_blend[label_features] = 0.8*preds_blend[label_features] + 0.2*z[label_features]
    preds_blend.to_csv('output/blend_final.csv', index=False)
