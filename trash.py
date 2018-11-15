#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:07:22 2018

@author: dslasdoce
"""

def getFullDataTS(ts_data, meta_data, label=False):

    meta_set = pd.DataFrame(columns=['object_id'])
    meta_set['object_id'] = ts_data['object_id'].unique()
#    full_data['ts'] = np.nan
#    full_data['ts'] = full_data['ts'].astype('object')
#    ############## full train data ##############
#    aggs = {'mjd': ['min', 'max', 'size'],
#            'passband': ['min', 'max', 'mean', 'median', 'std'],
#            'flux': ['min', 'max', 'mean', 'median', 'std'],
#            'flux_err': ['min', 'max', 'mean', 'median', 'std'],
#            'detected': ['min', 'max', 'mean', 'median', 'std']}
#    full_data = ts_data.groupby('object_id').agg(aggs)
#    new_columns = [
#        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
#    ]
#    full_data.columns = new_columns
#    
#    #remove mjd diff
#    full_data['mjd_length'] = full_data['mjd_max'] - full_data['mjd_min']
#    full_data = full_data.drop(['mjd_max', 'mjd_min'], axis=1)
##    new_columns = agg_train.columns
#    
#    full_data = full_data.reset_index().merge(
#        right=meta_data,
#        how='left',
#        on='object_id'
#    )
#    del full_data['distmod'], full_data['hostgal_specz']
#    excluded_features = ['target', 'target_id', 'y', 'object_id']
#    train_features = [f for f in full_data.columns if f not in excluded_features]
#
##    del agg_ts
#    gc.collect()
    train_features_ts = ['mjd', 'passband', 'flux', 'flux_err', 'detected']
#    ts_arr = np.zeros((ts_data['object_id'].nunique(),
#                       len(train_features_ts)))
    ts_dataset = []
    for oid in meta_set['object_id']:
        tmp_df = ts_data.loc[ts_data['object_id']==oid, train_features_ts]\
                        .sort_values('mjd').reset_index(drop=True)
        tmp_df = tmp_df.fillna(0)
        ts_dataset.append(tmp_df)
#        meta_dataset.append(ts_data\
#                          .loc[ts_data['object_id']==oid, train_features_ts]\
#                          .sort_values('mjd').reset_index(drop=True))

#    for gr in ts_data.groupby('object_id'):
#        idx_list = full_data.index[full_data['object_id']==gr[0]].tolist()
#        if len(idx_list) != 1:
#            print("duplicate error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#        for idx in idx_list:
#            full_data.at[idx, 'ts'] = gr[1][train_features_ts].values.tolist()
#        del gr
#        gc.collect()
    ts_dataset = pd.Series(ts_dataset)
    meta_set = meta_set.merge(meta_data, how='left', on='object_id').reset_index()
    del meta_set['distmod'], meta_set['hostgal_specz']
    train_features_meta = [f for f in meta_set.columns if f not in excluded_features]
    return ts_dataset, meta_set, train_features_ts, train_features_meta

def getMax(df_main):
    df_main = df_main.reset_index(drop=True)
#    print(df_main)
    flmax_idx = df_main['flux'].idxmax()
    flmax_mag = df_main['flux'].max()
    flmax_mjd = df_main['mjd'].loc[flmax_idx]
    
    left_d = len(df_main.loc[df_main['mjd'] < flmax_mjd])
    right_d = len(df_main.loc[df_main['mjd'] > flmax_mjd])
    
    if right_d > left_d:
        df_temp = df_main.iloc[flmax_idx:, :]
        flmin_mag = df_main['flux'].min()
        fl_lower = flmax_mag - (flmax_mag - flmin_mag)*0.5
        try:
            flmin_mjd = df_temp['mjd'].loc[df_temp['flux'] < fl_lower].iloc[0]
        except IndexError:
#            print(df_main)
            flmin_mjd = np.nan
    else:
        df_temp = df_main.iloc[:flmax_idx, :]
        flmin_mag = df_main['flux'].min()
        fl_lower = flmax_mag - (flmax_mag - flmin_mag)*0.5
        try:
            flmin_mjd = df_temp.loc[df_temp['flux'] < fl_lower]['mjd'].iloc[-1]
        except IndexError:
            flmin_mjd = np.nan
        
    return pd.Series([abs(flmax_mjd - flmin_mjd)], index=['15decay'])

    
def get15decay(df_main):
    df_main = df_main.reset_index(drop=True)
#    print(df_main)
    flmax_idx = df_main['flux'].idxmax()
    flmax_mag = df_main['flux'].max()
    flmax_mjd = df_main['mjd'].loc[flmax_idx]
    
    left_d = len(df_main.loc[df_main['mjd'] < flmax_mjd])
    right_d = len(df_main.loc[df_main['mjd'] > flmax_mjd])
    
    try:
        df_temp = df_main.iloc[flmax_idx:, :]
        flmin_mag = df_temp['flux'].loc[df_temp['mjd'] > flmax_mjd + 15].iloc[0]
        flmin_mjd = df_temp['mjd'].loc[df_temp['mjd'] > flmax_mjd + 15].iloc[0]
    except IndexError:
        flmin_mag = flmin_mjd = np.nan
#        df_temp = df_main.iloc[:flmax_idx, :]
#        flmin_mag = df_temp['flux'].loc[df_temp['mjd'] < flmax_mjd - 15].iloc[-1]
#        flmin_mjd = df_temp['mjd'].loc[df_temp['mjd'] < flmax_mjd - 15].iloc[-1]
        
        
    return pd.Series([abs((flmax_mag-flmin_mag)/(flmax_mjd-flmin_mjd))],
                      index=['15decay'])

#def get15decay(df_main):
#    df_main = df_main.reset_index(drop=True)
##    print(df_main)
#    flmax_idx = df_main['flux'].idxmax()
#    flmax_mag = df_main['flux'].max()
#    flmax_mjd = df_main['mjd'].loc[flmax_idx]
#    
#    left_d = len(df_main.loc[df_main['mjd'] < flmax_mjd])
#    right_d = len(df_main.loc[df_main['mjd'] > flmax_mjd])
#    
#    if right_d > left_d:
#        df_temp = df_main.iloc[flmax_idx:, :]
#        flmin_mag = df_temp['flux'].loc[df_temp['mjd'] > flmax_mjd + 15].iloc[0]
#        flmin_mjd = df_temp['mjd'].loc[df_temp['mjd'] > flmax_mjd + 15].iloc[0]
#    else:
#        df_temp = df_main.iloc[:flmax_idx, :]
#        flmin_mag = df_temp['flux'].loc[df_temp['mjd'] < flmax_mjd - 15].iloc[-1]
#        flmin_mjd = df_temp['mjd'].loc[df_temp['mjd'] < flmax_mjd - 15].iloc[-1]
#        
#    return pd.Series([abs((flmax_mag-flmin_mag)/flmax_mag)],
#                      index=['15decay'])
    
#def getFullData(ts_data, meta_data):
#    ############## full train data ##############
#    aggs = {'mjd': ['min', 'max', 'size'],
#            'passband': ['min', 'max', 'mean', 'std'],
#            'flux': ['min', 'max', 'mean', 'median', 'std', 'size'],
#            'flux_err': ['min', 'max', 'mean', 'median', 'std'],
#            'detected': ['mean']}
#    
#    period_df = ts_data.groupby('object_id')\
#            .apply(lambda grp: pd.Series({'period': lcperiod(grp)})).reset_index()
#    full_data = ts_data.groupby('object_id').agg(aggs)
#    new_columns = [
#        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
#    ]
#    full_data.columns = new_columns
#    
#    #remove mjd diff
#    full_data['mjd_length'] = full_data['mjd_max'] - full_data['mjd_min']
#    full_data = full_data.drop(['mjd_max', 'mjd_min'], axis=1)
#    full_data['flux_per_mjd'] = (full_data['flux_max'] - full_data['flux_min'])\
#                                /full_data['mjd_length']
#    full_data['flux_err_per_mjd'] = (full_data['flux_err_max'] - full_data['flux_err_min'])\
#                                /full_data['mjd_length']
#    full_data['passband_per_mjd'] = (full_data['passband_max'] - full_data['passband_min'])\
#                                /full_data['mjd_length']
#    full_data['flux_standard_err'] = full_data['flux_mean']/np.sqrt(full_data['flux_size'])
#
##    new_columns = agg_train.columns
#    
#    full_data = full_data.reset_index().merge(
#        right=meta_data,
#        how='left',
#        on='object_id'
#    )
#    del full_data['distmod'], full_data['hostgal_specz']
#    del full_data['passband_min'], full_data['passband_max']
#    del full_data['flux_size']
##    del full_data['passband_min'], full_data['passband_max']
#    
#    #add interpolateed periods
#    full_data = full_data.merge(period_df, how='left', on='object_id')
#        
#    train_features = [f for f in full_data.columns if f not in excluded_features]
#
##    del agg_ts
#    gc.collect()


##################################################
    ############################## one class SVM ################################### 
#test_meta_data = test_meta_data.fillna(0)
#train_full_svm = train_full.fillna(0)

#import sklearn.svm as svm
#oneclass = svm.OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05, verbose=True)
#oneclass.fit(train_full_svm[train_features])
#del train_full_svm
#gc.collect()
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