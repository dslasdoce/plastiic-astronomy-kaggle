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
