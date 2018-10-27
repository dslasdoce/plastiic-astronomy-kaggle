#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 21:32:31 2018

@author: dslasdoce
"""

import pandas as pd
import operator
import numpy as np
import gc
#from keras.utils import Sequence
from gatspy.periodic import LombScargleMultiband
from astropy.stats import LombScargle

#class mapping from actual class to target_id
target_map = {6: 0, 15:1, 16:2, 42:3, 52: 4, 53: 5, 62: 6, 64: 7,
              65: 8, 67: 9, 88: 10, 90: 11, 92: 12, 95: 13, 99: 14}
target_map = dict(sorted(target_map.items(), key=operator.itemgetter(1)))
excluded_features = ['target', 'target_id', 'y', 'object_id', 'passband',
                     'hostgal_specz']

    
def getDataParameters():
    #actual class list
    all_classes = np.array(list(target_map.keys()))

    #class weights
    all_class_weights = {'class_' + str(cl):1 for cl in all_classes}
    all_class_weights['class_99'] = 2
    all_class_weights['class_64'] = 2
    all_class_weights['class_15'] = 2
    
    #class name based labels
    label_features = ['class_' + str(cl) for cl in all_classes]
    
    return target_map, label_features, all_classes, all_class_weights

def lcperiod(df_main):
    try:
        frequency, power = LombScargle(df_main['mjd'], df_main['flux'],
                                       dy=df_main['flux_err']).autopower(nyquist_factor=1)
        period = 1/frequency[np.argmax(power)]
        power = power.mean()
    except ValueError:
        period = 0
        power = 0
    period = pd.Series([period, power], index=['period', 'pow'])
    return period

def lscargleTrans(df_main):
    try:
        frequency, power = LombScargle(df_main['mjd'], df_main['flux'])\
                                       .autopower(nyquist_factor=1)
        period = 1/frequency[np.argmax(power)]
    except ValueError:
        period = 0
    period = pd.Series([period, power.mean()], index=['period', 'pow'])
#    freq_df = pd.Series(df_main['mjd']/period)%1, )
    return period
    
def passbandToCols(df):
    try:
        df = df.drop('object_id', axis=1)
    except KeyError:
        pass
    df = df.set_index('passband').unstack()
    return pd.DataFrame(columns=["{0}-{1}".format(idx[0], idx[1])\
                                 for idx in df.index.tolist()],
                        data=[df.values])
    
def getFullData(ts_data, meta_data):
    aggs = {'mjd': ['min', 'max', 'size'],
            'flux': ['min', 'max', 'mean', 'median', 'std', 'size', 'skew'],
            'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
            'detected': ['mean'],
            'flux_ratio_sq': ['sum', 'skew'],
            'flux_by_flux_ratio_sq': ['sum', 'skew']}
    
    ts_data['flux_ratio_sq'] = np.power(ts_data['flux'] / ts_data['flux_err'], 2.0)
    ts_data['flux_by_flux_ratio_sq'] = ts_data['flux'] * ts_data['flux_ratio_sq']

    #feature aggregation per passband
    full_data = ts_data.groupby(['object_id','passband']).agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    full_data.columns = new_columns
    full_data = full_data.reset_index()
    
#    full_data['mjd_length'] = full_data['mjd_max'] - full_data['mjd_min']
    full_data = full_data.drop(['mjd_max', 'mjd_min'], axis=1)
#    full_data['flux_per_mjd'] = (full_data['flux_max'] - full_data['flux_min'])\
#                                /full_data['mjd_length']
#    full_data['flux_err_per_mjd'] = (full_data['flux_err_max'] - full_data['flux_err_min'])\
#                                /full_data['mjd_length']
    full_data['flux_standard_err']\
        = full_data['flux_mean']/np.sqrt(full_data['flux_size'])
    del full_data['flux_size']
#    del full_data['mjd_size']
    
    #period calculation if period is per passband
    period_df = ts_data.groupby(['object_id', 'passband'])\
                .apply(lcperiod).reset_index()
    full_data = full_data.merge(period_df, how='left',
                                on=['object_id', 'passband'])
    
    full_data = full_data.groupby('object_id').apply(passbandToCols)
    #bring back the object_id column but not the passband
    full_data = full_data.reset_index(level=0).reset_index(drop=True)
    full_data = full_data.merge(
        right=meta_data,
        how='left',
        on='object_id'
    )
    del full_data['distmod']#, full_data['hostgal_specz']

    #period calculation if period is average per object_id
#    period_df = ts_data.groupby(['object_id', 'passband'])\
#                .apply(lcperiod).reset_index()\
#                .groupby('object_id').mean().reset_index()
#    full_data = full_data.merge(period_df, how='left', on='object_id')
#    del full_data['passband']
        
    train_features = [f for f in full_data.columns if f not in excluded_features]

#    del agg_ts
    gc.collect()
    return full_data, train_features

def getMetaData():
    train_meta = pd.read_csv('training_set_metadata.csv')
    test_meta_data = pd.read_csv('test_set_metadata.csv')
    
#    train_meta.loc[train_meta['hostgal_specz'] == 0, 'is_galactic'] = 0
#    train_meta.loc[train_meta['hostgal_specz'] > 0, 'is_galactic'] = 1
#    
#    test_meta_data.loc[test_meta_data['hostgal_specz'] == 0, 'is_galactic'] = 0
#    test_meta_data.loc[test_meta_data['hostgal_specz'] > 0, 'is_galactic'] = 1
    
    #period merging to meta data if period is calculated as average
#    train_periods = pd.read_csv('train_periods_saved.csv')
#    test_periods = pd.read_csv('test_periods_saved.csv')
#    train_meta = train_meta.merge(train_periods,
#                                  on='object_id', how='left')
#    test_meta_data = test_meta_data.merge(test_periods,
#                                  on='object_id', how='left')
    
    # create 'target_id' column to map with 'target' classes
    # target_id is the index defined in previous step: see dictionary target_map
    # this column will be used later as index for the columns in the final submission
    #target_ids = [target_map[i] for i in train_meta['target']]
    train_meta['target_id'] = train_meta['target'].map(target_map)
    #train_meta = train_full.drop('target', axis=1)
    
#    from sklearn.neighbors import KernelDensity
#    kde = KernelDensity(bandwidth=0.5).fit(train_meta['mwebv'].values.reshape(-1,1))
#    train_meta['logdens'] = kde.score_samples(train_meta['mwebv'].values.reshape(-1,1))
    
    return train_meta, test_meta_data

#    return full_data, train_features
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
#    from time import sleep
    ts_data = pd.read_csv('training_set.csv')
#    period_df = train.groupby(['object_id', 'passband']).apply(lcperiod).reset_index()
#    
#    obj_id = 615
#    passband = 3
#    z = train.set_index(['object_id', 'passband'])
#    z = z.loc[obj_id, passband].sort_values('mjd')
#    period = period_df['period'].loc[(period_df['object_id']==obj_id)\
#                       & (period_df['passband']==passband)].values[0]
#    plt.scatter((z['mjd']/period)%1, z['flux'])
#    plt.show(block=False)
    import seaborn as sns
#train_meta['lomb_error'] = np.exp(train_meta['lomb_error'])
    train_meta, test_meta_data = getMetaData()
    color_map = {92: '#75bbfd', 88: '#929591', 42: '#89fe05', 90: '#bf77f6',
                 65: '#d1b26f', 16: '#00ffff', 67: '#13eac9', 95: '#35063e',
                 62: '#0504aa', 15: '#c7fdb5', 6: '#cb416b', 52: '#fdaa48',
                 64: '#040273', 53: '#ffff84'}
    train_meta['cl_color'] = train_meta['target'].map(color_map)       
##    sns.scatterplot(x='target', y='mwebv', data=train_meta,
##                palette=train_meta['cl_color'])
#    plt.scatter(x=train_meta['target'], y=train_meta['mwebv'], c=train_meta['cl_color'])
#    
#    from sklearn.neighbors import KernelDensity
#    kde = KernelDensity(bandwidth=0.5).fit(train_meta['mwebv'].values.reshape(-1,1))
#    log_dens = kde.score_samples(train_meta['mwebv'].values.reshape(-1,1))
#    fig, ax = plt.subplots()
#    ax.scatter(train_meta.index.values, np.exp(log_dens), c=train_meta['cl_color'])
#    plt.show(block=False)
    
    import feets
    obj_id = 615
    passband = 3
    sample = ts_data.loc[(ts_data['object_id']==obj_id) & (ts_data['passband']==passband)]
    sample = sample.sort_values('mjd')
    fs = feets.FeatureSpace(only=['Std','Amplitude'])
    features, values = fs.extract(time=sample['mjd'],
                                  magnitude=sample['flux'],
                                  error=sample['flux_err'])
#    lc(time=time, magnitude=mag, error=error)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    