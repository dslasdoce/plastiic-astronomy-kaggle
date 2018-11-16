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
#from gatspy.periodic import LombScargleMultiband
from astropy.stats import LombScargle
import feets
#import matplotlib.pyplot as plt
#import seaborn as sns

#class mapping from actual class to target_id
target_map = {6: 0, 15:1, 16:2, 42:3, 52: 4, 53: 5, 62: 6, 64: 7,
              65: 8, 67: 9, 88: 10, 90: 11, 92: 12, 95: 13, 99: 14}
target_map = dict(sorted(target_map.items(), key=operator.itemgetter(1)))
excluded_features = ['target', 'target_id', 'y', 'object_id', 'passband',
                     'hostgal_specz', 'distmod','index',
                     'ra', 'decl',
                     'gal_l', 'gal_b',
                     'ddf']
excluded_features += []
    
def getDataParameters():
    # actual class list
    all_classes = np.array(list(target_map.keys()))

    # class weights
    all_class_weights = {'class_' + str(cl):1 for cl in all_classes}
    all_class_weights['class_99'] = 2
    all_class_weights['class_64'] = 2
    all_class_weights['class_15'] = 2
    
    #class name based labels
    label_features = ['class_' + str(cl) for cl in all_classes]
    
    return target_map, label_features, all_classes, all_class_weights
#Q31
try:
    import sys
    ft = sys.argv[1]
except IndexError:
    print("using default feets")
    ft = ['Eta_e', 'LinearTrend', 'MaxSlope', 'Q31', 'StructureFunction_index_21']
#    ft = ['Eta_e', 'Amplitude', 'Autocor_length', 'Beyond1Std']
fs = feets.FeatureSpace(only=ft)
print(ft)

def lcperiod(df_main):
    df_main = df_main.sort_values('mjd')
    try:
        frequency, power = LombScargle(df_main['mjd'], df_main['flux'],
                                       dy=df_main['flux_err']).autopower(nyquist_factor=1)
        period = 1/frequency[np.argmax(power)]
        power = power.mean()
    except ValueError:
        period = 0
        power = 0
        
#    features, values = fs.extract(time=sample['mjd'],
#                              magnitude=sample['flux'],
#                              error=sample['flux_err'])
    
    period = pd.Series([period, power], index=['period', 'power'])
#    period = pd.Series([power], index=['power'])
    return period

from cesium.time_series import TimeSeries
import cesium.featurize as featurize
import warnings
#warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)
def lcFreq(df_main):
    df_main = df_main.sort_values('mjd').reset_index(drop=True)
    groups = df_main.groupby('passband')
    t_list = groups.apply(lambda gr: gr['mjd'].values).tolist()
    flx_list = groups.apply(lambda gr: gr['flux'].values).tolist()
    flxer_list = groups.apply(lambda gr: gr['flux_err'].values).tolist()
    feats = featurize.featurize_time_series(times=t_list, values=flx_list, errors=flxer_list,
                                          features_to_use=['freq1_freq'],
                                          scheduler=None)
    feats.columns = feats.columns.droplevel(1)
    feats['freq1_freq'].mean()
    feats['freq1_freq'].std()
    return pd.Series([feats['freq1_freq'].mean(), feats['freq1_freq'].std()],
               index=['freq', 'freq_std'])
    

def lcFeatures(df_main):
    df_main = df_main.sort_values('mjd')
    try:
        frequency, power = LombScargle(df_main['mjd'], df_main['flux'],
                                       dy=df_main['flux_err'])\
                           .autopower(nyquist_factor=1)
        period = 1/frequency[np.argmax(power)]
        power = power.mean()
    except ValueError:
        period = 0
        power = 0
        
    features, values = fs.extract(time=df_main['mjd'],
                              magnitude=df_main['flux'],
                              error=df_main['flux_err'])
    
    features_ser = pd.Series([period, power], index=['period', 'power'])
    features_ser = pd.concat([features_ser, pd.Series(values, index=features)])
#    features_ser = pd.Series(values, index=features)
    return features_ser

def lcFeaturesEx(df_main):
    df_main = df_main.sort_values('mjd')
        
    features, values = fs.extract(time=df_main['mjd'],
                              magnitude=df_main['flux'],
                              error=df_main['flux_err'])
    
    features_ser = pd.Series(values, index=features)
#    features_ser = pd.Series(values, index=features)
    return features_ser

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
    return pd.DataFrame(columns=["{0}--{1}".format(idx[0], idx[1])\
                                 for idx in df.index.tolist()],
                        data=[df.values])
from tsfresh.feature_extraction import extract_features   
def getFullData(ts_data, meta_data, perpb=False):
    aggs = {'flux': ['min', 'max', 'mean', 'median', 'std', 'size', 'skew'],
            'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
            'detected': ['mean'],
            'flux_ratio_sq': ['sum', 'skew'],
            'flux_by_flux_ratio_sq': ['sum'],
            'amag' : ['min', 'max', 'mean', 'std', 'skew']}
    
    ts_data['flux_ratio_sq'] = np.power(ts_data['flux'] / ts_data['flux_err'], 2.0)
    ts_data['flux_by_flux_ratio_sq'] = ts_data['flux'] * ts_data['flux_ratio_sq']

#    
#    vmag = {0: 1.810, 1:3.730, 2:4.490, 3:4.760, 4:4.810, 5: 1.600}
#    ts_data['amag'] = ts_data['passband'].map(vmag)
#    ts_data['amag'] = np.power(10, ts_data['flux']/-2.15)*ts_data['amag']
#    ts_data.loc[ts_data['flux'] < 0, 'flux_adj'] = 0
#    ts_data['vmag'] = -2.5*np.log10(abs(ts_data['flux_adj']/ts_data['vmag']))
    ts_data = ts_data.merge(meta_data[['hostgal_photoz', 'mwebv',
                                       'object_id']], on='object_id', how='left')
    ts_data['amag'] = np.power(ts_data['hostgal_photoz'], 2)*4*np.pi* ts_data['flux']
#    ts_data['amag'] = ts_data['flux'] - ts_data['distmod'] - ts_data['mwebv']
#    ts_data['amag'] = np.power(10, (ts_data['vmag'] - np.log10(abs(ts_data['flux']) + 1) - ts_data['mwebv'])/5) + 1
#    ts_data['br'] = np.power(10, 0.4*ts_data['flux'])
#    ts_data['rd'] = 1/np.power(ts_data['flux']*np.power(1+ts_data['hostgal_photoz'], 2), 0.5)
    #feature aggregation per passband
    if perpb is True:
        full_data = ts_data.groupby(['object_id','passband']).agg(aggs) #orig
    else:
        full_data = ts_data.groupby(['object_id']).agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    full_data.columns = new_columns
    full_data = full_data.reset_index()
    
    full_data['flux_diff'] = full_data['flux_max'] - full_data['flux_min']
    full_data['flux_dif2'] = (full_data['flux_max'] - full_data['flux_min'])\
                             / full_data['flux_mean']
    full_data['flux_w_mean'] = full_data['flux_by_flux_ratio_sq_sum']\
                               / full_data['flux_ratio_sq_sum']
    full_data['flux_dif3'] = (full_data['flux_max'] - full_data['flux_min'])\
                             / full_data['flux_w_mean']
    
#    full_data['amp'] = full_data['flux_max'] - full_data['flux_min']
    ####################### loaded data up to here #############################
    #period calculation if period is per passband
#    period_df = ts_data.groupby(['object_id', 'passband'])\
#                .apply(lcperiod).reset_index()
#    full_data = full_data.merge(period_df, how='left',
#                                on=['object_id', 'passband'])
    
    # lc feats per passband and object_id #
#    lc_feats = ts_data.groupby(['object_id', 'passband'])\
#                .apply(lcFeaturesEx).reset_index()
#    full_data = full_data.merge(lc_feats, how='left',
#                                on=['object_id', 'passband'])
    
    ############################################################################
    # make values for each column VS passband a separate column
    # e.g. period of passband 1 becomes period-1 column then passband column is omitted
    if perpb is True:
        full_data = full_data.groupby('object_id').apply(passbandToCols) #orig
    #bring back the object_id column but not the passband
    full_data = full_data.reset_index(level=0).reset_index(drop=True)
    gc.collect()
    
    ############################
    fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},
                               {'coeff': 1, 'attr': 'abs'}],
           'kurtosis' : None, 'skewness' : None}
    agg_df_ts = extract_features(ts_data, column_id='object_id', column_sort='mjd',
                                     column_kind='passband', column_value='flux',
                                     default_fc_parameters=fcp, n_jobs=2)\
                .reset_index()
    agg_df_ts = agg_df_ts.rename({'id':'object_id'}, axis=1)
    full_data = full_data.merge(agg_df_ts, how='left', on='object_id')
    del agg_df_ts
    gc.collect()
    ###########################
#    df_15decay = ts_data.groupby(['object_id','passband']).apply(get15decay).reset_index()
#    df_15decay = ts_data.groupby(['object_id','passband']).apply(getMax).reset_index()
    
#    aggs = {'15decay': ['min', 'max', 'mean', 'std']}
#    df_15decay = df_15decay.groupby('object_id').agg(aggs)
#    new_columns = [
#            k + '_' + agg for k in aggs.keys() for agg in aggs[k]
#        ]
#    df_15decay.columns = new_columns
#    full_data = full_data.merge(df_15decay, how='left', on='object_id')   
    
    ############################################################################
    #lc feats per passband and object_id #
#    lc_feats = ts_data.groupby(['object_id']).apply(lcFeaturesEx).reset_index()
#    aggs = {'Beyond1Std': ['min', 'max', 'mean', 'std']}
#    lc_feats = lc_feats.groupby('object_id').agg(aggs)
#    new_columns = [
#            k + '_' + agg for k in aggs.keys() for agg in aggs[k]
#        ]
#    lc_feats.columns = new_columns

#    full_data = full_data.merge(lc_feats, how='left', on='object_id')
    
    ############################################################################
    aggs = {'flux_by_flux_ratio_sq': ['skew']}
    ts_data['flux_by_flux_ratio_sq'] = ts_data['flux'] * ts_data['flux_ratio_sq']

    #feature aggregation per passband
    full_data2 = ts_data.groupby('object_id').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    full_data2.columns = new_columns
    full_data2 = full_data2.reset_index()
    full_data = full_data.merge(full_data2, how='left', on='object_id')    
    del full_data2
    gc.collect()
    ############################################################################
    full_data = full_data.merge(
        right=meta_data,
        how='left',
        on='object_id'
    )
    del full_data['distmod']#, full_data['hostgal_specz']
    
    #######################################
#     detected-based mjd length feature for each object id only
    z = ts_data.loc[ts_data.detected==1].groupby('object_id')\
        .apply(lambda df: pd.Series({'mjd_det': max(df.mjd) - min(df.mjd),
                                     'mjd_decay':\
                                         df.mjd.loc[df.flux.idxmax()]\
                                         - df.mjd.loc[df.flux.idxmin()],
                                     'flux_amp': df.flux.max()\
                                                 - df.flux.min()}))\
        .reset_index()
        
    full_data = full_data.merge(z, how='left', on='object_id')
    del z
    full_data['flux_amp'] = full_data['flux_amp']/full_data['mjd_det']

#    full_data = full_data.merge(z, how='left', on='object_id')
#    del z
    ################## frequency
#    freq = ts_data.groupby(['object_id']).apply(lcFreq).reset_index()
#    full_data = full_data.merge(period_df, how='left',
#                                on=['object_id', 'passband'])

################## redshift calibration 
#    flx = ['flux_mean--' + str(i) for i in range(6)]
#    full_data['new'] = full_data['flux_mean']/np.power(full_data['hostgal_photoz'] + 1, 2)
#    aggs = {'flux': ['mean']}
#    full_data['new'] = 1/np.power(full_data['flux_ratio_sq_sum']*np.power(1+full_data['hostgal_photoz'], 2), 0.5)
#    excluded_features += flx
#    del full_data[flx]
#    for f in flx:
#        full_data['dist-' + f] = 1/np.power(full_data[f]*np.power(1+full_data['hostgal_photoz'], 2), 0.5)
#        del full_data[f]    
    
############## feature interactions
#    period_df = meta_data[['object_id', 'period--0', 'period--1', 'period--2',
#                     'period--3', 'period--4', 'period--5']]\
#        .set_index('object_id').unstack().reset_index()
#    period_map = {k: int(k.split('--')[1]) for k in ['period--' + str(i) for i in range(6)]}
#    period_df.columns = ['passband', 'object_id', 'period']
#    period_df['passband'] = period_df['passband'].map(period_map)
#    ts_data = ts_data.merge(period_df, on=['object_id', 'passband'], how='inner')
#    ts_data['mjd'] = ts_data['mjd']%ts_data['period']
#    z = ts_data.loc[ts_data.detected==1].groupby('object_id').apply(lambda df: df['flux'].abs().min())
#    full_data['correction_diff'] = z - full_data.set_index('object_id')['mwebv']
#    full_data['correction_diff'] = full_data['flux_mean'] - full_data['mwebv']
    
#    

#    ts_data = ts_data.groupby(['object_id', 'passband', 'mjd']).mean().reset_index()
    
    #train features
    train_features = [f for f in full_data.columns if f not in excluded_features]
#    del agg_ts
    gc.collect()
    return full_data, train_features

def col_index(cidx):
    if cidx > -0.17 and cidx < 0.15:
        return 0
    elif cidx >= 0.15 and cidx < 0.44:
        return 1
    elif cidx >= 0.44 and cidx < 0.68:
        return 2
    elif cidx >= 0.68 and cidx < 1.15:
        return 3
    elif cidx >= 1.15 and cidx < 1.64:
        return 4
    else:
        return 5
        
def getMetaData():
    train_meta = pd.read_csv('input/training_set_metadata.csv')
    test_meta_data = pd.read_csv('input/test_set_metadata.csv')
    
    #mjd_det saved
#    mjd_det = pd.read_csv('mjd_det.csv')
#    train_meta = train_meta.merge(mjd_det, on='object_id', how='left')
    
    # create 'target_id' column to map with 'target' classes
    # target_id is the index defined in previous step: see dictionary target_map
    # this column will be used later as index for the columns in the final submission
    #target_ids = [target_map[i] for i in train_meta['target']]
    train_meta['target_id'] = train_meta['target'].map(target_map)
    #train_meta = train_full.drop('target', axis=1)
    
    #saved extracted features from LC
#    calc_feats_test
    lc_feats_train = pd.read_csv('input/calc_lcfeats_train.csv')
    lc_feats_test = pd.read_csv('input/calc_lcfeats_test.csv')
    train_meta = train_meta.merge(lc_feats_train,
                                  on='object_id', how='left')
    test_meta_data = test_meta_data.merge(lc_feats_test,
                                  on='object_id', how='left')
    
#    points = [[169*(np.pi/180), 60*(np.pi/180)],
#              [225*(np.pi/180), 54*(np.pi/180)], #
#              [236*(np.pi/180), 42*(np.pi/180)], #
#              [320*(np.pi/180), 50*(np.pi/180)],#
#              [329*(np.pi/180), 68*(np.pi/180)]]#
#    points = [[169*(np.pi/180), 60*(np.pi/180)]]
#    for i, p in enumerate(points):
#        lmbda1 = p[0]
#        phi1 = p[1]
#        phi = train_meta['gal_b'] * (np.pi/180)
#        lmbda = train_meta['gal_l'] * (np.pi/180)
#        train_meta['d-p' + str(i)]\
#            = 2*np.arcsin(np.power(np.power(np.sin((phi-phi1)/2), 2)\
#                                   + np.cos(lmbda)*np.cos(lmbda1)\
#                                     * np.power(np.sin((lmbda-lmbda1)/2), 2),
#                                   0.5))
#        
#    d0 = 63 * (np.pi/180)
#    a0 = 350 * (np.pi/180)
    delta = train_meta['decl']*(np.pi/180)
    alpha = train_meta['ra']*(np.pi/180)
    x = np.cos(delta)*np.cos(alpha)
    y = np.cos(delta)*np.sin(alpha)
    z = np.sin(delta)
#    x0 = np.cos(d0)*np.cos(a0)
#    y0 = np.cos(d0)*np.sin(a0)
#    train_meta['d_sky'] = np.sqrt(np.square(x-x0) + np.square(y-y0))
    train_meta['d_sky'] = np.sqrt(np.square(x) + np.square(y) + np.square(z)
                                  + np.square(train_meta['hostgal_photoz']))
    
    delta = test_meta_data['decl']*(np.pi/180)
    alpha = test_meta_data['ra']*(np.pi/180)
    x = np.cos(delta)*np.cos(alpha)
    y = np.cos(delta)*np.sin(alpha)
    z = np.sin(delta)
    test_meta_data['d_sky'] = np.sqrt(np.square(x) + np.square(y) + np.square(z)
                              + np.square(test_meta_data['hostgal_photoz']))
    
#    train_meta['c_index'] = train_meta['mwebv'].map(col_index)
#    train_meta['gal_pyt2'] = np.power(np.power(train_meta['gal_l'], 2) + np.power(train_meta['gal_b'], 2), 0.5)
#    train_meta['gal_pyt'] = np.power(np.power(train_meta['decl'], 2) + np.power(train_meta['ra'], 2), 0.5)
#    train_meta['hostgal_photoz_sq'] = np.power(train_meta['hostgal_photoz'], 2.0)
#    train_meta = train_meta.merge(pd.read_csv('input/frq.csv'),
#                                  on='object_id',
#                                  how='inner')
#    train_meta['hostgal_photoz_sq'] = np.power(train_meta['hostgal_photoz']/(train_meta['hostgal_photoz_err']+1), 2.0)
#    from sklearn.neighbors import KernelDensity
#    kde = KernelDensity(bandwidth=0.5).fit(train_meta['mwebv'].values.reshape(-1,1))
#    train_meta['logdens'] = kde.score_samples(train_meta['mwebv'].values.reshape(-1,1))
    
    return train_meta, test_meta_data

#    return full_data, train_features
    
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
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

    print(cm)
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
    
if __name__ == "__main__":
    target_map, label_features, all_classes, all_class_weights \
        = getDataParameters()
    train_meta, test_meta_data = getMetaData()
    
    from sklearn.metrics import confusion_matrix
    oof_preds_nn = pd.read_csv('output/oof_nn.csv')
    oof_preds_lgbm = pd.read_csv('output/oof_lgbm.csv')
    oof_preds_xgb = pd.read_csv('output/oof_xgb.csv')
    
    cnf_matrix_nn = confusion_matrix(train_meta['target_id'],
                                     np.argmax(oof_preds_nn[label_features].values,
                                               axis=-1))
    cnf_matrix_lgb = confusion_matrix(train_meta['target_id'],
                                     np.argmax(oof_preds_lgbm[label_features].values,
                                               axis=-1))
    cnf_matrix_xgb = confusion_matrix(train_meta['target_id'],
                                     np.argmax(oof_preds_xgb[label_features].values,
                                               axis=-1))
    plot_confusion_matrix(cnf_matrix_lgb, classes=label_features,normalize=True,
                      title='Confusion matrix')
    plt.show(block=False)
    
    
    
    ############################################################################
#    import multiprocessing
#    from multiprocessing import Pool
#    from time import time
#    t = time()
#    def worker(df_main, i):
#        print("Starting chunk {0}...".format(i))
#        freq = df_main.groupby(['object_id']).apply(lcFreq).reset_index()
#        freq.to_csv("freqs/freq-{0}.csv".format(i), index=False)
#        print("=== Chunk {0} done ===".format(i))
#        return
#    
#    jobs = []
#    chunk_last = pd.DataFrame()
#    chunks = 10000
#    test_row_num = 1421705 
#    total_steps = int(np.ceil(test_row_num/chunks))
#    df_list = []
#    for i_c, df in enumerate(pd.read_csv('training_set.csv',
#                                         chunksize=chunks, iterator=True)):
#        df = pd.concat([chunk_last, df], ignore_index=True)
#        if i_c+1<total_steps:
#            #get the last object id
#            id_last = df['object_id'].values[-1] 
#            #get boolean indeces of rows with object_id == id_last
#            mask_last = (df['object_id']==id_last).values 
#            #get the rows with last object_id
#            chunk_last = df[mask_last] 
#            #remove the rows of the last object_id from the dataset
#            df = df[~mask_last]   
#        df_list.append(df)
##        p = multiprocessing.Process(target=worker, args=(df, i_c))
##        jobs.append(p)
##        p.start()
#        if i_c % 4 ==0 or i_c+1 == total_steps:
#            with Pool(processes=len(df_list)) as pool:
#                pool.starmap(worker, zip(df_list, range(len(df_list))))
#            df_list = []
#        else:
#            continue
##    pool = Pool(processes=len(df_list))
#    
#    print("{0} mins".format((time()-t)/60))
##    period_df = train.groupby(['object_id', 'passband']).apply(lcperiod).reset_index()
##    
##    obj_id = 615
##    passband = 3
##    z = train.set_index(['object_id', 'passband'])
##    z = z.loc[obj_id, passband].sort_values('mjd')
##    period = period_df['period'].loc[(period_df['object_id']==obj_id)\
##                       & (period_df['passband']==passband)].values[0]
##    plt.scatter((z['mjd']/period)%1, z['flux'])
##    plt.show(block=False)
#    
##train_meta['lomb_error'] = np.exp(train_meta['lomb_error'])
#    
#    from os import listdir
#    flist = listdir('freqs/')
#    df = pd.DataFrame()
#    for f in flist:
#        df = pd.concat([pd.read_csv('freqs/' + f), df])
#        
#    df.to_csv('input/frq.csv', index=False)
    
    ############################################################################
    #    from time import sleep
#    ts_data = pd.read_csv('training_set.csv')
#    train_meta, test_meta_data = getMetaData()
#    color_map = {92: '#75bbfd', 88: '#929591', 42: '#89fe05', 90: '#bf77f6',
#                 65: '#d1b26f', 16: '#00ffff', 67: '#13eac9', 95: '#35063e',
#                 62: '#0504aa', 15: '#c7fdb5', 6: '#cb416b', 52: '#fdaa48',
#                 64: '#040273', 53: '#ffff84'}
#    train_meta['cl_color'] = train_meta['target'].map(color_map)
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
    
#    import feets
#    obj_id = 615
#    passband = 3
#    sample = ts_data.loc[(ts_data['object_id']==obj_id) & (ts_data['passband']==passband)]
#    sample = sample.sort_values('mjd')
#    fs = feets.FeatureSpace(only=['Std','Amplitude'])
#    features, values = fs.extract(time=sample['mjd'],
#                                  magnitude=sample['flux'],
#                                  error=sample['flux_err'])
#    sample = ts_data.iloc[:50000].reset_index().copy()
#    z = ts_data.groupby(['object_id', 'passband']).apply(lcFeaturesEx).reset_index()
#    z.to_csv('Autocor_length.csv', index=False)
#    print(features)
#    lc(time=time, magnitude=mag, error=error)
#    a = ts_data[(ts_data.detected==1), mjd_diff:=max(mjd)-min(mjd), by=object_id]
#    z = ts_data.loc[ts_data.detected==1].groupby('object_id')\
#            .apply(lambda df: pd.Series({'mjd_det': max(df.mjd) - min(df.mjd)}))\
#            .reset_index()
#    z.to_csv('mjd_det.csv', index=False)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    