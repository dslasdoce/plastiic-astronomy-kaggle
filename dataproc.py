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
from astropy.stats import LombScargle
import feets
import sncosmo
from astropy.table import Table
import peakutils
import cesium.featurize as featurize
import warnings
from tsfresh.feature_extraction import extract_features
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter('ignore', RuntimeWarning)
#class mapping from actual class to target_id
target_map = {6: 0, 15:1, 16:2, 42:3, 52: 4, 53: 5, 62: 6, 64: 7,
              65: 8, 67: 9, 88: 10, 90: 11, 92: 12, 95: 13, 99: 14}
target_map = dict(sorted(target_map.items(), key=operator.itemgetter(1)))
excluded_features = ['target', 'target_id', 'y', 'object_id', 'passband',
                     'hostgal_specz', 'distmod','index', 'level_1'
                     'ra', 'decl',
                     'gal_l', 'gal_b',
                     'ddf']
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

try:
    import sys
    ft = sys.argv[1]
except IndexError:
    print("using default feets")
#    ft = ['Eta_e', 'LinearTrend', 'MaxSlope', 'Q31', 'StructureFunction_index_21']
    ft = ['LinearTrend', 'MaxSlope', 'Q31', 'Amplitude', 'Autocor_length',
          'Beyond1Std', 'FluxPercentileRatioMid20', 'FluxPercentileRatioMid80']

fs = feets.FeatureSpace(only=ft)

def peakAndValley(df):
    df = df.reset_index(drop=True)
    indices_peak = peakutils.indexes(df['flux'], min_dist=50)
    indices_valley = peakutils.indexes(-df['flux'], min_dist=50)
    try:
        indices_peak = indices_peak[0]
        indices_valley = [i for i in indices_valley if i > indices_peak][0]
        val = df['mjd'].iloc[indices_peak] - df['mjd'].iloc[indices_valley]
    except:
        val = 0
    
    return pd.Series([val], index=['decay_period'])
        
    
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
    
    period = pd.Series([period, power], index=['period', 'power'])\
    
    return period

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
    return features_ser

def lcFeaturesEx(df_main):
    df_main = df_main.sort_values('mjd')
        
    features, values = fs.extract(time=df_main['mjd'],
                              magnitude=df_main['flux'],
                              error=df_main['flux_err'])
    
    features_ser = pd.Series(values, index=features)
    
    return features_ser

def lscargleTrans(df_main):
    try:
        frequency, power = LombScargle(df_main['mjd'], df_main['flux'])\
                                       .autopower(nyquist_factor=1)
        period = 1/frequency[np.argmax(power)]
    except ValueError:
        period = 0
    period = pd.Series([period, power.mean()], index=['period', 'pow'])
    
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

def boundaryFreq(df):
    q_low = df['flux'].quantile(0.90)
    vals_upper = df.loc[df['flux'] > q_low]
    q_count = vals_upper['mjd'].count()
    q_up_mjd_diff = vals_upper['mjd'].max() - vals_upper['mjd'].min()
    q_up_flux_diff = vals_upper['flux'].max() - vals_upper['flux'].min()
    q_up_ratio = q_up_flux_diff/q_up_mjd_diff
    
    q_low = df['flux'].quantile(0.45)
    q_up = df['flux'].quantile(0.55)
    vals_mid = df.loc[(df['flux'] > q_low) & (df['flux'] < q_up)]
    q_mid_count = vals_mid['mjd'].count()
    q_mid_mjd_diff = vals_mid['mjd'].max() - vals_mid['mjd'].min()
    q_mid_flux_diff = vals_mid['flux'].max() - vals_mid['flux'].min()
    q_mid_ratio = q_mid_flux_diff/q_mid_mjd_diff
    
    q_up = df['flux'].quantile(0.1)
    vals_low = df.loc[df['flux'] < q_up]
    q_low_count = vals_low['mjd'].count()
    q_low_mjd_diff = vals_low['mjd'].max() - vals_low['mjd'].min()
    q_low_flux_diff = vals_low['flux'].max() - vals_low['flux'].min()
    q_low_ratio = q_low_flux_diff/q_low_mjd_diff
    
    r_um = q_count/q_mid_count
    r_ul = q_count/q_low_count
    
    
    
    return pd.Series({'q_count': q_count, 'q_up_ratio':q_up_ratio,
                      'q_mid_count': q_mid_count, 'q_mid_ratio': q_mid_ratio,
                      'q_low_count': q_low_count, 'q_low_ratio': q_low_ratio,
                      'r_um': r_um, 'r_ul': r_ul})

pb_mapping = {0:'sdssu', 1:'sdssg', 2:'sdssr', 3:'sdssi', 4:'sdssz', 5:'desy'}
model = sncosmo.Model(source='salt2')
def lcFit(df_main):
    df_main = df_main.rename({'mjd': 'time', 'passband': 'band'}, axis=1)
    df_main['band'] = df_main['band'].map(pb_mapping)
    df_main['zp'] = 25
    df_main['zpsys'] = 'ab'
    data = Table.from_pandas(df_main)
    
    try:
        res, fitted_model = sncosmo.fit_lc(data, model, ['t0', 'x0', 'x1', 'c'],
                                           minsnr=3)
    except (RuntimeError, sncosmo.fitting.DataQualityError):
        res = {}
        res['parameters'] = [0,0,0,0,0]
        res['param_names'] = ['z', 't0', 'x0', 'x1', 'c']
    ret = pd.Series(res['parameters'], index=res['param_names'])
    ret.drop('z', inplace=True)
    ret.drop('t0', inplace=True)

    return ret

def getAveDeclineRate(df_main):
    df_main = df_main.sort_values('mjd').reset_index(drop=True)
    df_main = df_main.loc[df_main['flux'].idxmax():]
    df_main = df_main.loc[:df_main['flux'].idxmin()]
    ave_decrate = {'ave_decrate':
                    (df_main['flux'].diff()/df_main['mjd'].diff()).mean()}
    return pd.Series(ave_decrate)

def getFullData(ts_data, meta_data, perpb=False):
    aggs = {'flux': ['min', 'max', 'mean', 'median', 'std', 'size', 'skew'],
            'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
            'detected': ['mean'],
            'flux_ratio_sq': ['sum', 'skew'],
            'flux_by_flux_ratio_sq': ['sum'],
            'amag' : ['min', 'max', 'mean', 'std', 'skew']}
    
    ts_data['flux_ratio_sq'] = np.power(ts_data['flux'] / ts_data['flux_err'], 2.0)
    ts_data['flux_by_flux_ratio_sq'] = ts_data['flux'] * ts_data['flux_ratio_sq']

    ts_data = ts_data.merge(meta_data[['hostgal_photoz', 'mwebv',
                                       'object_id']], on='object_id', how='left')
    ts_data['amag'] = np.power(ts_data['hostgal_photoz'], 2)*4*np.pi* ts_data['flux']
    
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
                             
    z = ts_data.loc[ts_data['detected']==1]\
        .groupby('object_id')\
        .apply(lambda df: pd.Series(df.loc[df['flux'].idxmax(), 'passband'], index=['pbm']))\
        .reset_index()
    full_data = full_data.merge(z, how='left', on='object_id')
    
    ####################### loaded data up to here #############################
    #uncomment this section if you wish to calculate the light curve period
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
#    fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},
#                               {'coeff': 1, 'attr': 'abs'}],
#           'kurtosis' : None, 'skewness' : None}
    fcp = {
        'flux': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
            'mean_change': None,
            'mean_abs_change': None,
            'length': None,
        },
                
        'flux_by_flux_ratio_sq': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,       
        },
                
        'flux_passband': {
            'fft_coefficient': [
                    {'coeff': 0, 'attr': 'abs'}, 
                    {'coeff': 1, 'attr': 'abs'}
                ],
            'kurtosis' : None, 
            'skewness' : None,
        },
                
        'mjd': {
            'mean_change': None,
            'mean_abs_change': None,
        },
    }
    agg_df_ts = extract_features(ts_data, column_id='object_id', column_sort='mjd',
                                     column_kind='passband', column_value='flux',
                                     default_fc_parameters=fcp['flux_passband'],
                                     n_jobs=2).reset_index()
    agg_df_ts = agg_df_ts.rename({'id':'object_id'}, axis=1)
    full_data = full_data.merge(agg_df_ts, how='left', on='object_id')
    del agg_df_ts
    gc.collect()
    
    agg_df_ts_flux = extract_features(ts_data, 
                                      column_id='object_id', 
                                      column_value='flux', 
                                      default_fc_parameters=fcp['flux'],
                                      n_jobs=2).reset_index()
    agg_df_ts_flux = agg_df_ts_flux.rename({'id':'object_id'}, axis=1)
    full_data = full_data.merge(agg_df_ts_flux, how='left', on='object_id')
    del agg_df_ts_flux
    
    agg_df_ts_flux_by_flux_ratio_sq = extract_features(ts_data, 
                                      column_id='object_id', 
                                      column_value='flux_by_flux_ratio_sq', 
                                      default_fc_parameters=fcp['flux_by_flux_ratio_sq'],
                                      n_jobs=2).reset_index()
    agg_df_ts_flux_by_flux_ratio_sq = agg_df_ts_flux_by_flux_ratio_sq\
                                      .rename({'id':'object_id'}, axis=1)
    full_data = full_data.merge(agg_df_ts_flux_by_flux_ratio_sq, how='left',
                                on='object_id')
    del agg_df_ts_flux_by_flux_ratio_sq
    
    agg_df_ts = extract_features(ts_data, column_id='object_id', column_sort='mjd',
                                 column_kind='passband', column_value='amag',
                                 default_fc_parameters=fcp['flux_passband'],
                                 n_jobs=2).reset_index()
    agg_df_ts = agg_df_ts.rename({'id':'object_id'}, axis=1)
    full_data = full_data.merge(agg_df_ts, how='left', on='object_id')
    del agg_df_ts
    gc.collect()
    
    ############################################################################
    #lc feats per passband and object_id #
    lc_feats = ts_data.groupby(['object_id']).apply(lcFeaturesEx).reset_index()
    full_data = full_data.merge(lc_feats, how='left', on='object_id')
    
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
    
    #######################################
#     detected-based mjd length feature for each object id only
    z = ts_data.loc[ts_data.detected==1].groupby('object_id')\
        .apply(lambda df: pd.Series({'mjd_det': max(df.mjd) - min(df.mjd),
                                     'mjd_decay':\
                                         df.mjd.loc[df.flux.idxmax()]\
                                         - df.mjd.loc[df.flux.idxmin()],
                                     'flux_amp': df.flux.max()\
                                                 - df.flux.min(),
                                     'flux_amp_median': df.flux.max()\
                                                 - df.flux.median()}))\
        .reset_index()
        
    full_data = full_data.merge(z, how='left', on='object_id')
    del z
    full_data['flux_amp'] = full_data['flux_amp']/full_data['mjd_det']
    
    full_data['r'] = np.power(full_data['distmod']/5, 10)*10
    full_data.loc[full_data['r'].isna(), 'r'] = 0
    full_data['r']  = full_data['r']/np.square(1+full_data['hostgal_photoz'] )

    full_data['flux_diff_mm']  = (full_data['flux_max']\
                                  - full_data['flux_median'])\
                                 /full_data['flux_max']
    
    z = ts_data.loc[ts_data['detected']==1]\
        .groupby(['object_id', 'passband'])['flux'].agg(np.max).reset_index()
    z.columns = ['object_id', 'passband', 'fcmax']#, 'fcmin']

    z['fcolmax_flux'] = 2.5*np.log10(z['fcmax']).diff()

    del z['fcmax']
    z = z.loc[z['passband']!=0]
    z = z.groupby('object_id').apply(passbandToCols).reset_index()
    del z['level_1']
    full_data = full_data.merge(z, how='left', on='object_id')
    
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

def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) from 
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    #Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    #Implementing Haversine Formula: 
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),  
                          np.multiply(np.cos(lat1), 
                                      np.multiply(np.cos(lat2), 
                                                  np.power(np.sin(np.divide(dlon, 2)), 2))))
    
    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    return {
        'haversine': haversine, 
        'latlon1': np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2)), 
   }

      
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
    
    return train_meta, test_meta_data

    
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    