#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 23:41:32 2018

@author: dslasdoce
"""
import pandas as pd
import cesium.featurize as featurize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dataproc as dproc
import sncosmo
from astropy.table import Table

train = pd.read_csv('input/training_set.csv')
train_meta = pd.read_csv('input/training_set_metadata.csv')
pb_mapping = {0:'sdssu', 1:'sdssg', 2:'sdssr', 3:'sdssi', 4:'sdssz', 5:'desy'}
target_class = 67
target_object_ids = train_meta.loc[train_meta.target==target_class, 'object_id']
target_df = train.loc[train.object_id.isin(target_object_ids)]
target_meta = train_meta.loc[train_meta.object_id.isin(target_object_ids)]

obj = 2922

def getAveDeclineRate(df_main):
    df_main = df_main.sort_values('mjd').reset_index(drop=True)
    df_main = df_main.loc[df_main['flux'].idxmax():]
    df_main = df_main.loc[:df_main['flux'].idxmin()]
    ave_decrate = {'ave_decrate':
                    (df_main['flux'].diff()/df_main['mjd'].diff()).mean()}
    return pd.Series(ave_decrate)

z = train.groupby(['object_id', 'passband'])\
          .apply(getAveDeclineRate).reset_index()
          






#model = sncosmo.Model(source='salt2')
#def lcFit(df_main):
#    df_main = df_main.rename({'mjd': 'time', 'passband': 'band'}, axis=1)
#    df_main['band'] = df_main['band'].map(pb_mapping)
#    df_main['zp'] = 25
#    df_main['zpsys'] = 'ab'
#    data = Table.from_pandas(df_main)
#    
##    model.set(z=0.5)
#    try:
#        res, fitted_model = sncosmo.fit_lc(data, model, ['t0', 'x0', 'x1', 'c'],
#                                           minsnr=3)
#    except (RuntimeError, sncosmo.fitting.DataQualityError):
#        res = {}
#        res['parameters'] = [0,0,0,0,0]
#        res['param_names'] = ['z', 't0', 'x0', 'x1', 'c']
#    ret = pd.Series(res['parameters'], index=res['param_names'])
#    ret.drop('z', inplace=True)
#    ret.drop('t0', inplace=True)
##    if ret['t0'] != 0:
##        print(df_main.loc[df_main['time']==ret['t0'], 'flux'])
#    return ret
#train['sn'] = np.abs(train['flux']/train['flux_err'])
#import time
#start = time.time()
#z = train.loc[train['detected']==1].groupby('object_id').apply(lcFit)
#end = time.time() - start
##z = z.transform()
##
#import pywt
#df_main = train.loc[(train.object_id==obj)].sort_values('mjd').reset_index(drop=True)
#coeffs2 = pywt.dwt2(df_main, 'bior1.3')
#df_main = df_main.rename({'mjd': 'time', 'passband': 'band'}, axis=1)
#df_main['band'] = df_main['band'].map(pb_mapping)
#df_main['zp'] = 25
#df_main['zpsys'] = 'ab'
#data = Table.from_pandas(df_main)
#model = sncosmo.Model(source='salt2')
#model.set(z=0.5)
#res, fitted_model = sncosmo.fit_lc(data, model, ['t0', 'x0', 'x1', 'c'])
#
#mflux = model.bandflux(data['band'], data['time'])


#train.groupby('object_id')