#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:58:25 2018

@author: dslasdoce
"""

from gatspy import datasets, periodic
from gatspy.periodic import LombScargleMultiband
import pandas as pd
import numpy as np
from astropy.stats import LombScargle
from sklearn.metrics import mean_squared_error

train = pd.read_csv('training_set.csv')
train_meta = pd.read_csv('training_set_metadata.csv')
pb_mapping = {0:'u', 1:'g', 2:'r', 3:'i', 4:'z', 5:'y'}
#from sklearn import mixture
#rrlyrae = datasets.fetch_rrlyrae()
#lcid = rrlyrae.ids[0]
#t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
#model = periodic.LombScargleMultiband(fit_period=True)
#model.optimizer.period_range=(0.5, 0.7)
#model.fit(t, mag, dmag, filts)

##168952
#DFlc = train.groupby('object_id').groups[168952]
#DFlc = train.loc[DFlc].sort_values('mjd')
#DFlc['filts'] = DFlc['passband'].map(pb_mapping)
#t_min = max(np.median(np.diff(sorted(DFlc['mjd']))), 0.1)
#t_max = min(10., (DFlc['mjd'].max() - DFlc['mjd'].min())/2.)
#
#model = LombScargleMultiband(fit_period=True)
#model.optimizer.set(period_range=(t_min, t_max), first_pass_coverage=5)
#model.fit(DFlc['mjd'], DFlc['flux'], dy=DFlc['flux_err'], filts=DFlc['filts'])
#a = model.predict(DFlc.mjd,
#              filts=DFlc['filts'],
#              period=model.best_period)
#np.log(mean_squared_error(y_true=DFlc.mjd, y_pred=a))
##615:                    22.014871611804605
##615 + 168952 -- 168952: 22.010882417427815
##615 + 168952 -- 615:    22.010687129319862
##168952: 


target_class = 92
target_object_ids = train_meta.loc[train_meta.target==target_class, 'object_id']
target_df = train.loc[train.object_id.isin(target_object_ids)]
models = []
for i in range(5):
    target_obj = target_object_ids.sample().values[0]
    DFlc = target_df.loc[target_df.object_id==target_obj].copy()
    DFlc['filts'] = DFlc['passband'].map(pb_mapping)
    model = LombScargleMultiband(fit_period=True)
    t_min = max(np.median(np.diff(sorted(DFlc['mjd']))), 0.1)
    t_max = min(10., (DFlc['mjd'].max() - DFlc['mjd'].min())/2.)
    model.optimizer.set(period_range=(t_min, t_max), first_pass_coverage=5)
    model.fit(DFlc['mjd'], DFlc['flux'], dy=DFlc['flux_err'], filts=DFlc['filts'])
    models.append(model)
    
#calculate lomb error for each object_id
for obj_id, DFlc in train.groupby('object_id'):
    DFlc = DFlc.sort_values('mjd')
    DFlc['filts'] = DFlc['passband'].map(pb_mapping)
    
    y_pred = np.zeros((DFlc.shape[0],))
    for mdl in models:
        y_pred += mdl.predict(DFlc.mjd,
                               filts=DFlc['filts'],
                               period=mdl.best_period)\
                  /(len(models))
        
    lomb_error = np.log(mean_squared_error(y_true=DFlc.mjd, y_pred=y_pred))
    train_meta.loc[train_meta.object_id==obj_id, 'lomb_error'] = lomb_error
    
import seaborn as sns
#train_meta['lomb_error'] = np.exp(train_meta['lomb_error'])
color_map = {92: '#75bbfd', 88: '#929591', 42: '#89fe05', 90: '#bf77f6',
             65: '#d1b26f', 16: '#00ffff', 67: '#13eac9', 95: '#35063e',
             62: '#0504aa', 15: '#c7fdb5', 6: '#cb416b', 52: '#fdaa48',
             64: '#040273', 53: '#ffff84'}
train_meta['cl_color'] = train_meta['target'].map(color_map)
sns.distplot(train_meta['lomb_error'])
import matplotlib.pyplot as plt
plt.scatter(x=train_meta['target'], y=train_meta['lomb_error'],
            c=train_meta['cl_color'])
sns.scatterplot(x='target', y='lomb_error', data=train_meta,
                palette=train_meta['cl_color'])
def passbandToCols(df):

    try:
        df = df.drop('object_id', axis=1)
    except KeyError:
        pass
    df = df.set_index('passband').unstack()
    return pd.DataFrame(columns=["{0}-{1}".format(idx[0], idx[1])\
                                 for idx in df.index.tolist()],
                        data=[df.values])


#model = LombScargleMultiband(fit_period=True)
#model.optimizer.set(period_range=(t_min, t_max), first_pass_coverage=5)
#model.fit(cl_lc['mjd'], cl_lc['flux'], dy=cl_lc['flux_err'], filts=cl_lc['passband'])
#period = model.best_period
#magfit = model.predict(cl_lc_test['mjd'], filts=cl_lc_test['passband'])

#from scipy import signal
#signal.bspline(cl_lc_test['flux'], 2).sum()
#signal.bspline(cl_lc['flux'], 2).sum()

#class_sample = train_meta[['object_id', 'target']].groupby('target').first()
#class_sample = class_sample.reset_index()
#class_sample_t = train_meta[['object_id', 'target']].groupby('target').last()
#class_sample_t = class_sample_t.reset_index()
#i = 3
#cl_lc = train.loc[train['object_id']==class_sample['object_id'].iloc[i]]\
#        .sort_values('mjd')
#cl_lc_test = train.loc[train['object_id']==class_sample_t['object_id'].iloc[i]]\
#        .sort_values('mjd')
#t_min = max(np.median(np.diff(sorted(cl_lc['mjd']))), 0.1)
#t_max = min(5., (cl_lc['mjd'].max() - cl_lc['mjd'].min())/2.)
#import peakutils
#a = len(peakutils.indexes(cl_lc['flux'].values, thres=0, min_dist=5))/(cl_lc_test['mjd'].max() - cl_lc_test['mjd'].min())
#b = len(peakutils.indexes(cl_lc_test['flux'].values, thres=0, min_dist=5))/(cl_lc_test['mjd'].max() - cl_lc_test['mjd'].min())
#
#from astropy.stats import LombScargle
#frequency, power = LombScargle(cl_lc['mjd'], cl_lc['flux']).autopower()
#best_period1 = 1/frequency[np.argmax(power)]
#
#frequency, power = LombScargle(cl_lc_test['mjd'], cl_lc_test['flux']).autopower()
#best_period2 = 1/frequency[np.argmax(power)]

chunks = 100000
feats = ['period', 'power', 'Eta_e']
labels = ['object_id']
for f in feats:
    for i in range(6):
        labels.append(f + '--' + str(i))
        
for i_c, df in enumerate(pd.read_csv('full_test_saved.csv',
                                     chunksize=chunks, iterator=True)):
    if i_c == 0:
        df[labels].to_csv('calc_feats.csv', index=False)
    else: 
        df[labels].to_csv('calc_feats.csv', 
                          header=False, mode='a', index=False)
z = pd.read_csv('calc_feats.csv')
