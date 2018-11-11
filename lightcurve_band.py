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
import cesium.featurize as featurize
from cesium.time_series import TimeSeries

train = pd.read_csv('input/training_set.csv')
train_meta = pd.read_csv('input/training_set_metadata.csv')
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


target_class = 52
target_object_ids = train_meta.loc[train_meta.target==target_class, 'object_id']
target_df = train.loc[train.object_id.isin(target_object_ids)]
models = []
obj = 615
df_main = train.loc[(train.object_id==obj)].sort_values('mjd').reset_index(drop=True)
#df_main = train.loc[(train.object_id==obj) & (train.passband==2)].sort_values('mjd').reset_index(drop=True)
#for i in range(6):
#    tmp = df_main.loc[df_main.passband==i, 'flux']
#    df_main.loc[df_main.passband==i, 'flux'] = (tmp -tmp.mean())/tmp.std()
#    tmp = df_main.loc[df_main.passband==i, 'flux_err']
#    df_main.loc[df_main.passband==i, 'flux_err'] = (tmp -tmp.mean())/tmp.std()
#frequency, power = LombScargle(df_main['mjd'], df_main['flux'],
#                               dy=df_main['flux_err']).autopower()
#period = 1/frequency[np.argmax(power)]
#power = power.mean()
groups = df_main.groupby('passband')
t_list = groups.apply(lambda gr: gr['mjd'].values).tolist()
flx_list = groups.apply(lambda gr: gr['flux'].values).tolist()
flxer_list = groups.apply(lambda gr: gr['flux_err'].values).tolist()
#ts = TimeSeries(t=df_main['mjd'], m=df_main['flux'])
#feats = featurize.featurize_single_ts(ts=ts,
#                                      features_to_use=['freq1_freq',
#                                                    'freq1_signif',
#                                                    'freq1_amplitude1'])
feats = featurize.featurize_time_series(times=t_list, values=flx_list, errors=flxer_list,
                                      features_to_use=['freq1_freq'],
                                                scheduler=None)
feats.loc['freq1_freq']
feats.columns = feats.columns.droplevel(1)
frq = feats['freq1_freq'][0]
import seaborn as sns
#sns.scatterplot(x='mjd', y='flux', data=df_main)

#df_main['mjd1'] = df_main['mjd']%period
#sns.scatterplot(x='mjd1', y='flux', data=df_main.loc[df_main.passband==0 & (df_main.detected==True)])

#periods = len(df_main)/feats['freq1_freq'][0]
df_main['mjd2'] = (df_main['mjd']*frq) % 1
#df_main['passband'] = df_main['passband'].astype(str)
sns.catplot(x='mjd2', y='flux', hue='passband', palette='Set1', data=df_main)

from sklearn import preprocessing
for i, df in df_main.groupby('passband'):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['flux1'] = min_max_scaler.fit_transform(df['flux'].values.reshape(-1,1))
    df['flux_err1'] = min_max_scaler.fit_transform(df['flux_err'].values.reshape(-1,1))
    frequency, power = LombScargle(df['mjd'], df['flux1'],
                               dy=df['flux_err1']).autopower()
    period = 1/frequency[np.argmax(power)]
    df['mjd1'] = df['mjd']%period
    sns.scatterplot(x='mjd1', y='flux1', data=df)



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

def normalise(ts):
    return (ts - ts.mean()) / ts.std

groups = train.groupby(['object_id', 'passband'])
times = groups.apply(
    lambda block: block['mjd'].values).reset_index().rename(columns={0: 'seq'})
flux = groups.apply(
    lambda block: normalise(block['flux']).values
).reset_index().rename(columns={0: 'seq'})
err = groups.apply(
    lambda block: (block['flux_err'] / block['flux'].std()).values
).reset_index().rename(columns={0: 'seq'})
det = groups.apply(
    lambda block: block['detected'].astype(bool).values
).reset_index().rename(columns={0: 'seq'})
times_list = times.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
flux_list = flux.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
err_list = err.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
det_list = det.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
