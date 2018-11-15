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
train = pd.read_csv('input/training_set.csv')
train_meta = pd.read_csv('input/training_set_metadata.csv')
pb_mapping = {0:'u', 1:'g', 2:'r', 3:'i', 4:'z', 5:'y'}
target_class = 67
target_object_ids = train_meta.loc[train_meta.target==target_class, 'object_id']
target_df = train.loc[train.object_id.isin(target_object_ids)]
target_meta = train_meta.loc[train_meta.object_id.isin(target_object_ids)]

############ symmetry
def getSym(df):    
    max_mjd = df.loc[df['flux'].idxmax(), 'mjd']
    left_d = len(df.loc[df['mjd'] < max_mjd])
    right_d = len(df.loc[df['mjd'] > max_mjd])
    try:
        z = left_d/right_d
    except ZeroDivisionError:
        z = np.nan
    return pd.Series([z], index=['symmetry'])

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
        fl_lower = flmax_mag - (flmax_mag - flmin_mag)*0.8
        try:
            flmin_mjd = df_temp['mjd'].loc[df_temp['flux'] < fl_lower].iloc[0]
        except IndexError:
#            print(df_main)
            flmin_mjd = np.nan
    else:
        df_temp = df_main.iloc[:flmax_idx, :]
        flmin_mag = df_main['flux'].min()
        fl_lower = flmax_mag - (flmax_mag - flmin_mag)*0.8
        try:
            flmin_mjd = df_temp.loc[df_temp['flux'] < fl_lower]['mjd'].iloc[-1]
        except IndexError:
            flmin_mjd = np.nan
        
    return pd.Series([abs(flmax_mjd - flmin_mjd)],
                      index=['15decay'])
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

train_meta['is_gal'] = 0
train_meta.loc[train_meta['hostgal_photoz']==0, 'is_gal'] = 1
plot_meta = train_meta

plot_meta['target_l'] = train_meta['target']\
                        .map({i:'cl-'+str(i) for i in  train_meta['target'].unique()})

sns.scatterplot(x='ra', y='decl', data=plot_meta, hue='target_l',
                palette='Set1', alpha=0.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.show(block=False)

d0 = 63 * (np.pi/180)
a0 = 350 * (np.pi/180)
delta = train_meta['decl']*(np.pi/180)
alpha = train_meta['ra']*(np.pi/180)
x = np.cos(delta)*np.cos(alpha)
y = np.cos(delta)*np.sin(alpha)
x0 = np.cos(d0)*np.cos(a0)
y0 = np.cos(d0)*np.sin(a0)
train_meta['d_sky'] = np.sqrt(np.square(x-x0) + np.square(y-y0))

points = [[169*(np.pi/180), 60*(np.pi/180)]]
for i, p in enumerate(points):
    lmbda1 = p[0]
    phi1 = p[1]
    phi = train_meta['gal_b'] * (np.pi/180)
    lmbda = train_meta['gal_l'] * (np.pi/180)
    train_meta['d-p' + str(i)]\
        = 2*np.arcsin(np.power(np.power(np.sin((phi-phi1)/2), 2)\
                               + np.cos(lmbda)*np.cos(lmbda1)\
                                 * np.power(np.sin((lmbda-lmbda1)/2), 2),
                               0.5))
sns.scatterplot(x='d_sky', y='d-p0', data=plot_meta, hue='target_l',
                palette='Set1', alpha=0.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.show(block=False)
#agg_df_ts = pd.merge(agg_df_ts, agg_df_mjd, on = 'id')
# tsfresh returns a dataframe with an index name='id'
#agg_df_ts.index.rename('object_id',inplace=True)
#agg_df = pd.merge(agg_df, agg_df_ts, on='object_id')
##obj = 128895075
#obj = 2922
#df_main = train.loc[(train.object_id==obj)].sort_values('mjd').reset_index(drop=True)
#df_main = df_main.loc[df_main['passband']==5]
##df_main = target_df.groupby('object_id').apply(getSym)
#df_main = df_main.loc[(df_main['mjd'] <= flmax_mjd) & (df_main['mjd'] >= flmin_mjd)]
#
#
#
#df_15decay = target_df.groupby(['object_id','passband']).apply(getMax).reset_index()
#df_15decay = target_df.groupby(['object_id','passband']).apply(get15decay).reset_index()
##df_15decay = df_15decay.groupby('object_id').mean()
##df_15decay2 = df_15decay.loc[df_15decay['passband']==0]
##df_15decay2['15decay'] = np.log(df_15decay2['15decay'])
#aggs = {'15decay': ['min', 'max', 'mean', 'std']}
#df_15decay2 = df_15decay.groupby('object_id').agg(aggs)
#new_columns = [
#        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
#    ]
#df_15decay2.columns = new_columns
#
#sns.catplot(x='mjd', y='flux', data=df_main, hue='passband', palette='Set1')
#plt.title("Orig")
#plt.tight_layout()
#
#df_main['unix'] = ((df_main['mjd'] - 40587)*86400)
#df_main['unix'] = df_main['unix'].map(datetime.datetime.utcfromtimestamp)
#
#sns.scatterplot(x='target', y='hostgal_photoz', data=train_meta)
#
#sym_df = target_df.groupby(['object_id', 'passband']).apply(getSym).reset_index()
#import datetime
#datetime.datetime.utcfromtimestamp(df_main['unix'].iloc[2])
############ frequency
#groups = df_main.groupby('passband')
#t_list = groups.apply(lambda gr: gr['mjd'].values).tolist()
#flx_list = groups.apply(lambda gr: gr['flux'].values).tolist()
#flxer_list = groups.apply(lambda gr: gr['flux_err'].values).tolist()
#feats = featurize.featurize_time_series(times=t_list, values=flx_list, errors=flxer_list,
#                                      features_to_use=['freq1_freq'],
#                                                scheduler=None)
#feats.columns = feats.columns.droplevel(1)
#frq = feats['freq1_freq'][0]
##frq = len(df_main)/6/frq
#df_main['mjd2'] = (df_main['mjd']*frq) % 1
#

#sns.catplot(x='mjd2', y='flux', data=df_main, hue='passband', palette='Set1')
#plt.title("Phase")
#plt.tight_layout()