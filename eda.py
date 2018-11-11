#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 23:41:32 2018

@author: dslasdoce
"""
import pandas as pd
import cesium.featurize as featurize
train = pd.read_csv('training_set.csv')
train_meta = pd.read_csv('training_set_metadata.csv')
pb_mapping = {0:'u', 1:'g', 2:'r', 3:'i', 4:'z', 5:'y'}
target_class = 52
target_object_ids = train_meta.loc[train_meta.target==target_class, 'object_id']
target_df = train.loc[train.object_id.isin(target_object_ids)]

obj = 10757
df_main = train.loc[(train.object_id==obj)].sort_values('mjd').reset_index(drop=True)

groups = df_main.groupby('passband')
t_list = groups.apply(lambda gr: gr['mjd'].values).tolist()
flx_list = groups.apply(lambda gr: gr['flux'].values).tolist()
flxer_list = groups.apply(lambda gr: gr['flux_err'].values).tolist()
feats = featurize.featurize_time_series(times=t_list, values=flx_list, errors=flxer_list,
                                      features_to_use=['freq1_freq'],
                                                scheduler=None)
feats.columns = feats.columns.droplevel(1)
frq = feats['freq1_freq'][0]
#frq = len(df_main)/6/frq
df_main['mjd2'] = (df_main['mjd']*frq) % 1

import seaborn as sns
#sns.catplot(x='mjd', y='flux', data=df_main, hue='passband', palette='Set1')
sns.catplot(x='mjd2', y='flux', data=df_main, hue='passband', palette='Set1')