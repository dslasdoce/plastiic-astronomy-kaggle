from keras.utils.vis_utils import plot_model
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.models import load_model
import dataproc as dproc
from keras import regularizers
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
import seaborn as sns
#import matplotlib.pyplot as plt   
import pandas as pd 

target_map, label_features, all_classes, all_class_weights \
    = dproc.getDataParameters()

train_meta, test_meta_data = dproc.getMetaData()
train_full = pd.read_csv('input/training_set.csv')
train_full, train_features = dproc.getFullData(ts_data=train_full,
                                               meta_data=train_meta)

train_mean = train_full.mean(axis=0)
train_full.fillna(train_mean, inplace=True)
# this is the size of our encoded representations
encoding_dim = 64

# this is our input placeholder
input_layer = Input(shape=(len(train_features),))
# "encoded" is the encoded representation of the input
encoded = Dense(256, activation='relu')(input_layer)
encoded = Dropout(0.5)(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dropout(0.5)(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dropout(0.5)(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(128, activation='relu')(encoded)
decoded = Dropout(0.5)(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dropout(0.5)(decoded)
decoded = Dense(len(train_features), activation='relu')(decoded)

from keras.optimizers import Adam
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
           amsgrad=False)
dau_list = []
def getFolds(ser_target=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get folds
    folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=13)
#    idx = np.arange(df.shape[0])
    fold_idx = []
    for train_idx, val_idx in folds.split(X=ser_target, y=ser_target):
        fold_idx.append([train_idx, val_idx])

    return fold_idx

folds = getFolds(train_full['target_id'])

oof_reconstructed = np.zeros((train_full.shape[0], len(train_features)))
for i, (train_idx, cv_idx) in enumerate(folds):
    X_train = train_full[train_features].iloc[train_idx]
#    Y_train = train_full['target_id'].iloc[train_idx]
    X_cv = train_full[train_features].iloc[cv_idx]
#    Y_cv = train_full['target_id'].iloc[cv_idx]
    
    #for classification
#    Y_train = onehot.transform(Y_train.values.reshape(-1, 1)).toarray()
#    Y_cv = onehot.transform(Y_cv.values.reshape(-1, 1)).toarray()
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input_layer, decoded)
    plot_model(autoencoder, show_shapes=True, to_file='model_multi.png')
    autoencoder.compile(optimizer=opt, loss='mse')
    modelfile = "auto_enc.h5"
    model_checkpoint = ModelCheckpoint(modelfile, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(patience=20)
    autoencoder.fit(X_train, X_train,
                    epochs=200,
                    batch_size=256,
                    validation_data=(X_train, X_train),
                    callbacks=[model_checkpoint, early_stop],
                    verbose=True)
    autoencoder = load_model(modelfile)
    dau_list.append(autoencoder)
    oof_reconstructed[cv_idx] = autoencoder.predict(X_cv)

train_data = train_full[train_features]
dist = np.log10(np.linalg.norm(train_data-oof_reconstructed, axis=1) + 1)
sns.distplot(dist)
desc = pd.DataFrame(columns=['error', 'z-score'])
desc['error'] = dist
desc['z-score'] =  (dist - dist.mean())/dist.std()

import gc
import time
train_mean = train_full.mean(axis=0)
del train_full, train_idx
gc.collect()
start = time.time()
chunks = 5000000q
chunk_last = pd.DataFrame() 
test_row_num = 453653104 
total_steps = int(np.ceil(test_row_num/chunks))

#del train_full

#load all autoencoder distance predictions
preds_dau = pd.read_csv('predictions_dau.csv')
#calculae z-score
preds_dau['z-score'] \
    =  (preds_dau['L2DistLog'] - (preds_dau['L2DistLog'].mean()))\
       /preds_dau['L2DistLog'].std()
sns.distplot(preds_dau['L2DistLog'])

#set histogram outliers to 1 / class_99
preds_all = pd.read_csv('sfd_predictions_xgb_naive99.csv')
preds_all.loc[preds_all['object_id']\
              .isin(preds_dau.loc[preds_dau['z-score']>2.5, 'object_id']),
              'class_99'] = 1
              
preds_all.to_csv('predictions_comb_naive_dau.csv', index=False)

#
preds_nn = pd.read_csv('output/ld-nn_predictions_nn_scgal.csv')
preds_lgbm = pd.read_csv('output/gb_predictions_comb_scgal.csv')
preds_blend = preds_nn
preds_blend[label_features] = 0.4*preds_blend[label_features] + 0.6*preds_lgbm[label_features]

preds_blend.to_csv('output/blend.csv', index=False)