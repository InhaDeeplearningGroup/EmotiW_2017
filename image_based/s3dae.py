__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-
from keras.layers import Dense, Activation 
from keras.layers.convolutional import UpSampling3D
from keras.layers import GlobalMaxPooling3D
from keras.layers.convolutional import Conv3D
from keras.engine import Input, Model
import scipy as sp
from keras import backend as K
import keras
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

from keras.utils import np_utils
import numpy as np
np.random.seed(2 ** 10)
from six.moves import xrange
import logging
logging.basicConfig(level=logging.DEBUG)

import sys
sys.stdout = sys.stderr
sys.setrecursionlimit(2 ** 20)

def load_from_npz(data_name):
    with sp.load(data_name) as f:
        values = [f['arr_%d' % i] for i in range(len(f.files))][0]
        return values

# Keras specific
if K.image_dim_ordering() == "th":
    logging.debug("image_dim_ordering = 'th'")
    channel_axis = 1
    input_shape = (1, 224, 224)
else:
    logging.debug("image_dim_ordering = 'tf'")
    channel_axis = -1
    input_shape = (224, 224, 1)


#%% Load dataset

X_data_name_1 = '/home/kdh/Desktop/AFEW/X_train.npz'
y_data_name_1 = '/home/kdh/Desktop/AFEW/y_train.npz'
X_data_name_2 = '/home/kdh/Desktop/AFEW/X_test.npz'
y_data_name_2 = '/home/kdh/Desktop/AFEW/y_test.npz'
y_data_name_3 = '/home/kdh/Desktop/AFEW/y_train_frame.npz'
y_data_name_4 = '/home/kdh/Desktop/AFEW/y_teset_frame.npz'
X_train, y_train = load_from_npz(X_data_name_1), load_from_npz(y_data_name_1)
X_test, y_test = load_from_npz(X_data_name_2), load_from_npz(y_data_name_2)
y_train_frame, y_test_frame = load_from_npz(y_data_name_3), load_from_npz(y_data_name_4)
 
X_train = X_train / 255.
X_test = X_test / 255.
 
mc = keras.callbacks.ModelCheckpoint('/path/to/here/weights.{epoch:02d}-{loss:.2f}.h5', 
                                     monitor='loss', verbose=1, save_best_only=False, 
                                     save_weights_only=True, mode='auto', period=1)
data_augmentation = True


nb_classes = 7
y_train_frame = np_utils.to_categorical(y_train_frame, nb_classes).astype(np.float32)
y_test_frame = np_utils.to_categorical(y_test_frame, nb_classes).astype(np.float32)

frame_tr = len(X_train) // 40
frame_te = len(X_test) // 40
X_train_c3d = np.zeros(shape=(frame_tr, 40, 112, 112, 1))
X_test_c3d = np.zeros(shape=(frame_te, 40, 112, 112, 1))

for i in xrange(len(X_train_c3d)):
    etr=[0 + i*40, 1 + i*40, 2 + i*40, 3 + i*40, 4 + i*40, 5 + i*40, 6 + i*40, 7 + i*40, 8 + i*40, 9 + i*40, 10 + i*40,
         11 + i*40, 12 + i*40, 13 + i*40, 14 + i*40, 15 + i*40, 16 + i*40, 17 + i*40, 18 + i*40, 19 + i*40, 20 + i*40,
         21 + i*40, 22 + i*40, 23 + i*40, 24 + i*40, 25 + i*40, 26 + i*40, 27 + i*40, 28 + i*40, 29 + i*40, 30 + i*40, 
         31 + i*40, 32 + i*40, 33 + i*40, 34 + i*40, 35 + i*40, 36 + i*40, 37 + i*40, 38 + i*40, 39 + i*40]
    for j, w in enumerate(etr):
        X_train_c3d[i,j,:] = X_train[w,:]
X_train_c3d = X_train_c3d.astype(np.float32)

for i in xrange(len(X_test_c3d)):
    etr=[0 + i*40, 1 + i*40, 2 + i*40, 3 + i*40, 4 + i*40, 5 + i*40, 6 + i*40, 7 + i*40, 8 + i*40, 9 + i*40, 10 + i*40,
         11 + i*40, 12 + i*40, 13 + i*40, 14 + i*40, 15 + i*40, 16 + i*40, 17 + i*40, 18 + i*40, 19 + i*40, 20 + i*40,
         21 + i*40, 22 + i*40, 23 + i*40, 24 + i*40, 25 + i*40, 26 + i*40, 27 + i*40, 28 + i*40, 29 + i*40, 30 + i*40, 
         31 + i*40, 32 + i*40, 33 + i*40, 34 + i*40, 35 + i*40, 36 + i*40, 37 + i*40, 38 + i*40, 39 + i*40]
    for j, w in enumerate(etr):
        X_test_c3d[i,j,:] = X_test[w,:]
X_test_c3d = X_test_c3d.astype(np.float32)


# %% Build network

logging.debug("Loading network/training configuration...")

def c3da_ae(x):
    
    # encoder
    x = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', 
               kernel_initializer='he_normal', data_format='channels_last')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same', 
               kernel_initializer='he_normal', data_format='channels_last')(x)
    
    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x_1 = Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding='same',
                 kernel_initializer='he_normal', data_format='channels_last')(x)

    x = Conv3D(256, (3, 3, 3), strides=(2, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x_1)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Conv3D(256, (3, 3, 3), strides=(2, 2, 2), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    
    
    # decoder-1
    y = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x_1)
    y = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(y)
    y = BatchNormalization(axis=channel_axis)(y)
    y = Activation('relu')(y)
    y = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(y)
    y = BatchNormalization(axis=channel_axis)(y)
    y = Activation('relu')(y)
    
    y = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(y)
    y = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(y)
    y = BatchNormalization(axis=channel_axis)(y)
    y = Activation('relu')(y)
    
    y = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(y)
    y = Conv3D(1, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(y)
    y = BatchNormalization(axis=channel_axis)(y)
    out1 = Activation('sigmoid')(y)
    
    
    # decoder-2
    out2 = GlobalMaxPooling3D(data_format='channels_last')(x)
    out2 = Dense(7, activation='softmax')(out2)
    
    return out1, out2


# %% S3DAE network

inputs = Input(input_shape)

out1, out2 = c3da_ae(inputs)

model_fi = Model(inputs=inputs, outputs=[out1, out2])
model_fi.summary()

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model_fi.compile(optimizer=sgd,
                 loss=['binary_crossentropy', 'categorical_crossentropy'], 
                 loss_weights=[0.7, 1], 
                 metrics=['accuracy'])


# %% Network training

print('Not using data augmentation.')
logging.debug("Running training...")
model_fi.fit(X_train_c3d, [X_train_c3d, y_train_frame], 
             batch_size=32, epochs=100, verbose=1,
             callbacks=[mc])


# %% Prediction

model_fi.load_weights('/home/kdh/Desktop/s3dae/checkpoint/s3dae_weights.h5')
proba_fi = model_fi.predict(X_test_c3d, batch_size=4, verbose=1)
y_pred = [np.argmax(prob) for prob in proba_fi[1]]
y_true = [np.argmax(true) for true in y_test_frame]
    
count = 0
for i in range(len(y_pred)):
    if y_test_frame[i] == y_pred[i]:
        count += 1


