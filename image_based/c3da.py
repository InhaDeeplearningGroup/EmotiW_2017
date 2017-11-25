__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-
from six.moves import xrange
from keras.layers import Dense, Activation
from keras.layers import GlobalMaxPooling3D
from keras.layers.convolutional import Conv3D
from keras.engine import Input, Model
import scipy as sp
from keras.utils import np_utils
import keras
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import numpy as np
np.random.seed(2 ** 10)

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
 
mc = keras.callbacks.ModelCheckpoint('/path/to/here/weights.{epoch:02d}-{loss:.2f}.h5', 
                                     monitor='loss', verbose=1, save_best_only=False, 
                                     save_weights_only=True, mode='auto', period=1)

nb_classes = 7
img_rows, img_cols = 224, 224
img_channels = 1

data_augmentation = True

depth = 28
k = 10
dropout_probability = 0

weight_decay = 0.0005
use_bias = True
weight_init="he_normal"
nb_classes = 7

y_train_frame = np_utils.to_categorical(y_train_frame, nb_classes).astype(np.float32)
y_test_frame = np_utils.to_categorical(y_test_frame, nb_classes).astype(np.float32)

# train
X_train /= 255.
frame_tr = len(X_train) // 40
X_train_c3d = np.zeros(shape=(frame_tr, 40, 224, 224, 1))
for i in xrange(len(X_train_c3d)):
    etr=[0 + i*40, 1 + i*40, 2 + i*40, 3 + i*40, 4 + i*40, 5 + i*40, 6 + i*40, 7 + i*40, 8 + i*40, 9 + i*40, 10 + i*40,
         11 + i*40, 12 + i*40, 13 + i*40, 14 + i*40, 15 + i*40, 16 + i*40, 17 + i*40, 18 + i*40, 19 + i*40, 20 + i*40,
         21 + i*40, 22 + i*40, 23 + i*40, 24 + i*40, 25 + i*40, 26 + i*40, 27 + i*40, 28 + i*40, 29 + i*40, 30 + i*40, 
         31 + i*40, 32 + i*40, 33 + i*40, 34 + i*40, 35 + i*40, 36 + i*40, 37 + i*40, 38 + i*40, 39 + i*40]
    for j, w in enumerate(etr):
        X_train_c3d[i,j,:] = X_train[w,:]
X_train_c3d = X_train_c3d.astype(np.float32)

# test
X_test /= 255.
frame_te = len(X_test) // 40
X_test_c3d = np.zeros(shape=(frame_te, 40, 224, 224, 1))
for i in xrange(len(X_test_c3d)):
    etr=[0 + i*40, 1 + i*40, 2 + i*40, 3 + i*40, 4 + i*40, 5 + i*40, 6 + i*40, 7 + i*40, 8 + i*40, 9 + i*40, 10 + i*40,
         11 + i*40, 12 + i*40, 13 + i*40, 14 + i*40, 15 + i*40, 16 + i*40, 17 + i*40, 18 + i*40, 19 + i*40, 20 + i*40,
         21 + i*40, 22 + i*40, 23 + i*40, 24 + i*40, 25 + i*40, 26 + i*40, 27 + i*40, 28 + i*40, 29 + i*40, 30 + i*40, 
         31 + i*40, 32 + i*40, 33 + i*40, 34 + i*40, 35 + i*40, 36 + i*40, 37 + i*40, 38 + i*40, 39 + i*40]
    for j, w in enumerate(etr):
        X_test_c3d[i,j,:] = X_test[w,:]
X_test_c3d = X_test_c3d.astype(np.float32)


#%% Build network

logging.debug("Loading network/training configuration...")

def c3d(x):
    x = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Conv3D(648, (3, 3, 3), strides=(2, 2, 2), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    
    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same', 
               kernel_initializer='he_normal', data_format='channels_last')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    _x = Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding='same', 
               kernel_initializer='he_normal', data_format='channels_last')(x)
    
    a = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(_x)    
    a = BatchNormalization(axis=channel_axis)(a)
    a = Activation('relu')(a)
    a = GlobalMaxPooling3D(data_format='channels_last')(a)
# first auxiliary network
    out1 = Dense(7, activation='softmax')(a)
    
    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(_x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    __x = Conv3D(512, (3, 3, 3), strides=(2, 2, 2), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)

    c = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(__x)
    c = BatchNormalization(axis=channel_axis)(c)
    c = Activation('relu')(c)
    c = GlobalMaxPooling3D(data_format='channels_last')(c)
# second auxiliary network
    out2 = Dense(7, activation='softmax')(c)
    
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(__x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    ___x = Conv3D(512, (3, 3, 3), strides=(2, 2, 2), padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(x)
    
    b = Conv3D(1024, (3, 3, 3), strides=(1, 1, 1), padding='same', 
               kernel_initializer='he_normal', data_format='channels_last')(___x)
    b = BatchNormalization(axis=channel_axis)(b)
    b = Activation('relu')(b)
    b = Conv3D(1024, (3, 3, 3), strides=(1, 1, 1), padding='same', 
               kernel_initializer='he_normal', data_format='channels_last')(b)
    b = BatchNormalization(axis=channel_axis)(b)
    b = Activation('relu')(b)
    b = GlobalMaxPooling3D(data_format='channels_last')(b)

    out3 = Dense(7, activation='softmax')(b)
    
    return out1, out2, out3
 
 
# %% C3DA network

inputs = Input(shape=(40, 224, 224, 1))

out1, out2, out3 = c3d(inputs)

model_c3d = Model(inputs=inputs, outputs=[out1, out2, out3])
# model_c3d.summary()

model_c3d.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# %% Network training

print('Not using data augmentation.')
logging.debug("Running training...")
hist = model_c3d.fit(X_train_c3d, [y_train_frame, y_train_frame, y_train_frame], 
                     batch_size=32, epochs=100, verbose=1, 
                     shuffle=True, callbacks=[mc]) 


# %% Prediction    

logging.debug("Running test...")
model_c3d.load_weights('/home/kdh/Desktop/c3da/checkpoint/c3da_weights.h5')
proba_c3d = model_c3d.predict(X_test_c3d, batch_size=4, verbose=1)
y_pred = [np.argmax(prob) for prob in proba_c3d[2]]
y_true = [np.argmax(true) for true in y_test_frame]
    
count = 0
for i in range(len(y_pred)):
    if y_test[i] == y_pred[i]:
        count += 1

