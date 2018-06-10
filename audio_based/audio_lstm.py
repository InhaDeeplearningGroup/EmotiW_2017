__author__ = "kdhht5022@gmail.com"
from keras.regularizers import l2
from sklearn.cross_validation import train_test_split
from six.moves import xrange
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import MaxPooling1D, MaxPooling2D, AveragePooling2D, MaxPooling3D
from keras.layers import Conv1D, Conv2D, Conv3D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D, ConvRecurrent2D
from keras.engine import Input, Model

from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import json
import keras
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras import backend as K

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.utils.data_utils import get_file

from keras.layers.wrappers import TimeDistributed, Bidirectional
import numpy as np

from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import SimpleRNN, GRU

import warnings
from keras.utils import layer_utils



if __name__ == "__main__":

    from util import Util as util
    X_data_name_1 = '/your/path/train/x_train.npz'
    y_data_name_1 = '/your/path/train/y_train.npz'
    X_data_name_2 = '/your/path/test/x_test.npz'
    y_data_name_2 = '/your/path/test/y_test.npz'
    X_train, y_train = util.load_from_npz(X_data_name_1), util.load_from_npz(y_data_name_1)
    X_test, y_test = util.load_from_npz(X_data_name_2), util.load_from_npz(y_data_name_2)

    def normalize(X_train, X_test):
        mean = np.mean(X_train,axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test
    
    # data normalize
    X_train, X_test = normalize(X_train, X_test)
    
    X_train = X_train.reshape(1464,1,577)
    X_test = X_test.reshape(383,1,577)

    
    # one-hot
    from keras.utils import np_utils
    nb_classes = 7
    y_train = np_utils.to_categorical(y_train, nb_classes).astype(np.float32)
    y_test = np_utils.to_categorical(y_test, nb_classes).astype(np.float32)

    from keras.layers.merge import concatenate, add
    def audio_clstm(inputs):
        x = GRU(577, return_sequences=True)(inputs)
        x = Dropout(0.8)(x)
        # x = concatenate( [x1, x2] )
        x = GRU(577, return_sequences=True)(x)
        x = Dropout(0.8)(x)
        x = GRU(577, return_sequences=True)(x)
        x = Dropout(0.8)(x)
        x = GRU(577)(x)
        x = Dropout(0.8)(x)
        x = Dense(7, activation='softmax')(x)
        
        return x
    
    # Case 1: CLSTM
    # from keras.layers.merge import concatenate, add
    inputs = Input(shape=(1,577))

    out = audio_clstm(inputs)
    model_clstm = Model(inputs=[inputs], outputs=[out])
    model_clstm.summary()

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model_clstm.compile(optimizer='adam', 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])
    
    mc = keras.callbacks.ModelCheckpoint('/your/path/checkpoint-clstm/model_clstm_weights.{epoch:02d}-{loss:.2f}-{val_acc:.2f}.h5', monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
    model_clstm.fit(X_train_2nd, y_train_2nd, batch_size=4, validation_data=(X_test, y_test), 
                    shuffle=True, epochs=50, callbacks=[mc])
    
    open('/your/path/checkpoint-clstm/model_audio_clstm.yaml', 'w').write(model_clstm.to_yaml())
    model_clstm.load_weights('/your/path/checkpoint-clstm/model_clstm_weights.07-1.72-0.35.h5')
    proba_clstm = model_clstm.predict_on_batch(X_test)
    
    # Case 2: Bidirectional LSTM model
    model_Bilstm = Sequential()
    model_Bilstm.add(LSTM(577, return_sequences=True,
                   input_shape=(1,577)))
    model_Bilstm.add(Dropout(0.8))
    model_Bilstm.add(Bidirectional(LSTM(577, return_sequences=True)))
    model_Bilstm.add(Dropout(0.8))
    model_Bilstm.add(Bidirectional(LSTM(577)))
    model_Bilstm.add(Dropout(0.8))
    model_Bilstm.add(Dense(7, activation='softmax'))
    model_Bilstm.summary()
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model_Bilstm.compile(optimizer=sgd, 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])
    
    mc = keras.callbacks.ModelCheckpoint('/your/path/checkpoint-Bilstm/model_Bilstm_weights.{epoch:02d}-{loss:.2f}-{val_acc:.2f}.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    model_Bilstm.fit(X_train_2nd, y_train_2nd, batch_size=4, validation_data=(X_test, y_test), 
                    shuffle=True, epochs=50, callbacks=[mc])
    
    # Case 3: LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(577, return_sequences=True,
                   input_shape=(1,577)))
    model_lstm.add(Dropout(0.8))
    model_lstm.add(LSTM(577, return_sequences=True))
    model_lstm.add(Dropout(0.8))
    model_lstm.add(LSTM(577))
    model_lstm.add(Dropout(0.8))
    model_lstm.add(Dense(7, activation='softmax'))
    model_lstm.summary()
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model_lstm.compile(optimizer=sgd, 
                       loss='categorical_crossentropy', 
                       metrics=['accuracy'])
    
    mc = keras.callbacks.ModelCheckpoint('/your/path/checkpoint-lstm/model_lstm_weights.{epoch:02d}-{loss:.2f}-{val_acc:.2f}.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    model_lstm.fit(X_train, y_train, batch_size=4, validation_data=(X_test, y_test), 
                   shuffle=True, epochs=200, callbacks=[mc])

    
    # Case 3: LSTM
    model_lstm = Sequential()
    model_lstm.add(LSTM(577, return_sequences=True,
                   input_shape=(1,577)))
    model_lstm.add(Dropout(0.8))
    model_lstm.add(LSTM(577, return_sequences=True))
    model_lstm.add(Dropout(0.8))
    model_lstm.add(LSTM(577))
    model_lstm.add(Dropout(0.8))
    model_lstm.add(Dense(7, activation='softmax'))
    model_lstm.summary()
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model_lstm.compile(optimizer=sgd, 
                       loss='categorical_crossentropy', 
                       metrics=['accuracy'])
    
    mc = keras.callbacks.ModelCheckpoint('/home/kdh/바탕화면/dfd/Concate_Network/checkpoint-lstm/model_lstm_weights.{epoch:02d}-{loss:.2f}.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
    model_lstm.fit(X_train_2nd, y_train_2nd, batch_size=4, validation_data=(X_test, y_test), 
                   shuffle=True, epochs=50, callbacks=[mc])
