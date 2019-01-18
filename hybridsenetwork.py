import training_utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import keras
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, Add, Multiply, RepeatVector, Activation, MaxPool1D
from keras.layers import GlobalAveragePooling1D, Reshape, multiply, Dropout
from keras.regularizers import l2
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, SGD, Adagrad
from attention import AttLayer
import numpy as np
import random

import hybridmodelbase
from hybridmodelbase import HybridModelBase

"""
part from:
squeeze and exciting network
"""

def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    filters = init._keras_shape[-1]
    se_shape = (1, filters)
    se = GlobalAveragePooling1D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = multiply([init, se])
    return x


class HybridSEModel(HybridModelBase):

    def feed_forward2(self, x, train_top):
        #6层网络
        filter_size = 256
        x = Dense(filter_size)(x)
        xs_till_now = []
        xs_till_now.append(x)
        x1 = Activation('relu')(x)
        x1 = Conv1D(filter_size, 3, padding='same', trainable=train_top)(x1)
        x1 = squeeze_excite_block( x1 )
        xs_till_now.append(x1)
        x = Concatenate()(xs_till_now)
        # x2 = Conv1D(filter_size, 3, padding='same', trainable=train_top)(x1)
        # x = Add()([xmap, x2])
        # x = MaxPool1D(pool_size=3, strides=2)(x)

        for _ in range(5):
            # x = BatchNormalization()(x)
            x1 = Activation('relu')(x)
            x1 = Conv1D(filter_size, 3, padding='same', trainable=train_top)(x1)
            x1 = squeeze_excite_block( x1 )
            xs_till_now.append(x1)
            x = Concatenate()(xs_till_now)
            # x2 = Conv1D(filter_size, 3, padding='same',trainable=train_top)(x1)
            # x = Add()([x, x2])
            # x = MaxPool1D(pool_size=3, strides=2)(x)
        x = GlobalMaxPool1D()(x)
        return x


    def feed_forward(self, x, train_top):
        filter_sz = 250
        weight_decay = 5e-4
        x1 = Activation('relu')(x)
        x1 = Conv1D(filter_sz, 3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay),trainable=train_top)(x1)
        x2 = Conv1D(filter_sz, 3, padding='same', kernel_regularizer=l2(weight_decay), trainable=train_top)(x1)
        xmap = Dense(filter_sz)(x)
        x = Add()([xmap, x2])
        x = MaxPool1D(pool_size=3, strides=2)(x)
        x = squeeze_excite_block(x)
        for _ in range(5):
            x = BatchNormalization()(x)
            x1 = Activation('relu')(x)
            x1 = Conv1D(filter_sz, 3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay), trainable=train_top)(x1)
            x2 = Conv1D(filter_sz, 3, padding='same', kernel_regularizer=l2(weight_decay), trainable=train_top)(x1)
            x = Add()([x, x2])
            x = MaxPool1D(pool_size=3, strides=2)(x)
            x = squeeze_excite_block(x)
        x = GlobalMaxPool1D()(x)
        return x

    def __init__(self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='hybridsemodel.h5', train_embed=False,
                 train_top=True):
        super(HybridSEModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                 name=name, train_embed=train_embed,
                 train_top=train_top)



if __name__ == '__main__':
    main = hybridmodelbase.BaseMain('hybridsemodel', HybridSEModel)
    main.main()