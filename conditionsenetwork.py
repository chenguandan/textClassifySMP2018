import training_utils
import data_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import RepeatVector, Flatten, Bidirectional, GRU, Add, Activation, MaxPool1D, Reshape, multiply
from attention import AttLayer
from keras.regularizers import l2
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import random
import conditionmodelbase
from conditionmodelbase import ConditionModelBase
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
class ConditionSENetwork(ConditionModelBase):
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
                 name='conditionsemodel.h5', train_embed=False,
                 train_top=True):
        super(ConditionSENetwork, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix,
                                                 term_embed_matrix=term_embed_matrix,
                                                 name=name, train_embed=train_embed,
                                                 train_top=train_top)


if __name__ == '__main__':
    main = conditionmodelbase.BaseMain('conditionsemodel', ConditionSENetwork)
    main.main()
