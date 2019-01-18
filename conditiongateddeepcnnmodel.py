"""
usage: python conditionconvmodel.py pe
"""
import training_utils
import data_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import RepeatVector, Flatten, Bidirectional, GRU, Add, Activation, MaxPool1D, Multiply
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

class ConditionGatedDeepCNNModel(ConditionModelBase):
    def feed_forward(self, inputs, train_top):
        x = Conv1D(128, 2, padding='same', activation='sigmoid', trainable=train_top)(inputs)
        xo = Conv1D(128, 2, padding='same', trainable=train_top)(inputs)
        inputs = Multiply()([x, xo])
        for li in range(3):
            x = Conv1D(128, 2, padding='same', activation='sigmoid', trainable=train_top)(inputs)
            xo = Conv1D(128, 2, padding='same', activation='relu', trainable=train_top)(inputs)
            x = Multiply()([x, xo])
            inputs = Add()([x, inputs])
            # inputs = BatchNormalization()(inputs)
            # inputs = Activation('relu')(inputs)
        h = inputs
        # h = GlobalMaxPool1D()(h)
        h1 = GlobalMaxPool1D()(h)
        h2 = GlobalAveragePooling1D()(h)
        h = Concatenate()([h1, h2])
        return h

    def __init__(self,conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='conditiongateddeepcnnmodel.h5', train_embed=False,
                 train_top=True):
        super(ConditionGatedDeepCNNModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix,
                                                  term_embed_matrix=term_embed_matrix,
                                                  name=name, train_embed=train_embed,
                                                  train_top=train_top)

if __name__ == '__main__':
    main = conditionmodelbase.BaseMain('conditiongateddeepcnnmodel', ConditionGatedDeepCNNModel)
    main.main()