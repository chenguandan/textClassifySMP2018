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

class ConditionGatedConvModel(ConditionModelBase):
    def feed_forward(self, x, train_top):
        kss = [2, 3, 4, 5]
        hs = []
        for ks in kss:
            g = Conv1D(128, ks, padding='same', activation='sigmoid', trainable=train_top)(x)
            xo = Conv1D(128, ks, padding='same', trainable=train_top)(x)
            h = Multiply()([g, xo])
            # h = GlobalMaxPool1D()(h)
            h1 = GlobalMaxPool1D()(h)
            h2 = GlobalAveragePooling1D()(h)
            h = Concatenate()([h1, h2])
            hs.append(h)
        return Concatenate()(hs)


    def __init__(self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='conditiongatedconvmodel.h5', train_embed=False,
                 train_top=True):
        super(ConditionGatedConvModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix,
                                                  term_embed_matrix=term_embed_matrix,
                                                  name=name, train_embed=train_embed,
                                                  train_top=train_top)

if __name__ == '__main__':
    main = conditionmodelbase.BaseMain('conditiongatedconvmodel', ConditionGatedConvModel)
    main.main()
