"""
usage: python conditionconvmodel.py pe
"""
import training_utils
import data_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import RepeatVector, Flatten, Bidirectional, GRU, Add, Activation, MaxPool1D
from attention import AttLayer
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

class ConditionDPCNNModel(ConditionModelBase):
    def feed_forward(self, x, train_top):
        x1 = Activation('relu')(x)
        x1 = Conv1D(250, 3, padding='same', activation='relu', trainable=train_top)(x1)
        x2 = Conv1D(250, 3, padding='same', trainable=train_top)(x1)
        xmap = Dense(250)(x)
        x = Add()([xmap, x2])
        x = MaxPool1D(pool_size=3, strides=2)(x)
        for _ in range(5):
            x1 = Activation('relu')(x)
            x1 = Conv1D(250, 3, padding='same', activation='relu', trainable=train_top)(x1)
            x2 = Conv1D(250, 3, padding='same', trainable=train_top)(x1)
            x = Add()([x, x2])
            x = MaxPool1D(pool_size=3, strides=2)(x)
        x = GlobalMaxPool1D()(x)
        return x

    def __init__(self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='conditiondpcnnmodel.h5', train_embed=False,
                 train_top=True):
        super(ConditionDPCNNModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix,
                                              term_embed_matrix=term_embed_matrix,
                                              name=name, train_embed=train_embed,
                                              train_top=train_top)

class ConditionDPCNNMain(conditionmodelbase.BaseMain):

    def get_model_conf(self):
        model_conf = conditionmodelbase.ModelConfigure()
        model_conf.lr = 0.0005
        return model_conf

if __name__ == '__main__':
    main = ConditionDPCNNMain('conditiondpcnnmodel', ConditionDPCNNModel)
    main.main()
