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

class ConditionRCNNModel(ConditionModelBase):
    def feed_forward(self, x, train_top):
        h = Bidirectional(GRU(128, return_sequences=True, trainable=train_top), trainable=train_top)(x)
        # h2 = Bidirectional(GRU(64, return_sequences=True, trainable=train_top), trainable=train_top)(h1)
        # x = Add()([h1, h2])
        x = h
        kss = [2, 3, 4, 5]
        hs = []
        for ks in kss:
            h = Conv1D(128, ks, activation='relu', padding='same')(x)
            # h = GlobalMaxPool1D()(h)
            h1 = GlobalMaxPool1D()(h)
            h2 = GlobalAveragePooling1D()(h)
            h = Concatenate()([h1, h2])
            hs.append(h)
        hs = Concatenate()(hs)
        return hs

    def __init__(self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='conditionrcnnmodel.h5', train_embed=False,
                 train_top=True):
        super(ConditionRCNNModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix,
                                                  term_embed_matrix=term_embed_matrix,
                                                  name=name, train_embed=train_embed,
                                                  train_top=train_top)


class ConditionAttMain(conditionmodelbase.BaseMain):

    def get_model_conf(self):
        model_conf = conditionmodelbase.ModelConfigure()
        model_conf.batch_size = 512
        return model_conf

if __name__ == '__main__':
    main = ConditionAttMain('conditionrcnnmodel', ConditionRCNNModel)
    main.main()