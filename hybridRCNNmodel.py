import training_utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, Add, Multiply, RepeatVector, Bidirectional, GRU
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

class HybridRCNNModel(HybridModelBase):

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
                 name='hybridrcnnmodel.h5', train_embed=False,
                 train_top=True):
        super(HybridRCNNModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                 name=name, train_embed=train_embed,
                 train_top=train_top)

class HybridAttMain(hybridmodelbase.BaseMain):

    def get_model_conf(self):
        model_conf = hybridmodelbase.ModelConfigure()
        model_conf.batch_size = 512
        return model_conf

if __name__ == '__main__':
    main = HybridAttMain('hybridrcnnmodel', HybridRCNNModel)
    main.main()