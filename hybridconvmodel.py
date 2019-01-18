import training_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, Add, Multiply, RepeatVector
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, SGD, Adagrad
from attention import AttLayer
import numpy as np
import random
from time_embedding import TimeEmbedding
from keras.regularizers import l1
import hybridmodelbase
from hybridmodelbase import HybridModelBase


class HybridConvModel(HybridModelBase):

    def feed_forward(self, inputs, train_top):
        kss = [2, 3, 4, 5]
        hs = []
        for ks in kss:
            h = Conv1D(128, ks, activation='relu', padding='same', trainable=train_top)(inputs)
            # h = GlobalMaxPool1D()(h)
            h1 = GlobalMaxPool1D()(h)
            h2 = GlobalAveragePooling1D()(h)
            h = Concatenate()([h1, h2])
            hs.append(h)
        hs = Concatenate()(hs)
        return hs

    def __init__(self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='hybridconvmodel.h5', train_embed=False,
                 train_top=True):
        super(HybridConvModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                 name=name, train_embed=train_embed,
                 train_top=train_top)


if __name__ == '__main__':
    main = hybridmodelbase.BaseMain('hybridconvmodel', HybridConvModel)
    main.main()