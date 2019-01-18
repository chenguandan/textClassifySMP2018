import training_utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import hybridmodelbase
from hybridmodelbase import HybridModelBase
"""
xn+1 = ( sigmoid(conv(xn))*conv(xn)+xn ) * sqrt(0.5)
"""

class HybridGatedConvTopicModel(HybridModelBase):

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
            hs.append( h )
        return Concatenate()(hs)

    def __init__(self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='hybridgatedconvtopicmodel.h5', train_embed=False,
                 train_top=True):
        super(HybridGatedConvTopicModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                 name=name, train_embed=train_embed,
                 train_top=train_top)

if __name__ == '__main__':
    main = hybridmodelbase.BaseMain('hybridgatedconvtopicmodel', HybridGatedConvTopicModel)
    main.main()
