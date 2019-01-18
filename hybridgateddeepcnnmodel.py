import training_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, Add, Multiply, RepeatVector, Activation
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

class HybridGatedDeepCNNModel(HybridModelBase):

    def feed_forward(self, inputs, train_top):
        kernel_size = 3
        dim = 128
        x = Conv1D(dim, kernel_size, padding='same', activation='sigmoid', trainable=train_top)(inputs)
        xo = Conv1D(dim, kernel_size, padding='same', trainable=train_top)(inputs)
        inputs = Multiply()([x, xo])
        for li in range(3):
            x = Conv1D(dim, kernel_size, padding='same', activation='sigmoid', trainable=train_top)(inputs)
            xo = Conv1D(dim, kernel_size, padding='same', trainable=train_top)(inputs)
            x = Multiply()([x, xo])
            inputs = Add()([x, inputs])
            inputs = BatchNormalization()(inputs)
            inputs = Activation('relu')(inputs)
        h = inputs
        # h = GlobalMaxPool1D()(h)
        h1 = GlobalMaxPool1D()(h)
        h2 = GlobalAveragePooling1D()(h)
        h = Concatenate()([h1, h2])
        return h

    def __init__(self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='hybridgateddeepcnnmodel.h5', train_embed=False,
                 train_top=True):
        super(HybridGatedDeepCNNModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                 name=name, train_embed=train_embed,
                 train_top=train_top)



def train_model( ):
    print('load data')
    import data_utils, training_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    print('loading embed ...')
    vocab_dict = data_utils.pickle_load(conf.char_dict)
    char_embed_matrix = data_utils.load_embedding(vocab_dict,
                                             'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                             dump_path='data/char_embed.pkl')

    MAX_LEN = 600
    x = data_dict['x']
    xterm = data_utils.pickle_load(conf.term_file)
    term_vocab_dict = data_utils.pickle_load(conf.term_dict)
    term_embed_matrix = data_utils.load_embedding(term_vocab_dict,
                                             'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                             dump_path='data/term_embed.pkl')
    MAX_LEN_TERM = 300
    name = 'hybridgateddeepcnnmodel.h5'
    print('load embed done.')
    y = to_categorical(data_dict['y'])
    xt = data_utils.pickle_load('data/lda_vec.pkl')

    xfeat = data_utils.pickle_load(conf.feat_file)
    # normalization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(xfeat)
    data_utils.pickle_dump(scaler, conf.feat_norm)
    xfeat = scaler.transform(xfeat)

    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xterm, xfeat, xt], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = HybridGatedDeepCNNModel(char_embed_matrix=char_embed_matrix,
                                    term_embed_matrix = term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8,
                                    name= name)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = HybridGatedDeepCNNModel(char_embed_matrix=char_embed_matrix,
                                    term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8,
                                    name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights( )
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


if __name__ == '__main__':
    main = hybridmodelbase.BaseMain('hybridgateddeepcnnmodel', HybridGatedDeepCNNModel)
    main.main()