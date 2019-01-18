import training_utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, Add, Multiply, RepeatVector, Activation, MaxPool1D
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
Deep Pyramid Convolutional Neural Networks for Text Categorization
"""

class HybridDPCNNModel(HybridModelBase):

    def feed_forward(self, x, train_top):
        #总共15层weight layers： embed+ conv * 2 * 6 + 2* dense
        #preactivation
        x1 = Activation('relu')(x)
        x1 = Conv1D(250, 3, padding='same', activation='relu', trainable=train_top)(x1)
        x2 = Conv1D(250, 3, padding='same', trainable=train_top)(x1)
        xmap = Dense(250)(x)
        x = Add()([xmap, x2])
        x = MaxPool1D(pool_size=3, strides=2)(x)
        for _ in range(5):
            x1 = Activation('relu')(x)
            x1 = Conv1D(250, 3, padding='same', activation='relu', trainable=train_top)(x1)
            x2 = Conv1D(250, 3, padding='same',trainable=train_top)(x1)
            x = Add()([x, x2])
            x = MaxPool1D(pool_size=3, strides=2)(x)
        x = GlobalMaxPool1D()(x)
        return x

    def __init__(self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='hybriddpcnnmodel.h5', train_embed=False,
                 train_top=True):
        super(HybridDPCNNModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
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
    name = 'hybriddpcnnmodel.h5'
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
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix,
                             term_embed_matrix = term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8,
                             name= name)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix,
                             term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8,
                             name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights( )
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_pe( ):
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
    name = 'hybriddpcnnmodel_PE.h5'
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

    xe = [[i for i in range(MAX_LEN)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    xe_term = [[i for i in range(MAX_LEN_TERM)] for _ in range(y.shape[0])]
    xe_term = np.array(xe_term)

    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xt], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix,
                             term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                             name=name)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix,
                             term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                             name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_peoe( ):
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

    char_embeds = []
    char_embed_matrix_oe = data_utils.load_our_embedding(vocab_dict)
    char_embeds.append( char_embed_matrix_oe )
    for windows in [3, 5, 8]:
        sg = 1
        # for sg in [0,1]:
        embed_file = 'data/char_embed_{}_{}.model'.format(windows, sg)
        char_embed_tmp = data_utils.load_our_embedding(vocab_dict, model_file = embed_file,
                                                       dump_path = 'data/our_char_embed_{}_{}.pkl'.format(windows, sg))
        char_embeds.append(char_embed_tmp)

    term_embeds = []
    term_embed_matrix_oe = data_utils.load_our_embedding(term_vocab_dict, model_file='data/term_embed.model',
                                                         dump_path='data/our_term_embed.pkl')
    term_embeds.append( term_embed_matrix_oe )
    for windows in [3, 5, 8]:
        sg = 1
        # for sg in [0,1]:
        embed_file = 'data/term_embed_{}_{}.model'.format(windows, sg)
        term_embed_tmp = data_utils.load_our_embedding(term_vocab_dict, model_file=embed_file,
                                                       dump_path='data/our_term_embed_{}_{}.pkl'.format(windows,
                                                                                                        sg))
        term_embeds.append(term_embed_tmp)


    MAX_LEN_TERM = 300
    name = 'hybriddpcnnmodel_PEOE.h5'
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

    xe = [[i for i in range(MAX_LEN)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    xe_term = [[i for i in range(MAX_LEN_TERM)] for _ in range(y.shape[0])]
    xe_term = np.array(xe_term)

    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xt], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    #加入更多embedding模型以后，学习率要降低才能正常学习
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix,
                             term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                             name=name, char_embeds=char_embeds, term_embeds = term_embeds, lr=0.0004)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix,
                             term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                             name=name, train_embed=True, train_top=False, lr=0.001, char_embeds=char_embeds, term_embeds = term_embeds)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_n200( ):
    print('load data')
    import data_utils200 as data_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    print('loading embed ...')
    vocab_dict = data_utils.pickle_load(conf.char_dict)
    char_embed_matrix = data_utils.load_embedding(vocab_dict,
                                                  'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                  dump_path='data/char_embed.pkl')

    MAX_LEN = conf.MAX_LEN
    x = data_dict['x']
    xterm = data_utils.pickle_load(conf.term_file)
    term_vocab_dict = data_utils.pickle_load(conf.term_dict)
    term_embed_matrix = data_utils.load_embedding(term_vocab_dict,
                                                  'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                  dump_path='data/term_embed.pkl')
    MAX_LEN_TERM = conf.MAX_LEN
    name = 'hybriddpcnnmodel_n200.h5'
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

    xe = [[i for i in range(MAX_LEN)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    xe_term = [[i for i in range(MAX_LEN_TERM)] for _ in range(y.shape[0])]
    xe_term = np.array(xe_term)

    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xt], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                            MAX_LEN=MAX_LEN, MAX_LEN_TERM=MAX_LEN_TERM, NUM_FEAT=8, PE=True,
                            name=name)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                            MAX_LEN=MAX_LEN, MAX_LEN_TERM=MAX_LEN_TERM, NUM_FEAT=8, PE=True,
                            name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

def train_model_n100( ):
    print('load data')
    import data_utils100 as data_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    print('loading embed ...')
    vocab_dict = data_utils.pickle_load(conf.char_dict)
    char_embed_matrix = data_utils.load_embedding(vocab_dict,
                                                  'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                  dump_path='data/char_embed.pkl')

    MAX_LEN = 100
    x = data_dict['x']
    xterm = data_utils.pickle_load(conf.term_file)
    term_vocab_dict = data_utils.pickle_load(conf.term_dict)
    term_embed_matrix = data_utils.load_embedding(term_vocab_dict,
                                                  'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                  dump_path='data/term_embed.pkl')
    MAX_LEN_TERM = 100
    name = 'hybriddpcnnmodel_n100.h5'
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

    xe = [[i for i in range(MAX_LEN)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    xe_term = [[i for i in range(MAX_LEN_TERM)] for _ in range(y.shape[0])]
    xe_term = np.array(xe_term)

    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xt], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                            MAX_LEN=MAX_LEN, MAX_LEN_TERM=MAX_LEN_TERM, NUM_FEAT=8, PE=True,
                            name=name)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                            MAX_LEN=MAX_LEN, MAX_LEN_TERM=MAX_LEN_TERM, NUM_FEAT=8, PE=True,
                            name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
def train_model_cv( cv_index, cv_num ):
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
    name = 'hybriddpcnnmodel_cv{}.h5'.format(cv_index)
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

    xe = [[i for i in range(MAX_LEN)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    xe_term = [[i for i in range(MAX_LEN_TERM)] for _ in range(y.shape[0])]
    xe_term = np.array(xe_term)

    x_tn, y_tn, x_ts, y_ts = training_utils.split_cv([x, xe, xterm, xe_term, xfeat, xt], y, cv_index=cv_index,cv_num=cv_num)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                            MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True, name=name)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = HybridDPCNNModel(char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                            MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                            name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

if __name__ == '__main__':
    main = hybridmodelbase.BaseMain('hybriddpcnnmodel', HybridDPCNNModel)
    main.main()