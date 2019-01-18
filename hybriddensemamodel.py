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
Dense connected network
part from:
Densely Connected CNN with Multi-scale Feature Attention for Text Classification
"""

class HybridDenseMAModel(HybridModelBase):

    def feed_forward(self, x, train_top):
        #6层网络
        xs_till_now = []
        xs_till_now.append(x)
        filter_size = 250
        x = Dense(filter_size)(x)
        x1 = Activation('relu')(x)
        x1 = Conv1D(filter_size, 3, padding='same', trainable=train_top)(x1)
        xs_till_now.append(x1)
        x = Concatenate()(xs_till_now)
        # x2 = Conv1D(filter_size, 3, padding='same', trainable=train_top)(x1)
        # x = Add()([xmap, x2])
        # x = MaxPool1D(pool_size=3, strides=2)(x)

        for _ in range(5):
            x1 = Activation('relu')(x)
            x1 = Conv1D(filter_size, 3, padding='same', trainable=train_top)(x1)
            xs_till_now.append(x1)
            x = Concatenate()(xs_till_now)
            # x2 = Conv1D(filter_size, 3, padding='same',trainable=train_top)(x1)
            # x = Add()([x, x2])
            # x = MaxPool1D(pool_size=3, strides=2)(x)
        hs = []
        for xi in xs_till_now:
            hs.append( AttLayer()(xi) )
        x = GlobalMaxPool1D()(x)
        hs.append( x )
        return Concatenate()(hs)

    def __init__(self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='hybriddensemodelma.h5', train_embed=False,
                 train_top=True):
        super(HybridDenseMAModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                 name=name, train_embed=train_embed,
                 train_top=train_top)




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
    name = 'hybriddensemodelma_PEOE.h5'
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
    model = HybridDenseMAModel(char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                               name=name, char_embeds=char_embeds, term_embeds = term_embeds, lr=0.0004)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = HybridDenseMAModel(char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                               name=name, train_embed=True, train_top=False, lr=0.001, char_embeds=char_embeds, term_embeds = term_embeds)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


class HybridDenseMain(hybridmodelbase.BaseMain):

    def get_model_conf(self):
        model_conf = hybridmodelbase.ModelConfigure()
        model_conf.batch_size = 128
        return model_conf
if __name__ == '__main__':
    main = HybridDenseMain('hybriddensemodelma', HybridDenseMAModel)
    main.main()