import training_utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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

class HybridAttModel(HybridModelBase):

    def feed_forward(self, x, train_top):
        h1 = Bidirectional(GRU(128, return_sequences=True, trainable=train_top), trainable=train_top)(x)
        # h2 = Bidirectional(GRU(64, return_sequences=True, trainable=train_top), trainable=train_top)(h1)
        # h = Add()([h1, h2])
        h = h1
        z = AttLayer()(h)
        return z

    def __init__( self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='hybridattmodel.h5', train_embed=False,
                 train_top=True):
        super(HybridAttModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
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
    name = 'hybridattmodel.h5'
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
    model = HybridAttModel(char_embed_matrix=char_embed_matrix,
                           term_embed_matrix = term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8,
                           name= name)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = HybridAttModel(char_embed_matrix=char_embed_matrix,
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
    name = 'hybridattmodel_PE.h5'
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
    model = HybridAttModel(char_embed_matrix=char_embed_matrix,
                           term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                           name=name)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = HybridAttModel(char_embed_matrix=char_embed_matrix,
                           term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                           name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

class HybridAttMain(hybridmodelbase.BaseMain):

    def get_model_conf(self):
        model_conf = hybridmodelbase.ModelConfigure()
        model_conf.batch_size = 512
        return model_conf

if __name__ == '__main__':
    main = HybridAttMain('hybridattmodel', HybridAttModel)
    main.main()