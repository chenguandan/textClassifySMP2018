"""
usage: python conditionconvmodel.py pe
"""
import training_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import RepeatVector, Flatten
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
def convert_y(y):
    yc = [np.argmax(yi) for yi in y]
    return np.array(yc)

class ConditionConvModel(ConditionModelBase):
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
                 name='conditionconvmodel.h5', train_embed=False,
                 train_top=True):
        super(ConditionConvModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix,
                                              term_embed_matrix=term_embed_matrix,
                                              name=name, train_embed=train_embed,
                                              train_top=train_top)


def train_model( use_char = True):
    print('train condition conv model.\nload data')
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
    name = 'conditionconvmodel.h5'
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
    model = ConditionConvModel(char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8,
                               name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ConditionConvModel(char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8,
                               name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

def train_model_pe( ):
    print('train condition conv model with PE\n load data')
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
    name = 'conditionconvmodel_PE.h5'
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
    model = ConditionConvModel(char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                               name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ConditionConvModel(char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                               name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_n100( ):
    print('train condition att model with PE\n load data')
    import data_utils100 as data_utils
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
    name = 'conditionconvmodel_n100.h5'
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
    model = ConditionConvModel(char_embed_matrix=char_embed_matrix,
                              term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, MAX_LEN_TERM=MAX_LEN_TERM, NUM_FEAT=8, PE=True,
                              name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ConditionConvModel(char_embed_matrix=char_embed_matrix,
                              term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, MAX_LEN_TERM=MAX_LEN_TERM, NUM_FEAT=8, PE=True,
                              name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_n200( ):
    print('train condition att model with PE\n load data')
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
    name = 'conditionconvmodel_n200.h5'
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
    model = ConditionConvModel(char_embed_matrix=char_embed_matrix,
                              term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, MAX_LEN_TERM=MAX_LEN_TERM, NUM_FEAT=8, PE=True,
                              name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ConditionConvModel(char_embed_matrix=char_embed_matrix,
                              term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, MAX_LEN_TERM=MAX_LEN_TERM, NUM_FEAT=8, PE=True,
                              name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_cv( cv_index, cv_num ):
    print('train condition conv model with PE\n load data')
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
    name = 'conditionconvmodel_cv{}.h5'.format(cv_index)
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

    x_tn, y_tn, x_ts, y_ts = training_utils.split_cv([x, xe, xterm, xe_term, xfeat, xt], y,cv_index=cv_index,cv_num=cv_num)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = ConditionConvModel(char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                               name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ConditionConvModel(char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix, MAX_LEN=MAX_LEN, NUM_FEAT=8, PE=True,
                               name=name, train_embed=True, train_top=False, lr=0.001)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

if __name__ == '__main__':
    main = conditionmodelbase.BaseMain('conditionconvmodel', ConditionConvModel)
    main.main()