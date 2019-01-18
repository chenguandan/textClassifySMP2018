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

def convert_y(y):
    yc = [np.argmax(yi) for yi in y]
    return np.array(yc)


class ModelConfigure(object):
    def __init__(self):
        self.C = 4
        self.V = 40000
        self.MAX_LEN = 600
        self.MAX_LEN_TERM = 300
        self.NUM_FEAT = 8
        self.PE = False
        self.CPE = False
        self.use_tfidf = False
        self.lr = 0.001
        self.batch_size = 256

class ConditionModelBase(object):
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

    def __init__(self, conf,
                 char_embed_matrix=None, term_embed_matrix=None,
                 name='conditionconvmodel.h5', train_embed=False,
                 train_top=True):
        self.C = conf.C
        self.MAX_LEN = conf.MAX_LEN
        self.PE = conf.PE
        self.batch_size = conf.batch_size
        self.name = name
        # char
        input = Input(shape=(self.MAX_LEN,), dtype='int32')
        label_in = Input(shape=(self.C,),dtype='float32')
        cond_x = Dense(32)(label_in)
        topic_in = Input(shape=(20,), dtype='float32')
        if char_embed_matrix is None:
            x = Embedding(conf.V, 32)(input)
        else:
            embed1 = Embedding(char_embed_matrix.shape[0],
                               char_embed_matrix.shape[1],
                               weights=[char_embed_matrix],
                               trainable=train_embed)
            x = embed1(input)
            xt_repeat = RepeatVector(conf.MAX_LEN)(topic_in)
            repeat_cond_x = RepeatVector(conf.MAX_LEN)(cond_x)
            x = Concatenate()([x, xt_repeat, repeat_cond_x])
        if self.PE:
            e_input = Input(shape=(conf.MAX_LEN,), dtype='int32', name='PE_in')
            ex = Embedding(conf.MAX_LEN, 32,
                           name='PE')(e_input)
            x = Concatenate()([x, ex])
        hs_char = self.feed_forward(x, train_top)

        input_term = Input(shape=(conf.MAX_LEN_TERM,), dtype='int32')
        if term_embed_matrix is None:
            xterm = Embedding(conf.V, 32)(input_term)
        else:
            embed1 = Embedding(term_embed_matrix.shape[0],
                               term_embed_matrix.shape[1],
                               weights=[term_embed_matrix],
                               trainable=train_embed)
            xterm = embed1(input_term)
            xt_repeat = RepeatVector(conf.MAX_LEN_TERM)(topic_in)
            cond_x_repeat = RepeatVector(conf.MAX_LEN_TERM)(cond_x)
            xterm = Concatenate()([xterm, xt_repeat, cond_x_repeat])
            # xterm = Dense(64, activation='relu')(xterm)
        if self.PE:
            eterm_input = Input(shape=(conf.MAX_LEN_TERM,), dtype='int32', name='PE_term_in')
            ex_term = Embedding(conf.MAX_LEN_TERM, 32,
                                name='PEterm')(eterm_input)
            xterm = Concatenate()([xterm, ex_term])
        hs_term = self.feed_forward(xterm, train_top)

        input_feat = Input(shape=(conf.NUM_FEAT,), dtype='float32')
        hfeat = Dense(8, activation='relu', trainable=train_top)(input_feat)
        #TODO 加入label_in, 不加入0.991
        hs = Concatenate()([hs_char, hs_term, hfeat, topic_in, label_in])
        # hs = BatchNormalization()(hs)
        z = Dense(128, activation='relu', trainable=train_top)(hs)
        # z = BatchNormalization()(z)
        z = Dense(1, activation='sigmoid', trainable=train_top)(z)
        if self.PE:
            model = Model([input, e_input, input_term, eterm_input, input_feat, topic_in, label_in], z)
        else:
            model = Model([input, input_term, input_feat, topic_in, label_in], z)
        # opt = Adagrad(lr=lr)
        opt = Adam(lr=conf.lr)
        model.compile(opt, 'binary_crossentropy', metrics=['acc'])
        self.model = model

    def train(self, x, y, x_val, y_val, x_ts, y_ts):
        early_stop = EarlyStopping(patience=2)
        save_path = self.name
        save_best = ModelCheckpoint( save_path, save_best_only=True)
        self.model.fit(x, y, validation_data=[x_val, y_val], batch_size=self.batch_size,
                       epochs=20, callbacks=[early_stop, save_best])
        self.evaluate(x_ts, y_ts)
        self.model.load_weights( save_path )
        self.evaluate(x_ts, y_ts)

    def train_exp(self,x_tn, y_tn, x_val, y_val, x_ts, y_ts):
        x_tn, y_tn = self.gen_train(x_tn, y_tn)
        x_val, y_val = self.gen_train(x_val, y_val)
        self.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

    def evaluate(self, x, y):
        x_cond, _ = self.gen_train(x, y)
        print(x_cond[0].shape)
        print(x_cond[1].shape)
        y_hat = self.model.predict(x_cond, batch_size=self.batch_size)
        C = y.shape[1]
        y_real = []
        for i in range(y.shape[0]):
            yi = []
            for c in range(C):
                yi.append( y_hat[i*C+c])
            ind = np.argmax( yi )
            y_real.append(ind)
        np.array(y_real)
        acc = accuracy_score(convert_y(y), y_real)
        print(acc)
        cnf_matrix = confusion_matrix(convert_y(y), y_real)
        print(cnf_matrix)


    def gen_train(self, x, y=None):
        #(x, one-hot label) -> {0,1}
        xmap = []
        xcond = []
        ymap = []
        C = y.shape[1]
        if isinstance(x, list):
            xmap = [list() for _ in range(len(x) )]
            for i in range(x[0].shape[0]):
                for c in range(C):
                    label = y[i][c]
                    for xmapi in range(len(x)):
                        xmap[xmapi].append(x[xmapi][i])
                    xcond.append([1.0 if yindex == c else 0.0 for yindex in range(C)])
                    if label > 0.5:
                        ymap.append(1.0)
                    else:
                        ymap.append(0.0)
            for xmapi in range(len(x)):
                xmap[xmapi] = np.asarray( xmap[xmapi] )
            xcond = np.asarray( xcond )
            ymap = np.array(ymap)
            xmap.append(xcond)
            return xmap, ymap
        else:
            for i in range(x.shape[0]):
                for c in range(C):
                    label = y[i][c]
                    xmap.append(x[i])
                    xcond.append([1.0 if yindex == c else 0.0 for yindex in range(C)])
                    if label > 0.5:
                        ymap.append(1.0)
                    else:
                        ymap.append(0.0)
            xmap = np.asarray( xmap )
            xcond = np.asarray( xcond )
            ymap = np.array(ymap)
            return [xmap, xcond], ymap

    def gen_val(self, x):
        #(x, one-hot label) -> {0,1}
        C = self.C
        if isinstance(x, list):
            xmap = [list() for _ in range(len(x) )]
            xcond = []
            for i in range(x[0].shape[0]):
                for c in range(C):
                    for xmapi in range(len(x)):
                        xmap[xmapi].append(x[xmapi][i])
                    xcond.append([1.0 if yindex == c else 0.0 for yindex in range(C)])
            for xmapi in range(len(x)):
                xmap[xmapi] = np.asarray( xmap[xmapi] )
            xcond = np.asarray( xcond )
            xmap.append(xcond)
            return xmap
        else:
            xmap = []
            xcond = []
            for i in range(x.shape[0]):
                for c in range(C):
                    xmap.append(x[i])
                    xcond.append([1.0 if yindex == c else 0.0 for yindex in range(C)])
            xmap = np.asarray( xmap )
            xcond = np.asarray( xcond )
            return [xmap, xcond]

    def load_weights(self, name=None):
        if name is None:
            save_path = self.name
        else:
            save_path = name
        self.model.load_weights(save_path)

    def predict(self, x):
        x_cond = self.gen_val( x )
        y_hat = self.model.predict(x_cond, batch_size=self.batch_size)
        C = self.C
        y_real = []
        N = len(x)
        if isinstance(x, list):
            N = x[0].shape[0]
        for i in range(N):
            yi = []
            for c in range(C):
                yi.append(y_hat[i * C + c])
            y_real.append( yi )
        np.array(y_real)
        return y_real


def train_model( model_conf, name, ModelClass):
    print(name)
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
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix,
                               name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix,
                               name=name, train_embed=True, train_top=False)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

def train_model_pe( model_conf, name, ModelClass ):
    print(name)
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

    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xt], y, split_ratio=0.95, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, split_ratio=0.95, shuffle=False)
    print('train')
    print('define model')
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix,
                               name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix,
                               name=name, train_embed=True, train_top=False)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_oe( model_conf, name, ModelClass, char_embed_file, term_embed_file ):
    print(name)
    import data_utils, training_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    print('loading embed ...')
    vocab_dict = data_utils.pickle_load(conf.char_dict)
    # char_embed_matrix = data_utils.load_embedding(vocab_dict,
    #                                               'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
    #                                               dump_path='data/char_embed.pkl')

    MAX_LEN = 600
    x = data_dict['x']
    xterm = data_utils.pickle_load(conf.term_file)
    term_vocab_dict = data_utils.pickle_load(conf.term_dict)

    char_embed_matrix = data_utils.load_our_embedding(vocab_dict, model_file = char_embed_file,
                                                   dump_path = 'data/our_char_embed_{}.pkl'.format(char_embed_file[5:]))

    term_embed_matrix = data_utils.load_our_embedding(term_vocab_dict, model_file=term_embed_file,
                                                   dump_path='data/our_term_embed_{}.pkl'.format(term_embed_file[5:]))
    MAX_LEN_TERM = 300
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

    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xt], y, split_ratio=0.95, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, split_ratio=0.95, shuffle=False)
    print('train')
    print('define model')
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix,
                               name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix,
                               name=name, train_embed=True, train_top=False)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

def train_model_ftoe( model_conf, name, ModelClass,char_embed_file = None, term_embed_file=None ):
    print(name)
    import data_utils, training_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    print('loading embed ...')
    vocab_dict = data_utils.pickle_load(conf.char_dict)
    # char_embed_matrix = data_utils.load_embedding(vocab_dict,
    #                                               'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
    #                                               dump_path='data/char_embed.pkl')

    MAX_LEN = 600
    x = data_dict['x']
    xterm = data_utils.pickle_load(conf.term_file)
    term_vocab_dict = data_utils.pickle_load(conf.term_dict)
    char_embed_matrix = data_utils.load_embedding(vocab_dict, char_embed_file,
                                                  dump_path='data/{}.pkl'.format(char_embed_file[5:]))
    term_embed_matrix = data_utils.load_embedding(term_vocab_dict, term_embed_file,
                                                  dump_path='data/{}.pkl'.format(term_embed_file[5:]))
    MAX_LEN_TERM = 300
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

    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xt], y, split_ratio=0.95, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, split_ratio=0.95, shuffle=False)
    print('train')
    print('define model')
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix,
                               name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix,
                               name=name, train_embed=True, train_top=False)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

def train_model_n100( model_conf, name, ModelClass  ):
    print(name)
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
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                              term_embed_matrix=term_embed_matrix,
                              name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                              term_embed_matrix=term_embed_matrix,
                              name=name, train_embed=True, train_top=False)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_n200( model_conf, name, ModelClass ):
    print(name)
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
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                              term_embed_matrix=term_embed_matrix,
                              name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                              term_embed_matrix=term_embed_matrix,
                              name=name, train_embed=True, train_top=False)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_cv( model_conf, cv_index, cv_num, model_name, ModelClass  ):
    print(model_name)
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
    name = model_name.format(cv_index)
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
    model = ModelClass(model_conf,char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix,
                               name=name)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = ModelClass(model_conf, char_embed_matrix=char_embed_matrix,
                               term_embed_matrix=term_embed_matrix,
                               name=name, train_embed=True, train_top=False)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

class BaseMain(object):
    def __init__(self, model_name_prefix, ModelClass):
        self.model_name_prefix = model_name_prefix
        self.ModelClass = ModelClass

    def get_model_conf(self):
        model_conf = ModelConfigure()
        return model_conf

    def main(self):
        SEED = 88
        np.random.seed(SEED)
        random.seed(SEED)
        model_conf = self.get_model_conf()
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == 'pe':
            model_conf.PE = True
            self.train_model_pe(model_conf)
        elif len(sys.argv) > 1 and sys.argv[1] == 'cpe':
            print('model with cosine position embedding')
            model_conf.PE = True
            model_conf.CPE = True
            self.train_model_pe(model_conf)
        elif len(sys.argv) > 1 and sys.argv[1] == 'tfidf':
            print('model with tfidf feature')
            model_conf.PE = True
            model_conf.CPE = False
            model_conf.use_tfidf = True
            self.train_model_tfidf(model_conf)
        elif len(sys.argv) > 1 and sys.argv[1] == 'oe':
            model_conf.PE = True
            print('oe')
            labelindex = None
            for windows in [3, 5, 8]:
                for sg in [0, 1]:
                    self.train_model_oe(model_conf, labelindex, windows, sg, 8, 1)
            windows = 10
            sg = 1
            for labelindex in range(4):
                self.train_model_oe(model_conf, labelindex, 8, 1, windows, sg)
            labelindex = None
            for windows in [3, 5, 8]:
                for sg in [0, 1]:
                    self.train_model_oe(model_conf, labelindex, 8, 1, windows, sg)
            windows = 10
            sg = 1
            for labelindex in range(4):
                self.train_model_oe(model_conf, labelindex, 8, 1, windows, sg)
        elif len(sys.argv) > 1 and sys.argv[1] == 'ftoe':
            model_conf.PE = True
            print('ftoe')
            windows1 = 8
            for windows2 in [3, 5, 8]:
                self.train_model_ftoe(model_conf, windows1, windows2)
            windows2 = 8
            for windows1 in [3, 5, 8]:
                self.train_model_ftoe(model_conf, windows1, windows2)

        elif len(sys.argv) > 1 and sys.argv[1] == 'n200':
            print('model max len 200')
            self.train_model_n200(model_conf)
        elif len(sys.argv) > 1 and sys.argv[1] == 'n100':
            print('model max len 100')
            self.train_model_n100(model_conf)
        elif len(sys.argv) > 1 and sys.argv[1] == 'cv':
            print('model cv')
            cv_num = 5
            for cv_index in range(cv_num):
                print('cv index', cv_index, '/', cv_num)
                self.train_model_cv(model_conf, cv_index, cv_num)
        else:
            self.train_model(model_conf)

    def train_model_ftoe(self, model_conf, windows1, windows2):
        char_embed_file = 'data/char_ft_embed_{}_{}.model.vec'.format(windows1, 1)
        term_embed_file = 'data/term_ft_embed_{}_{}.model.vec'.format(windows2, 1)
        name = '{}_ftoe_w{}_w{}.h5'.format(self.model_name_prefix, windows1,windows2)
        train_model_ftoe(model_conf, name, self.ModelClass, char_embed_file=char_embed_file, term_embed_file=term_embed_file)

    def train_model_oe(self, model_conf, labelindex, windows1, sg1, windows2, sg2):
        if labelindex is None:
            char_embed_file = 'data/char_embed_{}_{}.model'.format(windows1, sg1)
        else:
            char_embed_file = 'data/char_embed_{}_{}_{}.model'.format(labelindex, windows1, sg1)
        if labelindex is None:
            term_embed_file = 'data/term_embed_{}_{}.model'.format(windows2, sg2)
        else:
            term_embed_file = 'data/term_embed_{}_{}_{}.model'.format(labelindex, windows2, sg2)
        if labelindex is None:
            name = '{}_oe_w{}_sg{}_w{}_sg{}.h5'.format(self.model_name_prefix, windows1, sg1, windows2, sg2)
        else:
            name = '{}_oe_l{}_w{}_sg{}_w{}_sg{}.h5'.format(self.model_name_prefix, labelindex, windows1, sg1,windows2, sg2)

        train_model_oe(model_conf, name, self.ModelClass, char_embed_file, term_embed_file)

    def train_model(self, model_conf ):
        name = '{}.h5'.format(self.model_name_prefix)
        train_model(model_conf, name, self.ModelClass)

    def train_model_pe(self, model_conf):
        if model_conf.CPE:
            name = '{}_CPE.h5'.format(self.model_name_prefix)
        else:
            name = '{}_PE.h5'.format(self.model_name_prefix)
        train_model_pe(model_conf, name, self.ModelClass)

    def train_model_tfidf(self, model_conf):
        name = '{}_tfidf.h5'.format(self.model_name_prefix)
        train_model_pe(model_conf, name, self.ModelClass)

    def train_model_n200(self, model_conf):
        name = '{}_n200.h5'.format(self.model_name_prefix)
        train_model_n200(model_conf, name, self.ModelClass)

    def train_model_n100(self, model_conf):
        name = '{}_n100.h5'.format(self.model_name_prefix)
        train_model_n100(model_conf, name, self.ModelClass)

    def train_model_cv(self, model_conf, cv_index, cv_num):
        name = '{}_cv{}.h5'.format(self.model_name_prefix, cv_index)
        train_model_cv(model_conf, cv_index, cv_num, name, self.ModelClass)

if __name__ == '__main__':
    main = BaseMain('hybridconvmodel', ConditionModelBase)
    main.main()