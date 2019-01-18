from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import RepeatVector, Flatten
from keras.layers import BatchNormalization
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

class ConditionModel(object):
    def __init__(self, C = 4, V = 40000, MAX_LEN= 600, embed_matrix = None,
                 name='conditionmodel.h5', PE=False):
        self.name = name
        self.C = C
        input = Input(shape=(MAX_LEN,),dtype='int32')
        label_in = Input(shape=(C,),dtype='float32')
        # cond_x = Embedding(C, 32)(label_in)
        # cond_x = Flatten()(cond_x)
        #bs x d
        cond_x = Dense(32)(label_in)
        cond_x = RepeatVector(MAX_LEN)(cond_x)
        #CNN不支持mask，即 mask_zero=True
        if embed_matrix is None:
            x = Embedding(V, 32)(input)
        else:
            embed1 = Embedding(embed_matrix.shape[0],
                               embed_matrix.shape[1],
                               weights=[embed_matrix],
                               trainable=False)
            # embed2 = Embedding(embed_matrix.shape[0],
            #                   embed_matrix.shape[1],
            #                   weights=[embed_matrix], trainable=True)
            embed2 = Embedding(embed_matrix.shape[0],
                               32, trainable=True)
            x = embed1(input)
            x2 = embed2(input)
            x = Concatenate()([x, x2])
        if PE:
            e_input = Input(shape=(MAX_LEN,), dtype='int32', name='PE_in')
            ex = Embedding(MAX_LEN, 32,
                           name='PE')(e_input)
            x = Concatenate()([x, ex])
        x = Concatenate()([x, cond_x])
        kss = [2, 3, 4, 5]
        hs = []
        for ks in kss:
            h = Conv1D(64, ks, activation='relu', padding='same')(x)
            h = GlobalMaxPool1D()(h)
            hs.append( h )
        hs = Concatenate()(hs)
        z = BatchNormalization()(hs)
        z = Dense(128, activation='relu')(hs)
        # z = BatchNormalization()(z)
        z = Dense(1, activation='sigmoid')(z)
        if PE:
            model = Model([input, e_input, label_in], z)
        else:
            model = Model([input, label_in], z)
        model.compile('adam', 'binary_crossentropy', metrics=['acc'])
        self.model = model

    def train(self, x, y, x_val, y_val, x_ts, y_ts):
        early_stop = EarlyStopping(min_delta=0.01, patience=2)
        save_path = self.name
        save_best = ModelCheckpoint( save_path, save_best_only=True)
        self.model.fit(x, y, validation_data=[x_val, y_val], batch_size=128,
                       epochs=20, callbacks=[early_stop, save_best])
        self.evaluate(x_ts, y_ts)
        self.model.load_weights( save_path )
        self.evaluate(x_ts, y_ts)

    def evaluate(self, x, y):
        x_cond, _ = self.gen_train(x, y)
        print(x_cond[0].shape)
        print(x_cond[1].shape)
        y_hat = self.model.predict(x_cond)
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
        y_hat = self.model.predict(x_cond)
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


def train_model( use_char = True):
    print('define model')
    print('load data')
    import data_utils, training_utils
    tn_conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load('data.dict')
    y = to_categorical(data_dict['y'])
    if use_char:
        vocab_dict = data_utils.pickle_load(tn_conf.char_dict)
        embed_matrix = data_utils.load_embedding(vocab_dict,
                                                 'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                 dump_path='data/char_embed.pkl')
        x = data_dict['x']
        MAX_LEN = 600
        name = 'conditionmodel.h5'
    else:
        x = data_utils.pickle_load(tn_conf.term_file)
        vocab_dict = data_utils.pickle_load(tn_conf.term_dict)
        embed_matrix = data_utils.load_embedding(vocab_dict,
                                                 'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                 dump_path='data/term_embed.pkl')
        MAX_LEN = 300
        name = 'conditionmodel_term.h5'
    x_tn, y_tn, x_ts, y_ts = training_utils.split(x, y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    model = ConditionModel(embed_matrix=embed_matrix, name=name, MAX_LEN = MAX_LEN)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

def train_model_pe( use_char = True):
    print('define model')
    print('load data')
    import data_utils, training_utils
    tn_conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load('data.dict')
    y = to_categorical(data_dict['y'])
    if use_char:
        vocab_dict = data_utils.pickle_load(tn_conf.char_dict)
        embed_matrix = data_utils.load_embedding(vocab_dict,
                                                      'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                      dump_path='data/char_embed.pkl')
        x = data_dict['x']
        MAX_LEN = 600
        name = 'conditionmodel_PE.h5'
    else:
        x = data_utils.pickle_load(tn_conf.term_file)
        vocab_dict = data_utils.pickle_load(tn_conf.term_dict)
        embed_matrix = data_utils.load_embedding(vocab_dict,
                                                      'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                      dump_path='data/term_embed.pkl')
        MAX_LEN = 300
        name = 'conditionmodel_term_PE.h5'
    xe = [[i for i in range(MAX_LEN)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')

    model = ConditionModel(embed_matrix=embed_matrix, MAX_LEN=MAX_LEN, name=name, PE=True)
    x_tn, y_tn = model.gen_train(x_tn, y_tn)
    x_val, y_val = model.gen_train(x_val, y_val)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

if __name__ == '__main__':
    SEED = 88
    np.random.seed(SEED)
    random.seed(SEED)
    import sys
    """
    usage: python conditionmodel.py char pe
    python conditionmodel.py term pe
    """
    if len(sys.argv)>1 and sys.argv[1] == 'char':
        if len(sys.argv) > 2 and sys.argv[2] == 'pe':
            train_model_pe( )
        else:
            train_model( )
    else:
        if len(sys.argv) > 2 and sys.argv[2] == 'pe':
            train_model_pe( use_char=False)
        else:
            train_model( use_char=False)