import training_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, Add, Multiply
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, SGD, Adagrad
from attention import AttLayer
import numpy as np
import random
"""
xn+1 = ( sigmoid(conv(xn))*conv(xn)+xn ) * sqrt(0.5)
"""
def convert_y(y):
    yc = [np.argmax(yi) for yi in y]
    return np.array(yc)

class GatedConvModel(object):
    def __init__(self, C = 4, V = 40000, MAX_LEN= 600, embed_matrix = None,
                 name='gatedconvmodel.h5', PE= False, train_embed=False):
        self.MAX_LEN = MAX_LEN
        self.PE = PE
        self.name = name
        input = Input(shape=(MAX_LEN,),dtype='int32')

        #CNN不支持mask，即 mask_zero=True
        if embed_matrix is None:
            x = Embedding(V, 32)(input)
        else:
            embed1 = Embedding(embed_matrix.shape[0],
                              embed_matrix.shape[1],
                              weights=[embed_matrix],
                              trainable=train_embed)
            x = embed1(input)
        if self.PE:
            e_input = Input(shape=(MAX_LEN,), dtype='int32', name='PE_in')
            ex = Embedding(self.MAX_LEN, 32,
                           name='PE')(e_input)
            x = Concatenate()([x, ex])
        kss = [2, 3, 4, 5]
        hs = []
        for ks in kss:
            g = Conv1D(128, ks, padding='same', activation='sigmoid')(x)
            xo = Conv1D(128, ks, padding='same')(x)
            h = Multiply()([g, xo])
            # h = GlobalMaxPool1D()(h)
            h1 = GlobalMaxPool1D()(h)
            h2 = GlobalAveragePooling1D()(h)
            h = Concatenate()([h1, h2])
            hs.append( h )
        hs = Concatenate()(hs)
        # hs = BatchNormalization()(hs)
        z = Dense(128, activation='relu')(hs)
        # z = BatchNormalization()(z)
        z = Dense(C, activation='softmax')(z)
        if self.PE:
            model = Model([input, e_input], z)
        else:
            model = Model(input, z)
        opt = Adagrad(lr=0.005)
        model.compile(opt, 'categorical_crossentropy', metrics=['acc'])
        self.model = model

    def train(self, x, y, x_val, y_val, x_ts, y_ts):
        early_stop = EarlyStopping(min_delta=0.01, patience=2)
        save_path = self.name
        save_best = ModelCheckpoint( save_path, save_best_only=True)
        self.model.fit(x, y, validation_data=[x_val, y_val], batch_size=128,
                       epochs=20, callbacks=[early_stop, save_best])
        metric = self.model.evaluate( x_ts, y_ts )
        print(metric)
        self.model.load_weights( save_path )
        metric = self.model.evaluate( x_ts, y_ts, batch_size=512 )
        print(metric)
        y_pred = self.model.predict(x_ts, batch_size=512)

        cnf_matrix = confusion_matrix(convert_y(y_ts), convert_y(y_pred) )
        print(cnf_matrix)

    def load_weights(self, name = None):
        if name is None:
            save_path = self.name
        else:
            save_path = name
        self.model.load_weights(save_path)

    def predict(self, x):
        y_pred = self.model.predict(x, batch_size=512)
        return y_pred



def train_model( use_char = False):
    print('load data')
    import data_utils, training_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    print('loading embed ...')
    if use_char:
        vocab_dict = data_utils.pickle_load(conf.char_dict)
        embed_matrix = data_utils.load_embedding(vocab_dict,
                                                 'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                 dump_path='data/char_embed.pkl')

        MAX_LEN = 600
        name = 'gatedconvmodel.h5'
        x = data_dict['x']
    else:
        x = data_utils.pickle_load(conf.term_file)
        vocab_dict = data_utils.pickle_load(conf.term_dict)
        embed_matrix = data_utils.load_embedding(vocab_dict,
                                                 'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                 dump_path='data/term_embed.pkl')
        MAX_LEN = 300
        name = 'gatedconvmodel_term.h5'
    print('load embed done.')
    y = to_categorical(data_dict['y'])
    x_tn, y_tn, x_ts, y_ts = training_utils.split(x, y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = GatedConvModel(embed_matrix=embed_matrix, MAX_LEN=MAX_LEN, name=name)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = GatedConvModel(embed_matrix=embed_matrix, train_embed=True, MAX_LEN=MAX_LEN, name=name)
    model.load_weights( )
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_pe( use_char=False ):
    print('load data')
    import data_utils, training_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    print('loading embed ...')
    if use_char:
        vocab_dict = data_utils.pickle_load(conf.char_dict)
        embed_matrix = data_utils.load_embedding(vocab_dict,
                                                 'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                 dump_path='data/char_embed.pkl')

        MAX_LEN = 600
        name = 'gatedconvmodel_PE.h5'
        x = data_dict['x']
    else:
        x = data_utils.pickle_load(conf.term_file)
        vocab_dict = data_utils.pickle_load(conf.term_dict)
        embed_matrix = data_utils.load_embedding(vocab_dict,
                                                 'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                 dump_path='data/term_embed.pkl')
        MAX_LEN = 300
        name = 'gatedconvmodel_term_PE.h5'
    print('load embed done.')
    y = to_categorical(data_dict['y'])

    xe = [[i for i in range(MAX_LEN)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe] , y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = GatedConvModel(embed_matrix=embed_matrix, MAX_LEN=MAX_LEN, name=name, PE=True)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    del model
    model = GatedConvModel(embed_matrix=embed_matrix, MAX_LEN=MAX_LEN, name=name, PE=True, train_embed=True)
    model.load_weights()
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

if __name__ == '__main__':
    SEED = 88
    np.random.seed(SEED)
    random.seed(SEED)
    import sys
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