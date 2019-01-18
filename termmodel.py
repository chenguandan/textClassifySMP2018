import training_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from training_utils import convert_y
import random

class TermModel(object):
    def __init__(self, C = 4, V = 40000, MAX_LEN_TERM = 300,
                 embed_matrix=None, PE=False, name='termmodel.h5'):
        self.name = name
        kss = [2, 3, 4, 5]
        input_term = Input(shape=(MAX_LEN_TERM,), dtype='int32')
        # xterm = Embedding(V, 32)(input_term)
        if embed_matrix is None:
            xterm = Embedding(V, 32)(input_term)
        else:
            embed1 = Embedding(embed_matrix.shape[0],
                              embed_matrix.shape[1],
                              weights=[embed_matrix],
                              trainable=False)
            embed2 = Embedding(embed_matrix.shape[0],
                              embed_matrix.shape[1],
                              weights=[embed_matrix], trainable=True)
            xterm = embed1(input_term)
            xterm2 = embed2(input_term)
            xterm = Concatenate()([xterm, xterm2])
        if PE:
            e_input = Input(shape=(MAX_LEN_TERM,), dtype='int32', name='PE_in')
            ex = Embedding(MAX_LEN_TERM, 32,
                           name='PE')(e_input)
            xterm = Concatenate()([xterm, ex])
        hsterm = []
        for ks in kss:
            h = Conv1D(64, ks, activation='relu', padding='same')(xterm)
            h = GlobalMaxPool1D()(h)
            hsterm.append(h)
        hsterm = Concatenate()(hsterm)
        # hsterm = Dense(64, activation='relu')(hsterm)
        # hs = BatchNormalization()(hs)
        z = Dense(128, activation='sigmoid')(hsterm)
        # z = BatchNormalization()(z)
        z = Dense(C, activation='softmax')(z)
        if PE:
            model = Model([input_term, e_input], z)
        else:
            model = Model(input_term, z)
        model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
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


def train_model():
    print('load data')
    import data_utils, training_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    y = to_categorical(data_dict['y'])
    x = data_dict['x']
    xterm = data_utils.pickle_load(conf.term_file)
    print('loading embed ...')
    vocab_dict = data_utils.pickle_load(conf.term_dict)
    embed_matrix = data_utils.load_embedding(vocab_dict,
                                             'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                             dump_path='data/term_embed.pkl')
    print('load embed done.')
    x_tn, y_tn, x_ts, y_ts = training_utils.split(xterm, y, shuffle=True)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=True)
    print('train')
    print('define model')
    model = TermModel(embed_matrix=embed_matrix)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_pe():
    print('load data')
    import data_utils, training_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    y = to_categorical(data_dict['y'])
    x = data_dict['x']
    xterm = data_utils.pickle_load(conf.term_file)
    print('loading embed ...')
    vocab_dict = data_utils.pickle_load(conf.term_dict)
    embed_matrix = data_utils.load_embedding(vocab_dict,
                                             'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                             dump_path='data/term_embed.pkl')
    print('load embed done.')

    xe = [[i for i in range(300)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    x_tn, y_tn, x_ts, y_ts = training_utils.split([xterm, xe] , y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = TermModel(embed_matrix=embed_matrix, name='termmodel_PE.h5', PE=True)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


if __name__ == '__main__':
    SEED = 88
    np.random.seed(SEED)
    random.seed(SEED)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'pe':
        train_model_pe( )
    else:
        train_model( )