"""
hierarchical pooling

"""

from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, GRU, Bidirectional, Add, TimeDistributed, AveragePooling1D, MaxPool1D
from keras.layers import Dropout
from attention import AttLayer
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adagrad
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import random
from data_utils import TrainConfigure, ValidConfigure
"""
人类作者 48018
自动摘要 31034
机器作者 31163
机器翻译 36206
"""

def convert_y(y):
    yc = [np.argmax(yi) for yi in y]
    return np.array(yc)

class HpoolingModel(object):
    def __init__(self, C = 4, V = 40000, MAX_LEN= 600, name='termhpoolingmodel.h5', embed_matrix = None):
        self.name = name
        input = Input(shape=(MAX_LEN,),dtype='int32')
        #RNN支持mask
        if embed_matrix is None:
            x = Embedding(V, 32, mask_zero=False)(input)
        else:
            embed1 = Embedding(embed_matrix.shape[0],
                               embed_matrix.shape[1],
                               weights=[embed_matrix],
                               trainable=False, mask_zero=False)
            embed2 = Embedding(embed_matrix.shape[0],
                               embed_matrix.shape[1],
                               weights=[embed_matrix],
                               trainable=True, mask_zero=False)
            x = embed1(input)
            x2 = embed2(input)
            x2 = Dropout(0.05)(x2)
            x = Concatenate()([x, x2])
            #加conv使用activation 不使用activation
            x = TimeDistributed(Dense(64, activation='relu'))(x)
        #0.954 0.957 pool3 0.954 #max+avg 0.963 dropout=0.1 0.962
        h1 = AveragePooling1D( pool_size=2, strides=None )(x)
        z1 = GlobalMaxPool1D()(h1)
        z2 = GlobalAveragePooling1D()(h1)
        z = Concatenate()([z1, z2])
        # z = Dense(128, activation='relu')(z)
        # z = BatchNormalization()(z)
        # z = Dropout(0.3)(z)
        z = Dense(C, activation='softmax')(z)
        model = Model(input, z)
        opt = Adagrad(lr=0.005)
        model.compile(opt, 'categorical_crossentropy', metrics=['acc'])
        self.model = model

    def train(self, x, y, x_val, y_val, x_ts, y_ts):
        early_stop = EarlyStopping(min_delta=0.01, patience=2)
        save_path = self.name
        save_best = ModelCheckpoint( save_path, save_best_only=True)
        self.model.fit(x, y, validation_data=[x_val, y_val], batch_size=512,
                       epochs=20, callbacks=[early_stop, save_best])
        metric = self.model.evaluate( x_ts, y_ts )
        print(metric)
        self.model.load_weights( save_path )
        metric = self.model.evaluate( x_ts, y_ts, batch_size=512 )
        print(metric)
        y_pred = self.model.predict(x_ts, batch_size=512)

        cnf_matrix = confusion_matrix(convert_y(y_ts), convert_y(y_pred) )
        print(cnf_matrix)


if __name__ == '__main__':
    SEED = 88
    np.random.seed(SEED)
    random.seed(SEED)
    import sys
    tn_conf = TrainConfigure()
    if len(sys.argv)>1 and sys.argv[1] == 'char':
        print('define char model')
        print('load data')
        import data_utils, training_utils
        data_dict = data_utils.pickle_load(tn_conf.char_file)
        y = to_categorical(data_dict['y'])
        char_vocab_dict = data_utils.pickle_load(tn_conf.char_dict)
        char_embed_matrix = data_utils.load_embedding(char_vocab_dict,
                                                      'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                      dump_path='data/char_embed.pkl')
        x_tn, y_tn, x_ts, y_ts = training_utils.split(data_dict['x'], y, shuffle=False)
        x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
        print('train')
        model = HpoolingModel(MAX_LEN=600, name='charhpoolingmodel.h5', embed_matrix=char_embed_matrix)
        model.train( x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    else:
        print('define term model')
        print('load data')
        import data_utils, training_utils

        data_dict = data_utils.pickle_load(tn_conf.char_file)
        y = to_categorical(data_dict['y'])
        xterm = data_utils.pickle_load(tn_conf.term_file)
        term_vocab_dict = data_utils.pickle_load(tn_conf.term_dict)
        term_embed_matrix = data_utils.load_embedding(term_vocab_dict,
                                                      'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                      dump_path='data/term_embed.pkl')

        print('load embed done.')
        x_tn, y_tn, x_ts, y_ts = training_utils.split(xterm, y, shuffle=False)
        x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
        print('train')

        model = HpoolingModel(MAX_LEN=300, embed_matrix=term_embed_matrix)
        model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)