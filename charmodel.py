import training_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, Add
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
人类作者 48018
自动摘要 31034
机器作者 31163
机器翻译 36206
"""
def convert_y(y):
    yc = [np.argmax(yi) for yi in y]
    return np.array(yc)

class CharModel(object):
    #+conv 0.978
    #不加0.982
    #去除dense 0.980; 去除Batch norm 0.982
    def __init__(self, C = 4, V = 40000, MAX_LEN= 600, embed_matrix = None,
                 name='charmodel.h5', PE= False):
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
                              trainable=False)
            # embed2 = Embedding(embed_matrix.shape[0],
            #                   embed_matrix.shape[1],
            #                   weights=[embed_matrix], trainable=True)
            embed2 = Embedding(embed_matrix.shape[0],
                               32, trainable=True)
            x = embed1(input)
            x2 = embed2(input)
            #concat: 0.982 add: 0.981
            # dynamic embed dim: 32 0.980
            x = Concatenate()([x, x2])
            #先使用1x1的Conv降低维数
            # x = Conv1D(32, 1, activation='relu', padding='same')(x)
        if self.PE:
            e_input = Input(shape=(MAX_LEN,), dtype='int32', name='PE_in')
            ex = Embedding(self.MAX_LEN, 128,
                           name='PE')(e_input)
            x = Concatenate()([x, ex])
        kss = [2, 3, 4, 5]
        hs = []
        for ks in kss:
            h = Conv1D(128, ks, activation='relu', padding='same')(x)
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



def train_model():
    print('load data')
    import data_utils, training_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    print('loading embed ...')
    vocab_dict = data_utils.pickle_load(conf.char_dict)
    embed_matrix = data_utils.load_embedding(vocab_dict,
                                             'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                             dump_path='data/char_embed.pkl')
    print('load embed done.')
    y = to_categorical(data_dict['y'])
    x_tn, y_tn, x_ts, y_ts = training_utils.split(data_dict['x'], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = CharModel(embed_matrix=embed_matrix)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

def train_model_pe():
    print('load data')
    import data_utils, training_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    print('loading embed ...')
    vocab_dict = data_utils.pickle_load(conf.char_dict)
    embed_matrix = data_utils.load_embedding(vocab_dict,
                                             'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                             dump_path='data/char_embed.pkl')
    print('load embed done.')
    y = to_categorical(data_dict['y'])

    xe = [[i for i in range(600)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    x_tn, y_tn, x_ts, y_ts = training_utils.split([data_dict['x'], xe] , y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = CharModel(embed_matrix=embed_matrix, name='charmodel_PE.h5', PE=True)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)


def train_model_oe_pe():
    """
    使用自己训练的Embedding,使用Position embedding
    :return:
    """
    print('load data')
    import data_utils, training_utils
    conf = data_utils.TrainConfigure()
    data_dict = data_utils.pickle_load(conf.char_file)
    print('loading embed ...')
    vocab_dict = data_utils.pickle_load(conf.char_dict)
    embed_matrix = data_utils.load_our_embedding(vocab_dict)
    print('load embed done.')
    y = to_categorical(data_dict['y'])

    xe = [[i for i in range(600)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    x_tn, y_tn, x_ts, y_ts = training_utils.split([data_dict['x'], xe] , y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = CharModel(embed_matrix=embed_matrix, name='charmodel_PE_OE.h5', PE=True)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

if __name__ == '__main__':
    SEED = 88
    np.random.seed(SEED)
    random.seed(SEED)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'pe':
        train_model_pe( )
    elif len(sys.argv) > 1 and sys.argv[1] == 'oepe':
        train_model_oe_pe( )
    else:
        train_model()