import training_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, Add
from keras import Model
from keras.layers import Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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


def build_block(inputs, filters, strides, projection_shortcut=None):
    #TODO projection shortcut
    shortcut = inputs
    inputs = BatchNormalization()(inputs)
    inputs = Activation('relu')(inputs)
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = Conv1D(filters, 3, strides=strides, padding='same')(inputs)
    inputs = BatchNormalization()(inputs)
    inputs = Activation('relu')(inputs)
    inputs = Conv1D(filters, 3, strides=1, padding='same')(inputs)
    return Add()([inputs, shortcut])


def block_layer(inputs, filters, blocks, strides):
    def projection_shortcut(inputs):
        return Conv1D(filters=filters, kernel_size=1, strides=strides)(inputs)
    inputs = build_block(inputs, filters, strides, projection_shortcut)
    for _ in range(1, blocks):
        inputs = build_block(inputs, filters, 1)
    return inputs

class DeepCNNModel(object):
    """
    1. 减少C_in ，即在3*1的卷积前，用1*1的卷积把input channels减少
    2个block：0.975

    """
    def __init__(self, C = 4, V = 40000, MAX_LEN= 600, embed_matrix=None,
                 name='deepcnn_model.h5', PE=False):
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
            embed2 = Embedding(embed_matrix.shape[0],
                              embed_matrix.shape[1],
                              weights=[embed_matrix], trainable=True)
            x = embed1(input)
            x2 = embed2(input)
            x = Concatenate()([x, x2])
            #先使用1x1的Conv降低维数
            # x = Conv1D(32, 1, activation='relu', padding='same')(x)
        if PE:
            e_input = Input(shape=(MAX_LEN,), dtype='int32', name='PE_in')
            ex = Embedding(MAX_LEN, 32,
                           name='PE')(e_input)
            x = Concatenate()([x, ex])
        #原始形式
        # inputs = Conv1D(64, 7, strides=2)(x)
        # inputs = block_layer(inputs, 64, blocks=3, strides=1)
        # inputs = block_layer(inputs, 128, blocks=4, strides=2)
        # inputs = block_layer(inputs, 256, blocks=6, strides=2)
        # inputs = block_layer(inputs, 512, blocks=3, strides=2)
        kss = [2, 3, 4]
        hs = []
        for ks in kss:
            h = Conv1D(32, ks, activation='relu', padding='same')(x)
            hs.append(h)
        inputs = Concatenate()(hs)
        # inputs = x
        #blocks = 2,2,2,2
        #blocks=1,...
        inputs = block_layer(inputs, 128, blocks=1, strides=1)
        inputs = block_layer(inputs, 64, blocks=1, strides=2)
        # inputs = block_layer(inputs, 32, blocks=1, strides=2)
        # inputs = block_layer(inputs, 16, blocks=1, strides=2)
        inputs = BatchNormalization()(inputs)
        inputs = Activation('relu')(inputs)
        h = GlobalMaxPool1D()(inputs)
        z = Dense(8, activation='relu')(h)
        # z = BatchNormalization()(z)
        z = Dense(C, activation='softmax')(z)
        if PE:
            model = Model([input, e_input], z)
        else:
            model = Model(input, z)
        model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
        self.model = model

    def train(self, x, y, x_val, y_val, x_ts, y_ts):
        early_stop = EarlyStopping(min_delta=0.001, patience=2)
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

    def load_weights(self, name=None):
        if name is None:
            save_path = self.name
        else:
            save_path = name
        self.model.load_weights(save_path)

    def predict(self, x):
        y_pred = self.model.predict(x, batch_size=512)
        return y_pred


def train_model(use_char = False ):
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
        name = 'deepcnn_model.h5'
        x = data_dict['x']
    else:
        x = data_utils.pickle_load(conf.term_file)
        vocab_dict = data_utils.pickle_load(conf.term_dict)
        embed_matrix = data_utils.load_embedding(vocab_dict,
                                                 'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                 dump_path='data/term_embed.pkl')
        MAX_LEN = 300
        name = 'deepcnn_model_term.h5'
    print('load embed done.')
    y = to_categorical(data_dict['y'])
    x_tn, y_tn, x_ts, y_ts = training_utils.split(x, y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = DeepCNNModel(embed_matrix=embed_matrix, MAX_LEN=MAX_LEN, name=name)
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

def train_model_pe( use_char = False ):
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
        name = 'deepcnn_model_PE.h5'
        x = data_dict['x']
    else:
        x = data_utils.pickle_load(conf.term_file)
        vocab_dict = data_utils.pickle_load(conf.term_dict)
        embed_matrix = data_utils.load_embedding(vocab_dict,
                                                 'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                 dump_path='data/term_embed.pkl')
        MAX_LEN = 300
        name = 'deepcnn_model_term_PE.h5'
    print('load embed done.')
    y = to_categorical(data_dict['y'])
    xe = [[i for i in range(MAX_LEN)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    print('define model')
    model = DeepCNNModel(embed_matrix=embed_matrix, MAX_LEN=MAX_LEN, name=name, PE=True)
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
