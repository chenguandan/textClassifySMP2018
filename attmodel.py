import training_utils
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, GRU, Bidirectional, Add, TimeDistributed
from attention import AttLayer
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adagrad, Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
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

class AttModel(object):
    def __init__(self, C = 4, V = 40000, MAX_LEN= 600, embed_matrix = None, PE= False,
                 name='attmodel.h5'):
        self.MAX_LEN = MAX_LEN
        self.name = name
        input = Input(shape=(MAX_LEN,),dtype='int32')
        #RNN支持mask
        if embed_matrix is None:
            x = Embedding(V, 32, mask_zero=True)(input)
        else:
            embed1 = Embedding(embed_matrix.shape[0],
                               embed_matrix.shape[1],
                               weights=[embed_matrix],
                               trainable=False, mask_zero=True)
            embed2 = Embedding(embed_matrix.shape[0],
                               embed_matrix.shape[1],
                               weights=[embed_matrix],
                               trainable=True, mask_zero=True)
            x = embed1(input)
            x2 = embed2(input)
            x = Concatenate()([x, x2])
            #先使用1x1的conv降低维度
            #不加conv:0.980 加：0.983
            # x = TimeDistributed(Dense(64))(x)
        if PE:
            e_input = Input(shape=(MAX_LEN,), dtype='int32', name='PE_in')
            ex = Embedding(self.MAX_LEN, 32,
                           name='PE')(e_input)
            x = Concatenate()([x, ex])
        h1 = Bidirectional(GRU(64, return_sequences=True))(x)
        h2 = Bidirectional(GRU(64, return_sequences=True))(h1)
        #加入add: 0.983 不加：
        h = Add( )([h1, h2])
        z = AttLayer()(h)
        #不使用attention
        # h1 = Bidirectional(GRU(64))(x)
        # h2 = Bidirectional(GRU(64))(h1)
        # # 加入add: 0.983 不加：
        # z = Add()([h1, h2])
        z = Dense(128, activation='relu')(z)
        # z = BatchNormalization()(z)
        z = Dense(C, activation='softmax')(z)
        if PE:
            model = Model([input, e_input], z)
        else:
            model = Model(input, z)
        opt = Adam()#Adagrad(lr=0.005)
        model.compile(opt, 'categorical_crossentropy', metrics=['acc'])
        self.model = model

    def train(self, x, y, x_val, y_val, x_ts, y_ts):
        early_stop = EarlyStopping(min_delta=0.001, patience=2)
        save_path = self.name
        lr_scheduler = LearningRateScheduler(training_utils.scheduler([(1, 0.005), (3, 0.0003), (5, 0.0002), (8, 0.0001)]))
        save_best = ModelCheckpoint( save_path, save_best_only=True)
        self.model.fit(x, y, validation_data=[x_val, y_val], batch_size=512,
                       epochs=20, callbacks=[early_stop, save_best, lr_scheduler])
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

if __name__ == '__main__':
    import sys
    tn_conf = TrainConfigure()
    if len(sys.argv)>1 and sys.argv[1] == 'char':
        if len(sys.argv) > 2 and sys.argv[2] == 'pe':
            print('define char model with position embedding')
            print('load data')
            import data_utils, training_utils

            data_dict = data_utils.pickle_load(tn_conf.char_file)
            y = to_categorical(data_dict['y'])
            char_vocab_dict = data_utils.pickle_load(tn_conf.char_dict)
            char_embed_matrix = data_utils.load_embedding(char_vocab_dict,
                                                          'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                          dump_path='data/char_embed.pkl')
            xe = [[i for i in range(600)] for _ in range(y.shape[0])]
            xe = np.array(xe)
            x_tn, y_tn, x_ts, y_ts = training_utils.split([data_dict['x'], xe], y, shuffle=False)
            x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
            print('train')
            model = AttModel(MAX_LEN=600, name='charattmodel_PE.h5', embed_matrix=char_embed_matrix, PE=True)
            model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
        else:
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
            model = AttModel(MAX_LEN=600, name='charattmodel.h5', embed_matrix=char_embed_matrix)
            model.train( x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    else:
        if len(sys.argv) > 2 and sys.argv[2] == 'pe':
            print('define term model with position embedding')
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
            xe = [[i for i in range(300)] for _ in range(y.shape[0])]
            xe = np.array(xe)
            x_tn, y_tn, x_ts, y_ts = training_utils.split([xterm, xe], y, shuffle=False)
            x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
            print('train')
            model = AttModel(MAX_LEN=300, embed_matrix=term_embed_matrix, PE=True, name='attmodel_PE.h5')
            model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
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

            model = AttModel(MAX_LEN=300, embed_matrix=term_embed_matrix)
            model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)