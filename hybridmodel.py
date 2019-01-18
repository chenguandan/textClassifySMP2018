from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.optimizers import SGD, Adam, Adagrad
import numpy as np
from training_utils import convert_y
import random
from data_utils import TrainConfigure, ValidConfigure
"""
加大CNN的filter个数、dense的size有一定的收益。
word-word的embedding效果稍微差一些
人类作者 48018
自动摘要 31034
机器作者 31163
机器翻译 36206
"""



class HybridModel(object):
    def __init__(self, C = 4, V = 40000, MAX_LEN= 600, MAX_LEN_TERM = 300,NUM_FEAT= 8,
                 char_embed_matrix=None, term_embed_matrix=None, use_multi_task =False,
                 name= 'hybridmodel.h5', PE=False):
        #+bn2   0.975 +bn1  0.986
        #+bn1,max+avg pool  0.987
        #squeeze embedding (128)0.985  (64+conv64)0.983
        #去除子网络的dense 0.987   squeeze embedding+relu 0.985
        #conv 64 0.987  conv 128 0.988
        self.name = name
        self.use_multi_task = use_multi_task
        input = Input(shape=(MAX_LEN,),dtype='int32')
        #CNN不支持mask，即 mask_zero=True
        if char_embed_matrix is None:
            x = Embedding(V, 32)(input)
        else:
            embed1 = Embedding(char_embed_matrix.shape[0],
                               char_embed_matrix.shape[1],
                              weights=[char_embed_matrix],
                              trainable=False)
            embed2 = Embedding(char_embed_matrix.shape[0],
                               char_embed_matrix.shape[1],
                              weights=[char_embed_matrix], trainable=True)
            x = embed1(input)
            x2 = embed2(input)
            x = Concatenate()([x, x2])
            # x = Dense(64, activation='relu')(x)
        if PE:
            echar_input = Input(shape=(MAX_LEN,), dtype='int32', name='PE_char_in')
            ex_char = Embedding(MAX_LEN, 32,
                           name='PEchar')(echar_input)
            x = Concatenate()([x, ex_char])
        kss = [2, 3, 4, 5]
        hs = []
        for ks in kss:
            h = Conv1D(128, ks, activation='relu', padding='same')(x)
            h1 = GlobalMaxPool1D()(h)
            h2 = GlobalAveragePooling1D()(h)
            hs.append( h1 )
            hs.append( h2 )
        hs = Concatenate()(hs)
        # hs = Dense(128, activation='relu')(hs)
        if self.use_multi_task:
            y1 = Dense(C, activation='softmax',name='y1')(hs)

        input_term = Input(shape=(MAX_LEN_TERM,), dtype='int32')
        if term_embed_matrix is None:
            xterm = Embedding(V, 32)(input_term)
        else:
            embed1 = Embedding(term_embed_matrix.shape[0],
                               term_embed_matrix.shape[1],
                              weights=[term_embed_matrix],
                              trainable=False)
            embed2 = Embedding(term_embed_matrix.shape[0],
                               term_embed_matrix.shape[1],
                              weights=[term_embed_matrix], trainable=True)
            xterm = embed1(input_term)
            xterm2 = embed2(input_term)
            xterm = Concatenate()([xterm, xterm2])
            # xterm = Dense(64, activation='relu')(xterm)
        if PE:
            eterm_input = Input(shape=(MAX_LEN_TERM,), dtype='int32', name='PE_term_in')
            ex_term = Embedding(MAX_LEN_TERM, 32,
                           name='PEterm')(eterm_input)
            xterm = Concatenate()([xterm, ex_term])
        hsterm = []
        for ks in kss:
            h = Conv1D(128, ks, activation='relu', padding='same')(xterm)
            h1 = GlobalMaxPool1D()(h)
            h2 = GlobalAveragePooling1D()(h)
            hsterm.append(h1)
            hsterm.append(h2)
        hsterm = Concatenate()(hsterm)
        # hsterm = Dense(128, activation='relu')(hsterm)

        input_feat = Input(shape=(NUM_FEAT,),dtype='float32')
        hfeat = Dense(8, activation='relu')(input_feat)

        hs = Concatenate()([hs, hsterm, hfeat])

        hs = BatchNormalization()(hs)
        z = Dense(128, activation='relu')(hs)
        # z = BatchNormalization()(z)
        z = Dense(C, activation='softmax', name='y')(z)
        if PE:
            model = Model([input, input_term, input_feat, echar_input, eterm_input], z)
        else:
            model = Model([input,input_term, input_feat], z)
        opt = Adagrad(lr=0.005)
        # opt = Adam()
        model.compile(opt, 'categorical_crossentropy', metrics=['acc'])
        self.model = model
        if self.use_multi_task:
            y2 = Dense(C, activation='softmax',name='y2')(hsterm)
            y3 = Dense(C, activation='softmax',name='y3')(hfeat)
            if PE:
                self.train_model = Model([input, input_term, input_feat, echar_input, eterm_input], [z, y1, y2, y3])
            else:
                self.train_model = Model([input, input_term, input_feat], [z, y1, y2, y3])
            self.train_model.compile(opt, 'categorical_crossentropy', metrics=['acc'])

    def load_weights(self, name = None):
        if name is None:
            save_path = self.name
        else:
            save_path = name
        if self.use_multi_task:
            self.train_model.load_weights(save_path)
        else:
            self.model.load_weights(save_path)

    def train(self, x, y, x_val, y_val, x_ts, y_ts):
        early_stop = EarlyStopping(min_delta=0.01, patience=2)
        save_path = self.name
        save_best = ModelCheckpoint( save_path, save_best_only=True)
        if self.use_multi_task:
            self.train_model.fit(x, [y, y, y, y], validation_data=[x_val, [y_val, y_val, y_val, y_val]], batch_size=128,
                                 epochs=20, callbacks=[early_stop, save_best])
        else:
            self.model.fit(x, y, validation_data=[x_val, y_val], batch_size=128,
                           epochs=20, callbacks=[early_stop, save_best])

        metric = self.model.evaluate( x_ts, y_ts )
        print(metric)
        self.load_weights( )
        metric = self.model.evaluate( x_ts, y_ts, batch_size=512 )
        print(metric)
        y_pred = self.model.predict(x_ts, batch_size=512)

        cnf_matrix = confusion_matrix(convert_y(y_ts), convert_y(y_pred) )
        print(cnf_matrix)

    def test(self, x, ids, out_file):
        labels = ['人类作者', '自动摘要', '机器作者', '机器翻译']
        y_pred = self.model.predict(x, batch_size=512)
        y_pred = convert_y(y_pred)
        with open(out_file, 'w', encoding='utf-8') as fout:
            for id, yi in zip(ids, y_pred):
                label = labels[yi]
                fout.write('{},{}\n'.format(id, label))
        print('done.')

    def predict(self, x):
        y_pred = self.model.predict(x, batch_size=512)
        return y_pred

    def error_analysis(self, x_ts, y_ts, texts, start_index ):
        labels = ['人类作者', '自动摘要', '机器作者', '机器翻译']
        y_pred = self.model.predict(x_ts, batch_size=512)
        y_ts, y_pred = convert_y(y_ts), convert_y(y_pred)
        with open('error.txt','w') as fout:
            for i in range(y_ts.shape[0]):
                if y_ts[i] != y_pred[i]:
                    fout.write('*****\n{}\n正确标签：{}   分类标签：{}\n'.format(texts[start_index+i],
                                labels[y_ts[i]], labels[y_pred[i]]) )
        print('output error done.')



def train_main():
    print('load data')
    import data_utils, training_utils
    tn_conf = TrainConfigure()
    data_dict = data_utils.pickle_load(tn_conf.char_file)
    y = to_categorical(data_dict['y'])
    x = data_dict['x']
    xterm = data_utils.pickle_load(tn_conf.term_file)
    xfeat = data_utils.pickle_load(tn_conf.feat_file)
    # normalization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(xfeat)
    data_utils.pickle_dump(scaler, tn_conf.feat_norm)
    xfeat = scaler.transform(xfeat)

    print('loading embed ...')
    term_vocab_dict = data_utils.pickle_load(tn_conf.term_dict)
    term_embed_matrix = data_utils.load_embedding(term_vocab_dict,
                                                  'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                  dump_path='data/term_embed.pkl')
    # term_embed_matrix = data_utils.load_embedding(term_vocab_dict,
    #                                               'data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5',
    #                                               dump_path='data/term_embed_ww.pkl')
    char_vocab_dict = data_utils.pickle_load(tn_conf.char_dict)
    char_embed_matrix = data_utils.load_embedding(char_vocab_dict,
                                                  'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                  dump_path='data/char_embed.pkl')
    print('load embed done.')

    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xterm, xfeat], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('define model')
    model = HybridModel(char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix, NUM_FEAT=8)  # +37
    print('feat shape', xfeat.shape)
    import sys
    if len(sys.argv) <= 1 or sys.argv[1] == 'train':
        print('train')
        model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)
    if len(sys.argv) > 1 and sys.argv[1] == 'val':
        val_conf = ValidConfigure()
        data_dict = data_utils.pickle_load(val_conf.char_file)
        y = to_categorical(data_dict['y'])
        x = data_dict['x']
        ids = data_dict['id']
        xterm = data_utils.pickle_load(val_conf.term_file)
        xfeat = data_utils.pickle_load(val_conf.feat_file)
        xfeat = scaler.transform(xfeat)
        model.load_weights()
        model.test([x, xterm, xfeat], ids, val_conf.out_file)

    if len(sys.argv) > 1 and sys.argv[1] == 'error':
        start_index = y_tn.shape[0] + y_val.shape[0]
        texts = data_utils.load_all_text(tn_conf)
        model.load_weights()
        model.error_analysis(x_ts, y_ts, texts, start_index)

def train_main_pe():
    print('load data')
    import data_utils, training_utils
    tn_conf = TrainConfigure()
    data_dict = data_utils.pickle_load(tn_conf.char_file)
    y = to_categorical(data_dict['y'])
    x = data_dict['x']
    xterm = data_utils.pickle_load(tn_conf.term_file)
    xfeat = data_utils.pickle_load(tn_conf.feat_file)
    # normalization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(xfeat)
    data_utils.pickle_dump(scaler, tn_conf.feat_norm)
    xfeat = scaler.transform(xfeat)

    print('loading embed ...')
    term_vocab_dict = data_utils.pickle_load(tn_conf.term_dict)
    term_embed_matrix = data_utils.load_embedding(term_vocab_dict,
                                                  'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                  dump_path='data/term_embed.pkl')
    # term_embed_matrix = data_utils.load_embedding(term_vocab_dict,
    #                                               'data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5',
    #                                               dump_path='data/term_embed_ww.pkl')
    char_vocab_dict = data_utils.pickle_load(tn_conf.char_dict)
    char_embed_matrix = data_utils.load_embedding(char_vocab_dict,
                                                  'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',
                                                  dump_path='data/char_embed.pkl')
    print('load embed done.')
    xe_char = [[i for i in range(600)] for _ in range(y.shape[0])]
    xe_char = np.array(xe_char)
    xe_term = [[i for i in range(300)] for _ in range(y.shape[0])]
    xe_term = np.array(xe_term)
    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xterm, xfeat, xe_char, xe_term], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('define model')
    model = HybridModel(char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix, NUM_FEAT=8,
                        name='hybridmodel_PE.h5', PE=True)  # +37
    print('feat shape', xfeat.shape)
    import sys
    print('train')
    model.train(x_tn, y_tn, x_val, y_val, x_ts, y_ts)

if __name__ == '__main__':
    SEED = 88
    np.random.seed(SEED)
    random.seed(SEED)
    import sys
    use_pe = len(sys.argv) > 2 and (sys.argv[2] == 'pe' or sys.argv[1] == 'pe')
    use_pe = use_pe or (len(sys.argv)>1 and sys.argv[1] == 'pe')
    if use_pe:
        print('train model with position embedding')
        train_main_pe( )
    else:
        train_main( )
