from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, GRU, Bidirectional, TimeDistributed, Masking
from attention import AttLayer
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
"""
人类作者 48018
自动摘要 31034
机器作者 31163
机器翻译 36206
"""

def convert_y(y):
    yc = [np.argmax(yi) for yi in y]
    return np.array(yc)

class HanModel(object):
    def __init__(self, C = 4, V = 40000, MAX_SENT= 20, MAX_LEN= 100, name='hanmodel.h5'):
        self.name = name
        input = Input(shape=(MAX_LEN,),dtype='int32', name='input')
        #RNN支持mask
        x = Embedding(V, 32, mask_zero=True)(input)
        h = Bidirectional(GRU(64, return_sequences=True))(x)
        z = AttLayer()(h)
        sent_model = Model(input, z)
        sent_input = Input(shape=(MAX_SENT, MAX_LEN), dtype='int32', name='sent_input')
        h = TimeDistributed(sent_model)(sent_input)
        h = Masking()(h)
        h = Bidirectional(GRU(64,return_sequences=True))(h)
        z = AttLayer()(h)
        z = Dense(128, activation='relu')(z)
        # z = BatchNormalization()(z)
        z = Dense(C, activation='softmax')(z)
        model = Model(sent_input, z)
        model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
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
    import sys
    print('define char model')
    model = HanModel(name='hanmodel.h5')
    print('load data')
    import data_utils, training_utils
    data_dict = data_utils.pickle_load('data_sent.dict')
    y = to_categorical(data_dict['y'])
    x_tn, y_tn, x_ts, y_ts = training_utils.split(data_dict['x'], y, shuffle=False)
    x_tn, y_tn, x_val, y_val = training_utils.split(x_tn, y_tn, shuffle=False)
    print('train')
    model.train( x_tn, y_tn, x_val, y_val, x_ts, y_ts)