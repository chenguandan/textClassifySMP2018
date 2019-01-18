# coding=utf-8

'''
Created by lrr on 2018/7/3
'''

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

from hybridconvmodel import HybridConvModel
from hybriddpcnnmodel import HybridDPCNNModel
from hybridgateddeepcnnmodel import HybridGatedDeepCNNModel
from hybridgatedconvmodel import HybridGatedConvTopicModel
from conditionconvmodel import ConditionConvModel
from conditiondpcnnmodel import ConditionDPCNNModel
from conditiongatedconvmodel import ConditionGatedConvModel
from conditiongateddeepcnnmodel import ConditionGatedDeepCNNModel
import data_utils
from data_utils import TrainConfigure, ValidConfigure
from keras.utils import to_categorical
import training_utils
from sklearn.externals import joblib
import conditionmodelbase
import hybridmodelbase

class stacking(object):
    def __init__(self, n_fold, name = 'model/stack_model.pkl', is_condition=False):
        self.is_condition = is_condition
        self.name = name
        self.n_fold = n_fold
        self.base_models = []
        self.X_train = None
        self.y_train = None
        self.top_model = None

    def add_model(self, constructor, param):
        models = []
        for i in range(self.n_fold):
            models.append(constructor(**param))
        #num_model x n_fold
        self.base_models.append(models)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=0)
        # i = 0
        y_test_list = []
        y_pred_list = []
        # for train_idx, test_idx in skf.split(self.X_train, self.y_train):
        #     X_train_s, y_train_s = self.X_train[train_idx], self.y_train[train_idx]
        #     X_test_s, y_test_s = self.X_train[test_idx], self.y_train[test_idx]
        for i in range(self.n_fold):
            print('cross ', i+1, '/', self.n_fold)
            X_train_s, y_train_s, X_test_s, y_test_s = training_utils.split_cv(X_train, y_train, cv_num=self.n_fold, cv_index=i)
            # X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_s, y_train_s, test_size=0.1,
            #                                                           stratify=y_train_s)
            X_train_s, y_train_s, X_val_s, y_val_s = training_utils.split(X_train_s, y_train_s, split_ratio=0.95)
            y_pred_s = None
            for models in self.base_models:
                model = models[i]
                print(model.name)
                if self.is_condition:
                    model.train_exp(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s)
                else:
                    model.train(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s)
                #作为特征的时候使用one-hot表示
                if y_pred_s is None:
                    y_pred_s = model.predict(X_test_s)
                else:
                    # import ipdb
                    # ipdb.set_trace( )
                    y_pred_s = np.hstack( (y_pred_s, model.predict(X_test_s) ) )
            i += 1
            y_test_list.append(y_test_s)
            y_pred_list.append(y_pred_s)
        # 使用y_pred_list做特征，y_test_list做目标，再次训练模型
        X_top = np.vstack(y_pred_list)
        y_top = np.vstack(y_test_list)
        y_top = training_utils.convert_y( y_top )
        if self.is_condition:
            X_top = np.squeeze(X_top, axis=-1)
        print(X_top.shape, y_top.shape)
        self.top_model = LogisticRegression()
        self.top_model.fit(X_top, y_top)
        print(X_top)

    def fit_tmp(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        y_test_list = []
        y_pred_list = []
        for i in range(self.n_fold):
            print('cross ', i+1, '/', self.n_fold)
            X_train_s, y_train_s, X_test_s, y_test_s = training_utils.split_cv(X_train, y_train, cv_num=self.n_fold, cv_index=i)
            X_train_s, y_train_s, X_val_s, y_val_s = training_utils.split(X_train_s, y_train_s, split_ratio=0.9)
            y_pred_s = None
            for models in self.base_models:
                model = models[i]
                print(model.name)
                # if self.is_condition:
                #     model.train_exp(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s)
                # else:
                #     model.train(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s)
                #作为特征的时候使用one-hot表示
                if y_pred_s is None:
                    y_pred_s = model.predict(X_test_s)
                else:
                    # import ipdb
                    # ipdb.set_trace( )
                    y_pred_s = np.hstack( (y_pred_s, model.predict(X_test_s) ) )
            i += 1
            y_test_list.append(y_test_s)
            y_pred_list.append(y_pred_s)
        # 使用y_pred_list做特征，y_test_list做目标，再次训练模型
        X_top = np.vstack(y_pred_list)
        if self.is_condition:
            X_top = np.squeeze(X_top, axis=-1)
        y_top = np.vstack(y_test_list)
        y_top = training_utils.convert_y( y_top )
        print(X_top.shape, y_top.shape)
        self.top_model = LogisticRegression()
        self.top_model.fit(X_top, y_top)
        print(X_top)
        joblib.dump(self.top_model, self.name)

    def save(self):
        for k in range(self.n_fold):
            for models in self.base_models:
                model = models[k]
                model.model.save_weights( model.name+'_'+str(k) )
        joblib.dump(self.top_model,self.name)

    def load(self):
        for k in range(self.n_fold):
            for models in self.base_models:
                model = models[k]
                model.model.load_weights( model.name+'_'+str(k) )
        self.top_model = joblib.load(self.name)

    def predict(self, X_test):
        X_top = None
        for models in self.base_models:
            X = []
            #n_fold
            for model in models:
                X.append(model.predict(X_test))
            # X = np.vstack(X)
            X = np.mean(X, axis=0)#, axis=1
            if X_top is None:
                X_top = X
            else:
                X_top = np.hstack((X_top, X))
        print(X_top)
        print(X_top.shape)
        if self.is_condition:
            X_top = np.squeeze(X_top, axis=-1)
        y_pred = self.top_model.predict_proba(X_top)
        return to_categorical(y_pred)



def stacking_main_hybrid():
    print('load data')
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
    xe = [[i for i in range(600)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    xe_term = [[i for i in range(300)] for _ in range(y.shape[0])]
    xe_term = np.array(xe_term)
    xtopic = data_utils.pickle_load('data/lda_vec.pkl')

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

    name = 'model/stack_model.pkl'
    model_dir = 'model/stack/'
    n_fold = 3
    name = 'model/stack_model5.pkl'
    model_dir = 'model/stack5/'
    n_fold = 5
    stk_model = stacking(n_fold, name=name)
    conf = hybridmodelbase.ModelConfigure()
    conf.PE = True
    stk_model.add_model(HybridConvModel, {"conf":conf,"char_embed_matrix":char_embed_matrix,
                            "term_embed_matrix":term_embed_matrix,
                                          "name":model_dir+'hybridconvmodel_PE.h5'})
    stk_model.add_model(HybridGatedConvTopicModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                          "term_embed_matrix": term_embed_matrix,
                                                    "name": model_dir+'hybridgatedconvmodel_PE.h5'})

    stk_model.add_model(HybridGatedDeepCNNModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                          "term_embed_matrix": term_embed_matrix,
                                                  "name": model_dir+'hybridgateddeepcnnmodel_PE.h5'})
    conf.lr = 0.0005
    stk_model.add_model(HybridDPCNNModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                          "term_embed_matrix": term_embed_matrix,
                                           "name": model_dir+'hybriddpcnnmodel_PE.h5'})
    # stk_model.add_model(HybridDPCNNModel, char_embed_matrix=char_embed_matrix,
    #                          term_embed_matrix=term_embed_matrix, NUM_FEAT=8,
    #                          PE=True, name='model/stack/hybriddpcnnmodel_PE.h5')
    #采样0.1用于测试
    # x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xtopic], y, split_ratio=0.01, shuffle=False)
    # x_tn, y_tn, x_ts, y_ts = training_utils.split(x_tn, y_tn, shuffle=False)
    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xtopic], y, split_ratio=0.95)
    stk_model.fit(x_tn, y_tn)
    # joblib.dump(stk_model, 'model/stack_model_3.pkl')
    y_pred = stk_model.predict(x_ts)
    acc = accuracy_score(training_utils.convert_y(y_pred), training_utils.convert_y(y_ts) )
    print(acc)
    cnf_matrix = confusion_matrix(training_utils.convert_y(y_pred), training_utils.convert_y(y_ts) )
    print(cnf_matrix)
    stk_model.save( )

def load_stacking_model(isfold5=True):
    tn_conf = TrainConfigure()
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
    name = 'model/stack_model.pkl'
    model_dir = 'model/stack/'
    n_fold = 3
    if isfold5:
        name = 'model/stack_model5.pkl'
        model_dir = 'model/stack5/'
        n_fold = 5
    conf = hybridmodelbase.ModelConfigure()
    stk_model = stacking(n_fold, name=name)
    stk_model.add_model(HybridConvModel, {"conf":conf,"char_embed_matrix":char_embed_matrix,
                            "term_embed_matrix":term_embed_matrix,
                            "name":model_dir+'hybridconvmodel_PE.h5'})
    stk_model.add_model(HybridGatedConvTopicModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                          "term_embed_matrix": term_embed_matrix,
                                            "name": model_dir+'hybridgatedconvmodel_PE.h5'})
    stk_model.add_model(HybridGatedDeepCNNModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                          "term_embed_matrix": term_embed_matrix,
                                                  "name": model_dir+'hybridgateddeepcnnmodel_PE.h5'})
    stk_model.add_model(HybridDPCNNModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                          "term_embed_matrix": term_embed_matrix,
                                           "name": model_dir+'hybriddpcnnmodel_PE.h5'})
    stk_model.load()
    return stk_model

def load_condition_stacking_main():
    tn_conf = TrainConfigure()
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
    name = 'model/stack_condition_model.pkl'
    model_dir = 'model/stack/'
    n_fold = 3
    name = 'model/stack_condition_model5.pkl'
    model_dir = 'model/stack5/'
    n_fold = 5
    conf = conditionmodelbase.ModelConfigure()
    stk_model = stacking(n_fold, name=name, is_condition=True)
    stk_model.add_model(ConditionConvModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                             "term_embed_matrix": term_embed_matrix,
                                             "name": model_dir + 'conditionconvmodel_PE.h5'})
    stk_model.add_model(ConditionDPCNNModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                              "term_embed_matrix": term_embed_matrix,
                                              "name": model_dir + 'conditiondpcnnmodel_PE.h5'})
    stk_model.add_model(ConditionGatedConvModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                                  "term_embed_matrix": term_embed_matrix,
                                                  "name": model_dir + 'conditiongatedconvmodel_PE.h5'})
    stk_model.add_model(ConditionGatedDeepCNNModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                                     "term_embed_matrix": term_embed_matrix,
                                                     "name": model_dir + 'conditiongateddeepcnnmodel_PE.h5'})
    stk_model.load( )
    return stk_model
    #continue train
    # data_dict = data_utils.pickle_load(tn_conf.char_file)
    # y = to_categorical(data_dict['y'])
    # x = data_dict['x']
    # xterm = data_utils.pickle_load(tn_conf.term_file)
    # xfeat = data_utils.pickle_load(tn_conf.feat_file)
    # # normalization
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # scaler.fit(xfeat)
    # data_utils.pickle_dump(scaler, tn_conf.feat_norm)
    # xfeat = scaler.transform(xfeat)
    # xe = [[i for i in range(600)] for _ in range(y.shape[0])]
    # xe = np.array(xe)
    # xe_term = [[i for i in range(300)] for _ in range(y.shape[0])]
    # xe_term = np.array(xe_term)
    # xtopic = data_utils.pickle_load('data/lda_vec.pkl')
    #
    # x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xtopic], y)
    # stk_model.fit_tmp(x_tn, y_tn)
    # y_pred = stk_model.predict(x_ts)
    # acc = accuracy_score(training_utils.convert_y(y_pred), training_utils.convert_y(y_ts))
    # print(acc)
    # cnf_matrix = confusion_matrix(training_utils.convert_y(y_pred), training_utils.convert_y(y_ts))
    # print(cnf_matrix)

def stacking_main_condition():
    print('load data')
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
    xe = [[i for i in range(600)] for _ in range(y.shape[0])]
    xe = np.array(xe)
    xe_term = [[i for i in range(300)] for _ in range(y.shape[0])]
    xe_term = np.array(xe_term)
    xtopic = data_utils.pickle_load('data/lda_vec.pkl')

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

    name = 'model/stack_condition_model.pkl'
    model_dir = 'model/stack/'
    n_fold = 3
    name = 'model/stack_condition_model5.pkl'
    model_dir = 'model/stack5/'
    n_fold = 5
    stk_model = stacking(n_fold, name=name, is_condition=True)
    conf = conditionmodelbase.ModelConfigure()
    conf.PE = True
    stk_model.add_model(ConditionConvModel, {"conf":conf,"char_embed_matrix":char_embed_matrix,
                            "term_embed_matrix":term_embed_matrix,
                                             "name":model_dir+'conditionconvmodel_PE.h5'})
    stk_model.add_model(ConditionGatedConvModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                          "term_embed_matrix": term_embed_matrix,
                                                  "name": model_dir+'conditiongatedconvmodel_PE.h5'})
    stk_model.add_model(ConditionGatedDeepCNNModel, {"conf":conf,"char_embed_matrix": char_embed_matrix,
                                          "term_embed_matrix": term_embed_matrix,
                                            "name": model_dir+'conditiongateddeepcnnmodel_PE.h5'})
    conf.lr = 0.0005
    stk_model.add_model(ConditionDPCNNModel, {"conf": conf, "char_embed_matrix": char_embed_matrix,
                                              "term_embed_matrix": term_embed_matrix,
                                              "name": model_dir + 'conditiondpcnnmodel_PE.h5'})
    #采样0.1用于测试
    # x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xtopic], y, split_ratio=0.005, shuffle=False)
    # x_tn, y_tn, x_ts, y_ts = training_utils.split(x_tn, y_tn, shuffle=False)
    x_tn, y_tn, x_ts, y_ts = training_utils.split([x, xe, xterm, xe_term, xfeat, xtopic],  y, split_ratio=0.95)
    stk_model.fit(x_tn, y_tn)
    # joblib.dump(stk_model, 'model/stack_model_3.pkl')
    y_pred = stk_model.predict(x_ts)
    acc = accuracy_score(training_utils.convert_y(y_pred), training_utils.convert_y(y_ts) )
    print(acc)
    cnf_matrix = confusion_matrix(training_utils.convert_y(y_pred), training_utils.convert_y(y_ts) )
    print(cnf_matrix)
    stk_model.save( )


if __name__ == '__main__':
    # import sys
    # class ExceptionHook:
    #     instance = None
    #
    #     def __call__(self, *args, **kwargs):
    #         if self.instance is None:
    #             from IPython.core import ultratb
    #             self.instance = ultratb.FormattedTB(mode='Plain',
    #                                                 color_scheme='Linux', call_pdb=1)
    #         return self.instance(*args, **kwargs)
    # sys.excepthook = ExceptionHook()

    stacking_main_hybrid()
    stacking_main_condition()
    # load_stacking_main( )