# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import tensorflow as tf
import numpy as np


import sys
def get_cur_info():
    print(sys._getframe().f_code.co_filename)#当前文件名
    print(sys._getframe(0).f_code.co_name)#当前函数名
    print(sys._getframe(1).f_code.co_name)#调用该函数的函数，如果没有，则返回module
    print(sys._getframe().f_lineno)#当前行号

def get_session():
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    set_session(sess)
    return sess

def scheduler(schedule):
    """
    usage:
    lr_scheduler = LearningRateScheduler(scheduler([(1, 0.001), (5, 0.0008), (10, 0.0005), (15, 0.0002)]))
    :param schedule:
    :return:
    """
    def lr_schedule(epoch):
        epoch += 1
        for ep, lr in sorted(schedule, reverse=True):
            if epoch >= ep:
                return lr
        raise ValueError()

    return lr_schedule

def split(xs, y, split_ratio=0.8, shuffle=False):
    num_tn = int(y.shape[0]*split_ratio)
    indices = np.arange(y.shape[0])
    if shuffle:
        np.random.shuffle( indices )
    tn_ind = indices[:num_tn]
    val_ind = indices[num_tn:]
    if isinstance(xs, list):
        x_tn = list()
        x_val = list()
        for x in xs:
            x_tn.append(x[tn_ind])
            x_val.append( x[val_ind])
    else:
        x_tn = xs[tn_ind]
        x_val = xs[val_ind]
    if isinstance(y, list):
        y_tn = list()
        y_val = list()
        for yi in y:
            y_tn.append( yi[tn_ind] )
            y_val.append( yi[val_ind] )
    else:
        y_tn= y[tn_ind]
        y_val = y[val_ind]
    return x_tn, y_tn, x_val, y_val

def split_cv(xs, y, cv_num, cv_index, split_ratio=0.8, shuffle=False):
    num_per_cv = y.shape[0]//cv_num
    indices = np.arange(y.shape[0])
    if shuffle:
        np.random.shuffle( indices )
    tn_ind1 = indices[:cv_index*num_per_cv]
    tn_ind2 =  indices[(cv_index+1)*num_per_cv:]
    tn_ind = np.concatenate([tn_ind1, tn_ind2])
    val_ind = indices[cv_index*num_per_cv:(cv_index+1)*num_per_cv]
    if isinstance(xs, list):
        x_tn = list()
        x_val = list()
        for x in xs:
            x_tn.append(x[tn_ind])
            x_val.append( x[val_ind])
    else:
        x_tn = xs[tn_ind]
        x_val = xs[val_ind]
    if isinstance(y, list):
        y_tn = list()
        y_val = list()
        for yi in y:
            y_tn.append( yi[tn_ind] )
            y_val.append( yi[val_ind] )
    else:
        y_tn= y[tn_ind]
        y_val = y[val_ind]
    return x_tn, y_tn, x_val, y_val

def random_shuffle(xs, y ):
    indices = np.arange(y.shape[0])
    np.random.shuffle( indices )
    if isinstance(xs, list):
        x_tn = list()
        for x in xs:
            x_tn.append(x[indices])
    else:
        x_tn = xs[indices]
    y_tn= y[indices]
    return x_tn, y_tn


import time
from functools import wraps
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1 - t0))
              )
        return result

    return function_timer

def convert_y(y):
    """
    将one-hot表示转化为label index的格式
    :param y:
    :return:
    """
    yc = [np.argmax(yi) for yi in y]
    return np.array(yc)
