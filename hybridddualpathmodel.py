import training_utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from keras.layers import BatchNormalization, Add, Multiply, RepeatVector, Activation, MaxPool1D, Lambda, concatenate, add
from keras import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, SGD, Adagrad
from attention import AttLayer
import numpy as np
import random

import hybridmodelbase
from hybridmodelbase import HybridModelBase
"""
Dual path network
"""

def _bn_relu_conv_block(input, filters, kernel=3, stride=1, weight_decay=5e-4):
    ''' Adds a Batchnorm-Relu-Conv block for DPN
    Args:
        input: input tensor
        filters: number of output filters
        kernel: convolution kernel size
        stride: stride of convolution
    Returns: a keras tensor
    '''
    x = Conv1D(filters, kernel, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), strides=stride)(input)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x

def _grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = input
    group_list = []
    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv1D(grouped_channels, 3, padding='same', use_bias=False, strides=strides,
                   kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(init)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, c * grouped_channels:(c + 1) * grouped_channels])(input)

        x = Conv1D(grouped_channels, 3, padding='same', use_bias=False, strides=strides,
                   kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis=-1)
    group_merge = BatchNormalization(axis=-1)(group_merge)
    group_merge = Activation('relu')(group_merge)
    return group_merge

def _dual_path_block(input, pointwise_filters_a, grouped_conv_filters_b, pointwise_filters_c,
                     filter_increment, cardinality, block_type='normal'):
    '''
    Creates a Dual Path Block. The first path is a ResNeXt type
    grouped convolution block. The second is a DenseNet type dense
    convolution block.
    Args:
        input: input tensor
        pointwise_filters_a: number of filters for the bottleneck
            pointwise convolution
        grouped_conv_filters_b: number of filters for the grouped
            convolution block
        pointwise_filters_c: number of filters for the bottleneck
            convolution block
        filter_increment: number of filters that will be added
        cardinality: cardinality factor
        block_type: determines what action the block will perform
            - `projection`: adds a projection connection
            - `downsample`: downsamples the spatial resolution
            - `normal`    : simple adds a dual path connection
    Returns: a list of two output tensors - one path of ResNeXt
        and another path for DenseNet
    '''
    grouped_channels = int(grouped_conv_filters_b / cardinality)

    init = concatenate(input, axis=-1) if isinstance(input, list) else input

    if block_type == 'projection':
        stride = 1
        projection = True
    elif block_type == 'downsample':
        stride = 2
        projection = True
    elif block_type == 'normal':
        stride = 1
        projection = False
    else:
        raise ValueError('`block_type` must be one of ["projection", "downsample", "normal"]. Given %s' % block_type)

    if projection:
        projection_path = _bn_relu_conv_block(init, filters=pointwise_filters_c + 2 * filter_increment,
                                              kernel=1, stride=stride)
        input_residual_path = Lambda(lambda z: z[:, :, :pointwise_filters_c])(projection_path)
        input_dense_path = Lambda(lambda z: z[:, :, pointwise_filters_c:])(projection_path)
    else:
        input_residual_path = input[0]
        input_dense_path = input[1]

    x = _bn_relu_conv_block(init, filters=pointwise_filters_a, kernel=1)
    x = _grouped_convolution_block(x, grouped_channels=grouped_channels, cardinality=cardinality, strides=stride)
    x = _bn_relu_conv_block(x, filters=pointwise_filters_c + filter_increment, kernel=1)

    output_residual_path = Lambda(lambda z: z[:, :, :pointwise_filters_c])(x)
    output_dense_path = Lambda(lambda z: z[:, :, pointwise_filters_c:])(x)

    residual_path = add([input_residual_path, output_residual_path])
    dense_path = concatenate([input_dense_path, output_dense_path], axis=-1)

    return [residual_path, dense_path]


class HybridDualPathModel(HybridModelBase):

    def feed_forward(self, x, train_top):
        """

                    depth=[3, 4, 20, 3],
                    filter_increment=[16, 32, 24, 128],
                    cardinality=32,
                    width=3,
        :param x:
        :param train_top:
        :return:
        """
        width = 3
        cardinality = 4#32
        depth = [1, 1, 1, 1]#[3, 4, 20, 3]#20太大
        N = list(depth)
        filter_increment = [16, 32, 24, 128]
        filter_inc = filter_increment[0]
        filters = int(cardinality * width)
        base_filters = 128#256

        x = _dual_path_block(x, pointwise_filters_a=filters,
                             grouped_conv_filters_b=filters,
                             pointwise_filters_c=base_filters,
                             filter_increment=filter_inc,
                             cardinality=cardinality,
                             block_type='projection')

        for i in range(N[0] - 1):
            x = _dual_path_block(x, pointwise_filters_a=filters,
                                 grouped_conv_filters_b=filters,
                                 pointwise_filters_c=base_filters,
                                 filter_increment=filter_inc,
                                 cardinality=cardinality,
                                 block_type='normal')

        # remaining blocks
        for k in range(1, len(N)):
            print("BLOCK %d" % (k + 1))
            filter_inc = filter_increment[k]
            filters *= 2
            base_filters *= 2

            x = _dual_path_block(x, pointwise_filters_a=filters,
                                 grouped_conv_filters_b=filters,
                                 pointwise_filters_c=base_filters,
                                 filter_increment=filter_inc,
                                 cardinality=cardinality,
                                 block_type='downsample')

            for i in range(N[k] - 1):
                x = _dual_path_block(x, pointwise_filters_a=filters,
                                     grouped_conv_filters_b=filters,
                                     pointwise_filters_c=base_filters,
                                     filter_increment=filter_inc,
                                     cardinality=cardinality,
                                     block_type='normal')

        x = concatenate(x, axis=-1)
        x = GlobalMaxPool1D()(x)
        return x

    def __init__(self, conf, char_embed_matrix=None, term_embed_matrix=None,
                 name='hybriddualpathmodel.h5', train_embed=False,
                 train_top=True):
        super(HybridDualPathModel, self).__init__(conf=conf, char_embed_matrix=char_embed_matrix, term_embed_matrix=term_embed_matrix,
                 name=name, train_embed=train_embed,
                 train_top=train_top)


class HybridDPMain(hybridmodelbase.BaseMain):

    def get_model_conf(self):
        model_conf = hybridmodelbase.ModelConfigure()
        model_conf.lr = 0.0002
        return model_conf


if __name__ == '__main__':
    main = HybridDPMain('hybriddualpathmodel', HybridDualPathModel)
    main.main()