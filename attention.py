import keras
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers,regularizers,constraints
from keras.optimizers import SGD, RMSprop, Adagrad
import numpy as np

use_scale = False
scale_init_const = 1.0
#用于初始化scale
def my_init(shape, name=None, dim_ordering='th'):
    a = np.zeros( shape=shape, dtype='float32')+scale_init_const
    return K.variable(value= a, name=name)

def my_init_2(shape, name=None, dim_ordering='th'):
    a = np.zeros( shape=shape, dtype='float32')+scale_init_const
    return K.variable(value= a, name=name)

class AttLayer(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, return_alpha =False, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.return_alpha = return_alpha
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],input_shape[-1]),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        if use_scale:
            self.scale = self.add_weight((1,),
                                         initializer=my_init,
                                         name='{}_scala'.format(self.name))
        self.uw = self.add_weight((input_shape[-1],),
                                  initializer=self.init,
                                  name='{}_u'.format(self.name))
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if use_scale:
            # print(x.shape)
            eij = K.dot(x, self.W)
            # print(eij.shape)
            if self.bias:
                eij += self.b
            eij = K.tanh(eij)
            # print(eij.shape)
            # norm_val = K.sum( self.uw**2 )+K.epsilon()
            # eij = self.scale[0] * K.dot(eij, K.l2_normalize( self.uw, axis=-1) )
            # a = K.exp(eij)

            eij = self.scale[0] * K.dot(eij, K.expand_dims(K.l2_normalize( self.uw, axis=-1) ))
            eij = K.squeeze(eij, axis=-1)
            a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))

            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                a *= K.cast(mask, K.floatx())
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a
            if self.return_alpha:
                return [K.sum(weighted_input, axis=1), K.squeeze(a, axis=-1)]
            return K.sum(weighted_input, axis=1)
        else:
            # print(x.shape)
            eij = K.dot( x, self.W)
            # print(eij.shape)
            if self.bias:
                eij += self.b
            eij = K.tanh(eij)
            # print(eij.shape)
            eij = K.dot(eij, K.expand_dims(self.uw))
            eij = K.squeeze(eij, axis=-1)
            a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))

            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                a *= K.cast(mask, K.floatx())
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a
            if self.return_alpha:
                return [K.sum(weighted_input, axis=1), K.squeeze( a, axis=-1 )]
            return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        if self.return_alpha:
            return [(input_shape[0], input_shape[-1]),(input_shape[0],input_shape[1])]
        return input_shape[0], input_shape[-1]
    def compute_output_shape(self, input_shape):
        if self.return_alpha:
            return [(input_shape[0], input_shape[-1]),(input_shape[0],input_shape[1])]
        return input_shape[0], input_shape[-1]


class AttLayerMask(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, return_alpha = False, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.return_alpha = return_alpha
        super(AttLayerMask, self).__init__(**kwargs)

    def build(self, input_shapes):
        input_shape = input_shapes[0]
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],input_shape[-1]),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.uw = self.add_weight((input_shape[-1],),
                                  initializer=self.init,
                                  name='{}_u'.format(self.name))
        if use_scale:
            self.scale = self.add_weight((1,),initializer=my_init, name='{}_scale'.format(self.name))
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x_and_mask, un_mask=None):
        if use_scale:
            x = x_and_mask[0]
            mask = x_and_mask[1]
            # print(x.shape)
            eij = K.dot(x, self.W)
            # print(eij.shape)
            if self.bias:
                eij += self.b
            eij = K.tanh(eij)

            eij = self.scale[0] * K.dot(eij, K.expand_dims(K.l2_normalize( self.uw, axis=-1)))
            eij = K.squeeze(eij, axis=-1)
            a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))

            # print(eij.shape)
            # norm_val = K.sum( self.uw**2 )+K.epsilon()
            # eij = self.scale[0] * K.dot(eij, K.l2_normalize( self.uw, axis=-1) )
            # a = K.exp(eij)

            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                a *= K.cast(mask, K.floatx())
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a
            if self.return_alpha:
                return [K.sum(weighted_input, axis=1), K.squeeze(a, axis=-1)]
            return K.sum(weighted_input, axis=1)
        else:
            # print(x.shape)
            x = x_and_mask[0]
            mask = x_and_mask[1]
            eij = K.dot( x, self.W)
            # print(eij.shape)
            if self.bias:
                eij += self.b
            eij = K.tanh(eij)
            # print(eij.shape)
            # eij = K.dot(eij, self.uw )
            eij = K.dot(eij, K.expand_dims(self.uw))
            eij = K.squeeze(eij, axis=-1)
            a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))

            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                a *= K.cast(mask, K.floatx())

            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a
            if self.return_alpha:
                return [K.sum(weighted_input, axis=1), K.squeeze( a, axis=-1 )]
            return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.return_alpha:
            return [(input_shape[0], input_shape[-1]),(input_shape[0],input_shape[1])]
        return input_shape[0], input_shape[-1]
    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        if self.return_alpha:
            return [(input_shape[0], input_shape[-1]), (input_shape[0],input_shape[1])]
        return input_shape[0], input_shape[-1]

class ContextAttention(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 Wc_regularizer=None,
                 Wc_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.Wc_regularizer = regularizers.get(Wc_regularizer)

        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.Wc_constraint = constraints.get(Wc_constraint)

        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(ContextAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        assert len(shape1) == 3

        self.W = self.add_weight((shape1[-1], shape1[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)


        self.Wc = self.add_weight((shape2[-1],shape1[-1], ),
                                 initializer=self.init,
                                 name='{}_Wc'.format(self.name),
                                 regularizer=self.Wc_regularizer,
                                 constraint=self.Wc_constraint)

        if self.bias:
            self.b = self.add_weight((shape1[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((shape1[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        if use_scale:
            self.scale = self.add_weight((1,),initializer=my_init, name='{}_scale'.format(self.name))
        super(ContextAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, input, mask=None):
        if use_scale:
            x = input[0]
            c = input[1]
            # compute from x
            uit_x = K.dot(x, self.W)
            # extra feature for attention
            c = K.expand_dims(c, axis=1)
            uit_c = K.dot(c, self.Wc)

            uit = uit_x + uit_c

            if self.bias:
                uit += self.b

            eij = K.tanh(uit)
            # print(eij.shape)
            # norm_val = K.sum( self.u **2 )+K.epsilon()
            eij = self.scale[0] * K.dot(eij, K.expand_dims(K.l2_normalize( self.u, axis=-1) ) )
            eij = K.squeeze(eij, axis=-1)
            a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))
            # apply mask after the exp. will be re-normalized next
            if mask is not None and mask[0] is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                a *= K.cast(mask[0], K.floatx())
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a
            return K.sum(weighted_input, axis=1)
        else:
            x = input[0]
            c = input[1]
            #compute from x
            uit_x = K.dot(x, self.W)
            #extra feature for attention
            c = K.expand_dims(c, axis=1)
            uit_c = K.dot(c, self.Wc)

            uit = uit_x + uit_c

            if self.bias:
                uit += self.b

            uit = K.tanh(uit)
            ait = K.dot(uit, K.expand_dims(self.u))
            ait = K.squeeze(ait, axis=-1)
            a = K.exp(ait - K.max(ait, axis=-1, keepdims=True))
            #it is a list
            if mask is not None and mask[0] is not None:
                a *= K.cast(mask[0], K.floatx())
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
            a = K.expand_dims(a)
            weighted_input = x * a
            return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        shape1 = input_shape[0]
        return shape1[0], shape1[-1]
    def compute_output_shape(self, input_shape):
        shape1 = input_shape[0]
        return shape1[0], shape1[-1]


# class AttentionWithContext(Layer):
#     """
#         Attention operation, with a context/query vector, for temporal data.
#         Supports Masking.
#         Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
#         "Hierarchical Attention Networks for Document Classification"
#         by using a context vector to assist the attention
#         # Input shape
#             3D tensor with shape: `(samples, steps, features)`.
#         # Output shape
#             2D tensor with shape: `(samples, features)`.
#         :param kwargs:
#         Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
#         The dimensions are inferred based on the output shape of the RNN.
#         Example:
#             model.add(LSTM(64, return_sequences=True))
#             model.add(AttentionWithContext())
#         """
#
#     def __init__(self,
#                  W_regularizer=None, u_regularizer=None, b_regularizer=None,
#                  W_constraint=None, u_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):
#
#         self.supports_masking = True
#         self.init = initializations.get('glorot_uniform')
#
#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.u_regularizer = regularizers.get(u_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)
#
#         self.W_constraint = constraints.get(W_constraint)
#         self.u_constraint = constraints.get(u_constraint)
#         self.b_constraint = constraints.get(b_constraint)
#
#         self.bias = bias
#         super(AttentionWithContext, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 3
#
#         self.W = self.add_weight((input_shape[-1], input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_W'.format(self.name),
#                                  regularizer=self.W_regularizer,
#                                  constraint=self.W_constraint)
#         if self.bias:
#             self.b = self.add_weight((input_shape[-1],),
#                                      initializer='zero',
#                                      name='{}_b'.format(self.name),
#                                      regularizer=self.b_regularizer,
#                                      constraint=self.b_constraint)
#
#         self.u = self.add_weight((input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_u'.format(self.name),
#                                  regularizer=self.u_regularizer,
#                                  constraint=self.u_constraint)
#
#         super(AttentionWithContext, self).build(input_shape)
#
#     def compute_mask(self, input, input_mask=None):
#         # do not pass the mask to the next layers
#         return None
#
#     def call(self, x, mask=None):
#         uit = K.dot(x, self.W)
#
#         if self.bias:
#             uit += self.b
#
#         uit = K.tanh(uit)
#         ait = K.dot(uit, self.u)
#
#         a = K.exp(ait)
#
#         # apply mask after the exp. will be re-normalized next
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             a *= K.cast(mask, K.floatx())
#
#         # in some cases especially in the early stages of training the sum may be almost zero
#         # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
#         # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
#         a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
#
#         a = K.expand_dims(a)
#         weighted_input = x * a
#         return K.sum(weighted_input, axis=1)
#
#     def get_output_shape_for(self, input_shape):
#         return input_shape[0], input_shape[-1]
#
#
# class Attention(Layer):
#     def __init__(self,
#                  W_regularizer=None, b_regularizer=None,
#                  W_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):
#         """
#         Keras Layer that implements an Attention mechanism for temporal data.
#         Supports Masking.
#         Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
#         # Input shape
#             3D tensor with shape: `(samples, steps, features)`.
#         # Output shape
#             2D tensor with shape: `(samples, features)`.
#         :param kwargs:
#         Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
#         The dimensions are inferred based on the output shape of the RNN.
#         Example:
#             model.add(LSTM(64, return_sequences=True))
#             model.add(Attention())
#         """
#         self.supports_masking = True
#         self.init = initializations.get('glorot_uniform')
#
#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)
#
#         self.W_constraint = constraints.get(W_constraint)
#         self.b_constraint = constraints.get(b_constraint)
#
#         self.bias = bias
#         super(Attention, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 3
#
#         self.W = self.add_weight((input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_W'.format(self.name),
#                                  regularizer=self.W_regularizer,
#                                  constraint=self.W_constraint)
#         if self.bias:
#             self.b = self.add_weight((input_shape[1],),
#                                      initializer='zero',
#                                      name='{}_b'.format(self.name),
#                                      regularizer=self.b_regularizer,
#                                      constraint=self.b_constraint)
#         else:
#             self.b = None
#
#         self.built = True
#
#     def compute_mask(self, input, input_mask=None):
#         # do not pass the mask to the next layers
#         return None
#
#     def call(self, x, mask=None):
#         eij = K.dot(x, self.W)
#
#         if self.bias:
#             eij += self.b
#
#         eij = K.tanh(eij)
#
#         a = K.exp(eij)
#
#         # apply mask after the exp. will be re-normalized next
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             a *= K.cast(mask, K.floatx())
#
#         # in some cases especially in the early stages of training the sum may be almost zero
#         # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
#         # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
#         a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
#
#         a = K.expand_dims(a)
#         weighted_input = x * a
#         return K.sum(weighted_input, axis=1)
#
#     def get_output_shape_for(self, input_shape):
#         return input_shape[0], input_shape[-1]


class ContextAttentionMask(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 Wc_regularizer=None,
                 Wc_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.Wc_regularizer = regularizers.get(Wc_regularizer)

        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.Wc_constraint = constraints.get(Wc_constraint)

        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(ContextAttentionMask, self).__init__(**kwargs)

    def build(self, input_shape):
        shape1 = input_shape[0]
        # shape_mask = input_shape[1]
        shape2 = input_shape[2]
        assert len(shape1) == 3

        self.W = self.add_weight((shape1[-1], shape1[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)


        self.Wc = self.add_weight((shape2[-1],shape1[-1], ),
                                 initializer=self.init,
                                 name='{}_Wc'.format(self.name),
                                 regularizer=self.Wc_regularizer,
                                 constraint=self.Wc_constraint)

        if self.bias:
            self.b = self.add_weight((shape1[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((shape1[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        if use_scale:
            self.scale = self.add_weight((1,),initializer=my_init, name='{}_scale'.format(self.name))

        super(ContextAttentionMask, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, input, un_mask=None):
        if use_scale:
            x = input[0]
            mask = input[1]
            c = input[2]
            # compute from x
            uit_x = K.dot(x, self.W)
            # extra feature for attention
            c = K.expand_dims(c, axis=1)
            uit_c = K.dot(c, self.Wc)

            uit = uit_x + uit_c

            if self.bias:
                uit += self.b

            eij = K.tanh(uit)
            # print(eij.shape)
            # norm_val = K.sum( self.u**2 )+K.epsilon()
            eij = self.scale[0] * K.dot(eij, K.expand_dims(K.l2_normalize( self.u, axis=-1) ) )
            eij = K.squeeze(eij, axis=-1)
            a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))
            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                a *= K.cast(mask, K.floatx())
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a
            return K.sum(weighted_input, axis=1)
        else:
            x = input[0]
            mask = input[1]
            c = input[2]
            #compute from x
            uit_x = K.dot(x, self.W)
            #extra feature for attention
            c = K.expand_dims(c, axis=1)
            uit_c = K.dot(c, self.Wc)

            uit = uit_x + uit_c

            if self.bias:
                uit += self.b

            uit = K.tanh(uit)
            ait = K.dot(uit, K.expand_dims(self.u))
            ait = K.squeeze(ait, axis=-1)
            a = K.exp(ait - K.max(ait, axis=-1, keepdims=True))
            if mask is not None:
                a *= K.cast(mask, K.floatx())
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
            a = K.expand_dims(a)
            weighted_input = x * a
            return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        shape1 = input_shape[0]
        return shape1[0], shape1[-1]
    def compute_output_shape(self, input_shape):
        shape1 = input_shape[0]
        return shape1[0], shape1[-1]

class ContextAttentionLuong(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 Wc_regularizer=None,
                 Wc_constraint=None,
                 return_alpha = False,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.Wc_regularizer = regularizers.get(Wc_regularizer)

        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.Wc_constraint = constraints.get(Wc_constraint)

        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.return_alpha = return_alpha
        super(ContextAttentionLuong, self).__init__(**kwargs)

    def build(self, input_shape):
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        assert len(shape1) == 3

        self.W = self.add_weight((shape1[-1], shape1[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)


        self.Wc = self.add_weight((shape2[-1],shape1[-1], ),
                                 initializer=self.init,
                                 name='{}_Wc'.format(self.name),
                                 regularizer=self.Wc_regularizer,
                                 constraint=self.Wc_constraint)

        if self.bias:
            self.b = self.add_weight((shape1[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        self.scale = self.add_weight((1,),initializer=my_init, name='{}_scale'.format(self.name))
        super(ContextAttentionLuong, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, input, mask=None):
        x = input[0]
        c = input[1]
        #bs x n x dim
        #TODO do not use mapping
        keys = K.dot(x, self.W)
        # keys = x
        # extra feature for attention
        #bs x 1 x dim
        c = K.expand_dims(c, axis=1)
        query = K.dot(c, self.Wc)
        if self.bias:
            query += self.b
        #TODO without nonlinear
        query = K.squeeze( K.tanh(query), axis=1)
        # query = K.squeeze( query, axis=1)
        query = K.expand_dims( query, axis=-1)
        # print(eij.shape)
        # norm_val = K.sum( self.uw**2 )+K.epsilon()
        # bs x n x dim * bs x dim x 1=> bs x n
        eij = self.scale[0] * K.batch_dot(query, keys, axes=[1,2])
        eij = K.squeeze( eij, axis=1)
        a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))

        # apply mask after the exp. will be re-normalized next
        if mask is not None and mask[0] is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask[0], K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        if self.return_alpha:
            return K.sum(weighted_input, axis=1), K.squeeze(a, axis=-1)
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        shape1 = input_shape[0]
        return shape1[0], shape1[-1]
    # def compute_output_shape(self, input_shape):
    #     shape1 = input_shape[0]
    #     return shape1[0], shape1[-1]
    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        if self.return_alpha:
            return [(input_shape[0], input_shape[-1]),(input_shape[0],input_shape[1])]
        return input_shape[0], input_shape[-1]

class ContextAttentionMaskLuong(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 Wc_regularizer=None,
                 Wc_constraint=None,
                 return_alpha = False,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.Wc_regularizer = regularizers.get(Wc_regularizer)

        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.Wc_constraint = constraints.get(Wc_constraint)

        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.return_alpha = return_alpha
        super(ContextAttentionMaskLuong, self).__init__(**kwargs)

    def build(self, input_shape):
        shape1 = input_shape[0]
        # shape_mask = input_shape[1]
        shape2 = input_shape[2]
        assert len(shape1) == 3

        self.W = self.add_weight((shape1[-1], shape1[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)


        self.Wc = self.add_weight((shape2[-1],shape1[-1], ),
                                 initializer=self.init,
                                 name='{}_Wc'.format(self.name),
                                 regularizer=self.Wc_regularizer,
                                 constraint=self.Wc_constraint)

        if self.bias:
            self.b = self.add_weight((shape1[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.scale = self.add_weight((1,),initializer=my_init_2, name='{}_scale'.format(self.name))

        super(ContextAttentionMaskLuong, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, input, un_mask=None):
        x = input[0]
        mask = input[1]
        c = input[2]
        # compute from x
        #TODO do not use a linear mapping
        keys = K.dot(x, self.W)
        # keys = x
        # extra feature for attention
        c = K.expand_dims(c, axis=1)
        query = K.dot(c, self.Wc)
        if self.bias:
            query += self.b
        #TODO without nonlinear
        query = K.squeeze(K.tanh(query), axis=1)
        # query = K.squeeze( query, axis=1)
        query = K.expand_dims(query, axis=-1)
        # print(eij.shape)
        # norm_val = K.sum( self.uw**2 )+K.epsilon()
        # bs x n x dim * bs x dim x 1=> bs x n
        eij = self.scale[0] * K.batch_dot(query, keys, axes=[1, 2])
        eij = K.squeeze(eij, axis=1)
        a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        if self.return_alpha:
            return K.sum(weighted_input, axis=1), K.squeeze(a, axis=-1)
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        shape1 = input_shape[0]
        return shape1[0], shape1[-1]
    # def compute_output_shape(self, input_shape):
    #     shape1 = input_shape[0]
    #     return shape1[0], shape1[-1]
    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        if self.return_alpha:
            return [(input_shape[0], input_shape[-1]),(input_shape[0],input_shape[1])]
        return input_shape[0], input_shape[-1]

class HighTopicAttention(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 Wc_regularizer=None,
                 Wc_constraint=None,
                 bias=True,K=32, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.Wc_regularizer = regularizers.get(Wc_regularizer)

        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.Wc_constraint = constraints.get(Wc_constraint)

        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.K = K
        self.bias = bias
        super(HighTopicAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        shape1 = input_shape[0]
        # shape_mask = input_shape[1]
        shape2 = input_shape[1]
        assert len(shape1) == 3

        self.W = self.add_weight((shape1[-1], shape1[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)


        self.Wc = self.add_weight((shape2[-1],shape1[-1], ),
                                 initializer=self.init,
                                 name='{}_Wc'.format(self.name),
                                 regularizer=self.Wc_regularizer,
                                 constraint=self.Wc_constraint)

        if self.bias:
            self.b = self.add_weight((shape1[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.bt = self.add_weight((shape1[-1],),
                                      initializer='zero',
                                      name='{}_bt'.format(self.name))

        self.Wt = self.add_weight((shape1[-1], shape1[-1]),
                                  initializer='glorot_uniform',
                                  name='{}_Wt'.format(self.name))
        self.topics = self.add_weight((self.K, shape1[-1]),
                                      initializer='glorot_uniform',
                                      name='{}_topics'.format(self.name))

        super(HighTopicAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def get_topic(self, ei ):
        tij = K.dot(ei, self.Wt)#bs x d; d x d
        if self.bias:
            tij += self.bt
        tij = K.tanh(tij)
        tij = K.dot(tij, K.expand_dims(self.topics) ) #bs x d; 64 x d x 1 =>bs x 64 x 1
        tij = K.squeeze(tij, axis=-1)
        # a = K.softmax( tij )
        a = K.exp(tij - K.max(tij, axis=-1, keepdims=True))
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        out = self.topics * a# 64 x d; bs x 64 x 1
        uw = K.sum(out, axis=1) #bs x d
        return uw#bs x d

    def call(self, x_and_w, mask=None):
        x = x_and_w[0]#bs x seq x d
        c = K.expand_dims(x_and_w[1], axis=1)
        uit_c = K.dot(c, self.Wc)

        # ei = K.expand_dims( x_and_w[1],dim=1)#bs x 1 x d
        # uit_t = K.dot(t, self.Wt)
        eij = K.dot(x, self.W) + uit_c
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)

        #read uw from topics
        uw = self.get_topic( K.squeeze( uit_c,axis=1 ) )

        eij = K.batch_dot(eij, K.expand_dims(uw) ) #bs x seq x d ; bs x d  => bs x seq x 1
        eij = K.squeeze(eij, axis=-1)
        #防止softmax产生过大的结果
        a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))#bs x seq_len x 64
        if mask is not None and mask[0] is not None:
            a *= K.cast(mask[0], K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        out = K.sum(weighted_input, axis=1)
        return out

    def get_output_shape_for(self, input_shape):
        shape1 = input_shape[0]
        return shape1[0], shape1[-1]

    def compute_output_shape(self, input_shape):
        shape1 = input_shape[0]
        return shape1[0], shape1[-1]

class HighTopicAttentionMask(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 Wc_regularizer=None,
                 Wc_constraint=None,
                 bias=True,K=32, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.Wc_regularizer = regularizers.get(Wc_regularizer)

        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.Wc_constraint = constraints.get(Wc_constraint)

        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.K = K
        self.bias = bias
        super(HighTopicAttentionMask, self).__init__(**kwargs)

    def build(self, input_shape):
        shape1 = input_shape[0]
        # shape_mask = input_shape[1]
        shape2 = input_shape[2]
        assert len(shape1) == 3

        self.W = self.add_weight((shape1[-1], shape1[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)


        self.Wc = self.add_weight((shape2[-1],shape1[-1], ),
                                 initializer=self.init,
                                 name='{}_Wc'.format(self.name),
                                 regularizer=self.Wc_regularizer,
                                 constraint=self.Wc_constraint)

        if self.bias:
            self.b = self.add_weight((shape1[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.bt = self.add_weight((shape1[-1],),
                                      initializer='zero',
                                      name='{}_bt'.format(self.name))

        self.Wt = self.add_weight((shape1[-1], shape1[-1]),
                                  initializer='glorot_uniform',
                                  name='{}_Wt'.format(self.name))
        self.topics = self.add_weight((self.K, shape1[-1]),
                                      initializer='glorot_uniform',
                                      name='{}_topics'.format(self.name))

        super(HighTopicAttentionMask, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def get_topic(self, ei ):
        tij = K.dot(ei, self.Wt)#bs x d; d x d
        if self.bias:
            tij += self.bt
        tij = K.tanh(tij)
        tij = K.dot(tij, K.expand_dims(self.topics) ) #bs x d; 64 x d x 1 =>bs x 64 x 1
        tij = K.squeeze(tij, axis=-1)
        # a = K.softmax( tij )
        a = K.exp(tij - K.max(tij, axis=-1, keepdims=True))
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        out = self.topics * a# 64 x d; bs x 64 x 1
        uw = K.sum(out, axis=1) #bs x d
        return uw#bs x d

    def call(self, x_and_w, un_mask=None):
        x = x_and_w[0]#bs x seq x d
        mask = x_and_w[1]
        c = K.expand_dims(x_and_w[2], axis=1)
        uit_c = K.dot(c, self.Wc)
        # ei = K.expand_dims( x_and_w[2],dim=1)#bs x 1 x d
        # uit_t = K.dot(t, self.Wt)
        eij = K.dot(x, self.W) + uit_c
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)

        #read uw from topics
        uw = self.get_topic( K.squeeze(uit_c, axis=1) )

        #TODO ???
        eij = K.batch_dot(eij, K.expand_dims(uw) ) #bs x seq x d ; bs x d  => bs x seq x 1
        eij = K.squeeze(eij, axis=-1)
        #防止softmax产生过大的结果
        a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))#bs x seq_len x 64
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        out = K.sum(weighted_input, axis=1)
        return out

    def get_output_shape_for(self, input_shape):
        shape1 = input_shape[0]
        return shape1[0], shape1[-1]

    def compute_output_shape(self, input_shape):
        shape1 = input_shape[0]
        return shape1[0], shape1[-1]