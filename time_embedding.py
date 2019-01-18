from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras.optimizers import SGD, RMSprop, Adagrad



class TimeEmbedding(Layer):
    def __init__(self, K = 32, **kwargs):
        """
        input: bs x len( t0 ~ tn-1 )
        return: bs x len x dim
        :param W_regularizer:
        :param b_regularizer:
        :param W_constraint:
        :param b_constraint:
        :param bias:
        :param kwargs:
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.K = K
        super(TimeEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.w = self.add_weight( (1,),
                                  initializer=self.init,
                                  name='{}_w'.format(self.name) )
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        x = K.cast(x, K.floatx())
        seq = K.arange( 1, self.K+1, dtype='float32')*self.w[0]
        # seq = K.expand_dims( seq, )
        x_exp = K.expand_dims(x, axis=-1)
        sin_x = K.sin( x_exp*seq )
        cos_x = K.cos( x_exp*seq )
        embed = K.concatenate([sin_x, cos_x],axis=-1)
        return embed

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], 2*self.K

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], 2*self.K