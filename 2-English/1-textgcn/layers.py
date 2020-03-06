"""
参考: https://www.cnblogs.com/think90/articles/11555381.html
定义基类Layer
属性: name(String) 定义了变量范围 logging(Boolean) 打开或关闭Tensorflow直方图日志记录
方法: init() 初始化 _call() 定义计算 call() 调用_call()函数, _log_vars()
定义Dense Layer类, 继承自Layer类
定义GraphCN类, 继承自Layer类
"""
from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]
# def gelu(input_tensor):
#     cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    
#     return input_tesnsor*cdf

def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x/ tf.sqrt(2.0)))
    return x*cdf

# 稀疏矩阵的dropout操作
def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

# 定义Layer层 对每层name做命名 还用一个参数决定是否log
class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs
    # __call__的作用让Layer的实例成为可调用对象
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

# 定义Dense Layer类, 继承自Layer类
class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act # 激活函数
        self.sparse_inputs = sparse_inputs # 是否稀疏
        self.featureless = featureless # 输入的数据带不带特征矩阵???
        self.bias = bias # 是否有偏置

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()
    # 重写了_call函数 其中对稀疏矩阵做sparse_dropout
    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)



# 定义GraphCN类, 继承自Layer类
# 与denseNet的区别是_call函数 和 __init__函数 self.support = placeholders['support']的初始化
class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support'] # 不同点 dense
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        # 下面是定义变量, 主要是通过调用utils.py中的glorot函数实现
        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):  # 不同点 dense
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output #output
        return self.act(output)

class GraphPageRank(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, hidden1_dim, hidden2_dim,alpha, propagations, placeholders, dropout=0.,
                 sparse_inputs=False, act1=tf.nn.relu, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphPageRank, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.act = act
        self.act1 = act1
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.alpha = alpha
        self.propagations = propagations
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
            
            self.vars['weights1'] = glorot([input_dim,self.hidden1_dim], name='weights1')
            self.vars['weights2'] = glorot([self.hidden1_dim,output_dim], name='weights2')
#             self.vars['weights2'] = glorot([self.hidden1_dim,self.hidden2_dim], name='weights2')
#             self.vars['weights3'] = glorot([self.hidden2_dim,output_dim], name='weights3')
        
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()


        for i in range(len(self.support)):
#             if not self.featureless:
#                 pre_sup = dot(x, self.vars['weights_' + str(i)],
#                               sparse=self.sparse_inputs)
#             else:
#                 pre_sup = self.vars['weights_' + str(i)]
            out1 = self.vars['weights1']
#             x = tf.nn.dropout(x, 1-self.dropout)
        
            out1relu = gelu(out1)
#             pre_sup = self.act1(pre_sup)
            
            out1relu = tf.nn.dropout(out1relu, 1-self.dropout)
            # 将out1转换为SparseTensor
        
#             out2 = dot(out1relu, self.vars['weights2'],sparse=self.sparse_inputs)
            out2 = dot(out1relu, self.vars['weights2'])
            out2relu = gelu(out2)
            # out2relu = out2
            
            
#             cdf = 0.5 * (1.0 + tf.erf(x/ tf.sqrt(2.0)))
    
#             x*cdf
#             out2relu = tf.nn.dropout(out2relu, 1-self.dropout)
            
#             out3 = dot(out2relu, self.vars['weights3'])
#             out3relu = self.act1(out3)
            
            tempA = self.support[i]
            support = out2relu
            
            for k in range(self.propagations):
                support = (1-self.alpha)*dot(tempA, support, sparse=True)+self.alpha*out2relu   #modify
                print('support',support)
            print('alpha is*******************************************************************\n',self.alpha)
            print('k is ======================================================================',self.propagations)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output #output
        return self.act(output)
#         return gelu(output)
