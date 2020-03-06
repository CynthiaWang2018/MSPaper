"""
参考：https://www.cnblogs.com/think90/articles/11555381.html
定义了一个model基类, 以及两个继承自model类的MLP, GCN类
"""
from layers import *
from metrics import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x/ tf.sqrt(2.0)))
    return x*cdf


# 根据Layer来建立Model, 主要是设置了self.layers, self.activations建立序列模型
# init中的比如 loss, accuracy, optimizer, opt_op等
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        # 在子类中可以看出, 通过_build方法append各个层
        # 保存每一个layer
        self.layers = []
        # 保存每一次的输入, 以及最后一层的输出
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
    # 定义私有方法, 只能被类中的函数调用, 不能在类外单独调用
    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)

        # 以两层GCN为例, 输入inputs是features
        # self.activations.append(self.inputs)初始化第一个元素为inputs, 也就是features
        # 第一层, hidden=layer(self.activations[-1]), 即hidden等于inputs的输出outputs, 并将第一层的输出hidden=outputs加入到activations中
        # 第二层同理
        # 最终activations存储了三个元素[第一层的输入, 第二层的输入(第一层的输出), 第二层的输出]
        for layer in self.layers:
            # Layer类重写了__call__函数, 可以把对象当作函数调用, __call__输入为inputs, 输出为outputs
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

# 继承Model的多层感知机, 主要是重写了基类中没有实现的函数
# 计算了网络第一层的权重衰减L2损失
# 因为是半监督学习, 还计算了掩码交叉熵masked_softmax_cross_entropy
class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        # 以key, value形式存储的字典
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss 正则化项
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error 交叉熵损失函数
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

# 继承Model的卷积模型GCN
class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
    # 损失计算
    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
    # 计算模型准确度
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)
    # 构建模型: 两层GCN
    def _build(self):
        # 第一层的输入维度: input_dim = 1433
        # 第一层的输出维度: ouput_dim = 16
        # 第一层的激活函数: relu
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            # act=tf.nn.relu,
                                            act=gelu,
                                            dropout=True,
                                            featureless=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        # 第二层的输入维度: input_dim = FLAGS.hidden1 = 16
        # 第二层的输出维度: ouput_dim = placeholders['labels'].get_shape().as_list()[1]=7
        # 第二层的激活函数: lambda x: x, 即没有激活函数
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x, #
                                            dropout=True,
                                            logging=self.logging))
    # 模型预测
    def predict(self):
        # 返回的tensor每一行和为1 [0.333, 0.333, 0.333]
        return tf.nn.softmax(self.outputs)


class APPNP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(APPNP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):

#         self.layers.append(GraphConvolution(input_dim=self.input_dim,
#                                             output_dim=FLAGS.hidden1,
#                                             placeholders=self.placeholders,
#                                             act=tf.nn.relu,
#                                             dropout=True,
#                                             featureless=True,
#                                             sparse_inputs=True,
#                                             logging=self.logging))

#         self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
#                                             output_dim=self.output_dim,
#                                             placeholders=self.placeholders,
#                                             act=lambda x: x, #
#                                             dropout=True,
#                                             logging=self.logging))

        self.layers.append(GraphPageRank(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            hidden1_dim=FLAGS.hidden1_dim,
                                            hidden2_dim=FLAGS.hidden2_dim,
                                            alpha=FLAGS.alpha, # modify
                                            propagations=FLAGS.propagations, #modify
                                            placeholders=self.placeholders,
                                            act1=tf.nn.relu,
                                            act=lambda x: x,
                                            dropout=True,
                                            featureless=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        print('model change------------------------------------------------------------------------')

    def predict(self):
        return tf.nn.softmax(self.outputs)