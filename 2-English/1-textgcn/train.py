"""
参考: https://www.cnblogs.com/think90/articles/11555381.html
以cora数据集为例
通过flags = tf.app.flags 模式设置参数, 可以在命令行运行时指定参数, 例如: python train.py --model gcn_cheby
提供了可供选择的三个模型: 'gcn', 'gcn_cheby', 'dense'. dense是由两层的MLP构成的
FLAGS.weight_decay(权重衰减): 在一定程度上减少模型过拟合问题
FLAGS.hidden1: 卷积层第一层的output dim, 第二层的input dim
FLAGS.max_degree: K阶的切比雪夫近似矩阵的参数K
FLAGS.dropout: 避免过拟合(按照一定的概率随机丢弃一部分神经元)
输入维度input_dim = features[2][1] (1433) 也就是每个节点的维度
"""

# this file
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from sklearn import metrics
from utils import *
from models import GCN, MLP, APPNP
import random
import os
import sys

if len(sys.argv) != 2:
	sys.exit("Use: python train.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'ag_news', 'dblp', 'TREC', 'WebKB','WebKB2'] # Modify
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")


# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
# flags.DEFINE_string('model', 'gcn', 'Model string.')
# flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')
flags.DEFINE_string('model', 'appnp', 'Model string.')

# flags.DEFINE_string('model', 'gcn', 'Model string.')

flags.DEFINE_float('learning_rate', 0.2, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
# 第一层的输出维度
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('alpha', 0.1, 'alpha of APPNP.') # modify
flags.DEFINE_integer('propagations', 2, 'propagations of APPNP.') # modify
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
# 权值衰减: 防止过拟合
# loss计算 (权值衰减*正则化) self.loss += FLAGS.weight_decay*tf.nn.l2_loss(var)
flags.DEFINE_float('weight_decay', 0.,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
# K阶的切比雪夫近似矩阵的参数
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    FLAGS.dataset)

# print(features)
# (0, 19) 1.0
# (0, 81) 1.0
# ...
# (2707, 1412) 1.0
# (2707, 1414) 1.0

# print(type(features))
# <class 'scipy.sparse.lil.lil_matrix'>
# print(adj)
# print(adj[0], adj[1])
# 定义为单位矩阵
# features = sp.identity(features.shape[0])  # featureless

# print('adj.shape', adj.shape)
# print('features.shape', features.shape)

# Some preprocessing
# 预处理特征矩阵: 将特征矩阵进行归一化并返回tuple(coords, values, shape)
features = preprocess_features(features)
# print(features)
# (array([[0, 1274],
#        [0, 1247],
#       [0, 1194],
#       ...
#       [2707, 19]
#       ], dtype=int32), array([0.11, 0.22, ..., 0.07], dtype=float32, (2708, 1433))

# print(type(features))
# <class 'tuple'>

# print("features[1]", features[1])
# features[1] [0.11 0.22 ... 0.07]

# print("features[1].shape", features[1].shape)
# features[1].shape (49216,)

if FLAGS.model == 'gcn':
    # support是邻接矩阵的归一化形式
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'appnp':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = APPNP
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    # 由于邻接矩阵是稀疏的, 并且用lil格式表示, 因此定义为一个tf.sparse_placeholder(tf.float32), 可以节省内存
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # 同上, shape = [2708, 1433]
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
# print(features[2][1])
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,
                     model.layers[0].embedding], feed_dict=feed_dict)
    # print("outs:",outs)
    # outs: [None, 0.57, 0.96]

    # Validation
    cost, acc, pred, labels, duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(
              outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

test_pred = []
test_labels = []
print(len(test_mask))
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(test_labels, test_pred, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

# doc and word embeddings
print('embeddings:')
word_embeddings = outs[3][train_size: adj.shape[0] - test_size]
train_doc_embeddings = outs[3][:train_size]  # include val docs
test_doc_embeddings = outs[3][adj.shape[0] - test_size:]

print(len(word_embeddings), len(train_doc_embeddings),
      len(test_doc_embeddings))
print(word_embeddings)

f = open('data/corpus/' + dataset + '_vocab.txt', 'r')
words = f.readlines()
f.close()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
f = open('data/' + dataset + '_word_vectors.txt', 'w')
f.write(word_embeddings_str)
f.close()

doc_vectors = []
doc_id = 0
for i in range(train_size):
    doc_vector = train_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

for i in range(test_size):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)
f = open('data/' + dataset + '_doc_vectors.txt', 'w')
f.write(doc_embeddings_str)
f.close()
