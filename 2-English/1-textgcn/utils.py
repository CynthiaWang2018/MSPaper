"""
lil(Row-based linked list format)基于行的链表格式
稀疏矩阵转化成两个链表data和rows 举例
import numpy as np
import scipy.sparse as sp

A=np.array([[1,0,2,0],[0,0,0,0],[3,0,0,0],[1,0,0,4]])
AS=sp.lil_matrix(A)

print(AS.data)
# [list([1, 2]) list([]) list([3]) list([1, 4])]
print(AS.rows)
# [list([0, 2]) list([]) list([0]) list([0, 3])]

# 载入数据的维度（以Cora数据集为例）
# adj(邻接矩阵)：由于比较稀疏，邻接矩阵格式是LIL的，并且shape为(2708, 2708)
# features（特征矩阵）：每个节点的特征向量也是稀疏的，也用LIL格式存储，features.shape: (2708, 1433)
# labels：ally, ty数据集叠加构成，labels.shape:(2708, 7)
# train_mask, val_mask, test_mask：shaped都为(2708, )的向量，但是train_mask中的[0,140)范围的是True，其余是False；val_mask中范围为(140, 640]范围为True，其余的是False；test_mask中范围为[1708,2707]范围是True，其余的是False
# y_train, y_val, y_test：shape都是(2708, 7) 。y_train的值为对应与labels中train_mask为True的行，其余全是0；y_val的值为对应与labels中val_mask为True的行，其余全是0；y_test的值为对应与labels中test_mask为True的行，其余全是0
# 特征矩阵进行归一化并返回一个格式为(coords, values, shape)的元组
# 将邻接矩阵加上自环以后，对称归一化，并存储为COO模式，最后返回格式为(coords, values, shape)的元组

"""
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# 数据的读取，这个预处理是把训练集（其中一部分带有标签），测试集，标签的位置，对应的掩码训练标签等返回。
def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    # print(test_idx_reorder)
    # [2488, 2644, 3261, 2804, 3176, 2432, 3310, 2410, 2812,...]
    test_idx_range = np.sort(test_idx_reorder)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    # training nodes are training docs, no initial features
    # print("x: ", x)
    # test nodes are training docs, no initial features
    # print("tx: ", tx)
    # both labeled and unlabeled training instances are training docs and words
    # print("allx: ", allx)
    # training labels are training doc labels
    # print("y: ", y)
    # test labels are test doc labels
    # print("ty: ", ty)
    # ally are labels for labels for allx, some will not have labels, i.e., all 0
    # print("ally: \n")
    # for i in ally:
    # if(sum(i) == 0):
    # print(i)
    # graph edge weight is the word co-occurence or doc word frequency
    # no need to build map, directly build csr_matrix
    # print('graph : ', graph)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # print(len(labels))

    idx_test = test_idx_range.tolist()
    # print(idx_test)
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    print(len(labels))

    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

# 处理特征:特征矩阵进行归一化并返回一个格式为(coords, values, shape)的元组
# 特征矩阵的每一行的每个元素除以行和，处理后的每一行元素之和为1
# 处理特征矩阵，跟谱图卷积的理论有关，目的是要把周围节点的特征和自身节点的特征都捕捉到，同时避免不同节点间度的不均衡带来的问题
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # >> > b = [[1.0, 3], [2, 4], [3, 5]]
    # >> > b = np.array(b)
    # >> > b
    # array([[1., 3.],
    #        [2., 4.],
    #        [3., 5.]])
    # >> > np.array(b.sum(1))
    # array([4., 6., 8.])
    # >> > c = np.array(b.sum(1))
    # >> > np.power(c, -1)
    # array([0.25, 0.16666667, 0.125])
    # >> > np.power(c, -1).flatten()
    # array([0.25, 0.16666667, 0.125])
    # >> > r_inv = np.power(c, -1).flatten()
    # >> > import scipy.sparse as sp
    # >> > r_mat_inv = sp.diags(r_inv)
    # >> > r_mat_inv
    # < 3x3 sparse matrix of type '<class 'numpy.float64 '>'
    # with 3 stored elements (1 diagonals) in DIAgonal format >
    # >> > r_mat_inv.toarray()
    # array([[0.25, 0., 0.],
    #        [0., 0.16666667, 0.],
    #        [0., 0., 0.125]])
    # >> > f = r_mat_inv.dot(b)
    # >> > f
    # array([[0.25, 0.75],
    #        [0.33333333, 0.66666667],
    #        [0.375, 0.625]])

    # a.sum()是将矩阵中所有的元素进行求和;a.sum(axis = 0)是每一列列相加;a.sum(axis = 1)是每一行相加

    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    # np.isnan(ndarray)返回一个判断是否是NaN的bool型数组
    r_inv[np.isinf(r_inv)] = 0.
    # sp.diags创建一个对角稀疏矩阵
    r_mat_inv = sp.diags(r_inv)
    # dot矩阵乘法
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

# 邻接矩阵adj对称归一化并返回coo存储模式
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# 将邻接矩阵加上自环以后，对称归一化，并存储为COO模式，最后返回元组格式
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # 加上自环，再对称归一化
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

# 构建输入字典并返回
# labels和labels_mask传入的是具体的值，例如
# labels=y_train,labels_mask=train_mask；
# labels=y_val,labels_mask=val_mask；
# labels=y_test,labels_mask=test_mask；
def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    # 由于邻接矩阵是稀疏的，并且用LIL格式表示，因此定义为一个tf.sparse_placeholder(tf.float32)，可以节省内存
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    # print(features)
    # (array([[   0, 1274],
    #        [   0, 1247],
    #        [   0, 1194],
    #        ...,
    #        [2707,  329],
    #        [2707,  186],
    #        [2707,   19]], dtype=int32), array([0.11111111, 0.11111111, 0.11111111, ..., 0.07692308, 0.07692308,
    #        0.07692308], dtype=float32), (2708, 1433))

    # print(type(features))
    # <class 'tuple'>

    # print("features[1]",features[1])
    # features[1] [0.11111111 0.11111111 0.11111111 ... 0.07692308 0.07692308 0.07692308]

    # print("features[1].shape",features[1].shape)
    # features[1].shape (49216,)
    # 49126是特征矩阵存储为coo模式后非零元素的个数（2078*1433里只有49126个非零，稀疏度达1.3%）
    return feed_dict

# 切比雪夫多项式近似:计算K阶的切比雪夫近似矩阵
def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj) # D^{-1/2}AD^{-1/2}
    laplacian = sp.eye(adj.shape[0]) - adj_normalized # L = I_N - D^{-1/2}AD^{-1/2}
    largest_eigval, _ = eigsh(laplacian, 1, which='LM') # \lambda_{max}
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
    # 2/\lambda_{max}L-I_N

    # 将切比雪夫多项式的 T_0(x) = 1和 T_1(x) = x 项加入到t_k中
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    # 依据公式 T_n(x) = 2xT_n(x) - T_{n-1}(x) 构造递归程序，计算T_2 -> T_k
    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

# in remove_words.py
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()