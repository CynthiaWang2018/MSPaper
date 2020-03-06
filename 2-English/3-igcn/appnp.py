from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.sparse as sp
import os
import numpy as np
from utils import *
from models import GCN_APPRO_Mix,APPNP

# Set random seed
# seed = 31
seed = random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', 'R8', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
# flags.DEFINE_string('model', 'gcn_mix', 'Model string.')  # 'gcn_mix', 'gcn_appr'
flags.DEFINE_string('model', 'appnp', 'Model string.')  # 'gcn_mix', 'gcn_appr'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.') # 5e-4
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


flags.DEFINE_float('alpha', 0.1, 'alpha.')
flags.DEFINE_integer('propagations', 3, 'propagations.')
def construct_feeddict_forMixlayers(AXfeatures, support, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['AXfeatures']: AXfeatures})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: AXfeatures[1].shape})
    return feed_dict

def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]


def main(rank1):

    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
    adj, features, y_train, y_val, y_test, y_vocab, train_mask, val_mask, test_mask, vocab_mask, _, _ = load_corpus(
        FLAGS.dataset)

    train_index = np.where(train_mask)[0]  # [10183]
    adj_train = adj[train_index, :][:, train_index] # [10183, 10183]
    train_mask = train_mask[train_index] # [61603] -> [10183]
    y_train = y_train[train_index] # [61603,20] -> [10183,20]

    ##modify
    vocab_index = np.where(vocab_mask)[0]  # [42757]
    y_vocab = y_vocab[vocab_index] # [42757,20]
    tmp_index = list(train_index) + list(vocab_index)  # [52940]
    train_index = tmp_index # modify
    # adj_train = adj[train_index, :][:, tmp_index] # [10183,52940]
    # adj_train_vocab = adj[tmp_index, :][:, tmp_index] # [52940,52940]
    adj_train = adj[tmp_index, :][:, tmp_index] # [52940,52940]
 ####   # y_train = y_train + y_vocab # [10183]+[42757]
    print('y_vocab type', type(y_vocab))
    # y_train = np.vstack(y_train, y_vocab)
    y_train = np.concatenate([y_train, y_vocab], axis=0)  # 按行
    ##

    val_index = np.where(val_mask)[0] # [1131]
    y_val = y_val[val_index] # [61603,20] -> [1131,20]

    test_index = np.where(test_mask)[0] # [7532]
    y_test = y_test[test_index] # [61603,20] -> [7532,20]

    # train_val_index = np.concatenate([train_index, val_index],axis=0) # 10183+1131 = 11000
    # train_test_idnex = np.concatenate([train_index, test_index],axis=0) # 10183+7532 = 12000

    ##modify
    train_val_index = np.concatenate([train_index, val_index],axis=0) # 52940+1131
    train_test_idnex = np.concatenate([train_index, test_index],axis=0) # 52940+7532
    ##

    numNode_train = adj_train.shape[0] # 10183   # 52940
    # print("numNode", numNode)


    # if FLAGS.model == 'gcn_mix':
    if FLAGS.model == 'appnp':
        normADJ_train = nontuple_preprocess_adj(adj_train)  # [52940,52940]
        # normADJ = nontuple_preprocess_adj(adj)


        normADJ_val = nontuple_preprocess_adj(adj[train_val_index,:][:,train_val_index])   #[53000,53000]
        normADJ_test = nontuple_preprocess_adj(adj[train_test_idnex,:][:,train_test_idnex]) #[54000,54000]

        num_supports = 2
        model_func = APPNP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Some preprocessing
    features = nontuple_preprocess_features(features).todense()  #[61603, 61603]

    train_features = normADJ_train.dot(features[train_index]) # [52940,52940]*[52940,61603]->[52940, 61603]
    val_features = normADJ_val.dot(features[train_val_index]) # [53000,53000]*[53000,61603]->[53000,61603]
    test_features = normADJ_test.dot(features[train_test_idnex]) # [54000,54000]*[54000,61603]->[54000,61603]

    nonzero_feature_number = len(np.nonzero(features)[0])
    nonzero_feature_number_train = len(np.nonzero(train_features)[0])


    # Define placeholders
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32) ,
        'AXfeatures': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features.shape[-1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feeddict_forMixlayers(features, support, labels, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    cost_val = []

    p0 = column_prop(normADJ_train)

    # testSupport = [sparse_to_tuple(normADJ), sparse_to_tuple(normADJ)]
    valSupport = sparse_to_tuple(normADJ_val[len(train_index):, :]) # [52940:,:]
    testSupport = sparse_to_tuple(normADJ_test[len(train_index):, :]) #[52940:,:]

    t = time.time()
    maxACC = 0.0
    # Train model
    for epoch in range(FLAGS.epochs):
        t1 = time.time()

        n = 0
        for batch in iterate_minibatches_listinputs([normADJ_train, y_train], batchsize=256, shuffle=True):
            [normADJ_batch, y_train_batch] = batch

            p1 = column_prop(normADJ_batch)
            if rank1 is None:
                support1 = sparse_to_tuple(normADJ_batch)
                features_inputs = train_features
            else:

                q1 = np.random.choice(np.arange(numNode_train), rank1, replace=False, p=p1)  # top layer

                support1 = sparse_to_tuple(normADJ_batch[:, q1].dot(sp.diags(1.0 / (p1[q1] * rank1))))

                features_inputs = train_features[q1, :]  # selected nodes for approximation
            # Construct feed dictionary
            feed_dict = construct_feeddict_forMixlayers(features_inputs, support1, y_train_batch,
                                            placeholders)  #[600,61603] [batch,600]
            # X1W1 [600,61603][61603,200]->[600,200]
            # A(X1W1)W2 [batch,600][600,200][200,20]->[batch,20]
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            n = n +1


        # Validation
        cost, acc, duration = evaluate(val_features, valSupport, y_val,  placeholders)
        cost_val.append(cost)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t1))

        # if epoch > 50 and acc>maxACC:
        #     maxACC = acc
        #     save_path = saver.save(sess, "tmp/tmp_MixModel.ckpt")

        # Print results
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
        #       "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
        #       "val_acc=", "{:.5f}".format(acc), "time per batch=", "{:.5f}".format((time.time() - t1)/n))

        if epoch > FLAGS.early_stopping and np.mean(cost_val[-2:]) > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            # print("Early stopping...")
            break


    train_duration = time.time() - t
    # Testing
    # if os.path.exists("tmp/pubmed_MixModel.ckpt"):
    #     saver.restore(sess, "tmp/pubmed_MixModel.ckpt")
    test_cost, test_acc, test_duration = evaluate(test_features, testSupport, y_test,
                                                  placeholders)
    print("rank1 = {}".format(rank1), "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "training time per epoch=", "{:.5f}".format(train_duration/(epoch+1)),
          "test time=", "{:.5f}".format(test_duration))

if __name__=="__main__":
    print("DATASET:", FLAGS.dataset)
    # for k in [25, 50, 100, 200, 400]:
    #     main(k)
    for k in [600]:
        main(k)