import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name)


def model(X, w_h, w_o):
    # h = tf.nn.sigmoid(tf.matmul(X, w_h), name="hfn") # this is a basic mlp, think 2 stacked logistic regressions
    h = tf.matmul(X, w_h)
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784], name="X")
Y = tf.placeholder("float", [None, 10], name="Y")

w_h = init_weights([784, 625], name="hiden_weight") # create symbolic variables
w_o = init_weights([625, 10], name="output_weight")

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
tf.scalar_summary("cost", cost)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/Users/peter/Documents/Work/tensorflow_test/tensorflow_log/test_net_1', tf.Graph.as_graph_def(sess.graph))

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX, Y: teY}))
    # summary_str = sess.run(merged_summary_op)
    # summary_writer.add_summary(summary_str, i)