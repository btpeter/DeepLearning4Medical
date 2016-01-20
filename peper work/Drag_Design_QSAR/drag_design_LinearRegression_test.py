import tensorflow as tf
import numpy as np
import data_loader

LOG_ADDRESS="/mnt/DeepLearning4Medical/tensorflow_log/drag_design"
#LOG_ADDRESS="/Users/peter/Documents/Work/DL4Medical_WorkTest/work_code/tensorflow_log/drag_design"
TRAINED_MODEL_ADDRESS="/mnt/DeepLearning4Medical/trained_model/drag_design/NK1_test_10"
#TRAINED_MODEL_ADDRESS="/Users/peter/Documents/Work/DL4Medical_WorkTest/work_code/trained_model/drag_design/test1"


CONSOLE_OUTPUT="/mnt/DeepLearning4Medical/console_output/NK1_test_10.txt"
STATISTIC_RESULT="/mnt/DeepLearning4Medical/statistic_result/NK1_test_10.txt"


NUM_CORES = 10


# set random seed
#tf.set_random_seed(5)

def init_weight(shape, name):
	return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01), name)

def init_bias(dim, name):
	return tf.Variable(tf.zeros([dim]), name)

def build_hidden_layer_ReLU(pre_layer, weight, bias):
	return tf.nn.relu(tf.matmul(pre_layer, weight)+bias)

def add_dropout(layer, p_drop_hidden):
	return tf.nn.dropout(layer, p_drop_hidden)

def build_model(X, w1, b1, w2, b2, w3, b3, w4, b4, wo, bo):
	pre_h1 = build_hidden_layer_ReLU(X, w1, b1)

	# input with no dropout
	h1 = add_dropout(pre_h1, 0.75)
	#h1 = pre_h1

	pre_h2 = build_hidden_layer_ReLU(h1, w2, b2)
	h2 = add_dropout(pre_h2, 0.75)

	pre_h3 = build_hidden_layer_ReLU(h2, w3, b3)
	h3 = add_dropout(pre_h3, 0.75)

	pre_h4 = build_hidden_layer_ReLU(h3, w4, b4)
	h4 = add_dropout(pre_h4, 0.9)

	model = tf.matmul(h4, wo)+bo
	#model = add_dropout(model, 0.9)

	return model


'''
	Load Dataset



# file address in docker envirement
data_file_train = "/mnt/DeepLearning4Medical/data/drag_design/NK1_training_disguised.csv"
data_file_test = "/mnt/DeepLearning4Medical/data/drag_design/NK1_test_disguised.csv"

#data_file_train = "/Users/peter/Documents/Work/data/drag_design/METAB_training_disguised.csv"
#data_file_test = "/Users/peter/Documents/Work/data/drag_design/METAB_test_disguised.csv"

# fill the file_dir
drag_data = data_loader.read_data_sets(data_file_train, data_file_test)
trX, trY, teX, teY = drag_data.train.descriptors, drag_data.train.activities, drag_data.test.descriptors, drag_data.test.activities
num_features = drag_data.train.num_features


print "NUM OF FEATURE: ", num_features

'''

tf.set_random_seed(1)

trX_temp = np.linspace(-1, 1, 101)

trX = np.array([1])
for x in trX_temp:
	x = np.array([x])
	trX = np.vstack((trX, x))

trY = trX * 2 + np.random.randn(*trX.shape) * 0.3 + 2 * np.random.rand() * trX.shape[0]
teX = trX
teY = trY

num_features = 1

'''
	Begin train
'''

# fill the dims
X = tf.placeholder("float", [None, num_features])
Y = tf.placeholder("float")


w1 = init_weight([num_features, 4000], "w_hidden_1")
b1 = init_bias(4000, "bias_1")

w2 = init_weight([4000, 2000], "w_hidden_2")
b2 = init_bias(2000, "bias_2")

w3 = init_weight([2000, 1000], "w_hidden_3")
b3 = init_bias(1000, "bias_3")

w4 = init_weight([1000, 1000], "w_hidden_4")
b4 = init_bias(1000, "bias_4")

# fill the out dim
wo = init_weight([1000, 1], "w_out")
bo = init_bias(1, "bias_out")


y_ = build_model(X, w1, b1, w2, b2, w3, b3, w4, b4, wo, bo)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, Y))
cost = tf.reduce_mean(tf.pow(Y-y_, 2))

#train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
train_op = tf.train.MomentumOptimizer(0.05, 0.9).minimize(cost)

# Add ops to save and restore all the variables
saver = tf.train.Saver()
predict_op = y_



init = tf.initialize_all_variables()
with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)) as sess:
	sess.run(init)

	for epoch in range(1000):
		print "Training in epoch: ", epoch
		sess.run(train_op, feed_dict={X: trX, Y: trY})
		cost_log = sess.run(cost, feed_dict={X: teX, Y: teY})
		print "Cost", cost_log
			

# predict option
	#batch_xs, batch_ys = drag_data.test.next_batch(128)
	predict_result = sess.run(predict_op, feed_dict={X: teX, Y: teY})
	print predict_result
	print "\n\nTRUE: \n"
	print teY


# evaluate with r^2
	R2 = data_loader.R2(np.array(predict_result), teY)
	print "R2 : ", R2




