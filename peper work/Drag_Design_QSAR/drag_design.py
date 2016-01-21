import tensorflow as tf
import numpy as np
import data_loader
import sys


# system_parameters
PARAM_TEST_OUTPUT_FILE_NAME = sys.argv[1]
PARAM_LEARNING_RATE = float(sys.argv[2])
#PARAM_MOMENTUM = float(sys.argv[2])
#PARAM_NON_ZEROS_CUTOFF = int(sys.argv[2])
PARAM_NON_ZEROS_CUTOFF = 0

LOG_ADDRESS="/mnt/DeepLearning4Medical/tensorflow_log/drag_design"
#LOG_ADDRESS="/Users/peter/Documents/Work/DL4Medical_WorkTest/work_code/tensorflow_log/drag_design"
TRAINED_MODEL_ADDRESS="/mnt/DeepLearning4Medical/trained_model/drag_design/160121/"+PARAM_TEST_OUTPUT_FILE_NAME+"_test"
#TRAINED_MODEL_ADDRESS="/Users/peter/Documents/Work/DL4Medical_WorkTest/work_code/trained_model/drag_design/test1"


CONSOLE_OUTPUT="/mnt/DeepLearning4Medical/console_output/160121/"+PARAM_TEST_OUTPUT_FILE_NAME+"_test2.txt"
STATISTIC_RESULT="/mnt/DeepLearning4Medical/statistic_result/160121/"+PARAM_TEST_OUTPUT_FILE_NAME+"_test2.txt"


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
'''


# file address in docker envirement
data_file_train = "/mnt/DeepLearning4Medical/data/drag_design/"+PARAM_TEST_OUTPUT_FILE_NAME+"_training_disguised.csv"
data_file_test = "/mnt/DeepLearning4Medical/data/drag_design/"+PARAM_TEST_OUTPUT_FILE_NAME+"_test_disguised.csv"

#data_file_train = "/Users/peter/Documents/Work/data/drag_design/METAB_training_disguised.csv"
#data_file_test = "/Users/peter/Documents/Work/data/drag_design/METAB_test_disguised.csv"

# fill the file_dir
drag_data = data_loader.read_data_sets(data_file_train, data_file_test, PARAM_NON_ZEROS_CUTOFF)
trX, trY, teX, teY = drag_data.train.descriptors, drag_data.train.activities, drag_data.test.descriptors, drag_data.test.activities
num_features = drag_data.train.num_features


#print "NUM OF FEATURE: ", num_features


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

train_op = tf.train.GradientDescentOptimizer(PARAM_LEARNING_RATE).minimize(cost)
#train_op = tf.train.MomentumOptimizer(0.05, PARAM_MOMENTUM).minimize(cost)
#train_op = tf.train.AdagradOptimizer(PARAM_LEARNING_RATE).minimize(cost)
#train_op = tf.train.AdamOptimizer(PARAM_LEARNING_RATE).minimize(cost)
#train_op = tf.train.FtrlOptimizer(PARAM_LEARNING_RATE).minimize(cost)
#train_op = tf.train.RMSPropOptimizer(PARAM_LEARNING_RATE, 0.9).minimize(cost)

# Add ops to save and restore all the variables
saver = tf.train.Saver()
predict_op = y_



# Create output log && statistic file objects
log_file_object = open(CONSOLE_OUTPUT, 'w')
#print "Write :", CONSOLE_OUTPUT
statistic_file_object = open(STATISTIC_RESULT, 'w')
#print "Write :", STATISTIC_RESULT




init = tf.initialize_all_variables()
with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)) as sess:
	sess.run(init)

	merged_summary_op = tf.merge_all_summaries()
	summary_writer = tf.train.SummaryWriter(LOG_ADDRESS, tf.Graph.as_graph_def(sess.graph))

	for epoch in range(350):
		#print "Training in epoch: ", epoch
		#log_file_object.write("Training in epoch: "+str(epoch)+"\r\n")
		for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
			sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
			cost_log = sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end]})
			log_file_object.write(str(cost_log)+"\r\n")
		log_file_object.flush()

		predict_result = sess.run(predict_op, feed_dict={X: teX, Y: teY})
		statistic_file_object.write("Training in epoch: "+str(epoch)+"\r\n")
		statistic_file_object.write("R2 : \r\n")
		R2 = data_loader.R2(np.array(predict_result), teY)
		statistic_file_object.write(str(R2)+"\r\n")
		statistic_file_object.flush()


# predict option
	#batch_xs, batch_ys = drag_data.test.next_batch(128)
	predict_result_temp = sess.run(predict_op, feed_dict={X: teX, Y: teY})
	#print predict_result

	predict_result = []
	for pred in predict_result_temp:
		predict_result.append(pred[0])

	# write file
	statistic_file_object.write("\r\nFinal  : \r\n")
	statistic_file_object.write("True Activities: \r\n")
	for true_y in teY:
		statistic_file_object.write(str(true_y)+"\r\n")
	statistic_file_object.write("Predict Activities: \r\n")
	for predict_y in predict_result:
		statistic_file_object.write(str(predict_y)+"\r\n")

# evaluate with r^2
	R2 = data_loader.R2(np.array(predict_result), teY)
	#print "R2 : ", R2

	# write file
	statistic_file_object.write("R2 : \r\n")
	statistic_file_object.write(str(R2)+"\r\n")

	# Save the variables to disk
	save_path = saver.save(sess, TRAINED_MODEL_ADDRESS)
	#print "Model saved in file: ", save_path


log_file_object.close()
#print "Close :", CONSOLE_OUTPUT
statistic_file_object.close()
#print "Close :", STATISTIC_RESULT




