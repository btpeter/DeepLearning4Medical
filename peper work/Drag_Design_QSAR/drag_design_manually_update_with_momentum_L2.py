import tensorflow as tf
import numpy as np
import data_loader
import sys
from parameterservermodel import ParameterServerModel


# system_parameters
PARAM_TEST_FILE_NAME = sys.argv[1]
PARAM_TEST_OUTPUT_FILE_NAME = sys.argv[2]
PARAM_LEARNING_RATE = float(sys.argv[3])
PARAM_MOMENTUM = float(sys.argv[4])
PARAM_COST_WEIGHT_STRENGTH = float(sys.argv[5])
PARAM_RANDOM_SEED = int(sys.argv[6])


PARAM_HIDDEN_LAYER_1 = int(sys.argv[7])
PARAM_HIDDEN_LAYER_2 = int(sys.argv[8])
PARAM_HIDDEN_LAYER_3 = int(sys.argv[9])
PARAM_HIDDEN_LAYER_4 = int(sys.argv[10])


#PARAM_NON_ZEROS_CUTOFF = int(sys.argv[2])
PARAM_NON_ZEROS_CUTOFF = 0

LOG_ADDRESS="/mnt/DeepLearning4Medical/tensorflow_log/drag_design"
#LOG_ADDRESS="/Users/peter/Documents/Work/DL4Medical_WorkTest/work_code/tensorflow_log/drag_design"
TRAINED_MODEL_ADDRESS="/mnt/DeepLearning4Medical/trained_model/drag_design/160222/"+PARAM_TEST_OUTPUT_FILE_NAME
#TRAINED_MODEL_ADDRESS="/Users/peter/Documents/Work/DL4Medical_WorkTest/work_code/trained_model/drag_design/test1"


COST_STEP_OUTPUT="/mnt/DeepLearning4Medical/console_output/160222/"+PARAM_TEST_OUTPUT_FILE_NAME+".step_cost.txt"
COST_EPOCH_OUTPUT="/mnt/DeepLearning4Medical/console_output/160222/"+PARAM_TEST_OUTPUT_FILE_NAME+".epoch_cost.txt"
STATISTIC_RESULT_TEST="/mnt/DeepLearning4Medical/statistic_result/160222/"+PARAM_TEST_OUTPUT_FILE_NAME+"_test.txt"
STATISTIC_RESULT_TRAINING="/mnt/DeepLearning4Medical/statistic_result/160222/"+PARAM_TEST_OUTPUT_FILE_NAME+"_train.txt"


NUM_CORES = 5


# set random seed
tf.set_random_seed(PARAM_RANDOM_SEED)

def init_weight(shape, name):
	return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01), name)

def init_bias(dim, name):
	return tf.Variable(tf.random_normal([dim], mean=0.0, stddev=0.01), name)

def build_hidden_layer_ReLU(pre_layer, weight, bias):
	return tf.nn.relu(tf.add(tf.matmul(pre_layer, weight), bias))

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
	#h4 = add_dropout(pre_h4, 0.75)
	h4 = add_dropout(pre_h4, 0.9)

	model = tf.add(tf.matmul(h4, wo), bo)
	#model = add_dropout(model, 0.9)

	return model


'''
	Load Dataset
'''


# file address in docker envirement
data_file_train = "/mnt/DeepLearning4Medical/data/drag_design/"+PARAM_TEST_FILE_NAME+"_training_disguised.csv"
data_file_test = "/mnt/DeepLearning4Medical/data/drag_design/"+PARAM_TEST_FILE_NAME+"_test_disguised.csv"

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


w1 = init_weight([num_features, PARAM_HIDDEN_LAYER_1], "w_hidden_1")
b1 = init_bias(PARAM_HIDDEN_LAYER_1, "bias_1")

w2 = init_weight([PARAM_HIDDEN_LAYER_1, PARAM_HIDDEN_LAYER_2], "w_hidden_2")
b2 = init_bias(PARAM_HIDDEN_LAYER_2, "bias_2")

w3 = init_weight([PARAM_HIDDEN_LAYER_2, PARAM_HIDDEN_LAYER_3], "w_hidden_3")
b3 = init_bias(PARAM_HIDDEN_LAYER_3, "bias_3")

w4 = init_weight([PARAM_HIDDEN_LAYER_3, PARAM_HIDDEN_LAYER_4], "w_hidden_4")
b4 = init_bias(PARAM_HIDDEN_LAYER_4, "bias_4")

# fill the out dim
wo = init_weight([PARAM_HIDDEN_LAYER_4, 1], "w_out")
bo = init_bias(1, "bias_out")


y_ = build_model(X, w1, b1, w2, b2, w3, b3, w4, b4, wo, bo)


''' Build variables and var_shape array '''
variables = [w1, b1, w2, b2, w3, b3, w4, b4, wo, bo]
variables_shapes = [[num_features,PARAM_HIDDEN_LAYER_1],[PARAM_HIDDEN_LAYER_1],[PARAM_HIDDEN_LAYER_1,PARAM_HIDDEN_LAYER_2],[PARAM_HIDDEN_LAYER_2],[PARAM_HIDDEN_LAYER_2,PARAM_HIDDEN_LAYER_3],[PARAM_HIDDEN_LAYER_3],[PARAM_HIDDEN_LAYER_3,PARAM_HIDDEN_LAYER_4],[PARAM_HIDDEN_LAYER_4],[PARAM_HIDDEN_LAYER_4,1],[1]]

''' Init cost function'''
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, Y))


''' Init L_x Losses'''
#l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
#l2_loss = (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(wo))

#train_op = tf.train.GradientDescentOptimizer(PARAM_LEARNING_RATE).minimize(cost)
#train_op = tf.train.MomentumOptimizer(PARAM_LEARNING_RATE, momentum=PARAM_MOMENTUM).minimize(cost)
#train_op = tf.train.AdagradOptimizer(PARAM_LEARNING_RATE).minimize(cost)
#train_op = tf.train.AdamOptimizer(PARAM_LEARNING_RATE).minimize(cost)
#train_op = tf.train.FtrlOptimizer(PARAM_LEARNING_RATE).minimize(cost)
#train_op = tf.train.RMSPropOptimizer(PARAM_LEARNING_RATE, decay=PARAM_DECAY, momentum=PARAM_MOMENTUM).minimize(cost)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=PARAM_LEARNING_RATE)
cost = tf.reduce_mean(tf.pow(Y-y_, 2))
compute_gradients = optimizer.compute_gradients(cost, variables)
apply_gradients = optimizer.apply_gradients(compute_gradients)
minimize = optimizer.minimize(cost)
correct_prediction = y_
accuracy = y_
error_rate = cost
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES))



# Add ops to save and restore all the variables
saver = tf.train.Saver()
predict_op = y_



''' Init output objects '''
log_step_cost_object = open(COST_STEP_OUTPUT, 'w')
log_epoch_cost_object = open(COST_EPOCH_OUTPUT, 'w')
statistic_file_test_object = open(STATISTIC_RESULT_TEST, 'w')
statistic_file_train_object = open(STATISTIC_RESULT_TRAINING, 'w')


''' console model parameters '''
log_step_cost_object.write("Learning rate: "+str(PARAM_LEARNING_RATE)+"  Momentum:  "+str(PARAM_MOMENTUM)+"  Cost Weight Strength:  "+str(PARAM_COST_WEIGHT_STRENGTH)+"  Random Seed:  "+str(PARAM_RANDOM_SEED)+"\r\n")
log_step_cost_object.write("Layer_1: "+str(PARAM_HIDDEN_LAYER_1)+"  Layer_2:  "+str(PARAM_HIDDEN_LAYER_2)+"  Layer_3:  "+str(PARAM_HIDDEN_LAYER_3)+"  Layer_4:  "+str(PARAM_HIDDEN_LAYER_4)+"\r\n")
log_epoch_cost_object.write("Learning rate: "+str(PARAM_LEARNING_RATE)+"  Momentum:  "+str(PARAM_MOMENTUM)+"  Cost Weight Strength:  "+str(PARAM_COST_WEIGHT_STRENGTH)+"  Random Seed:  "+str(PARAM_RANDOM_SEED)+"\r\n")
log_epoch_cost_object.write("Layer_1: "+str(PARAM_HIDDEN_LAYER_1)+"  Layer_2:  "+str(PARAM_HIDDEN_LAYER_2)+"  Layer_3:  "+str(PARAM_HIDDEN_LAYER_3)+"  Layer_4:  "+str(PARAM_HIDDEN_LAYER_4)+"\r\n")
statistic_file_test_object.write("Learning rate: "+str(PARAM_LEARNING_RATE)+"  Momentum:  "+str(PARAM_MOMENTUM)+"  Cost Weight Strength:  "+str(PARAM_COST_WEIGHT_STRENGTH)+"  Random Seed:  "+str(PARAM_RANDOM_SEED)+"\r\n")
statistic_file_test_object.write("Layer_1: "+str(PARAM_HIDDEN_LAYER_1)+"  Layer_2:  "+str(PARAM_HIDDEN_LAYER_2)+"  Layer_3:  "+str(PARAM_HIDDEN_LAYER_3)+"  Layer_4:  "+str(PARAM_HIDDEN_LAYER_4)+"\r\n")
statistic_file_train_object.write("Learning rate: "+str(PARAM_LEARNING_RATE)+"  Momentum:  "+str(PARAM_MOMENTUM)+"  Cost Weight Strength:  "+str(PARAM_COST_WEIGHT_STRENGTH)+"  Random Seed:  "+str(PARAM_RANDOM_SEED)+"\r\n")
statistic_file_train_object.write("Layer_1: "+str(PARAM_HIDDEN_LAYER_1)+"  Layer_2:  "+str(PARAM_HIDDEN_LAYER_2)+"  Layer_3:  "+str(PARAM_HIDDEN_LAYER_3)+"  Layer_4:  "+str(PARAM_HIDDEN_LAYER_4)+"\r\n")

merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(LOG_ADDRESS, tf.Graph.as_graph_def(sess.graph))

p = ParameterServerModel(X, Y, PARAM_MOMENTUM, PARAM_COST_WEIGHT_STRENGTH, variables_shapes, compute_gradients, apply_gradients, minimize, error_rate, sess, 128)

for epoch in range(350):
	#print "Training in epoch: ", epoch
	#log_step_cost_object.write("Training in epoch: "+str(epoch)+"\r\n")
	for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
		minibatch_features = trX[start:end]
		minibatch_activity = trY[start:end]
		p.train(minibatch_activity, minibatch_features)
		p.apply(p.gradients)
		step_cost = sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end]})
		log_step_cost_object.write(str(step_cost)+"\r\n")
		log_step_cost_object.flush()
	epoch_cost = sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end]})
	log_epoch_cost_object.write(str(epoch_cost)+"\r\n")
	log_epoch_cost_object.flush()
	predict_result = sess.run(predict_op, feed_dict={X: teX, Y: teY})
	R2 = data_loader.R2(np.array(predict_result), teY)
	statistic_file_test_object.write(str(R2)+"\r\n")
	statistic_file_test_object.flush()
	predict_result_2 = sess.run(predict_op, feed_dict={X: trX, Y: trY})
	R2 = data_loader.R2(np.array(predict_result_2), trY)
	statistic_file_train_object.write(str(R2)+"\r\n")
	statistic_file_train_object.flush()


# predict option
	#batch_xs, batch_ys = drag_data.test.next_batch(128)
predict_result_temp = sess.run(predict_op, feed_dict={X: teX, Y: teY})
	#print predict_result

predict_result = []
for pred in predict_result_temp:
	predict_result.append(pred[0])

	# write file
statistic_file_test_object.write("\r\nFinal  : \r\n")
statistic_file_test_object.write("True Activities: \r\n")
for true_y in teY:
	statistic_file_test_object.write(str(true_y)+"\r\n")
statistic_file_test_object.write("Predict Activities: \r\n")
for predict_y in predict_result:
	statistic_file_test_object.write(str(predict_y)+"\r\n")

# evaluate with r^2
R2 = data_loader.R2(np.array(predict_result), teY)
	#print "R2 : ", R2

	# write file
statistic_file_test_object.write("R2 : \r\n")
statistic_file_test_object.write(str(R2)+"\r\n")

	# Save the variables to disk
save_path = saver.save(sess, TRAINED_MODEL_ADDRESS)
	#print "Model saved in file: ", save_path


log_step_cost_object.close()
log_epoch_cost_object.close()
statistic_file_test_object.close()
statistic_file_train_object.close()



