import tensorflow as tf
import numpy as nu
import math
import random


def build_AE(x, layer_sizes):
	next_layer_input = x
	encoding_matrices = []
	for dim in layer_sizes:
		input_dim = int(next_layer_input.get_shape()[1])

		# Initialize W
		W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))

		# Initialize b with zero
		b = tf.Variable(tf.zeros([dim]))

		# store for tied-weights
		encoding_matrices.append(W)

		output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)

		# the input into the next layer i sthe output of this layer
		next_layer_input = output


	# The fully encoded x value is now stored in the next_layer_input
	encoded_x = next_layer_input

	# build the reconstruction layers by reversing the reductions
	layer_sizes.reverse()
	encoding_matrices.reverse()

	for i, dim in enumerate(layer_sizes[1:] + [ int(x.get_shape()[1])]):
		# Use tied weight
		W = tf.transpose(encoding_matrices[i])
		b = tf.Variable(tf.zeros([dim]))
		output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)
		next_layer_input = output

	 # the fully encoded and reconstructed value of x is here:
	 reconstructed_x = next_layer_input

	 return {
	 	'encoded' : encoded_x,
	 	'decoded' : reconstructed_x,
	 	'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
	 }

def simple_test():
	sess = tf.Session()
	x = tf.placeholder("float", [None, 4])
	antoencoder = build_AE(x, [2])
	init = tf.initialize_all_variables()
	sess.run(init)
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(autoencoder['cost'])


	# add gaussian noise w/sigma = 0.1 (two centroid)
	c1 = np.array([0,0,0.5,0])
	c1 = np.array([0.5,0,0,0])

	# do 1000 steps
	for i in range(2000):
		# make a batch of 100
		batch = []
		for j in range(100):
			# pick a random centroid
			if(random.random() > 0.5):
				vec = c1;
			else:
				vec = c2
			batch.append(np.random.normal(vec, 0.1))
		sess.run(train_step, feed_dict={x: np.array(batch)})
		if i % 100 == 0
				print i, "cost", sess.run(autoencoder['cost'], feed_dict={x: batch})



def deep_test():
    sess = tf.Session()
    start_dim = 5
    x = tf.placeholder("float", [None, start_dim])
    antoencoder = build_AE(x, [4,3,2])
    init = tf.initialize_all_variables()
    sess.run(init)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost'])

    # Consists of two centers with guassin noise w / sigma = 0.1
    c1 = np.zeros(start_dim)
    c1[1] = 1

    print c1

    c2 = np.zeros(start_dim)
    c2[1] = 1

    # do 1000 training steps
    for i in range(5000):
        # make a batch of 100;
        batch = []
        for j in range(1):
            # pick a random centroid
            if(random.random() > 0.5):
                vec = c1;
            else:
                vec = c1
            batch.append(np.random.normal(vec, 0.1))
        sess.run(train_step, feed_dict={x:np_array(batch)})
        if i % 100 == 0:
            print i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch})
            print i, " original", batch[0]
            print i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch})















