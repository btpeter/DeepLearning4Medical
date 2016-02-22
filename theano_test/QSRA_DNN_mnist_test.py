import theano
from theano import tensor as T
import numpy as np 
import input_data

PARAM_NON_ZEROS_CUTOFF = 0

OUTPUT = "/mnt/DeepLearning4Medical/Theano_test_output/mnist_output.txt"
out_object = open(OUTPUT, 'w')

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def init_bias(num):
    b_values = np.zeros((num, ), dtype=theano.config.floatX)
    return theano.shared(value=b_values, borrow=True)

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

### Gradient with momentum and weight decay ###

def gradient_updates_momentum_L2(cost, params, learning_rate, momentum, weight_cost_strength):
	# Make sure momentum is a sane value
	assert momentum < 1 and momentum >= 0
	# List of update steps for each parameter
	updates = []
	# Just gradient descent on cost
	for param in params:
		param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
		updates.append((param, param - (learning_rate*param_update + weight_cost_strength * param_update)))
		updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
	return updates



#### dropout ####

srng = T.shared_randomstreams.RandomStreams()

def drop(input_value, dropout):
	if T.gt(dropout, 0.):
		retain_prob = 1 - dropout
		mask = srng.binomial(n=1, p=retain_prob, size=input_value.shape, dtype='floatX')
		return input_value * mask / retain_prob
	else:
		return input_value

#################

def model(X, w1, b1, w2, b2, w3, b3, w4, b4, wo, bo):
	pre_h1 = T.nnet.relu(T.dot(X, w1)) + b1
	h1 = drop(pre_h1, 0.25)

	pre_h2 = T.nnet.relu(T.dot(h1, w2)) + b2
	h2 = drop(pre_h2, 0.25)

	pre_h3 = T.nnet.relu(T.dot(h2, w3)) + b3
	h3 = drop(pre_h3, 0.25)

	pre_h4 = T.nnet.relu(T.dot(h3, w4)) + b4
	h4 = drop(pre_h4, 0.1)

	pyx = T.nnet.softmax(T.dot(h4, wo)+bo)

	return pyx


#### Loading data sets #####


############################
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

############################

X = T.fmatrix()
Y = T.fmatrix()


w1 = init_weights((784,4000))
b1 = init_bias(4000)

w2 = init_weights((4000,2000))
b2 = init_bias(2000)

w3 = init_weights((2000,1000))
b3 = init_bias(1000)

w4 = init_weights((1000,1000))
b4 = init_bias(1000)

wo = init_weights((1000,10))
bo = init_bias(10)

y_ = model(X, w1, b1, w2, b2, w3, b3, w4, b4, wo, bo)
#p_y = y_
p_y = T.argmax(y_, axis=1)

#cost = T.mean(T.sqr(y_ - Y))
cost = T.mean(T.nnet.categorical_crossentropy(y_, Y))

params = [w1, b1, w2, b2, w3, b3, w4, b4, wo, bo]
#updates = sgd(cost, params)
updates = gradient_updates_momentum_L2(cost, params, 0.05, 0.9, 0.0001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=p_y, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    predict_result = np.mean(np.argmax(teY, axis=1) == predict(teX))
    out_object.write(str(predict_result)+"\r\n")
    out_object.flush()

out_object.close()








