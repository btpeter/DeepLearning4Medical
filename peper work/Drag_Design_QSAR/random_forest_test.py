import sys
import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle
import data_loader

data_file_train = "/Users/peter/Documents/Work/data/drag_design/NK1_training_disguised.csv"
data_file_test = "/Users/peter/Documents/Work/data/drag_design/NK1_test_disguised.csv"

drag_data = data_loader.read_data_sets(data_file_train, data_file_test, 1000)
trX, trY, teX, teY = drag_data.train.descriptors, drag_data.train.activities, drag_data.test.descriptors, drag_data.test.activities


# Random Forest Classifier
def random_forest_regressor(train_x, train_y):
	from sklearn.ensemble import RandomForestRegressor
	model = RandomForestRegressor(n_estimators=200)
	model.fit(train_x, train_y)
	return model


if __name__ == '__main__':
	num_train, num_feat = trX.shape
	num_test, num_feat = teX.shape

	model = random_forest_regressor(trX, trY)
	predict = model.predict(teX)

	#accuracy = metrics.accuracy_score(teY, predict)

	R2 = data_loader.R2(np.array(predict), teY)
	print "R : ", R2
	