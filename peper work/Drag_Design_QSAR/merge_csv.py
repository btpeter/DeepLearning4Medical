import tensorflow as tf
import numpy as np
import data_loader



#data_file_train = "/Users/peter/Documents/Work/data/drag_design/NK1_training_disguised.csv"
#data_file_test = "/Users/peter/Documents/Work/data/drag_design/NK1_test_disguised.csv"
data_file_train = "/mnt/DeepLearning4Medical/data/drag_design/NK1_training_disguised.csv"
data_file_test = "/mnt/DeepLearning4Medical/data/drag_design/NK1_test_disguised.csv"


# fill the file_dir
drag_data = data_loader.read_data_sets(data_file_train, data_file_test, 500)
trX, trY, teX, teY = drag_data.train.descriptors, drag_data.train.activities, drag_data.test.descriptors, drag_data.test.activities
num_features = drag_data.train.num_features

'''

OUTPUT = "/mnt/DeepLearning4Medical/data/work_code/DNN/test/NK1_filter.csv"

output_object = open(OUTPUT, "w")

sample_count = 0

for sample in trX:
	for desc in sample:
		output_object.write(str(desc))
		output_object.write(",")
	output_object.write(str(trY[sample_count]))
	output_object.write("\r\n")
	sample_count+=1

output_object.close()

'''
