import tensorflow as tf
import csv
import numpy as np
import math

'''
	load csv data
'''
def readcsv(filename):
	header = []
	activities = []
	descriptors = []
	with open(filename, 'r') as f:
		headings = next(f)
		header = headings.strip('\n').split(',')[2:]
		for line in f:			
			activities.append(float(line.strip('\n').split(',')[1]))
			desc_arr = []
			for desc in line.strip('\n').split(',')[2:]:
				desc_arr.append(float(desc))
			descriptors.append(desc_arr)	

	print "Finished load ",filename

	return header, descriptors, activities


def get_feature_normalization_arr(train_headers, test_headers):

	all_features = train_headers[:]
	all_features.extend(test_headers)
	all_features = list(set(all_features))

	# transform to digits
	all_features_digits = []
	for feature in all_features:
		digit = int(feature.split('_')[1])
		all_features_digits.append(digit)

	# sort features
	all_features_digits.sort()

	# start normalizating 
	# create result list which stored the index of each feature
	train_feature_index_arr = []
	test_feature_index_arr = []

	# normalization train data set
	for header in train_headers:
		digit = int(header.split('_')[1])
		for index in range(len(all_features_digits)):
			if digit == all_features_digits[index]:
				train_feature_index_arr.append(index)
				break


	# normalization test data set
	for header in test_headers:
		digit = int(header.split('_')[1])
		for index in range(len(all_features_digits)):
			if digit == all_features_digits[index]:
				test_feature_index_arr.append(index)
				break

	return len(all_features_digits), train_feature_index_arr, test_feature_index_arr

def contract_data_set(train_file, test_file):
	headers_train, descriptors_train_raw, activities_train = readcsv(train_file)
	headers_test, descriptors_test_raw, activities_test = readcsv(test_file)

	descriptors_train_raw = np.array(descriptors_train_raw)
	descriptors_test_raw = np.array(descriptors_test_raw)

	num_examples_train = descriptors_train_raw.shape[0]
	num_examples_test = descriptors_test_raw.shape[0]

	# normalization features
	num_feature, train_feature_index_arr, test_feature_index_arr = get_feature_normalization_arr(headers_train, headers_test)

	descriptors_train = np.zeros([num_examples_train, num_feature])
	descriptors_test = np.zeros([num_examples_test, num_feature])

	# build train_descriptors
	raw_count = 0
	for index in train_feature_index_arr:
		descriptors_train[:,index] = descriptors_train_raw[:,raw_count]
		raw_count+=1


	raw_count = 0
	# build test_descriptors
	for index in test_feature_index_arr:
		descriptors_test[:,index] = descriptors_test_raw[:,raw_count]
		raw_count+=1

	print "Normalization Done."
	return descriptors_train, activities_train, descriptors_test, activities_test



class DataSet(object):
	def __init__(self, descriptors, activities):
		assert descriptors.shape[0] == activities.shape[0], ("descriptors.shape: %s activities.shape: %s" % (descriptors.shape, activities.shape))
		self._num_examples = descriptors.shape[0]
		self._num_features = descriptors.shape[1]
		self._descriptors = descriptors
		self._activities = activities
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def descriptors(self):
		return self._descriptors
	
	@property 
	def activities(self):
		return self._activities

	@property 
	def num_examples(self):
		return self._num_examples

	@property 
	def num_features(self):
		return self._num_features

	@property 
	def num_classes(self):
		return self._num_classes

	@property 
	def epochs_completed(self):
		return self._epochs_completed


	def next_batch(self, batch_size):
		'''	Return the next batch_size examples from this data set.'''
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._descriptors = self._descriptors[perm]
			self._activities = self._activities[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._descriptors[start:end], self._activities[start:end]

def read_data_sets(file_dir_train, file_dir_test):
	class DataSets(object):
		pass
	data_sets = DataSets()

	'''
		load data files && format(log transform)
	'''

	train_descriptors_raw, train_activities_raw, test_descriptors_raw, test_activities_raw = contract_data_set(file_dir_train, file_dir_test)

	#train_descriptors = np.array(train_descriptors_raw)
	train_descriptors = np.array(log_transform_matrix(train_descriptors_raw))
	train_activities = zero_mean_and_unit_variance(train_activities_raw)
	#train_activities_one_hot = activity_format(train_activities_log, CLASS_NUM)
	
	#train_activities_one_hot = activity_format(train_activities_raw, CLASS_NUM)
	
	#train_activities = np.array(train_activities_one_hot)

	
	#test_descriptors = np.array(test_descriptors_raw)
	test_descriptors = np.array(log_transform_matrix(test_descriptors_raw))
	test_activities = zero_mean_and_unit_variance(test_activities_raw)
	#test_activities_one_hot = activity_format(test_activities_log, CLASS_NUM)
	
	#test_activities_one_hot = activity_format(test_activities_raw, CLASS_NUM)
	
	
	#test_activities = np.array(test_activities_one_hot)

	data_sets.train = DataSet(train_descriptors, train_activities)
	
	data_sets.test = DataSet(test_descriptors, test_activities)
	

	return data_sets



'''
	Matrix log transform
'''

def log_transform_matrix(matrix):
	matrix_log10_result = []

	for row in matrix:
		row_log = []
		row_max = max(row)
		row_max_log = math.log10(row_max + 1)
		for ele in row:
			ele_log = (math.log10(ele + 1))/row_max_log				
			row_log.append(ele_log)
		matrix_log10_result.append(row_log)

	print "Matrix Log Transform Done."
	return matrix_log10_result

def log_transform_vector(vector):
	vector_log10_result = []

	for ele in vector:
		vector_log10_result.append(math.log10(ele + 1))

	print "Vector Log Transform Done."
	return vector_log10_result


'''
	Because activities is a 1-D verctor
	We sort the array and then divide into 100 classes instead of using K-Means

'''


def zero_mean_and_unit_variance(data_list):
	data_list = np.array(data_list)
	mean = data_list.mean()
	std = data_list.std()
	bias = data_list - mean

	print "Zeros mean and unit variance Done."
	return bias/std

'''
	Calculate R^2 
'''
def R2(y_test, y_true):
	return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_test.mean())**2).sum()


def activity_format(data_list, num_class):
	data_list_temp = data_list[:]	# copy the array
	class_jenks_breaks = getClassJenksBreaks(data_list_temp, num_class)
	act_class_result = []
	for act in data_list:
		act_class_result.append(classify(act, class_jenks_breaks))
	
	# build one-hot matrix
	act_one_hot_matrix = np.zeros(num_class)
	for act in act_class_result:
		act_example = np.zeros(num_class)
		act_example[act-1] = 1
		act_one_hot_matrix = np.vstack((act_one_hot_matrix, act_example))

	# delete the first row
	return act_one_hot_matrix[1:]

def getClassJenksBreaks( dataList, numClass ):
  dataList.sort()
  mat1 = []
  for i in range(0,len(dataList)+1):
    temp = []
    for j in range(0,numClass+1):
      temp.append(0)
    mat1.append(temp)
  mat2 = []
  for i in range(0,len(dataList)+1):
    temp = []
    for j in range(0,numClass+1):
      temp.append(0)
    mat2.append(temp)
  for i in range(1,numClass+1):
    mat1[1][i] = 1
    mat2[1][i] = 0
    for j in range(2,len(dataList)+1):
      mat2[j][i] = float('inf')
  v = 0.0
  for l in range(2,len(dataList)+1):
    s1 = 0.0
    s2 = 0.0
    w = 0.0
    for m in range(1,l+1):
      i3 = l - m + 1
      val = float(dataList[i3-1])
      s2 += val * val
      s1 += val
      w += 1
      v = s2 - (s1 * s1) / w
      i4 = i3 - 1
      if i4 != 0:
        for j in range(2,numClass+1):
          if mat2[l][j] >= (v + mat2[i4][j - 1]):
            mat1[l][j] = i3
            mat2[l][j] = v + mat2[i4][j - 1]
    mat1[l][1] = 1
    mat2[l][1] = v
  k = len(dataList)
  kclass = []
  for i in range(0,numClass+1):
    kclass.append(0)
  kclass[numClass] = float(dataList[len(dataList) - 1])
  countNum = numClass
  while countNum >= 2:#print "rank = " + str(mat1[k][countNum])
    id = int((mat1[k][countNum]) - 2)
    #print "val = " + str(dataList[id])
    kclass[countNum - 1] = dataList[id]
    k = int((mat1[k][countNum] - 1))
    countNum -= 1
  return kclass

def classify(value, breaks):
  for i in range(1, len(breaks)):
    if value < breaks[i]:
      return i
  return len(breaks) - 1


def log_transform_2(matrix):
	matrix_log10_result = []
	#matrix_log10_result = np.array([])
	for row in matrix:
		row_max = np.max(row)
		if row_max == 0:
			#np.append(matrix_log10_result, row, axis=0)			
			matrix_log10_result.append(row)
			continue
		else:
			row_max_log10 = np.log10(row_max)
			if row_max_log10 == 0:
				row_log10_transform = np.log10(row)
				#np.append(matrix_log10_result, row_log10_transform, axis=0)
				matrix_log10_result.append(row_log10_transform)				
			else:
				row_log10 = np.log10(row)
				row_log10_transform = np.divide(row_log10, row_max_log10)
				#np.append(matrix_log10_result, row_log10_transform, axis=0)
				matrix_log10_result.append(row_log10_transform)
	return matrix_log10_result


