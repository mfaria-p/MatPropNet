import os
from os import path
import numpy as np
import csv
from bitarray import bitarray

def get_data_one(data_label = '', shuffle_seed = None, batch_size = 1, 
	data_split = 'cv', cv_folds = '1/1',	truncate_to = None, training_ratio = 0.9):
	'''This is a helper script to read the data file and return
	the training and test data sets separately. This is to allow for an
	already-trained model to be evaluated using the test data (i.e., which
	we know it hasn't seen before)'''

	# Roots
	data_label = data_label.lower()
	data_froot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

	###################################################################################
	### WHICH DATASET ARE WE TRYING TO USE?
	###################################################################################

	if data_label in ['density']:
		delimeter = '\t'
		dset = 'density'
		data_fpath = os.path.join(data_froot, 'density.fp')
		z_index = 0
		y_index = -1
		def y_func(x): return x
		y_label = 'Framework density (T/1000 Å^3)'
	elif data_label in ['acessible_volume']:
		delimeter = '\t'
		dset = 'acessible_volume'
		data_fpath = os.path.join(data_froot, 'acessible_volume.fp')
		z_index = 0
		y_index = -1
		def y_func(x): return x
		y_label = 'Accessible Volume (%)'
	elif data_label in ['max_diameter']:
		delimeter = '\t'
		dset = 'max_diameter'
		data_fpath = os.path.join(data_froot, 'max_diam_diff.fp')
		z_index = 0
		y_index = -1
		def y_func(x): return x
		y_label = 'Max. diameter of a sphere, that can diffuse along (the maximum between the 3) (Å)'

#63 columns 0 e 1
	# Other?
	else:
		print('Unrecognized data_label {}'.format(data_label))
		quit(1)
#tenho q append tds os zeros e uns
	###################################################################################
	### READ AND TRUNCATE DATA
	###################################################################################

	print('reading data...')
	data = []
	with open(data_fpath, 'r') as data_fid:
		reader = csv.reader(data_fid, delimiter = delimeter, quotechar = '"')
		for row in reader:
			data.append(row)
	print('done\n')
		
	# Truncate if necessary
	if truncate_to is not None:
		data = data[:truncate_to]
		print('truncated data to first {} samples'.format(truncate_to))

	# Get new shuffle seed if possible
	if shuffle_seed is not None:
		shuffle_seed = int(float(shuffle_seed))
		np.random.seed(shuffle_seed)

	###################################################################################
	### ITERATE THROUGH DATASET AND CREATE NECESSARY DATA LISTS
	###################################################################################

	cbu = []
	cbu_array = []
	y = []
	z = []
	print('processing data...')
	# Randomize
	np.random.shuffle(data)

	try:
		for j in range(251):
			z.append(data[j][z_index])
			cbu_array = []
			for i in range(1, len(data[j]) - 1):
				cbu_array.append(int(data[j][i]))
			cbu.append((np.array(cbu_array)))

			this_y = y_func(float(data[j][y_index]))
			y.append(this_y)  # Measured log(solubility M/L)

	except Exception as e:
		print('Error:', e)

	###################################################################################
	### DIVIDE DATA VIA RATIO OR CV
	###################################################################################

	if 'ratio' in data_split: # split train/notrain
		print('Using first fraction ({}) as training'.format(training_ratio))
		# Create training/development split
		division = int(len(y) * training_ratio)
		y_train = y[:division]
		y_notrain  = y[division:]
		cbu_train = cbu[:division]
		cbu_notrain = cbu[division:]
		z_train = z[:division]
		z_notrain = z[division:]

		# Split notrain up
		y_val       = y_notrain[:(len(y_notrain) / 2)] # first half
		cbu_val  = cbu_notrain[:(len(y_notrain) / 2)] # first half
		z_val  = z_notrain[:(len(z_notrain) / 2)] # first half
		y_test      = y_notrain[(len(y_notrain) / 2):] # second half
		cbu_test = cbu_notrain[(len(y_notrain) / 2):] # second half
		z_test = z_notrain[(len(z_notrain) / 2):] # second half
		print('Training size: {}'.format(len(y_train)))
		print('Validation size: {}'.format(len(y_val)))
		print('Testing size: {}'.format(len(y_test)))
	
	elif 'all_train' in data_split: # put everything in train 
		print('Using ALL as training')
		# Create training/development split
		cbu_train = cbu
		y_train = y
		z_train = z
		cbu_val    = []
		y_val       = []
		z_val = []
		cbu_test   = []
		y_test      = []
		z_test = []
		print('Training size: {}'.format(len(y_train)))
		print('Validation size: {}'.format(len(y_val)))
		print('Testing size: {}'.format(len(y_test)))

	elif 'cv' in data_split: # cross-validation
		# Default to first fold of 5-fold cross-validation
		folds = 5
		this_fold = 0

		# Read fold information
		try:
			folds = int(cv_folds.split('/')[1])
			this_fold = int(cv_folds.split('/')[0]) - 1
		except:
			pass

		# Get target size of each fold
		N = len(y)
		print('Total of {} zeolites'.format(N))
		target_fold_size = int(np.ceil(float(N) / folds))
		# Split up data
		folded_cbu 	= [cbu[x:x+target_fold_size]   for x in range(0, N, target_fold_size)]
		folded_y 		= [y[x:x+target_fold_size]      for x in range(0, N, target_fold_size)]
		folded_z 	= [z[x:x+target_fold_size] for x in range(0, N, target_fold_size)]
		print('Split data into {} folds'.format(folds))
		print('...using fold {}'.format(this_fold + 1))

		# Recombine into training and testing
		cbu_train   = [x for fold in (folded_cbu[:this_fold] + folded_cbu[(this_fold + 1):])     for x in fold]
		y_train      = [x for fold in (folded_y[:this_fold] + folded_y[(this_fold + 1):])           for x in fold]
		z_train = [x for fold in (folded_z[:this_fold] + folded_z[(this_fold + 1):]) for x in fold]
		# Test is this_fold
		cbu_test    = folded_cbu[this_fold]
		y_test       = folded_y[this_fold]
		z_test  = folded_z[this_fold]

		# Define validation set as random 10% of training
		training_indices = list(range(len(y_train)))
		np.random.shuffle(training_indices)
		training_ratio = float(training_ratio)
		split = int(len(training_indices) * training_ratio)
		cbu_train,   cbu_val    = [cbu_train[i] for i in training_indices[:split]],   [cbu_train[i] for i in training_indices[split:]]
		y_train,      y_val       = [y_train[i] for i in training_indices[:split]],      [y_train[i] for i in training_indices[split:]]
		z_train, z_val  = [z_train[i] for i in training_indices[:split]], [z_train[i] for i in training_indices[split:]]

		print('Total training: {}'.format(len(y_train)))
		print('Total validation: {}'.format(len(y_val)))
		print('Total testing: {}\n'.format(len(y_test)))

	else:
		print('Must specify a data_split type of "ratio" or "cv"')
		quit(1)

	###################################################################################
	### REPACKAGE AS DICTIONARIES
	###################################################################################
	if 'cv_full' in data_split: # cross-validation, but use 'test' as validation
		train = {}; train['cbu'] = cbu_train; train['y'] = y_train; train['z'] = z_train; train['y_label'] = y_label
		val   = {}; val['cbu']   = cbu_test;   val['y']   = y_test; val['z']   = z_test; val['y_label']   = y_label
		test  = {}; test['cbu']  = [];  test['y']  = [];  test['z']  = []; test['y_label']  = []

	else:

		train = {}; train['cbu'] = cbu_train; train['y'] = y_train; train['z'] = z_train; train['y_label'] = y_label
		val   = {}; val['cbu']   = cbu_val;   val['y']   = y_val;   val['z']   = z_val; val['y_label']   = y_label
		test  = {}; test['cbu']  = cbu_test;  test['y']  = y_test;  test['z']  = z_test; test['y_label']  = y_label

	return (train, val, test)