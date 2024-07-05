import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout  # TensorFlow/Keras layers for building the model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.utils import plot_model
import numpy as np
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
from sys import path
sys.path.append('/main')
from saving import save_model_history
from stats import ft_mse, ft_mae, ft_q, linreg

def build_model(embedding_size = 63, lr = 0.01, optimizer = 'adam', depth = 2, 
	scale_output = 0.05, padding = True, hidden = 0, hidden2 = 0, loss = 'mse', hidden_activation = 'tanh',
	output_activation = 'linear', dr1 = 0.0, dr2 = 0.0, output_size = 1, sum_after = False,
	molecular_attributes = False, use_fp = None, inner_rep = 32, verbose = True ):
	#if verbose is used to control whether additional logging information is printed.
    #When verbose is True, the script provides detailed output about the model construction process.

	'''Generates simple embedding model to use molecular tensor as
	input in order to predict a single-valued output (i.e., yield)

	inputs:
		embedding_size - size of fingerprint for GraphFP layer
		lr - learning rate to use (train_model overwrites this value)
		optimizer - optimization function to use
		depth - depth of the neural fingerprint (i.e., radius)
		scale_output - initial scale for output weights in GraphFP
		padding - whether or not molecular tensors will be padded (i.e., batch_size > 1)
		hidden - number of hidden tanh nodes after FP (0 is linear)
		hidden2 - number of hidden nodes after "hidden" layer
		hidden_activation - activation function used in hidden layers
		output_activation - activation function for final output nodes
		dr1 - dropout rate after embedding
		dr2 - dropout rate after hidden
		loss - loss function as a string (e.g., 'mse')
		sum_after - whether to sum neighbor contributions after passing
					through a single network layer, or to do so before
					passing them from the network layer (during updates)
		molecular_attributes - whether to include additional molecular 
					attributes in the atom-level features (recommended)
		use_fp - whether the representation used is actually a fingerprint
					and not a convolutional network (for benchmarking)

	outputs:
		model - a Keras model'''

	
	# Base model
	if type(use_fp) == type(None):
		print('We need a fingerprint input for this tests!')
	else:
		FPs = Input(shape = (63,), name = "input fingerprint")

    #Dropout is applied after dense layers. Dropout layers are used to prevent overfitting by randomly setting a fraction of the
    #neurons to zero during training, encouraging the network to learn more general and robust features.
	hidden = int(hidden)
	hidden2 = int(hidden2)
	if hidden > 0:
		h1 = Dense(hidden, activation = hidden_activation)(FPs)
		#Dropout(dr1)(h1): Adds a dropout layer with a dropout rate dr1. This layer is applied to the output of the first dense layer (h1).
        #The dropout rate dr1 specifies the fraction of neurons to be dropped out during each training iteration. 
        #For example, if dr1 = 0.5, then 50% of the neurons in h1 are set to zero randomly during training.
		h1d = Dropout(dr1)(h1)
		if verbose: print('    model: added {} Dense layer (-> {})'.format(hidden_activation, hidden))
		if hidden2 > 0:
			h2 = Dense(hidden2, activation = hidden_activation)(h1)
			if verbose: print('    model: added {} Dense layer (-> {})'.format(hidden_activation, hidden2))
			h = Dropout(dr2)(h2)
		else:
			h = h1d
	else:
		h = FPs	
		
	ypred = Dense(output_size, activation = output_activation)(h)
	if verbose: print('    model: added output Dense layer (-> {})'.format(output_size))
	
	model = Model(inputs = [FPs], outputs = [ypred])
	
	if verbose: model.summary()
	
    # Compile
	if optimizer == 'adam':
		optimizer = Adam(learning_rate = lr)
	elif optimizer == 'rmsprop':
		optimizer = RMSprop(learning_rate = lr)
	elif optimizer == 'adagrad':
		optimizer = Adagrad(learning_rate = lr)
	elif optimizer == 'adadelta':
		optimizer = Adadelta()
	else:
		print('Unrecognized optimizer')
		quit(1)

    #isto posso retirar do meu codigo dps, pk eu nao tenho NaN values!	
	if loss == 'custom':
		loss = 'mse'
	elif loss == 'custom2':
		loss = 'binary_crossnetropy'
		
	if verbose: print('compiling...',)
	model.compile(loss = loss, optimizer = optimizer)
	if verbose: print('done')

	return model

def train_model(model, data, nb_epoch = 0, batch_size = 1, lr_func = None, patience = 10, verbose = True):
	'''Trains the model.

	inputs:
		model - a Keras model
		data - three dictionaries for training,
				validation, and testing separately
		nb_epoch - number of epochs to train for
		batch_size - batch_size to use on the data. This must agree with what was
				specified for data (i.e., if tensors are padded or not)
		lr_func - string which is evaluated with 'epoch' to produce the learning 
				rate at each epoch 
		patience - number of epochs to wait when no progress is being made in 
				the validation loss. a patience of -1 means that the model will
				use weights from the best-performing model during training

	outputs:
		model - a trained Keras model
		loss - list of training losses corresponding to each epoch 
		val_loss - list of validation losses corresponding to each epoch'''
	
	# Unpack data 
	(train, val, test) = data
	cbu_train = train['cbu']; y_train = train['y']
	cbu_val   = val['cbu'];   y_val   = val['y']
	print('{} to train on'.format(len(y_train)))
	print('{} to validate on'.format(len(y_val)))
	#print('{} to test on'.format(len(smiles_val)))

	# Create learning rate function
	if lr_func:
		lr_func_string = 'def lr(epoch):\n    return {}\n'.format(lr_func)
		exec(lr_func_string)


	# Fit (allows keyboard interrupts in the middle)
	# Because molecular graph tensors are different sizes based on N_atoms, can only do one at a time
	# (alternative is to pad with zeros and try to add some masking feature to GraphFP)
	# -> this is why batch_size == 1 is treated distinctly
	try:
		loss = []
		val_loss = []
		nb_epoch = int(nb_epoch)
		batch_size = int(batch_size)
		patience = int(patience)

		# When the batch_size is larger than one, we have padded mol tensors
		# which  means we need to concatenate them but can use Keras' built-in
		# training functions with callbacks, validation_split, etc.
		if lr_func:
			callbacks = [LearningRateScheduler(lr)]
		else:
			callbacks = []
		if patience != -1:
			callbacks.append(EarlyStopping(patience = patience, verbose = 1))

			if cbu_val:
				cbu = np.vstack((cbu_train, cbu_val))
				y = np.concatenate((y_train, y_val))
				hist = model.fit(cbu, y, 
					epochs = nb_epoch, 
					batch_size = batch_size, 
					validation_split = (1 - float(len(y_train))/(len(y_val) + len(y_train))),
					verbose = verbose,
					callbacks = callbacks)	
			else:
				hist = model.fit(np.array(cbu_train), np.array(y_train), 
					epochs = nb_epoch, 
					batch_size = batch_size, 
					verbose = verbose,
					callbacks = callbacks)	
			
			loss = []; val_loss = []
			if 'loss' in hist.history: loss = hist.history['loss']
			if 'val_loss' in hist.history: val_loss = hist.history['val_loss']

	except KeyboardInterrupt:
		print('User terminated training early (intentionally)')

	return (model, loss, val_loss)

def save_model(model, loss, val_loss, fpath = '', config = {}, tstamp = ''):
	'''Saves NN model object and associated information.

	inputs:
		model - a Keras model
		loss - list of training losses 
		val_loss - list of validation losses
		fpath - root filepath to save everything to (with .json, h5, png, info)
		config - the configuration dictionary that defined this model 
		tstamp - current timestamp to log in info file'''
	
	# Dump data
	with open(fpath + '.json', 'w') as structure_fpath:
		json.dump(model.to_json(), structure_fpath)
	print('...saved structural information')

	# Dump weights
	model.save_weights(fpath + '.weights.h5', overwrite = True)
	print('...saved weights')

	# Dump image
	plot_model(model, to_file=fpath + '.png', show_shapes=True, show_layer_names=True)
	print('...saved image')

	# Dump history
	save_model_history(loss, val_loss, fpath + '.hist')
	print ('...saved history')

	# Write to info file
	info_fid = open(fpath + '.info', 'a')
	info_fid.write('{} saved {}\n\n'.format(fpath, tstamp))
	info_fid.write('Configuration details\n------------\n')
	info_fid.write('  {}\n'.format(config))
	info_fid.close()

	print('...saved model to {}.[json, h5, png, info]'.format(fpath))
	return True

def test_model(model, data, fpath, tstamp = 'no_time', batch_size = 128, return_test_MSE = False, verbose = True):
	'''This function evaluates model performance using test data.

	inputs:
		model - the trained Keras model
		data - three dictionaries for training,
					validation, and testing data. Each dictionary should have
					keys of 'cbu', a binary tensor with the composites building
					units of each zeolite, 'y', the target output, 
		fpath - folderpath to save test data to, will be appended with '/tstamp.test'
		tstamp - timestamp to add to the testing
		batch_size - batch_size to use while testing'''
	
	# Create folder to dump testing info to
	try:
		os.makedirs(fpath)
	except: # file exists
		pass
	test_fpath = os.path.join(fpath, tstamp)


	# Unpack
	(train, val, test) = data
	# Unpack
	cbu_train = train['cbu']; y_train = train['y']; z_train = train['z']
	cbu_val   = val['cbu'];   y_val   = val['y'];  z_val   = val['z']
	cbu_test  = test['cbu'];  y_test  = test['y'];  z_test  = test['z']

	y_train_pred = []
	y_val_pred = []
	y_test_pred = []

	y_train_pred = np.array([]); y_val_pred = np.array([]); y_test_pred = np.array([])
	if cbu_train: y_train_pred = model.predict(np.array(cbu_train), batch_size = batch_size, verbose = 1)
	if cbu_val: y_val_pred = model.predict(np.array(cbu_val), batch_size = batch_size, verbose = 1)
	if cbu_test: y_test_pred = model.predict(np.array(cbu_test), batch_size = batch_size, verbose = 1)

	def round3(x):
		return int(x * 1000) / 1000.0
	
	def parity_plot(true, pred, set_label):
		if len(true) == 0:
			print('skipping parity plot for empty dataset')
			return

		try:
			# Trim it to recorded values (not NaN)
			true = np.array(true).flatten()
			if verbose: print(true)
			if verbose: print(true.shape)
			pred = np.array(pred).flatten()
			if verbose: print(pred)
			if verbose: print(pred.shape)

			pred = pred[~np.isnan(true)]
			true = true[~np.isnan(true)]

			print('{}:'.format(set_label))

			AUC = 'N/A'
			min_y = np.min((true, pred))
			max_y = np.max((true, pred))
			mse = ft_mse(true, pred)
			mae = ft_mae(true, pred)
			q = ft_q(true, pred)
			(r2, a) = linreg(true, pred) # predicted v observed
			(r2p, ap) = linreg(pred, true) # observed v predicted

			# Print
			print('  mse = {}, mae = {}'.format(mse, mae))
			if verbose:
				print('  q = {}'.format(q))
				print('  r2 through origin = {} (pred v. true), {} (true v. pred)'.format(r2, r2p))
				print('  slope through origin = {} (pred v. true), {} (true v. pred)'.format(a[0], ap[0]))

			# Create parity plot
			plt.scatter(true, pred, alpha = 0.5)
			plt.xlabel('Actual')
			plt.ylabel('Predicted')
			plt.title('Parity plot for {} ({} set, N = {})'.format(y_label, set_label, len(true)) + 
				'\nMSE = {}, MAE = {}, q = {}, AUC = {}'.format(round3(mse), round3(mae), round3(q), AUC) + 
				'\na = {}, r^2 = {}'.format(round3(a), round3(r2)) + 
				'\na` = {}, r^2` = {}'.format(round3(ap), round3(r2p)))
			plt.grid(True)
			plt.plot(true, true * a, 'r--')
			plt.axis([min_y, max_y, min_y, max_y])	
			plt.savefig(test_fpath + ' {}.png'.format(set_label), bbox_inches = 'tight')
			plt.clf()

			if len(set(list(true))) <= 2:
				return AUC
			return mse

		except Exception as e:
			print(e)
			return 99999

		# Create plots for datasets
	if y_train:
		y_label = train['y_label']
		if type(y_train[0]) != type(0.0):
			num_targets = y_train[0].shape[-1]
		else:
			num_targets = 1
	elif y_val:
		y_label = val['y_label']
		if type(y_val[0]) != type(0.0):
			num_targets = y_val[0].shape[-1]
		else:
			num_targets = 1
	elif y_test:
		y_label = test['y_label']
		if type(y_test[0]) != type(0.0):
			num_targets = y_test[0].shape[-1]
		else:
			num_targets = 1
	else:
		raise ValueError('Nothing to evaluate?')
	
	# Save
	with open(test_fpath + '.test', 'w') as fid:
		fid.write('{} tested {}, predicting {}\n\n'.format(fpath, tstamp, y_label))		
		fid.write('test entry\tzeolite type\tactual\tpredicted\tactual - predicted\n')
		for i in range(len(y_test)):
			fid.write('{}\t{}\t{}\t{}\t{}\n'.format(i, 
				z_test[i],
				y_test[i], 
				y_test_pred[i],
				y_test[i] - y_test_pred[i]))

	test_MSE = 99999
	if y_train: 
		if type(y_train[0]) != type(0.): 
			num_targets = len(y_train[0])
			print('Number of targets: {}'.format(num_targets))
			for i in range(num_targets):
				parity_plot([x[i] for x in y_train], [x[0, i] for x in y_train_pred], 'train - ' + y_label[i])
		else:
			parity_plot(y_train, y_train_pred, 'train')
	if y_val: 
		if type(y_val[0]) != type(0.): 
			num_targets = len(y_val[0])
			print('Number of targets: {}'.format(num_targets))
			for i in range(num_targets):
				parity_plot([x[i] for x in y_val], [x[0, i] for x in y_val_pred], 'val - ' + y_label[i])
		else:
			parity_plot(y_val, y_val_pred, 'test')
	if y_test: 
		if type(y_test[0]) != type(0.): 
			num_targets = len(y_test[0])
			print('Number of targets: {}'.format(num_targets))
			test_MSE = 0.
			for i in range(num_targets):
				test_MSE += parity_plot([x[i] for x in y_test], [x[0, i] for x in y_test_pred], 'test - ' + y_label[i])
		else:
			test_MSE = parity_plot(y_test, y_test_pred, 'test')

	# train['residuals'] = np.array(y_train) - np.array(y_train_pred)
	# val['residuals'] = np.array(y_val) - np.array(y_val_pred)
	# test['residuals'] = np.array(y_test) - np.array(y_test_pred)

	if return_test_MSE: return test_MSE

	return (train, val, test)


