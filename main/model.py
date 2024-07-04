import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout  # TensorFlow/Keras layers for building the model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import numpy as np

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

	def train_step(single_z, single_y_as_array):
		return model.train_on_batch([single_z], single_y_as_array)

	def test_step(single_z, single_y_as_array):
		return model.test_on_batch([single_z], single_y_as_array)

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

