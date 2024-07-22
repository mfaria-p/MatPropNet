#i am using here the Sequential API

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
import keras_tuner as kt

#You can use this to write Python programs which can be customized by end users easily.
def read_config(file_path):
    import configparser  # For reading and parsing the configuration file
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def model_builder(hp):
  model = tf.keras.Sequential()

  units = hp.Int('units', min_value=32, max_value=512, step=32)
  layers = hp.Int('layers', min_value=2, max_value=5, step=1)
  dr = hp.Int('dropout rate', min_value=0, max_value=0.9, step=0.1)

  #model.add(Input(activation='tanh', input_shape=(63,)))
  model.add(Dense(units=units, activation='tanh', input_shape=(63,)))
  model.add(Dropout(dr))

  for _ in range(layers - 1):
            model.add(Dense(units=units, activation='tanh')) 
            model.add(Dropout(dr))

  model.add(Dense(1, activation = 'linear'))

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mse'])
  return model

if __name__ == '__main__':#to designate a section of code that should only be executed 
	#when the script is run directly, not when it is imported as a module in another script.
	if len(sys.argv) < 2:
		print('Usage: {} "settings.cfg"'.format(sys.argv[0])) #ve se tem dois args
		#aka mostra o usage dos argumentos quando se corre o codigo, my script nd the configuration file
		print(sys.argv[0])
		quit(1)

	# Load settings
	try:
	#Once the configuration file is read, the ConfigParser object (config) stores the data in a dictionary-like structure. 
    #Sections in the configuration file are represented as keys, and the settings within each section are stored as nested dictionaries.
		config = read_config(sys.argv[1])
	except:
		print('Could not read config file {}'.format(sys.argv[1]))
		quit(1)

	# Get model label
	#I think this is where the output of the model will put
	#or maybe where the model will get its input
	try:
		fpath = config['IO']['model_fpath']
	except KeyError:
		print('Must specify model_fpath in IO in config')
		quit(1)
	
	########## DEFINE THE DATA ##############################################
	data_kwargs = config['DATA']

	data =get_data_hyper(**data_kwargs)
	#fzr esta funcao

	tuner = kt.Hyperband(model_builder,
                        objective='val_accuracy',
                        max_epochs=10,
                        factor=3,
                        directory='my_conv_qsar')
	
	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

	tuner.search(cbu, y, 
		epochs = 50,  
		validation_split = 0.2,
		verbose = True
		callbacks = [stop_early])
			
    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

	print(f"""
	The hyperparameter search is complete. The optimal number of units in the first densely-connected
	layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
	is {best_hps.get('learning_rate')}.
	""")

	#model = tuner.hypermodel.build(best_hps)
	#history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)


""" def get_data_hyper(data_label = '')
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
	