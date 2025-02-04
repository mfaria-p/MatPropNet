import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
import keras_tuner as kt
import os
import sys
import csv
import numpy as np
from sklearn.model_selection import KFold
import datetime

def read_config(file_path):
    import configparser  # For reading and parsing the configuration file
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def model_builder(hp):
    model = tf.keras.Sequential()

    units = hp.Int('units', min_value=32, max_value=512, step=32)
    layers = hp.Int('layers', min_value=2, max_value=5, step=1)
    dr = hp.Float('dropout_rate', min_value=0, max_value=0.9, step=0.1)

    model.add(Dense(units=units, activation='tanh', input_shape=(63,)))
    model.add(Dropout(dr))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='tanh'))
        model.add(Dropout(dr))

    model.add(Dense(1, activation='linear'))
    
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])
    return model

def get_data_hyper(data_label= '', shuffle_seed=None, batch_size=1, data_split='cv', cv_folds='1/1', truncate_to=None, training_ratio=0.9):
    data_label = data_label.lower()
    data_froot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    if data_label == 'density':
        delimiter = '\t'
        data_fpath = os.path.join(data_froot, 'density.tsv')
        y_label = 'Framework density (T/1000 Å^3)'
    elif data_label == 'acessible_volume':
        delimiter = '\t'
        data_fpath = os.path.join(data_froot, 'acessible_volume.tsv')
        y_label = 'Accessible Volume (%)'
    elif data_label == 'max_diameter':
        delimiter = '\t'
        data_fpath = os.path.join(data_froot, 'max_diam_diff.tsv')
        y_label = 'Max. diameter of a sphere, that can diffuse along (the maximum between the 3) (Å)'
    else:
        print('Unrecognized data_label {}'.format(data_label))
        quit(1)

    print('reading data...')
    data = []
    labels = []
    with open(data_fpath, 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=delimiter, quotechar='"')
        for row in reader:
            labels.append(row[0])  # First column as label
            data.append(row[1:])  # Remaining columns as data
    print('done\n')

    data = np.array(data, dtype=float)
    x = data[:, :-1]  # Features (input data)
    y = data[:, -1]   # Target values (output data)
    return x, y

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} "settings.cfg"'.format(sys.argv[0]))
        quit(1)

    try:
        config = read_config(sys.argv[1])
    except:
        print('Could not read config file {}'.format(sys.argv[1]))
        quit(1)

    try:
        fpath = config['IO']['model_fpath']
    except KeyError:
        print('Must specify model_fpath in IO in config')
        quit(1)

    data_kwargs = config['DATA']
    
    if '__name__' in data_kwargs:
        del data_kwargs['__name__'] #  from configparser
    if 'batch_size' in config['TRAINING']:
        data_kwargs['batch_size'] = config['TRAINING']['batch_size']
    if 'truncate_to' in data_kwargs:
        data_kwargs['truncate_to'] = data_kwargs['truncate_to']
    if 'training_ratio' in data_kwargs:
        data_kwargs['training_ratio'] = data_kwargs['training_ratio']
    if 'molecular_attributes' in data_kwargs:
        del data_kwargs['molecular_attributes']

    x, y = get_data_hyper(**data_kwargs)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_best_hps = []
    
    tstamp = datetime.datetime.utcnow().strftime('%m-%d-%Y_%H-%M')

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        tuner = kt.Hyperband(model_builder,
                             objective='val_mse',
                             max_epochs=10,
                             factor=3,
                             directory = f"hyperband/{data_kwargs['data_label']}_{tstamp}",
                             project_name=f"fold_{len(all_best_hps) + 1}")

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(x_train, y_train,
                     epochs=30,
                     validation_data=(x_val, y_val),
                     callbacks=[stop_early])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        all_best_hps.append(best_hps)

    # Aggregate the best hyperparameters from each fold
    # Here, you can choose the most frequent or average values of the hyperparameters
    # For simplicity, let's just print the best hyperparameters from each fold
    
    from collections import Counter

    units_list = [hp.get('units') for hp in all_best_hps]
    layers_list = [hp.get('layers') for hp in all_best_hps]
    dropout_list = [hp.get('dropout_rate') for hp in all_best_hps]
    learning_rate_list = [hp.get('learning_rate') for hp in all_best_hps]

    avg_units = int(np.mean(units_list))
    avg_layers = int(np.mean(layers_list))
    avg_dropout = np.mean(dropout_list)
    avg_learning_rate = np.mean(learning_rate_list)

    print(f"Aggregated Hyperparameters:\nUnits: {avg_units}, Layers: {avg_layers}, Dropout Rate: {avg_dropout}, Learning Rate: {avg_learning_rate}")