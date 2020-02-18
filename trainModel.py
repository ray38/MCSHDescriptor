from __future__ import print_function
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

#import matplotlib
#import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.axes3d as p3
#import matplotlib.pyplot as plt

import numpy as np
import csv
import sys
import os
import time
import math
import json
from glob import glob
from sklearn import linear_model

import scipy
import h5py

import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle
#from subsampling import subsampling_system,random_subsampling,subsampling_system_with_PCA,subsampling_system_batch,subsampling_system_with_PCA_batch
from NNSubsampling import subsampling, subsampling_with_PCA, batch_subsampling, batch_subsampling_with_PCA, random_subsampling

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import keras

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import multi_gpu_model

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from keras's _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()

        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                fast_updates = (model.updates +
                                training_updates +
                                model.metrics_updates)

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs,
                    [model.total_loss] + model.metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R
                
                model.train_function = F


def fit_pca(data,filename):
    print( "start fitting pca")
    #pca = RandomizedPCA()
    pca = PCA( svd_solver = "randomized" )
    X_pca = pca.fit_transform(data)
    pickle.dump(pca, open(filename, 'wb'))
    print( "saved model")
    return pca

def transform_pca(pca_model,data,n_components):
    temp = pca_model.transform(data)
    return temp[:,:n_components]


def rsse(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true)))

def sae(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true))

def write(log_filename, text):
    with open(log_filename, "w") as myfile:
        myfile.write(text)
    return


def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return

def get_start_loss(log_filename,loss):
    
    with open(log_filename, 'r') as f:
        for line in f:
            pass
        temp = line
    
    if temp.strip().startswith('updated') and temp.split()[9] == loss:
        return float(temp.split()[2])
    else:
        raise ValueError


"""

def save_resulting_figure(n, y,predict_y,loss,loss_result):

    dens = n

    error = y - predict_y

    log_dens = np.log10(n)

    log_predict_y = np.log10(np.multiply(-1.,predict_y))
    log_y = np.log10(np.multiply(-1.,y))

    log_a_error = np.log10(np.abs(error))




    fig, axes = plt.subplots(2, 2, figsize=(100,100))
    ((ax1, ax2),(ax3,ax4)) = axes
    ax1.scatter(dens, y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    ax1.scatter(dens, predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    ax1.scatter(dens, error,            c= 'yellow',  lw = 0,label='error',alpha=1.0)
    legend = ax1.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax1.tick_params(labelsize=80)
    ax1.set_xlabel('density', fontsize=100)
    ax1.set_ylabel('energy density', fontsize=100)

    ax2.scatter(log_dens, y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    ax2.scatter(log_dens, predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    ax2.scatter(log_dens, error,            c= 'yellow',  lw = 0,label='error',alpha=1.0)
    legend = ax2.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax2.tick_params(labelsize=80)
    ax2.set_xlabel('log10 density', fontsize=100)
    ax2.set_ylabel('energy density', fontsize=100)

    ax3.scatter(dens, log_y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    try:
        ax3.scatter(dens, log_predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    except:
        pass
    legend = ax3.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax3.tick_params(labelsize=80)
    ax3.set_xlabel('density', fontsize=100)
    ax3.set_ylabel('log10 negative energy density', fontsize=100)

    ax4.scatter(log_dens, log_y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    try:
        ax4.scatter(log_dens, log_predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    except:
        pass
    legend = ax4.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax4.tick_params(labelsize=80)
    ax4.set_xlabel('log10 density', fontsize=100)
    ax4.set_ylabel('log10 negative energy density', fontsize=100)

    plt.savefig('result_plot_{}_{}.png'.format(loss,loss_result))


    fig, axes = plt.subplots(2, 2, figsize=(100,100))
    ((ax1, ax2),(ax3,ax4)) = axes
    ax1.scatter(dens, error,            c= 'red',  lw = 0,label='error',alpha=1.0)
    legend = ax1.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax1.tick_params(labelsize=80)
    ax1.set_xlabel('density', fontsize=100)
    ax1.set_ylabel('error', fontsize=100)

    ax2.scatter(log_dens, error,            c= 'red',  lw = 0,label='error',alpha=1.0)
    legend = ax2.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax2.tick_params(labelsize=80)
    ax2.set_xlabel('log10 density', fontsize=100)
    ax2.set_ylabel('error', fontsize=100)

    ax3.scatter(dens, log_a_error,            c= 'red',  lw = 0,label='absolute error',alpha=1.0)
    legend = ax3.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax3.tick_params(labelsize=80)
    ax3.set_xlabel('density', fontsize=100)
    ax3.set_ylabel('log10 absolute error', fontsize=100)

    ax4.scatter(log_dens, log_a_error,            c= 'red',  lw = 0,label='absolute error',alpha=1.0)
    legend = ax4.legend(loc="best", shadow=False, scatterpoints=1, fontsize=80, markerscale=10)
    ax4.tick_params(labelsize=80)
    ax4.set_xlabel('log10 density', fontsize=100)
    ax4.set_ylabel('log10 absolute error', fontsize=100)

    plt.savefig('error_plot_{}_{}.png'.format(loss,loss_result))




    return


"""
def fit_with_KerasNN2(X, y, setup, loss, tol, slowdown_factor, early_stop_trials, validation = False, val_set = []):

    loss_list = ["mse","sae","mean_squared_error", "rsse", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge", "categorical_hinge", "logcosh", "categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy", "kullback_leibler_divergence", "poisson", "cosine_proximity"]
    if loss not in loss_list:
        raise NotImplemented


    get_custom_objects().update({'swish': Swish(swish)})

    filename = "NN.h5"
    log_filename = "NN_fit_log.log"
    temp_check_filename = "NN_fit_checkpoint.log"
    num_samples = len(y)

    NN_setup = setup["NN_setup"]["structure"]

    if "batch_size" in setup["NN_setup"]:
        batch_size = setup["NN_setup"]["batch_size"]
    else:
        batch_size = 50000


    k = len(X[0])
    try:
        model = load_model(filename, custom_objects={'sae': sae, 'rsse': rsse})
        restart = True
        print( 'model loaded: ' + filename)
    except:

        restart = False
        input_img= Input(shape=(k,))
        layer_count = 1
        first_bool = True
        for layer_setup in NN_setup:
            number_nodes = layer_setup[0]
            #if layer_setup[1] == "prelu":
            #    activation = PReLU()
            #else:
            #    activation = layer_setup[1]
            activation = layer_setup[1]

            if first_bool:
                if activation == "prelu":
                    out = Dense(number_nodes)(input_img)
                    Act = PReLU()
                    out = Act(out)
                    first_bool = False

                else:
                    out = Dense(units=number_nodes, activation=activation)(input_img)
                    first_bool = False
            else:
                #out = Dense(units=number_nodes, activation=activation)(out)
                if activation == "prelu":
                    out = Dense(number_nodes)(out)
                    Act = PReLU()
                    out = Act(out)
                    first_bool = False

                else:
                    out = Dense(units=number_nodes, activation=activation)(out)
                    first_bool = False
            layer_count += 1

        model = Model(input_img, out)


    
    print( 'model set')
    default_lr = 0.001
    adam = keras.optimizers.Adam(lr=default_lr / slowdown_factor)
    if loss == "sae":
        model.compile(loss=sae,#loss='mse',#loss='mean_absolute_percentage_error',#custom_loss,
                  optimizer=adam)
                  #metrics=['mae'])
    elif loss == "rsse":
        model.compile(loss=rsse,#loss='mse',#loss='mean_absolute_percentage_error',#custom_loss,
                  optimizer=adam)
    else:
        model.compile(loss=loss,#loss='mean_absolute_percentage_error',#custom_loss,
                  optimizer=adam)

    lookahead = Lookahead(k=5, alpha=0.5)
    lookahead.inject(model)

    print( model.summary())
    print( model.get_config())

    callbacks = [EarlyStopping(monitor='loss', patience=early_stop_trials),
                    ModelCheckpoint(filepath='NN.h5', monitor='loss',save_best_only=True, verbose=1)]
    
    history = model.fit(X, y, epochs=50000, batch_size=batch_size, shuffle=True, callbacks=callbacks)
    
    best_loss = np.min(history.history["loss"])
    best_model = load_model("NN.h5", custom_objects={'sae': sae, 'rsse': rsse})
    best_model.save("NN_{}_{}_backup.h5".format(loss,best_loss))
    return best_model, best_loss



def fit_with_KerasNN(X, y, setup, loss, tol, slowdown_factor, early_stop_trials):

	loss_list = ["mse","sae", "rsse","mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge", "categorical_hinge", "logcosh", "categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy", "kullback_leibler_divergence", "poisson", "cosine_proximity"]
	if loss not in loss_list:
		raise NotImplemented


	filename = "NN.h5"
	log_filename = "NN_fit_log.log"
	temp_check_filename = "NN_fit_checkpoint.log"
	num_samples = len(y)

	n_layers = setup["NN_setup"]["number_layers"]
	n_per_layer = setup["NN_setup"]["number_neuron_per_layer"]
	activation = setup["NN_setup"]["activation"]



	try:
		model = load_model(filename, custom_objects={'sae': sae, 'rsse': rsse})
		restart = True
		print( 'model loaded: ' + filename)
	except:
		restart = False
		n = int(n_per_layer)
		k = len(X[0])
		print( n,k)
		model = Sequential()
		model.add(Dense(output_dim =n, input_dim = k, activation = activation))
    
		if n_layers > 1:        
			for i in range(int(n_layers-1)):
				model.add(Dense(input_dim = n, output_dim  = n, activation = activation))
		model.add(Dense(input_dim = n,output_dim =1, activation = 'linear'))
    
    #    model.add(Dense(input_dim = 1,output_dim =1, activation = 'linear',  init='uniform'))
    
	print( 'model set')
	print( "X shape: {}".format(X.shape))
	default_lr = 0.001
	adam = keras.optimizers.Adam(lr=default_lr / slowdown_factor)
	if loss == "sae":
		model.compile(loss=sae,#loss='mse',#loss='mean_absolute_percentage_error',#custom_loss,
                  optimizer=adam)
                  #metrics=['mae'])

	elif loss == "rsse":
		model.compile(loss=rsse,#loss='mse',#loss='mean_absolute_percentage_error',#custom_loss,
	              optimizer=adam)
	else:
		model.compile(loss=loss,#loss='mean_absolute_percentage_error',#custom_loss,
                  optimizer=adam)

	print( model.summary())
	print( model.get_config())
    
	history_callback_kickoff = model.fit(X, y, nb_epoch=1, batch_size=100000, shuffle=True)
	est_start = time.time()
	history_callback = model.fit(X, y, nb_epoch=1, batch_size=100000, shuffle=True)
	est_epoch_time = time.time()-est_start
	if est_epoch_time >= 20.:
		num_epoch = 1
	else:
		num_epoch = min(int(math.floor(20./est_epoch_time)),5)
	if restart == True:
		try:
			start_loss = get_start_loss(log_filename,loss)
		except:
			loss_history = history_callback.history["loss"]
			start_loss = np.array(loss_history)[0]
	else:
		loss_history = history_callback.history["loss"]
		start_loss = np.array(loss_history)[0]
    
	log(log_filename, "\n loss: {} \t start: {} \t slowdown: {} \t early stop: {} \t target tolerence: {}".format(loss, str(start_loss), slowdown_factor, early_stop_trials, tol))
    
	best_loss = start_loss
	best_model = model
	keep_going = True

	count_epochs = 0
	log(log_filename, "\n updated best: "+ str(start_loss) + " \t epochs since last update: " + str(count_epochs) + " \t loss: " + loss)
	while keep_going:
		count_epochs += 1
		print( count_epochs)
		history_callback = model.fit(X, y, nb_epoch=num_epoch, batch_size=100000, shuffle=True)
		loss_history = history_callback.history["loss"]
		new_loss = np.array(loss_history)[-1]
		write(temp_check_filename, "\n updated best: "+ str(new_loss) + " \t epochs since last update: " + str(count_epochs) + " \t loss: " + loss + "\t num_samples: " + str(num_samples))
		if new_loss < best_loss:
			model.save(filename)
			print( 'model saved')
			best_model = model
			if loss == "sae":
				log(log_filename, "\n updated best: "+ str(new_loss) + " \t epochs since last update: " + str(count_epochs) + " \t loss: " + loss + " \t projected error: " + str(((new_loss/1e6)*0.02*0.02*0.02*27.2114)*125/3)  )
			else:
				log(log_filename, "\n updated best: "+ str(new_loss) + " \t epochs since last update: " + str(count_epochs) + " \t loss: " + loss)
			best_loss = new_loss
			count_epochs = 0
		if new_loss < tol:
			keep_going = False
		if count_epochs >=early_stop_trials:
			keep_going = False
    

	best_model.save("NN_{}_{}_backup.h5".format(loss,best_loss))

	#save_resulting_figure(X[:,0]/1e6, y/1e6, best_model.predict(X)/1e6,loss,best_loss)

	return best_model, best_loss


def fit_with_KerasNN_continuous(training_data, setup):

	X,y = split(training_data)
	print(X.shape)
	print(y.shape)

	for fit_setup in setup["model_setup"]['fit_setup']:
		loss = fit_setup['loss']
		slowdown_factor = fit_setup['slowdown']
		early_stop_trials = fit_setup['early_stop']
		tol = fit_setup['tol']
		#fit_with_KerasNN(X*1e9, y*1e9, setup["model_setup"], loss, tol, slowdown_factor, early_stop_trials)
		fit_with_KerasNN2(X*1e9, y*1e9, setup["model_setup"], loss, tol, slowdown_factor, early_stop_trials)
	    

	return

def fit_with_PCA_KerasNN_continuous(training_data, setup):

	X_original,y = split(training_data)
	PCA_components = setup["model_setup"]["PCA_components"]
	PCA_model_filename = "PCA.sav"


	try:
		PCA_model = pickle.load(open(PCA_model_filename, 'rb'))
		X = transform_pca(PCA_model,X_original,PCA_components)

	except:
		PCA_model = fit_pca(X_original,PCA_model_filename)
		X = transform_pca(PCA_model,X_original,PCA_components)

	print( "Total explained variance: {}".format(sum(PCA_model.explained_variance_ratio_[:PCA_components])))

	for fit_setup in setup["model_setup"]['fit_setup']:
		loss = fit_setup['loss']
		slowdown_factor = fit_setup['slowdown']
		early_stop_trials = fit_setup['early_stop']
		tol = fit_setup['tol']
		#fit_with_KerasNN(X*1e9, y*1e9, setup["model_setup"], loss, tol, slowdown_factor, early_stop_trials)
		fit_with_KerasNN2(X*1e9, y*1e9, setup["model_setup"], loss, tol, slowdown_factor, early_stop_trials)


def fit_with_GPR():

	return




def combine(X,y):
	try:
		return np.hstack((y, X))
	except:
		return np.hstack((y.reshape((len(y),1)), X))


def split(overall):
	X = overall[:,1:]
	y = overall[:,0]
	return X,y

def load_data(filename, dirname, setup):

	num_random = int(setup["number_random"])
	if "subsample_bool" in setup:
		if setup["subsample_bool"] == True and "subsample_setup" in setup:
			subsample_bool = True
		else:
			print( "subsample bool is False, or subsample setup in empty")
			subsample_bool = False
	else:
		subsample_bool = False


	try:
		system_list = setup["system_list"]
	except:
		system_list = []

	

	cwd = os.getcwd()
	os.chdir(cwd + "/" + dirname)


	data =  h5py.File(filename,'r')

	if system_list == []:
		counter = 0
		for system in data.keys():
			print( "start loading {}".format(system))
			
			try:
				temp_uniform_data = combine(np.asarray(data[system]["X_uniform"]), np.asarray(data[system]["y_uniform"]))
				temp_random_data = combine(np.asarray(data[system]["X_random"]), np.asarray(data[system]["y_random"]))
				if counter == 0:
					uniform_data = temp_uniform_data
					random_data = temp_random_data
				else:
					uniform_data = np.concatenate((uniform_data,temp_uniform_data))
					random_data = np.concatenate((random_data,temp_random_data))

				print( "done loading {}".format(system))

				counter +=1
			except:
				print( "error loading: {}".format(system))




	else:
		counter = 0
		for system in system_list:
		
			print( "start loading {}".format(system))
			
			try:
				temp_uniform_data = combine(np.asarray(data[system]["X_uniform"]), np.asarray(data[system]["y_uniform"]))
				temp_random_data = combine(np.asarray(data[system]["X_random"]), np.asarray(data[system]["y_random"]))
				if counter == 0:
					uniform_data = temp_uniform_data
					random_data = temp_random_data
				else:
					uniform_data = np.concatenate((uniform_data,temp_uniform_data))
					random_data = np.concatenate((random_data,temp_random_data))

				print( "done loading {}".format(system))

				counter +=1
			except:
				print( "error loading: {}".format(system))


	print( "subsampled data size before subsampling: {}".format(uniform_data.shape))


	#X_uniform = np.asarray(data["X_uniform"])
	#y_uniform = np.asarray(data["y_uniform"])

	#uniform_data = combine(np.asarray(data["X_uniform"]), np.asarray(data["y_uniform"]))

	if subsample_bool:
		subsample_cutoff = float(setup["subsample_setup"]["subsample_cutoff_sig"])
		subsample_rate = float(setup["subsample_setup"]["subsample_rate"])
		if "PCA_subsample" in setup["subsample_setup"]:
			PCA_subsample = setup["subsample_setup"]["PCA_subsample"]
		else:
			PCA_subsample = False
		if "method" in setup["subsample_setup"]:
			method = setup["subsample_setup"]["method"]
		else:
			method = "pykdtree"
		if "verbose" in setup["subsample_setup"]:
			verbose = setup["subsample_setup"]["verbose"]
		else:
			verbose = 2
		if "standard_scale" in setup["subsample_setup"]:
			standard_scale = setup["subsample_setup"]["standard_scale"]
		else:
			standard_scale = True

		if "batch_size" in setup["subsample_setup"] and "recursive_level" in setup["subsample_setup"]:

			if "final_overall_subsample" in setup["subsample_setup"]:
				final_overall_subsample = setup["subsample_setup"]["final_overall_subsample"]
			else:
				final_overall_subsample = True

			if PCA_subsample:

				uniform_data = np.asarray(batch_subsampling_with_PCA(uniform_data, list_desc = [], cutoff_sig = subsample_cutoff, rate = subsample_rate, \
																	batch_size = setup["subsample_setup"]["batch_size"], recursive_level = int(setup["subsample_setup"]["recursive_level"]), \
																	start_trial_component = int(setup["subsample_setup"]["start_trial_component"]), max_component = int(setup["subsample_setup"]["max_component"]), \
																	standard_scale = standard_scale, method = method, verbose = verbose,\
																	final_overall_subsample = final_overall_subsample))
			else:
				uniform_data = np.asarray(batch_subsampling(uniform_data, list_desc = [], cutoff_sig = subsample_cutoff, rate = subsample_rate, \
															batch_size = setup["subsample_setup"]["batch_size"], recursive_level = int(setup["subsample_setup"]["recursive_level"]), \
															standard_scale = standard_scale, method = method, verbose = verbose,\
															final_overall_subsample = final_overall_subsample))

		else:
			if PCA_subsample:
				uniform_data = np.asarray(subsampling_with_PCA( uniform_data, list_desc = [], cutoff_sig = subsample_cutoff, rate = subsample_rate, \
																start_trial_component = int(setup["subsample_setup"]["start_trial_component"]), max_component = int(setup["subsample_setup"]["max_component"]), \
																standard_scale = standard_scale, method = method, verbose = verbose))
			else:
				uniform_data = np.asarray(subsampling(uniform_data, list_desc = [], cutoff_sig = subsample_cutoff, rate = subsample_rate, \
														standard_scale = standard_scale, method = method, verbose = verbose))


	print( "subsampled data size after subsampling: {}".format(uniform_data.shape))

	if num_random != 0:
		#random_data = combine(np.asarray(data["X_random"]), np.asarray(data["y_random"]))
		random_data = np.asarray(random_subsampling(random_data, num_random))
		training_data = np.vstack((uniform_data, random_data))
	else:
		training_data = uniform_data
	    
	os.chdir(cwd)
	print( "training data size: {}".format(training_data.shape))

	return training_data

def fit_with_model(data_filaname,data_dirname,model_name,model_dirname,setup):

	print("start fitting model")
	training_data = load_data(data_filaname,data_dirname, setup)
	print("loaded training data")
	cwd = os.getcwd()

	model_choice_list = ["KerasNN","PCA_KerasNN"]
	model_choice = setup["model_setup"]["model"]

	if model_choice not in model_choice_list:
		raise NotImplemented

	if os.path.isdir(model_dirname) == False:
		os.makedirs(model_dirname)
	os.chdir(cwd + "/" + model_dirname)

	if os.path.isdir(model_name) == False:
		os.makedirs(model_name)
	os.chdir(cwd + "/" + model_dirname + "/" + model_name)

	if model_choice == "KerasNN":
		
		K.set_floatx('float64')
		K.floatx()
		fit_with_KerasNN_continuous(training_data, setup)

	if model_choice == "PCA_KerasNN":
		
		K.set_floatx('float64')
		K.floatx()
		
		fit_with_PCA_KerasNN_continuous(training_data, setup)

	os.chdir(cwd)

	return




