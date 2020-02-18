

from __future__ import print_function
import h5py
import os
import sys
import numpy as np
try: import cPickle as pickle
except: import pickle
import math
import time
import os
import json
import csv

import itertools
import multiprocessing

import seaborn as sns
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
from numpy import linalg as LA


from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras import backend as K
import keras

def transform_pca(pca_model,data,n_components):
    temp = pca_model.transform(data)
    return temp[:,:n_components]

def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return
def rsse(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true)))

def sae(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true))


def prepare_visulization(y, LACO_y, NN_y, error, LCAO_error, filename, dim = None):
	if dim == None:
		dim = int(round(len(y) ** (1. / 3)))
		dimx = dimy = dimz = dim
		print( "dimension: {} {} {}".format(dimx, dimy, dimz))
	if isinstance(dim, tuple):
		dimx, dimy, dimz = dim
		print( "dimension: {} {} {}".format(dimx, dimy, dimz))
	if isinstance(dim, int):
		dimx = dimy = dimz = dim
		print( "dimension: {} {} {}".format(dimx, dimy, dimz))
	temp_error = error.copy().reshape((dimx, dimy, dimz))
	temp_LCAO_error = LCAO_error.copy().reshape((dimx, dimy, dimz))
	temp_y = y.copy().reshape((dimx, dimy, dimz))
	temp_LACO_y = LACO_y.copy().reshape((dimx, dimy, dimz))
	temp_NN_y = NN_y.copy().reshape((dimx, dimy, dimz))

	with open(filename, mode='w') as file:
		writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(["x","y","z","rho","LCAO_rho","NN_rho","error","LCAO_error"])
		for index, value in np.ndenumerate(temp_y):
			#print index
			writer.writerow([index[0], index[1], index[2],temp_y[index], temp_LACO_y[index], temp_NN_y[index], temp_error[index], temp_LCAO_error[index]])

	return


def read_raw_data_one_system(raw_data_filename, setup_dict):


    result_list = []
    data =  h5py.File(raw_data_filename,'r')

    temp_y = np.asarray(data['output'][setup_dict["output_image_name"]]["original"])
    dimension = temp_y.shape
    result_list.append(temp_y.flatten())

    for image_name in sorted(setup_dict["input_image_name_list"]):

        temp_data = np.asarray(data["input"][image_name]["original"])
        result_list.append(temp_data.flatten())

        try:
            MCSH_dict = setup_dict["input_image_MCSH_dict"][image_name]
        except:
            MCSH_dict = setup_dict["input_image_MCSH_dict"]["default"]

        for order in sorted(MCSH_dict.keys()):
            for r in sorted(MCSH_dict[order]):
                #print "read {} {}".format(order,r)

                if order == "0":
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["1"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())

                if order == "1":
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["1"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())

                if order == "2":
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["1"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["2"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())

                if order == "3":
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["1"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["2"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["3"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())

                if order == "4":
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["1"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["2"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["3"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["4"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())

                if order == "5":
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["1"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["2"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["3"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["4"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["5"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())

                if order == "6":
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["1"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["2"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["3"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["4"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["5"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["6"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())
                    temp_data = np.asarray(data["input"][image_name]["MCSH"][order]["7"]['{}'.format(str(r).replace('.','-'))])
                    result_list.append(temp_data.flatten())

    #result = zip(*result_list)

    result = np.stack(result_list, axis = 1)

    #result = np.array(zip(*result_list))
    print(result.shape)
    print(result[0])
    print( "done loading data, datasize: {}, {}".format(len(result),len(result[0])))

    X = []
    y = []


    for entry in result:
        X.append(list(entry[1:]))
        y.append(entry[0])
    

    return np.array(X),np.array(y), dimension

def predict_one_system_KerasNN(raw_data_filename,model, setup):

	os.chdir(setup["data_dir"])
	X,original_y, dimension = read_raw_data_one_system(raw_data_filename, setup)
	os.chdir(setup["cwd"])

	guess_y = X[:,0]

	predict_y = model.predict(X*1e9)/1e9

	predict_y = predict_y.flatten()
	print( predict_y.shape, guess_y.shape, original_y.shape)

	error = predict_y - original_y

	guess_error = guess_y - original_y

	os.chdir(setup["model_dir"])

	normFracNN = LA.norm(error) / LA.norm(original_y)
	normFracLCAO = LA.norm(guess_error) / LA.norm(original_y)

	log(setup["result_name"] + "_log.log", "\nfile:{}".format(raw_data_filename))
	log(setup["result_name"] + "_log.log", "\npredictResult\taveError:{}\taveAbsError:{}\tMinError:{}\tMaxError:{}\tMaxAbsError:{}\tnormFrac:{}\t".format(np.mean(error),np.mean(np.abs(error)),np.min(error), np.max(error), np.max(np.abs(error)), normFracNN))
	log(setup["result_name"] + "_log.log", "\noriginalResult\taveError:{}\taveAbsError:{}\tMinError:{}\tMaxError:{}\tMaxAbsError:{}\tnormFrac:{}\t".format(np.mean(guess_error),np.mean(np.abs(guess_error)),np.min(guess_error), np.max(guess_error), np.max(np.abs(guess_error)), normFracLCAO))

	try:
		if setup["output_csv"] == True:
			prepare_visulization(original_y, guess_y, predict_y, error, guess_error, raw_data_filename + "_visual.csv", dim = dimension)
	except:
		pass

	os.chdir(setup["cwd"])



	return guess_y, original_y, predict_y



def predict_one_system_PCA_KerasNN(raw_data_filename,PCA_model,model, setup):

	os.chdir(setup["data_dir"])
	X_before_PCA,original_y, dimension = read_raw_data_one_system(raw_data_filename, setup)
	PCA_components = setup["model_setup"]["PCA_components"]
	X = transform_pca(PCA_model,X_before_PCA,PCA_components)
	os.chdir(setup["cwd"])

	guess_y = X_before_PCA[:,0]

	predict_y = model.predict(X*1e9)/1e9

	predict_y = predict_y.flatten()
	print( predict_y.shape, guess_y.shape, original_y.shape)

	error = predict_y - original_y

	guess_error = guess_y - original_y

	os.chdir(setup["model_dir"])

	normFracNN = LA.norm(error) / LA.norm(original_y)
	normFracLCAO = LA.norm(guess_error) / LA.norm(original_y)

	log(setup["result_name"] + "_log.log", "\nfile:{}".format(raw_data_filename))
	log(setup["result_name"] + "_log.log", "\npredictResult\taveError:{}\taveAbsError:{}\tMinError:{}\tMaxError:{}\tMaxAbsError:{}\tnormFrac:{}\t".format(np.mean(error),np.mean(np.abs(error)),np.min(error), np.max(error), np.max(np.abs(error)), normFracNN))
	log(setup["result_name"] + "_log.log", "\noriginalResult\taveError:{}\taveAbsError:{}\tMinError:{}\tMaxError:{}\tMaxAbsError:{}\tnormFrac:{}\t".format(np.mean(guess_error),np.mean(np.abs(guess_error)),np.min(guess_error), np.max(guess_error), np.max(np.abs(guess_error)), normFracLCAO))

	#try:
	if setup["output_csv"] == True:
		prepare_visulization(original_y, guess_y, predict_y, error, guess_error, raw_data_filename + "_visual.csv", dim = dimension)
	#except:
	#	pass

	os.chdir(setup["cwd"])



	return guess_y, original_y, predict_y




def test_model_KerasNN(data_filaname_list,data_dirname,model_name,model_dirname,setup,result_name = "result"):
	cwd = os.getcwd()

	data_dir = cwd + "/" + data_dirname
	model_dir = cwd + "/" + model_dirname + "/" + model_name

	setup["cwd"] = cwd
	setup["data_dir"] = data_dir
	setup["model_dir"] = model_dir

	setup["result_name"] = result_name

	
	K.set_floatx('float64')
	K.floatx()
	os.chdir(model_dir)



	try:
		model = load_model("NN.h5", custom_objects={'sae': sae, 'rsse': rsse})
	except:
		model = load_model("NN.h5")

	os.chdir(cwd)

	
	first_bool = True

	for data_filename in data_filaname_list:
		try:
			temp_guess_y, temp_original_y, temp_predict_y = predict_one_system_KerasNN(data_filename,model, setup)

			if first_bool:
				guess_y = temp_guess_y
				original_y = temp_original_y
				predict_y = temp_predict_y
				first_bool = False
			else:
				guess_y = np.concatenate((guess_y,temp_guess_y))
				original_y = np.concatenate((original_y,temp_original_y))
				predict_y = np.concatenate((predict_y,temp_predict_y))
		except:
			pass

	error = predict_y - original_y

	guess_error = guess_y - original_y

	os.chdir(setup["model_dir"])

	normFracNN = LA.norm(error) / LA.norm(original_y)
	normFracLCAO = LA.norm(guess_error) / LA.norm(original_y)

	log(setup["result_name"] + "_log.log", "\n\noverall")
	log(setup["result_name"] + "_log.log", "\npredictResult\taveError:{}\taveAbsError:{}\tMinError:{}\tMaxError:{}\tMaxAbsError:{}\tnormFrac:{}\t".format(np.mean(error),np.mean(np.abs(error)),np.min(error), np.max(error), np.max(np.abs(error)), normFracNN))
	log(setup["result_name"] + "_log.log", "\noriginalResult\taveError:{}\taveAbsError:{}\tMinError:{}\tMaxError:{}\tMaxAbsError:{}\tnormFrac:{}\t".format(np.mean(guess_error),np.mean(np.abs(guess_error)),np.min(guess_error), np.max(guess_error), np.max(np.abs(guess_error)), normFracLCAO))


	plt.figure(figsize=(10,10))
	bins = np.linspace(-np.max(guess_error), np.max(guess_error), 50)

	plt.hist([error, guess_error], bins, label=['NN', 'LCAO'])
	plt.legend(loc='upper right')

	plt.savefig(setup["result_name"] + "_plot.png")


	log_abs_error = np.log10(np.abs(error))
	log_abs_guess_error = np.log10(np.abs(guess_error))

	plt.figure(figsize=(10,10))
	bins = np.linspace(np.min(log_abs_error), np.max(log_abs_error), 50)

	plt.hist([log_abs_error, log_abs_guess_error], bins, label=['NN', 'LCAO'])
	plt.legend(loc='upper right')

	plt.savefig(setup["result_name"] + "_log_abs_plot.png")


	os.chdir(setup["cwd"])


	return




def test_model_PCA_KerasNN(data_filaname_list,data_dirname,model_name,model_dirname,setup,result_name = "result"):
	cwd = os.getcwd()

	data_dir = cwd + "/" + data_dirname
	model_dir = cwd + "/" + model_dirname + "/" + model_name

	setup["cwd"] = cwd
	setup["data_dir"] = data_dir
	setup["model_dir"] = model_dir

	setup["result_name"] = result_name

	
	K.set_floatx('float64')
	K.floatx()
	os.chdir(model_dir)

	PC_component = setup["model_setup"]["PCA_components"]
	PCA_model_filename = "PCA.sav"
	PCA_model = pickle.load(open(PCA_model_filename, 'rb'))



	try:
		model = load_model("NN.h5", custom_objects={'sae': sae,'rsse':rsse})
	except:
		model = load_model("NN.h5")

	os.chdir(cwd)

	
	first_bool = True

	for data_filename in data_filaname_list:
		#try:
		temp_guess_y, temp_original_y, temp_predict_y = predict_one_system_PCA_KerasNN(data_filename,PCA_model,model, setup)

		if first_bool:
			guess_y = temp_guess_y
			original_y = temp_original_y
			predict_y = temp_predict_y
			first_bool = False
		else:
			guess_y = np.concatenate((guess_y,temp_guess_y))
			original_y = np.concatenate((original_y,temp_original_y))
			predict_y = np.concatenate((predict_y,temp_predict_y))
		#except:
		#	pass

	error = predict_y - original_y

	guess_error = guess_y - original_y

	os.chdir(setup["model_dir"])

	normFracNN = LA.norm(error) / LA.norm(original_y)
	normFracLCAO = LA.norm(guess_error) / LA.norm(original_y)

	log(setup["result_name"] + "_log.log", "\n\noverall")
	log(setup["result_name"] + "_log.log", "\npredictResult\taveError:{}\taveAbsError:{}\tMinError:{}\tMaxError:{}\tMaxAbsError:{}\tnormFrac:{}\t".format(np.mean(error),np.mean(np.abs(error)),np.min(error), np.max(error), np.max(np.abs(error)), normFracNN))
	log(setup["result_name"] + "_log.log", "\noriginalResult\taveError:{}\taveAbsError:{}\tMinError:{}\tMaxError:{}\tMaxAbsError:{}\tnormFrac:{}\t".format(np.mean(guess_error),np.mean(np.abs(guess_error)),np.min(guess_error), np.max(guess_error), np.max(np.abs(guess_error)), normFracLCAO))


	plt.figure(figsize=(10,10))
	bins = np.linspace(-np.max(guess_error), np.max(guess_error), 50)

	plt.hist([error, guess_error], bins, label=['NN', 'LCAO'])
	plt.legend(loc='upper right')

	plt.savefig(setup["result_name"] + "_plot.png")


	log_abs_error = np.log10(np.abs(error))
	log_abs_guess_error = np.log10(np.abs(guess_error))

	plt.figure(figsize=(10,10))
	bins = np.linspace(np.min(log_abs_error), np.max(log_abs_error), 50)

	plt.hist([log_abs_error, log_abs_guess_error], bins, label=['NN', 'LCAO'])
	plt.legend(loc='upper right')

	plt.savefig(setup["result_name"] + "_log_abs_plot.png")


	os.chdir(setup["cwd"])


    
	return




def test_model(data_filaname_list,data_dirname,model_name,model_dirname,setup,result_name = "result"):


	cwd = os.getcwd()

	model_choice_list = ["KerasNN","PCA_KerasNN"]
	model_choice = setup["model_setup"]["model"]

	if model_choice not in model_choice_list:
		raise NotImplemented


	if model_choice == "KerasNN":
		
		test_model_KerasNN(data_filaname_list,data_dirname,model_name,model_dirname,setup,result_name = result_name)

	if model_choice == "PCA_KerasNN":
		
		test_model_PCA_KerasNN(data_filaname_list,data_dirname,model_name,model_dirname,setup,result_name = result_name)

	os.chdir(cwd)

	return