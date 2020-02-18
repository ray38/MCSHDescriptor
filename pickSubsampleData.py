# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:34:20 2017

@author: ray
"""
from __future__ import print_function
import h5py
import os
import sys
import numpy as np
try: import cPickle as pickle
except: import pickle
#from subsampling import subsampling_system_with_PCA, random_subsampling, subsampling_system, subsampling_system_with_PCA_batch
from NNSubsampling import subsampling, subsampling_with_PCA, batch_subsampling, batch_subsampling_with_PCA, random_subsampling
import math
import time
import os
import json

import itertools
import multiprocessing


def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return





def transform_data(temp_data, transform):
    if transform == "real":
        return temp_data.flatten().tolist()
    elif transform == "log10":
        return np.log10(temp_data.flatten()).tolist()
    elif transform == "neglog10":
        return np.log10(np.multiply(-1., temp_data.flatten())).tolist()

        
"""

def read_sample_raw_data_one_system(raw_data_filename, setup_dict):

    result_list = []
    data =  h5py.File(raw_data_filename,'r')

    temp_y = np.asarray(data['output'][setup_dict["output_image_name"]]["original"])
    result_list.append(temp_y.flatten())

    for image_name in setup_dict["input_image_name_list"]:

        temp_data = np.asarray(data["input"][image_name]["original"])
        result_list.append(temp_data.flatten())

        try:
            MCSH_dict = setup_dict["input_image_MCSH_dict"][image_name]
        except:
            MCSH_dict = setup_dict["input_image_MCSH_dict"]["default"]

        for order in MCSH_dict.keys():
            for r in MCSH_dict[order]:

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

    result = zip(*result_list)

    print "done loading data, datasize: {}, {}".format(len(result),len(result[0]))

    subsample_result = subsampling_system_with_PCA(result, list_desc = [], cutoff_sig = float(setup_dict["subsample_cutoff_sig"]), rate = float(setup_dict["subsample_rate"]),start_trial_component = int(setup_dict["start_trial_component"]), max_component = int(setup_dict["max_component"]))

    #subsample_result = subsampling_system(result, list_desc = [], cutoff_sig = float(setup_dict["subsample_cutoff_sig"]), rate = float(setup_dict["subsample_rate"]))

    X_uniform = []
    y_uniform = []

    for entry in subsample_result:
        X_uniform.append(list(entry[1:]))
        y_uniform.append(entry[0])
    
    #X_uniform = np.asarray(X_uniform)
    #y_uniform = np.asarray(y_uniform).reshape((len(y_uniform),1))

    


    X_random = []
    y_random = []

    random_result = random_subsampling(result, setup_dict["random_pick_per_system"])

    for entry in random_result:
        X_random.append(list(entry[1:]))
        y_random.append(entry[0])
    
    #X_random = np.asarray(X_random)
    #y_random = np.asarray(y_random).reshape((len(y_random),1))

    return X_uniform, y_uniform, X_random, y_random




def read_sample_raw_data(raw_data_filename_list, raw_data_dirname, setup_dict):

    cwd = os.getcwd()

    os.chdir(cwd + "/" + raw_data_dirname)

    X_uniform_overall = []
    y_uniform_overall = []
    X_random_overall = []
    y_random_overall = []

    for raw_data_filename in raw_data_filename_list:
        temp_X_uniform, temp_y_uniform, temp_X_random, temp_y_random = read_sample_raw_data_one_system(raw_data_filename, setup_dict)

        X_uniform_overall += temp_X_uniform
        y_uniform_overall += temp_y_uniform
        X_random_overall += temp_X_random
        y_random_overall += temp_y_random

    os.chdir(cwd)

    X_uniform_overall = np.asarray(X_uniform_overall)
    y_uniform_overall = np.asarray(y_uniform_overall).reshape((len(y_uniform_overall),1))
    X_random_overall = np.asarray(X_random_overall)
    y_random_overall = np.asarray(y_random_overall).reshape((len(y_random_overall),1))

    return X_uniform_overall, y_uniform_overall, X_random_overall, y_random_overall





def write_to_file(data, result_data_filename, result_data_dirname, setup_dict):

    cwd = os.getcwd()

    if os.path.isdir(result_data_dirname) == False:
        os.makedirs(result_data_dirname)
    os.chdir(cwd + "/" + result_data_dirname)

    X_uniform_overall, y_uniform_overall, X_random_overall, y_random_overall = data


    with h5py.File(result_data_filename,'w') as database:

        database.create_dataset("X_uniform",data = X_uniform_overall)
        database.create_dataset("y_uniform",data = y_uniform_overall)
        database.create_dataset("X_random",data = X_random_overall)
        database.create_dataset("y_random",data = y_random_overall)

        
    os.chdir(cwd)
    return

def pick_subsample_data(raw_data_filename_list, raw_data_dirname, result_data_filename, result_data_dirname, setup_dict):

    data = read_sample_raw_data(raw_data_filename_list, raw_data_dirname, setup_dict)
    write_to_file(data, result_data_filename, result_data_dirname, setup_dict)

    return
"""

def write_to_file_per_system(system_name, data, result_data_filename, result_data_dirname, setup ):

    os.chdir(setup["result_data_dir"])


    X_uniform, y_uniform, X_random, y_random = data


    with h5py.File(result_data_filename,'a') as database:

        try:
            current_system_grp = database[system_name]
        except:
            current_system_grp = database.create_group(system_name)


        current_system_grp.create_dataset("X_uniform",data = X_uniform)
        current_system_grp.create_dataset("y_uniform",data = y_uniform)
        current_system_grp.create_dataset("X_random",data = X_random)
        current_system_grp.create_dataset("y_random",data = y_random)

        
    return


def create_system_grp(system_name, result_data_filename, result_data_dirname, setup ):
    os.chdir(setup["result_data_dir"])


    with h5py.File(result_data_filename,'a') as database:

        try:
            current_system_grp = database[system_name]
        except:
            current_system_grp = database.create_group(system_name)


    os.chdir(setup["raw_data_dir"])
    return




def read_sample_save_raw_data_one_system(system_name, raw_data_filename, result_data_filename, result_data_dirname, setup):

    create_system_grp(system_name, result_data_filename, result_data_dirname, setup )
    
    os.chdir(setup["raw_data_dir"])

    result_list = []
    #try:
    data =  h5py.File(raw_data_filename,'r')

    temp_y = np.asarray(data['output'][setup["output_image_name"]]["original"])
    result_list.append(temp_y.flatten())

    for image_name in sorted(setup["input_image_name_list"]):

        temp_data = np.asarray(data["input"][image_name]["original"])
        result_list.append(temp_data.flatten())

        try:
            MCSH_dict = setup["input_image_MCSH_dict"][image_name]
        except:
            MCSH_dict = setup["input_image_MCSH_dict"]["default"]

        print(MCSH_dict.keys())

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

    #except:
    #    print( "error loading, end")
    #    return
    for array in result_list:
        print(array.shape)
        print(array[0])

    result = np.stack(result_list, axis = 1)

    #result = np.array(zip(*result_list))
    print(result.shape)
    print(result[0])
    print( "done loading data, datasize: {}, {}".format(len(result),len(result[0])))

    #if len(result[0]) > int(setup["start_trial_component"]):
    #    subsample_result = subsampling_with_PCA(result, list_desc = [], cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]),start_trial_component = int(setup_dict["start_trial_component"]), max_component = int(setup_dict["max_component"]))

    #else:
    #    subsample_result = subsampling(result, list_desc = [], cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]))

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

            subsample_result = np.asarray(batch_subsampling_with_PCA(result, list_desc = [], cutoff_sig = subsample_cutoff, rate = subsample_rate, \
                                                                batch_size = setup["subsample_setup"]["batch_size"], recursive_level = int(setup["subsample_setup"]["recursive_level"]), \
                                                                start_trial_component = int(setup["subsample_setup"]["start_trial_component"]), max_component = int(setup["subsample_setup"]["max_component"]), \
                                                                standard_scale = standard_scale, method = method, verbose = verbose,\
                                                                final_overall_subsample = final_overall_subsample))
        else:
            subsample_result = np.asarray(batch_subsampling(result, list_desc = [], cutoff_sig = subsample_cutoff, rate = subsample_rate, \
                                                        batch_size = setup["subsample_setup"]["batch_size"], recursive_level = int(setup["subsample_setup"]["recursive_level"]), \
                                                        standard_scale = standard_scale, method = method, verbose = verbose,\
                                                        final_overall_subsample = final_overall_subsample))

    else:
        if PCA_subsample:
            subsample_result = np.asarray(subsampling_with_PCA( result, list_desc = [], cutoff_sig = subsample_cutoff, rate = subsample_rate, \
                                                            start_trial_component = int(setup["subsample_setup"]["start_trial_component"]), max_component = int(setup["subsample_setup"]["max_component"]), \
                                                            standard_scale = standard_scale, method = method, verbose = verbose))
        else:
            subsample_result = np.asarray(subsampling(result, list_desc = [], cutoff_sig = subsample_cutoff, rate = subsample_rate, \
                                                    standard_scale = standard_scale, method = method, verbose = verbose))


    X_uniform = []
    y_uniform = []

    for entry in subsample_result:
        X_uniform.append(list(entry[1:]))
        y_uniform.append(entry[0])
    
    #X_uniform = np.asarray(X_uniform)
    #y_uniform = np.asarray(y_uniform).reshape((len(y_uniform),1))

    


    X_random = []
    y_random = []

    random_result = random_subsampling(result, setup["random_pick_per_system"])

    for entry in random_result:
        X_random.append(list(entry[1:]))
        y_random.append(entry[0])
    
    #X_random = np.asarray(X_random)
    #y_random = np.asarray(y_random).reshape((len(y_random),1))

    data = (X_uniform, y_uniform, X_random, y_random)

    write_to_file_per_system(system_name, data, result_data_filename, result_data_dirname, setup)

    os.chdir(setup["cwd"])

    return 

def check_system_existence(system_name, result_data_filename, setup ):
    os.chdir(setup["result_data_dir"])

    test = False

    try:

        with h5py.File(result_data_filename,'r') as database:

            try:
                current_system_grp = database[system_name]
                test = True
            except:
                test = False

    except:
        test = False

    return test

    


def pick_subsample_data(raw_data_filename_dict, raw_data_dirname, result_data_filename, result_data_dirname, setup):

    cwd = os.getcwd()
    setup["cwd"] = cwd
    setup["raw_data_dir"] = cwd + "/" + raw_data_dirname
    setup["result_data_dir"] = cwd + "/" + result_data_dirname

    if os.path.isdir(setup["result_data_dir"]) == False:
        os.makedirs(setup["result_data_dir"])

    #os.chdir(cwd + "/" + raw_data_dirname)


    for system_name in raw_data_filename_dict.keys():
        print( "\nstart: {} ".format(system_name))
        system_exist_bool = check_system_existence(system_name, result_data_filename, setup )
        os.chdir(setup["cwd"])
        if system_exist_bool:
            print( "system already exist")
        else:
            raw_data_filename = raw_data_filename_dict[system_name]
            read_sample_save_raw_data_one_system(system_name, raw_data_filename, result_data_filename, result_data_dirname, setup)


    os.chdir(cwd)


    return