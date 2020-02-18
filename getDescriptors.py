# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:53:10 2017

@author: ray
"""
from __future__ import print_function
import numpy as np
import sys
import math

from nonorthogonalConvolutions import get_fftconv_with_known_stencil_no_wrap, get_fftconv_with_known_stencil_periodic, calc_MC_surface_harmonic_stencil_n, matrix_convolve2,  matrix_convolve3,matrix_convolve4
import h5py
import os
#from joblib import Parallel, delayed
#import multiprocessing
import itertools
import json
try: import cPickle as pickle
except: import pickle
from sklearn.preprocessing import normalize
from math import sqrt

    
def carve_out_matrix(matrix):
    old_shape_x, old_shape_y, old_shape_z = matrix.shape
    x = int(round(old_shape_x / 3. ,0))
    y = int(round(old_shape_y / 3. ,0))
    z = int(round(old_shape_z / 3. ,0))
   
    return matrix[x:2*x,y:2*y,z:2*z]

def get_homogeneous_gas_integral(n,r):
    return r*r*n*math.pi

def get_homo_nondimensional(int_arr, n_arr, r):
    temp = (4./3.)*r*r*r*math.pi
    result = np.divide(int_arr, n_arr)
    return np.divide(result, temp)

def get_homo_nondimensional_nave(int_arr, n_ave, r):
    temp = (4./3.)*r*r*r*math.pi*n_ave
    return np.divide(int_arr, temp)


def create_dataset(database, dataset_name, data):
    if dataset_name not in database.keys():
        database.create_dataset(dataset_name,data=data)
    return




def convolve_MCSH_subgroup(order_grp, order, subgroup_order, r_dict, MCSH_stencil_dict, n):
    print( "start order {} {}".format(order, subgroup_order))
    
    try:
        order_subgrp = order_grp[str(subgroup_order)]
    except:
        order_subgrp = order_grp.create_group(str(subgroup_order))

    for r in r_dict[str(order)]:

        dataset_name = '{}'.format(str(r).replace('.','-'))
        print( dataset_name)
        if dataset_name not in order_subgrp.keys():
            print( "start: {} MCSH {} {}".format(r,order, subgroup_order))
            #print MCSH_stencil_dict["{}_{}".format(order, subgroup_order)].keys()
            #print str(r)
            stencils = MCSH_stencil_dict["{}_{}".format(order, subgroup_order)][str(r)][0]
            pad = MCSH_stencil_dict["{}_{}".format(order, subgroup_order)][str(r)][1]

            temp_result = np.zeros_like(n)

            for temp_stencil in stencils:
                #temp_temp_result = get_fftconv_with_known_stencil_periodic(n,temp_stencil)
                temp_temp_result = matrix_convolve4(n,temp_stencil)
                temp_result = np.add(temp_result, np.square(temp_temp_result))

            temp_result = np.sqrt(temp_result)

            order_subgrp.create_dataset(dataset_name,data=temp_result)

    return



def convolve_one_input_image(n,name,h_dict,r_dict, MCSH_stencil_dict, filename):

    hx = h_dict["hx"]
    hy = h_dict["hy"]
    hz = h_dict["hz"]

    with h5py.File(filename,'a') as database:

        input_grp = database["input"]
        try:
            current_image_grp = input_grp[name]
        except:
            current_image_grp = input_grp.create_group(name)

        if "original" not in database.keys():
            current_image_grp.create_dataset("original",data=n)

        try:
            MCSH_grp = current_image_grp['MCSH']
        except:
            MCSH_grp = current_image_grp.create_group('MCSH')



        if "0" in r_dict.keys():
            
            try:
                order_grp = MCSH_grp['0']
            except:
                order_grp = MCSH_grp.create_group('0')

            convolve_MCSH_subgroup(order_grp, 0, 1, r_dict, MCSH_stencil_dict, n)


        if "1" in r_dict.keys():
            
            try:
                order_grp = MCSH_grp['1']
            except:
                order_grp = MCSH_grp.create_group('1')

            convolve_MCSH_subgroup(order_grp, 1, 1, r_dict, MCSH_stencil_dict, n)


        if "2" in r_dict.keys():
            
            try:
                order_grp = MCSH_grp['2']
            except:
                order_grp = MCSH_grp.create_group('2')

            convolve_MCSH_subgroup(order_grp, 2, 1, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 2, 2, r_dict, MCSH_stencil_dict, n)



        if "3" in r_dict.keys():
            
            try:
                order_grp = MCSH_grp['3']
            except:
                order_grp = MCSH_grp.create_group('3')

            convolve_MCSH_subgroup(order_grp, 3, 1, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 3, 2, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 3, 3, r_dict, MCSH_stencil_dict, n)


        if "4" in r_dict.keys():
            
            try:
                order_grp = MCSH_grp['4']
            except:
                order_grp = MCSH_grp.create_group('4')

            convolve_MCSH_subgroup(order_grp, 4, 1, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 4, 2, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 4, 3, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 4, 4, r_dict, MCSH_stencil_dict, n)


        if "5" in r_dict.keys():
            
            try:
                order_grp = MCSH_grp['5']
            except:
                order_grp = MCSH_grp.create_group('5')

            convolve_MCSH_subgroup(order_grp, 5, 1, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 5, 2, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 5, 3, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 5, 4, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 5, 5, r_dict, MCSH_stencil_dict, n)


        if "6" in r_dict.keys():
            
            try:
                order_grp = MCSH_grp['6']
            except:
                order_grp = MCSH_grp.create_group('6')

            convolve_MCSH_subgroup(order_grp, 6, 1, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 6, 2, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 6, 3, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 6, 4, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 6, 5, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 6, 6, r_dict, MCSH_stencil_dict, n)
            convolve_MCSH_subgroup(order_grp, 6, 7, r_dict, MCSH_stencil_dict, n)


    return





def get_MCSH_stencil_n(hx, hy, hz, r, l, n, accuracy = 6, stencil_lib = None, U = None):
    if U is None:
        U = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
    else:
        U = normalize(U)

    #print(U)
    print("calculating stencil: {} {} {} {} {} {}".format(hx, hy, hz, r, l, n))
    return calc_MC_surface_harmonic_stencil_n(hx, hy, hz, r, l, n, accuracy = accuracy, U = U)
    """
    if stencil_lib is None:
        print "calculating stencil: {} {} {} {} {} {}".format(hx, hy, hz, r, l, n)
        return calc_MC_surface_harmonic_stencil_n(hx, hy, hz, r, l, n, accuracy = accuracy, U = U)

    else:
        try:
            stencil = stencil_lib[hx][hy][hz][r][l][n][accuracy]["stencil"]
            pad = stencil_lib[hx][hy][hz][r][l][n][accuracy]["pad"]
            print "stencil loaded from lib"
            return stencil, pad
        except:
            print "calculating stencil: {} {} {} {} {} {}".format(hx, hy, hz, r, l, n)
            return calc_MC_surface_harmonic_stencil_n(hx, hy, hz, r, l, n, accuracy = accuracy, U = U)

    """


def prepare_MCSH_stencils(r_dict,system_dict, stencil_lib_filename = None):

    hx = system_dict["hx"]
    hy = system_dict["hy"]
    hz = system_dict["hz"]
    U  = np.array(system_dict["latvec"])
    print("U is")
    print(U)

    MCSH_stencil_dict = {}
    MCSH_stencil_dict["0_1"] = {}
    MCSH_stencil_dict["1_1"] = {}

    MCSH_stencil_dict["2_1"] = {}
    MCSH_stencil_dict["2_2"] = {}

    MCSH_stencil_dict["3_1"] = {}
    MCSH_stencil_dict["3_2"] = {}
    MCSH_stencil_dict["3_3"] = {}

    MCSH_stencil_dict["4_1"] = {}
    MCSH_stencil_dict["4_2"] = {}
    MCSH_stencil_dict["4_3"] = {}
    MCSH_stencil_dict["4_4"] = {}

    MCSH_stencil_dict["5_1"] = {}
    MCSH_stencil_dict["5_2"] = {}
    MCSH_stencil_dict["5_3"] = {}
    MCSH_stencil_dict["5_4"] = {}
    MCSH_stencil_dict["5_5"] = {}

    MCSH_stencil_dict["6_1"] = {}
    MCSH_stencil_dict["6_2"] = {}
    MCSH_stencil_dict["6_3"] = {}
    MCSH_stencil_dict["6_4"] = {}
    MCSH_stencil_dict["6_5"] = {}
    MCSH_stencil_dict["6_6"] = {}
    MCSH_stencil_dict["6_7"] = {}

    if stencil_lib_filename == None:
        print( "no stencel library loaded: Not specified")
        stencil_lib = None
    else:
        try:
            stencil_lib = pickle.load( open( stencil_lib_filename, "rb" ) )
        except:
            print( "no stencel library loaded: File no exist")
            stencil_lib = None

    try:
        for r in r_dict["0"]:
            stencil_Re_0_000, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 0, "000", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["0_1"][str(r)] = [[stencil_Re_0_000], pad ]
    except:
        pass





    try:
        for r in r_dict["1"]:
            stencil_Re_1_100, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 1, "100", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_1_010, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 1, "010", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_1_001, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 1, "001", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["1_1"][str(r)] = [[stencil_Re_1_100, stencil_Re_1_010, stencil_Re_1_001], pad ]
    except:
        pass



    try:
        for r in r_dict["2"]:
            stencil_Re_2_200, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 2, "200", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_2_020, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 2, "020", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_2_002, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 2, "002", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["2_1"][str(r)] = [[stencil_Re_2_200, stencil_Re_2_020, stencil_Re_2_002], pad ]

        for r in r_dict["2"]:
            stencil_Re_2_110, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 2, "110", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_2_101, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 2, "101", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_2_011, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 2, "011", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["2_2"][str(r)] = [[stencil_Re_2_110, stencil_Re_2_101, stencil_Re_2_011], pad ]
    except:
        pass



    try:
        for r in r_dict["3"]:
            stencil_Re_3_300, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 3, "300", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_3_030, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 3, "030", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_3_003, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 3, "003", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["3_1"][str(r)] = [[stencil_Re_3_300, stencil_Re_3_030, stencil_Re_3_003], pad ]

        for r in r_dict["3"]:
            stencil_Re_3_210, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 3, "210", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_3_201, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 3, "201", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_3_021, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 3, "021", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_3_120, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 3, "120", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_3_102, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 3, "102", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_3_012, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 3, "012", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["3_2"][str(r)] = [[stencil_Re_3_210, stencil_Re_3_201, stencil_Re_3_021, stencil_Re_3_120, stencil_Re_3_102, stencil_Re_3_012], pad ]

        for r in r_dict["3"]:
            stencil_Re_3_111, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 3, "111", accuracy = 6, stencil_lib = stencil_lib)
            MCSH_stencil_dict["3_3"][str(r)] = [[stencil_Re_3_111], pad ]
    except:
        pass



    try:
        for r in r_dict["4"]:
            stencil_Re_4_400, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "400", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_040, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "040", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_004, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "004", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["4_1"][str(r)] = [[stencil_Re_4_400, stencil_Re_4_040, stencil_Re_4_004], pad ]

        for r in r_dict["4"]:
            stencil_Re_4_310, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "310", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_301, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "301", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_031, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "031", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_130, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "130", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_103, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "103", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_013, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "013", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["4_2"][str(r)] = [[stencil_Re_4_310, stencil_Re_4_301, stencil_Re_4_031, stencil_Re_4_130, stencil_Re_4_103, stencil_Re_4_013], pad ]

        for r in r_dict["4"]:
            stencil_Re_4_220, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "220", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_202, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "202", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_022, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "022", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["4_3"][str(r)] = [[stencil_Re_4_220, stencil_Re_4_202, stencil_Re_4_022], pad ]

        for r in r_dict["4"]:
            stencil_Re_4_211, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "211", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_121, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "121", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_4_112, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 4, "112", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["4_4"][str(r)] = [[stencil_Re_4_211, stencil_Re_4_121, stencil_Re_4_112], pad ]
    except:
        pass



    try:
        for r in r_dict["5"]:
            stencil_Re_5_500, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "500", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_050, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "050", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_005, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "005", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["5_1"][str(r)] = [[stencil_Re_5_500, stencil_Re_5_050, stencil_Re_5_005], pad ]

        for r in r_dict["5"]:
            stencil_Re_5_410, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "410", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_401, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "401", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_041, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "041", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_140, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "140", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_104, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "104", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_014, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "014", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["5_2"][str(r)] = [[stencil_Re_5_410, stencil_Re_5_401, stencil_Re_5_041, stencil_Re_5_140, stencil_Re_5_104, stencil_Re_5_014], pad ]

        for r in r_dict["5"]:
            stencil_Re_5_320, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "320", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_302, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "302", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_032, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "032", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_230, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "230", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_203, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "203", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_023, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "023", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["5_3"][str(r)] = [[stencil_Re_5_320, stencil_Re_5_302, stencil_Re_5_032, stencil_Re_5_230, stencil_Re_5_203, stencil_Re_5_023], pad ]

        for r in r_dict["5"]:
            stencil_Re_5_311, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "311", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_131, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "131", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_113, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "113", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["5_4"][str(r)] = [[stencil_Re_5_311, stencil_Re_5_131, stencil_Re_5_113], pad ]

        for r in r_dict["5"]:
            stencil_Re_5_221, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "221", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_212, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "212", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_5_122, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 5, "122", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["5_5"][str(r)] = [[stencil_Re_5_221, stencil_Re_5_212, stencil_Re_5_122], pad ]
          
    except:
        pass



    try:
        for r in r_dict["6"]:
            stencil_Re_6_600, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "600", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_060, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "060", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_006, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "006", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["6_1"][str(r)] = [[stencil_Re_6_600, stencil_Re_6_060, stencil_Re_6_006], pad ]

        for r in r_dict["6"]:
            stencil_Re_6_510, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "510", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_501, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "501", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_051, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "051", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_150, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "150", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_105, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "105", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_015, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "015", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["6_2"][str(r)] = [[stencil_Re_6_510, stencil_Re_6_501, stencil_Re_6_051, stencil_Re_6_150, stencil_Re_6_105, stencil_Re_6_015], pad ]

        for r in r_dict["6"]:
            stencil_Re_6_420, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "420", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_402, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "402", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_042, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "042", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_240, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "240", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_204, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "204", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_024, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "024", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["6_3"][str(r)] = [[stencil_Re_6_420, stencil_Re_6_402, stencil_Re_6_042, stencil_Re_6_240, stencil_Re_6_204, stencil_Re_6_024], pad ]

        for r in r_dict["6"]:
            stencil_Re_6_411, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "411", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_141, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "141", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_114, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "114", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["6_4"][str(r)] = [[stencil_Re_6_411, stencil_Re_6_141, stencil_Re_6_114], pad ]

        for r in r_dict["6"]:
            stencil_Re_6_321, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "321", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_312, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "312", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_231, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "231", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_213, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "213", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_132, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "132", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_123, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "123", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["6_5"][str(r)] = [[stencil_Re_6_321, stencil_Re_6_312, stencil_Re_6_231, stencil_Re_6_213, stencil_Re_6_132, stencil_Re_6_123], pad ]

        for r in r_dict["6"]:
            stencil_Re_6_330, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "330", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_303, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "303", accuracy = 6, stencil_lib = stencil_lib, U = U)
            stencil_Re_6_033, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "033", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["6_6"][str(r)] = [[stencil_Re_6_330, stencil_Re_6_303, stencil_Re_6_033], pad ]

        for r in r_dict["6"]:
            stencil_Re_6_222, pad =  get_MCSH_stencil_n(hx, hy, hz, r, 6, "222", accuracy = 6, stencil_lib = stencil_lib, U = U)
            MCSH_stencil_dict["6_7"][str(r)] = [[stencil_Re_6_222], pad ]
      
    except:
        pass

    return MCSH_stencil_dict



def convolve_one_output_image(n,name,filename):


    with h5py.File(filename,'a') as database:

        output_grp = database["output"]
        try:
            current_image_grp = output_grp[name]
        except:
            current_image_grp = output_grp.create_group(name)

        if "original" not in database.keys():
            current_image_grp.create_dataset("original",data=n)


    return



def process_one_input_image(image,name,system_dict,r_dict, filename, U = None):

    if U is None:
        MCSH_stencil_dict = prepare_MCSH_stencils(r_dict,system_dict,  stencil_lib_filename = "stencil_library_n.pkl")
    else:
        MCSH_stencil_dict = prepare_MCSH_stencils(r_dict,system_dict,  stencil_lib_filename = "stencil_library_n.pkl", U = U)

    convolve_one_input_image(image,name,system_dict,r_dict,MCSH_stencil_dict, filename)

    return

def process_one_output_image(image,name, filename):
    with h5py.File(filename,'a') as database:

        output_grp = database["output"]
        try:
            current_image_grp = output_grp[name]
        except:
            current_image_grp = output_grp.create_group(name)

        if "original" not in database.keys():
            current_image_grp.create_dataset("original",data=image)
    return


def check_image_dimension_consistency(image_list):
    control_dimension = np.array(image_list[0]).shape

    for image in image_list:
        test_dimension = np.array(image).shape

        if test_dimension != control_dimension:
            return False

    return True

def initialize_database(filename):

    with h5py.File(filename,'a') as database:

        try:
            input_grp = database['input']
        except:
            input_grp = database.create_group('input')


        try:
            output_grp = database['output']
        except:
            output_grp = database.create_group('output')


        return

def process_one_system(image_list,setup_dict,system_dict,dirname = "scratch", sysname = "default", filename = "raw_data.h5"):
    cwd = os.getcwd()
    data_dir = cwd + "/" + dirname

    print("start processing")

    if os.path.isdir(data_dir) == False:
        os.makedirs(data_dir)

    os.chdir(data_dir)
    if os.path.isfile(filename):
        print( "file exist, skipped")

    else: 
        print( "file does not exist, started initialization")
        initialize_database(filename)

        if check_image_dimension_consistency(image_list) == False:
            print( "dimensions of the images are not consistent")
            raise ValueError


        name_list = setup_dict["name_list"]
        label_list = setup_dict["label_list"]
        r_dict_list = setup_dict["r_dict_list"]
        h_dict_list = setup_dict["h_dict_list"]

        for i, image in enumerate(image_list):
            temp_image = np.array(image)
            temp_name = name_list[i]
            temp_label = label_list[temp_name]
            if temp_label == "input":
                try:
                    temp_r_dict = r_dict_list[sysname]
                except:
                    temp_r_dict = r_dict_list["default"]

                try:
                    temp_system_dict = system_dict[sysname]
                except:
                    temp_system_dict = h_dict_list["default"]


                if "latvec" not in temp_system_dict:
                    try:
                        temp_system_dict["latvec"] = h_dict_list["default"]["latvec"]
                    except:
                        temp_system_dict["latvec"] = [[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]



                process_one_input_image(temp_image, temp_name, temp_system_dict, temp_r_dict, filename)


            if temp_label == "output":

                process_one_output_image(temp_image, temp_name, filename)


    os.chdir(cwd)


    return