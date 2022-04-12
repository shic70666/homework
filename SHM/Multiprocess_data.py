#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Run experiments on server
    Created by: Hao Bai (hao.bai@insa-rouen.fr)
    Copyright (C) 2021 Hao Bai
    This program comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions.
'''
import numpy as np
import multiprocessing as mp
import time
import json
import os
import platform
import datetime
from collections import OrderedDict
# internal imports
from utils import handle_error, handle_subprocess_error

import pandas as pd
import numpy as np
import scipy.io as scio
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import scipy.signal as signal
from pathlib import Path

#!------------------------------------------------------------------------------
#!                                GLOBAL VARIABLES
#!------------------------------------------------------------------------------
class GLOBAL(object):
    # make an ordered dict for class attributes (not necessary for Python > 3.5)
    ordered_attrs = OrderedDict()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        GLOBAL.ordered_attrs[name] = value

    def __init__(self):
        self.MODE = "manual"
        self.PLATFORM = platform.node()
        self.CORES = os.cpu_count()
        self.ID = ""

    def __repr__(self):
        return "\n".join(["`{}`: {}".format(k, v) for k, v in
                          GLOBAL.ordered_attrs.items()])

    @property
    def CORES(self):
        return self._CORES

    @CORES.setter
    def CORES(self, number_of_cores):
        self._CORES = number_of_cores
        if self.PLATFORM == "lmn-cs.insa-rouen.fr":
            self.USED_CORES = int(number_of_cores-1)
        elif self.PLATFORM == "admin.mydomain.org":
            self.USED_CORES = int(number_of_cores-1)
        elif self.PLATFORM == "DESKTOP-7RMMINT":
            self.USED_CORES = int(number_of_cores)
        else:
            self.USED_CORES = int(number_of_cores-1)
G = GLOBAL()



#!------------------------------------------------------------------------------
#!                                   FUNCTIONS
#!------------------------------------------------------------------------------
def run_ChenShi(data_path):
    print(data_path)
    dataFile = Path(data_path).expanduser() #采用文件的相对路径
    X = scio.loadmat(dataFile)
    data_all = X['dasy']

    colum_num = [str(i).rjust(2, '0') for i in range(1, 16+1) ] #返回一个原字符串右对齐,并使用0填充至长度2的新字符串 

    data_case = []
    for i in range(0, len(colum_num)):
        colum_name = 'DA'+colum_num[i] 
        data_case.append( data_all[0,0][colum_name] )

    case_num = data_path[-6]
    ns = 128 # length of frames
    nss = 16 # width of feature matrix

    each_case = pd.DataFrame() 
    for ch_num in range(0, len(data_case)): 
        name1 = 'C_{}_CH_{}_frames'.format(case_num, ch_num+1) #格式化字符串
        name2 = 'C_{}_CH_{}_feature_matrixs'.format(case_num, ch_num+1)
        name3 = 'C_{}_CH_{}_labels'.format(case_num, ch_num+1)
        label = 'C_{}_CH_{}'.format(case_num, ch_num+1)
        each_channel = pd.DataFrame( columns=[name1, name2, name3] ) 
        
        frames, feature_matrixs = [], []
        for i in range(0, len(data_case[ch_num])//ns):
            frame = data_case[ch_num][i*ns: (i+1)*ns]
            frames.append(frame)
            
            frame_temp = frame.reshape((len(frame)//nss, nss))
            feature_matrixs.append(frame_temp)
            each_channel = each_channel.append([{name1:frame, name2:frame_temp,
                                                name3:label}], ignore_index=True) 
        
        each_case = pd.concat([each_case, each_channel], axis=1) 
        
        filename = dataFile.stem
        each_case.to_csv(filename+".csv")


def run_multiprocess(list_data_path,):
    print("\nParallel computing toolbox v0.1 (April 11 2022)")
    print("Copyright (C) 2022 Hao Bai (hao.bai@insa-rouen.fr)")
    print("========== Multiprocessing Mode ==========")
    print("Please confirm the following parameters:\n{}".format(G))

    print("[INFO] {} tasks are unfinished and will be executed ...".format(
        len(list_data_path)))
    
    if G.MODE.lower() != "auto":
        confirm = input("Continue ? (y/n): ")
        if not (confirm.lower() == "y" or confirm.lower() == "yes"):
            exit()
    ## Execution
    print("[INFO] Running the calculation now ...")
    pool = mp.Pool(G.USED_CORES)
    list_result = [pool.apply_async(run_ChenShi,
                        args=(c,))
                   for c in list_data_path]
    pool.close()
    pool.join()
    
    for x in list_result:
        res = x.get()

    print("[INFO] Finished !")
    return



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
def main():
    #* ---- Configue problem
    G.MODE = "auto"
    G.ID = ""

    #* ---- Execution
    list_data_path = [
        '~/SHM_Data/Shaker/Random/shm01s.mat',
        '~/SHM_Data/Shaker/Random/shm02s.mat',
        '~/SHM_Data/Shaker/Random/shm03s.mat',
        '~/SHM_Data/Shaker/Random/shm04s.mat',
        '~/SHM_Data/Shaker/Random/shm05s.mat',
        '~/SHM_Data/Shaker/Random/shm06s.mat',
        '~/SHM_Data/Shaker/Random/shm07s.mat',
        '~/SHM_Data/Shaker/Random/shm08s.mat',
        '~/SHM_Data/Shaker/Random/shm09s.mat',
    ]

    tik = time.time()
    fn = run_multiprocess(list_data_path)
    tok = time.time()
    G.DURATION = str(datetime.timedelta(seconds=round(tok-tik)))
    print("Total consumed time:", G.DURATION)
    return fn

if __name__=="__main__":
    main()
