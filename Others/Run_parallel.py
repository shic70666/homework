#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 17:58:43 2022

@author: lshi
"""

#!------------------------------------------------------------------------------
#!                                   SETTINGS
#!------------------------------------------------------------------------------
import os
import platform
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path

from Simulation_Tran import filenumber2filepath, folderpath2filepath


#!------------------------------------------------------------------------------
#!                               GLOBAL VARIABLES
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
        elif self.PLATFORM == "admin.mydomain.org": # Changwu r730
            self.USED_CORES = int(number_of_cores-1)
        elif self.PLATFORM == "DESKTOP-7RMMINT": # Younes
            self.USED_CORES = int(number_of_cores-1)
        else:
            self.USED_CORES = int(number_of_cores-1)
G = GLOBAL()



#!------------------------------------------------------------------------------
#!                                  FUNCTION
#!------------------------------------------------------------------------------
def run_multiprocess(filenumber_list, folderpath, fs):
    print("\nNumerical Experiment on APDL v1.0 (Feb 15 2022)")
    print("Copyright (C) 2021-2022 Hao Bai (hao.bai@insa-rouen.fr)")
    print("========== Multiprocessing Mode ==========")
    ## Settings
    
    print("Experiment configurations:",
          "\n\tFolderpath:", folderpath, 
          "\n\tThe order number of specimens", filenumber_list,
          "\n\tThe frequency of sampling", fs,
         )
    print("Please confirm the following parameters:\n{}".format(G))

    # list_compo = [(x,) for x in variable]
    
    print("[INFO] {} tasks are unfinished and will be executed ...".format(len(filenumber_list)))
    
    
    if G.MODE.lower() != "auto":
        confirm = input("Continue ? (y/n): ")
        if not (confirm.lower() == "y" or confirm.lower() == "yes"):
            exit()
    
    ## Execution
    print("[INFO] Running the calculation now ...")
    pool = mp.Pool(G.USED_CORES)
    # list_result = [pool.apply_async(filenumber2filepath,
    #                                 args=(folderpath, fileordernumber, fs ))
    #                for fileordernumber in filenumber_list]
    list_result = [pool.apply_async(folderpath2filepath,
                                    args=(fileordernumber, fs ))
                   for fileordernumber in filenumber_list]
    pool.close()
    pool.join()

    # Plot figure
    # print("[INFO] Ploting figure ...")
    # plot_std_d(n)
    # print("[INFO] Finished !")


#!------------------------------------------------------------------------------
#!                                  TESTING
#!------------------------------------------------------------------------------
if __name__ == "__main__":
    fs = 2020
    folderpath = '../APDL_data/excitation data'
    folderpath = Path(folderpath)
    # fileordernumber = 12
    filenumber_list = np.array([3, 4])
    run_multiprocess(filenumber_list, folderpath, fs)

    
