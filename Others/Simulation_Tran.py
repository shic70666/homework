

#!/usr/bin/python
# coding: utf-8

'''
Function
input:   read real exciation 
output:  velocity and strain

'''

import numpy as np
import pandas as pd
from pyansys import Ansys
# from matplotlib import pyplot as plt
from pathlib import Path
# import sys
# sys.path.append("/home/lshi/Data/src") 
# import figureHB as hb 
# plt.style.use("/home/lshi/Data/src/figure.mplstyle")
import os
import shutil
import time
import datetime as dt


#%%
#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------
def run_tran_multifile(fileordernumber, fs) :
    folderpath = '/home/lshi/Data/excitation_30000'
    allname = sorted( os.listdir(folderpath) )
    filepath = folderpath+'/'+allname[fileordernumber-1]
    run_tran_singlefile(filepath, fs, fileordernumber) 

def run_tran_singlefile(filepath, fs, fileordernumber, save2disk= True )  :
    ## eg. filepath = '/home/lshi/Data/excitation_30000/S2_12_2119-68_30000.csv'
    
    #%% Convert the real test data
    exci_real = pd.read_csv(filepath, delimiter = ",", header = 0, skip_blank_lines = True )
    exci_real = np.array(exci_real)
    t = exci_real[: , 0 ]
    t = t - t[0]
    a = exci_real[: , 1 ]
    
    #  fft 
    nfft = 2**12
    amp1 = np.fft.fft(a, nfft) *2/nfft
    freq1 = np.fft.fftfreq(nfft, d = 1/fs) 
    freq = freq1[0: round(len(freq1)/2)+1]
    amp = np.abs(amp1[0: round(len(freq1)/2)+1])  
    
    # plt.plot(freq, amp)
    
    _ =np.where(amp ==max(amp))
    frequ = freq[_]
    print( 'the Fr is {} Hz'.format(frequ) )
    
    omegasq = (2 * np.pi * frequ) ** 2   
    exci_disp = -a * 9.8 / omegasq
    totaltime = t[-1]
    points = len(t)
    det = totaltime / points          #time step (s)

    tran_set = np.array([[1, totaltime],
                        [2, det],
                        [3, points]])
    # np.savetxt(r"G:\Simulation\tiaoshi\tran_set.txt", tran_set) 
    np.savetxt("tran_set.txt", tran_set) 
    
    my_uz = np.zeros((points, 3))
    ind = np.linspace(1, points, points)
    my_uz[:, 0] = ind                    # the 0 dimension is index  
    my_uz[:, 1] = t     
    my_uz[:, 2] = exci_disp 
    np.savetxt("my_uz.txt", my_uz) 
    

    #%% call ansys for transient analysis
    
    study = Ansys("Transient_bai.inp")
    # study.set("DMPRAT", damping)
    # study.save_inp()
    st=dt.datetime.now()
    study.run()
    et=dt.datetime.now()
    print('The CPU time for transient is {}'.format(et-st))   # get the simulation time
    
    if save2disk == True:
        outfile = 'result_tran.csv'
        newname = '/home/lshi/Codes/Lujie/APDL_output/Trans_results/result_tran_S'+str(fileordernumber)+'.csv'
        shutil.rename(outfile, newname)
    
    # study.rm("*.csv")
    
    # result_V.to_csv("Harmonic_bai_gather_Damping_VELO.csv")
    # result_E.to_csv("Harmonic_bai_gather_Damping_STRAIN.csv")
    
    study.autoclean()

#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
if __name__ == "__main__":
    filepath = '../APDL_data/excitation data/S2_12_2119-68.csv'
    filepath = Path(filepath)
    run_tran_singlefile(filepath, fs=2020, fileordernumber=12, save2disk= True )  

    
    