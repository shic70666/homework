''' Pre-Processsing simualtion Data
'''
# %%
import pandas as pd
import numpy as np
import scipy.io as scio
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import scipy.signal as signal
from pathlib import Path

dataFile = Path('~/SHM_Data/Shaker/Random/shm01s.mat').expanduser()
X = scio.loadmat(dataFile)
data_all = X['dasy']

colum_num = [str(i).rjust(2, '0') for i in range(1, 16+1) ] #返回一个原字符串右对齐,并使用0填充至长度2的新字符串 

data_case = []
for i in range(0, len(colum_num)):
    colum_name = 'DA'+colum_num[i] 
    data_case.append( data_all[0,0][colum_name] )


#%% 
case_num = 1

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







