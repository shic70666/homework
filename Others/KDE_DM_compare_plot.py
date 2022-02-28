#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Post-Processsing simualtion Data
'''
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
_ = Path("../src/figureplot.mplstyle").absolute()
plt.style.use(_)

import sys
sys.path.append("/media/lshi/DATA/Codes/staircase/src")
import figureHB as hb 

import os 
import math
from MLE_fun import para2dis


#%%------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------
def file_filter(filename, path):
    return sorted(Path(path).glob(filename))

def plot_d(n, verbose, truedist, data_dir="output/KDE_compare_normal", output_dir="figure/KDE_compare_normal", save_to_disk=True):
    global list_file, each_n_d, x_plot
    
    output_dir = Path(output_dir)
    
    
    if verbose == 1:
        truedis = truedist
        disname = 'normal'
    elif verbose == 2:
        truemean, truestd =  truedist[0],  truedist[1]
        mu = np.log(truemean) - 0.5*( np.log(1+ ((truestd**2)/(truemean**2) ) )  )
        # sigma = np.sqrt( np.log(1 + ((truestd**2)/(truemean**2)) ))
        median = np.exp(mu)
        print(median)
        truedis=[median, truestd]
        disname = 'lognorm'
        
    elif verbose == 3: 
        truemean, truestd =  truedist[0],  truedist[1] 
        shape = (truestd/truemean) **(-1.086)
        scale = truemean/ (1+1/shape)
        median = scale * ((np.log(2)) **( 1/shape ))
        print(median)
        truedis = [median, truestd]
        disname = 'weibull'
    
    
    
    
    
    # Load output files
    data_dir = Path(data_dir)
    filename = "STA_n{}_d*.csv".format(n)
    list_file = file_filter(filename, data_dir)

    
    u_dm_mean, u_dm_std = [], [] 
    u_kde_med_sc, u_kde_std_sc = [], []
    u_kde_med_sl, u_kde_std_sl = [], []
    u_kde_med_sj, u_kde_std_sj = [], []
    u_kde_med_pf, u_kde_std_pf = [], []
    
    u_kde_std_imp = []
    
    u_mle_med,    u_mle_std = [], []
    u_mle_med_ce, u_mle_std_ce = [], []
    u_mle_std_imp = []
    
    order = 0
    step = np.arange(0.1, 2.1, 0.1) * truedis[1]
    
    # figimp, axisimp = plt.subplots()
    for f in list_file:
        each_n_d = pd.read_csv(f, index_col=0)
        
        u_dm_mean_temp, u_dm_std_temp = [], [] 
        u_kde_med_sc_temp, u_kde_std_sc_temp = [], []
        u_kde_med_sl_temp, u_kde_std_sl_temp = [], []
        u_kde_med_sj_temp, u_kde_std_sj_temp = [], []
        u_kde_med_pf_temp, u_kde_std_pf_temp = [], []
        
        u_kde_std_imp_temp = []
        u_mle_std_imp_temp = []
        
        u_mle_med_temp, u_mle_std_temp = [], []
        u_mle_med_ce_temp, u_mle_std_ce_temp = [], []
        
        def u_matrix(colum_name, nor=truedis[0], inputDF = each_n_d):
            '''
            Parameters
            ----------
            inputDF : pd DateFrame
            colum_name : str
            nor : truedis[0] or truedis[1].

            '''
            u_temp = []
            u_temp.append( ( each_n_d[colum_name].mean() ) / nor )
            u_temp.append(  ( each_n_d[colum_name].std() )  )
            u_temp.append(  abs( each_n_d[colum_name].quantile(0.05) - each_n_d[colum_name].mean() ) / nor  )
            u_temp.append(  abs( each_n_d[colum_name].quantile(0.95) - each_n_d[colum_name].mean() ) / nor  )
            
            return u_temp
        
        u_dm_mean_temp = u_matrix('DM_mean', nor=truedis[0])
        u_dm_std_temp  = u_matrix('DM_std', nor=truedis[1])
        
        u_kde_med_sc_temp = u_matrix('KDE_med_sc', nor=truedis[0])
        u_kde_std_sc_temp = u_matrix('KDE_std_sc', nor=truedis[1])
        u_kde_med_sl_temp = u_matrix('KDE_med_sl', nor=truedis[0])
        u_kde_std_sl_temp = u_matrix('KDE_std_sl', nor=truedis[1])
        u_kde_med_sj_temp = u_matrix('KDE_med_sj', nor=truedis[0])
        u_kde_std_sj_temp = u_matrix('KDE_std_sj', nor=truedis[1])
        u_kde_med_pf_temp = u_matrix('KDE_med_pf', nor=truedis[0])
        u_kde_std_pf_temp = u_matrix('KDE_std_pf', nor=truedis[1])
        
        u_mle_med_temp = u_matrix('MLE_med', nor=truedis[0])
        u_mle_std_temp = u_matrix('MLE_std', nor=truedis[1])
        u_mle_med_ce_temp = u_matrix('MLE_med_ce', nor=truedis[0])
        # u_mle_std_ce_temp = u_matrix('MLE_std_ce', nor=truedis[1])
        
        #%% correction to the MLE
        d = step[order]
        order = order + 1
        
        ### delimite the sigular value 
        imp = np.array( each_n_d['MLE_std_ce'] )
        imp3 = []
        for iii in range(0, len(imp)):
            if imp[iii] < 5*d :
                imp3.append(imp[iii])                
        imp3 = np.array(imp3)                
        imp4 = imp3 *(n-1)/(n-6.5)
        
        u_mle_std_ce_temp.append(  ( np.mean(imp3) ) / truedis[1]  )
        u_mle_std_ce_temp.append(  ( np.std(imp3) )  )
        u_mle_std_ce_temp.append(  abs( np.quantile(imp3, 0.05) - np.mean(imp3) ) / truedis[1]  )
        u_mle_std_ce_temp.append(  abs( np.quantile(imp3, 0.95) - np.mean(imp3) ) / truedis[1]  )

        u_mle_std_imp_temp.append(  ( np.mean(imp4) ) / truedis[1]  )
        u_mle_std_imp_temp.append(  ( np.std(imp4) )  )
        u_mle_std_imp_temp.append(  abs( np.quantile(imp4, 0.05) - np.mean(imp4) ) / truedis[1]  )
        u_mle_std_imp_temp.append(  abs( np.quantile(imp4, 0.95) - np.mean(imp4) ) / truedis[1]  )
        
        #%% correction to the KDE 
        imp1 = np.array( each_n_d['KDE_std_pf'] )  # standard deviation 
        # imp2 = -np.tanh(- 1) + imp1
        # imp2 = -9 *((d / imp1) -1)  + imp1
        # imp2 = ((imp1/d)) ** 0.2
        # imp2 = ( 0.001 ** ( (imp1/d)-1 ) )   #smaller
        # imp2 = -1 * ( 0.01 ** ( (imp1/d)-1 ) ) + imp1
        
        #axisimp.plot( imp2, imp1, ".", label= str(int(d)) )
        #axisimp.set(xlim = [0, 25] , ylim= [0, 25])
        
        # axisimp.plot([d,]*len(imp1), imp1-d, ".", label='{:.2f}'.format(d))
        # plt.legend( loc='best')
        
        ### improvment ----------
        s_imp2 = []
        for s_imp1 in imp1 :
            if s_imp1 > d :
                s_imp2.append( ((s_imp1/d) + 13*s_imp1) ** 0.51 )
            else:
                s_imp2.append( ((s_imp1/d) +  7*s_imp1) ** 0.51 )
        imp2 = s_imp2
        #### ----------------------

        u_kde_std_imp_temp.append(  ( np.mean(imp2) ) / truedis[1]  )
        u_kde_std_imp_temp.append(  ( np.std(imp2) )  )
        u_kde_std_imp_temp.append(  abs( np.quantile(imp2, 0.05) - np.mean(imp2) ) / truedis[1]  )
        u_kde_std_imp_temp.append(  abs( np.quantile(imp2, 0.95) - np.mean(imp2) ) / truedis[1]  )
        
        u_dm_mean.append(u_dm_mean_temp)
        u_dm_std.append(u_dm_std_temp)
        
        u_kde_med_sc.append(u_kde_med_sc_temp)
        u_kde_std_sc.append(u_kde_std_sc_temp)
        u_kde_med_sl.append(u_kde_med_sl_temp)
        u_kde_std_sl.append(u_kde_std_sl_temp)
        u_kde_med_sj.append(u_kde_med_sj_temp)
        u_kde_std_sj.append(u_kde_std_sj_temp)
        u_kde_med_pf.append(u_kde_med_pf_temp)
        u_kde_std_pf.append(u_kde_std_pf_temp)
        u_kde_std_imp.append(u_kde_std_imp_temp)
        
        u_mle_med.append(u_mle_med_temp)
        u_mle_std.append(u_mle_std_temp)
        u_mle_med_ce.append(u_mle_med_ce_temp)
        u_mle_std_ce.append(u_mle_std_ce_temp)
        u_mle_std_imp.append(u_mle_std_imp_temp)
        
    # uncertainty data type
    u_dm_mean, u_dm_std = np.array(u_dm_mean), np.array(u_dm_std)
    u_kde_med_sc, u_kde_std_sc = np.array(u_kde_med_sc), np.array(u_kde_std_sc)
    u_kde_med_sl, u_kde_std_sl = np.array(u_kde_med_sl), np.array(u_kde_std_sl)
    u_kde_med_sj, u_kde_std_sj = np.array(u_kde_med_sj), np.array(u_kde_std_sj)
    u_kde_med_pf, u_kde_std_pf = np.array(u_kde_med_pf), np.array(u_kde_std_pf)
    u_kde_std_imp =  np.array(u_kde_std_imp)
    u_mle_med, u_mle_std = np.array(u_mle_med), np.array(u_mle_std)
    u_mle_med_ce, u_mle_std_ce = np.array(u_mle_med_ce), np.array(u_mle_std_ce)
    u_mle_std_imp =  np.array(u_mle_std_imp)
    

    #%% Figures 
    # x_plot = [round(i/truedis[1], 1) for i in range(1, len(list_file)+1)]
    x_plot = np.arange(0.1, 2.1, 0.1)

    #%% KDE final
    fig, axis = plt.subplots()
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')
    # axis.plot([min(x_plot), max(x_plot)], [1,1], color='forestgreen', ls='--', linewidth='0.5')

    axis.errorbar(x_plot, u_dm_std[:, 0].T, yerr=u_dm_std[:,[2,3]].T, alpha= 0.7, \
                  fmt='x', markersize=5, ecolor='green', color='darkgreen', elinewidth=1, capsize=5, label='DM')
    axis.plot(x_plot, u_dm_std[:, 0].T,  alpha= 0.7, color='green', ls='--')
    
    axis.errorbar(x_plot, u_mle_std_imp[:, 0].T, yerr=u_mle_std_imp[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=5, ecolor='blue', color='darkblue', elinewidth=1, capsize=4, label='MLE')
    axis.plot(x_plot, u_mle_std_imp[:, 0].T,  alpha= 0.7, color='blue', ls='--')
    
    axis.errorbar(x_plot, u_kde_std_imp[:, 0].T, yerr=u_kde_std_imp[:,[2,3]].T,  alpha= 0.7, \
              fmt='o', markersize=4, ecolor='red', color='darkred', elinewidth=1, capsize=6, label='KDE')
    axis.plot(x_plot, u_kde_std_imp[:, 0].T,  alpha= 0.7, color='red', ls='--')
        
    # axis.errorbar(x_plot, u_mean_lan[:, 4].T, yerr=u_mean_lan[:,[6,7]].T, \
        # fmt='v', markersize=4, ecolor='orange', color='darkorange', elinewidth=1, capsize=3, label='LAN-KDE-SL')
    axis.set(xlabel='Normalised step size (n=' + str(n) + ')',
             ylabel= 'Normalised standard deviation',
             # yticks=range(0, 20, 1)
             )
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_KDE_std_n{}.svg".format(n)))
    
        
    #%% MLE compare
    fig, axis = plt.subplots()
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')

    axis.errorbar(x_plot, u_mle_std[:, 0].T, yerr=u_mle_std[:,[2,3]].T,  alpha= 0.7, \
                  fmt='x', markersize=3, ecolor='green', color='green', elinewidth=1, capsize=5, label='MLE')
    axis.plot(x_plot, u_mle_std[:, 0].T,  alpha= 0.7, color='green', ls='--')  
    axis.errorbar(x_plot, u_mle_std_ce[:, 0].T, yerr=u_mle_std_ce[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=4, ecolor='blue', color='darkblue', elinewidth=1, capsize=6, label='MLE-Cen')
    axis.plot(x_plot, u_mle_std_ce[:, 0].T,  alpha= 0.7, color='blue', ls='--') 
    axis.errorbar(x_plot, u_mle_std_imp[:, 0].T, yerr=u_mle_std_imp[:,[2,3]].T,  alpha= 0.7, \
                  fmt='o', markersize=5, ecolor='red', color='darkred', elinewidth=1, capsize=4, label='MLE-Cor')
    axis.plot(x_plot, u_mle_std_imp[:, 0].T,  alpha= 0.7, color='red', ls='--')   
    # axis.errorbar(x_plot, u_dm_std[:, 0].T, yerr=u_dm_std[:,[2,3]].T,  alpha= 0.7, \
    #                fmt='v', markersize=6, ecolor='orange', color='darkorange', elinewidth=1, capsize=3, label='DM')
    axis.set(xlabel='Normalised step size (n=' + str(n) + ')',
              ylabel= 'Normalised standard deviation',
              # yticks=range(0, 20, 1)
              )
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_MLE_std_n{}.svg".format(n)))
    
    #%% KDE compare different bandwidth
    fig, axis = plt.subplots()
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')

    axis.errorbar(x_plot, u_kde_std_sc[:, 0].T, yerr=u_kde_std_sc[:,[2,3]].T,  alpha= 0.7, \
                  fmt='x', markersize=3, ecolor='green', color='darkgreen', elinewidth=1, capsize=5, label='KDE-SC')
    axis.plot(x_plot, u_kde_std_sc[:, 0].T,  alpha= 0.7, color='green', ls='--')  
    axis.errorbar(x_plot, u_kde_std_sl[:, 0].T, yerr=u_kde_std_sl[:,[2,3]].T,  alpha= 0.7, \
              fmt='o', markersize=4, ecolor='blue', color='darkblue', elinewidth=1, capsize=6, label='KDE-SI')
    axis.plot(x_plot, u_kde_std_sl[:, 0].T,  alpha= 0.7, color='blue', ls='--')
    axis.errorbar(x_plot, u_kde_std_sj[:, 0].T, yerr=u_kde_std_sj[:,[2,3]].T,  alpha= 0.7, \
                    fmt='v', markersize=6, ecolor='orange', color='darkorange', elinewidth=1, capsize=3, label='KDE-SJ')
    axis.plot(x_plot, u_kde_std_sj[:, 0].T,  alpha= 0.7, color='orange', ls='--')
    axis.errorbar(x_plot, u_kde_std_pf[:, 0].T, yerr=u_kde_std_pf[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=6, ecolor='red', color='darkred', elinewidth=1, capsize=3, label='KDE-PF')
    axis.plot(x_plot, u_kde_std_pf[:, 0].T,  alpha= 0.7, color='red', ls='--')    
        
    # axis.errorbar(x_plot, u_kde_std_imp[:, 0].T, yerr=u_kde_std_imp[:,[2,3]].T,  alpha= 0.7, \
    #          fmt='^', markersize=5, ecolor='cyan', color='darkcyan', elinewidth=1, capsize=4, label='KDE-Cor')
    
    axis.set(xlabel='Normalised step size (n=' + str(n) + ')',
             ylabel= 'Normalised standard deviation',
             # yticks=range(0, 20, 1)
             )
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_bw_std_n{}.svg".format(n)))
    
    
    
        
    #%%   mean (median)    
    # KDE final
    fig, axis = plt.subplots()
    plt.subplots_adjust(left=0.14, right=0.98)
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')
    
    axis.errorbar(x_plot, u_dm_mean[:, 0].T, yerr=u_dm_mean[:,[2,3]].T, alpha= 0.7, \
                  fmt='x', markersize=6, ecolor='green', color='darkgreen', elinewidth=1, capsize=6, label='DM')
    axis.plot(x_plot, u_dm_mean[:, 0].T,  alpha= 0.7, color='green', ls='--')
    axis.errorbar(x_plot, u_mle_med[:, 0].T, yerr=u_mle_med[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=5, ecolor='blue', color='darkblue', elinewidth=1, capsize=4, label='MLE')
    axis.plot(x_plot, u_mle_med[:, 0].T,  alpha= 0.7, color='blue', ls='--')
    axis.errorbar(x_plot, u_kde_med_pf[:, 0].T, yerr=u_kde_med_pf[:,[2,3]].T,  alpha= 0.7, \
              fmt='o', markersize=4, ecolor='red', color='darkred', elinewidth=1, capsize=6, label='KDE')
    axis.plot(x_plot, u_kde_med_pf[:, 0].T,  alpha= 0.7, color='red', ls='--')
    
    axis.set(xlabel='Normalised step size (n=' + str(n) + ')', 
             ylabel= 'Normalised median value' )
    # hb.set_axis_sci(axis, scilimits=(-1, -1))
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_KDE_med_n{}.svg".format(n)))
        

    #%% MLE compare mean 
    fig, axis = plt.subplots()
    plt.subplots_adjust(left=0.14, right=0.98)
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')

    axis.errorbar(x_plot, u_mle_med[:, 0].T, yerr=u_mle_med[:,[2,3]].T,  alpha= 0.7, \
                  fmt='x', markersize=3, ecolor='green', color='green', elinewidth=1, capsize=5, label='MLE')
    axis.plot(x_plot, u_mle_med[:, 0].T,  alpha= 0.7, color='green', ls='--')  
    axis.errorbar(x_plot, u_mle_med_ce[:, 0].T, yerr=u_mle_med_ce[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=4, ecolor='blue', color='darkblue', elinewidth=1, capsize=6, label='MLE-Cen')
    axis.plot(x_plot, u_mle_med_ce[:, 0].T,  alpha= 0.7, color='blue', ls='--') 
    # axis.errorbar(x_plot, u_mle_med_imp[:, 0].T, yerr=u_mle_med_imp[:,[2,3]].T,  alpha= 0.7, \
    #               fmt='o', markersize=5, ecolor='red', color='darkred', elinewidth=1, capsize=4, label='KDE-Cor')
    # axis.plot(x_plot, u_mle_med_imp[:, 0].T,  alpha= 0.7, color='red', ls='--')   
    # axis.errorbar(x_plot, u_dm_std[:, 0].T, yerr=u_dm_std[:,[2,3]].T,  alpha= 0.7, \
    #                fmt='v', markersize=6, ecolor='orange', color='darkorange', elinewidth=1, capsize=3, label='DM')
    axis.set(xlabel='Normalised step size (n=' + str(n) + ')',
              ylabel= 'Normalised median value',
              # yticks=range(0, 20, 1)
              )
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_MLE_med_n{}.svg".format(n)))
        
    
    #%% median : KDE compare different bandwidth
    fig, axis = plt.subplots()
    plt.subplots_adjust(left=0.13, right=0.98)
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')

    axis.errorbar(x_plot, u_kde_med_sc[:, 0].T, yerr=u_kde_med_sc[:,[2,3]].T,  alpha= 0.7, \
                  fmt='x', markersize=3, ecolor='green', color='darkgreen', elinewidth=1, capsize=5, label='KDE-SC')
    axis.plot(x_plot, u_kde_med_sc[:, 0].T,  alpha= 0.7, color='green', ls='--')  
    axis.errorbar(x_plot, u_kde_med_sl[:, 0].T, yerr=u_kde_med_sl[:,[2,3]].T,  alpha= 0.7, \
              fmt='o', markersize=4, ecolor='blue', color='darkblue', elinewidth=1, capsize=6, label='KDE-SI')
    axis.plot(x_plot, u_kde_med_sl[:, 0].T,  alpha= 0.7, color='blue', ls='--')
    axis.errorbar(x_plot, u_kde_med_sj[:, 0].T, yerr=u_kde_med_sj[:,[2,3]].T,  alpha= 0.7, \
                    fmt='v', markersize=6, ecolor='orange', color='darkorange', elinewidth=1, capsize=3, label='KDE-SJ')
    axis.plot(x_plot, u_kde_med_sj[:, 0].T,  alpha= 0.7, color='orange', ls='--')
    axis.errorbar(x_plot, u_kde_med_pf[:, 0].T, yerr=u_kde_med_pf[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=6, ecolor='red', color='darkred', elinewidth=1, capsize=3, label='KDE-PF')
    axis.plot(x_plot, u_kde_med_pf[:, 0].T,  alpha= 0.7, color='red', ls='--')    
        
    # axis.errorbar(x_plot, u_kde_std_imp[:, 0].T, yerr=u_kde_std_imp[:,[2,3]].T,  alpha= 0.7, \
    #          fmt='^', markersize=5, ecolor='cyan', color='darkcyan', elinewidth=1, capsize=4, label='KDE-Cor')
    
    axis.set(xlabel='Normalised step size (n=' + str(n) + ')',
             ylabel= 'Normalised median value',
             # yticks=range(0, 20, 1)
             )
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_bw_med_n{}.svg".format(n)))
        
        
#%%       
def plot_n(n_list, d, verbose=1, data_dir="output/KDE_compare_normal", output_dir="figure/KDE_compare_normal", save_to_disk=True):
    # global list_file, each_n_d, list_fil, u_kde_std, imp1, imp2 

    output_dir = Path(output_dir)
    truedist = [400, 20]
    
    if verbose == 1:
        truedis = truedist
        disname = 'normal'
    elif verbose == 2:
        truemean, truestd =  truedist[0],  truedist[1]
        mu = np.log(truemean) - 0.5*( np.log(1+ ((truestd**2)/(truemean**2) ) )  )
        # sigma = np.sqrt( np.log(1 + ((truestd**2)/(truemean**2)) ))
        median = np.exp(mu)
        print(median)
        truedis=[median, truestd]
        disname = 'lognorm'
        
    elif verbose == 3: 
        truemean, truestd =  truedist[0],  truedist[1] 
        shape = (truestd/truemean) **(-1.086)
        scale = truemean/ (1+1/shape)
        median = scale * ((np.log(2)) **( 1/shape ))
        print(median)
        truedis = [median, truestd]
        disname = 'weibull'
    
    
    
    
    # Load output files
    # data_dir = Path(data_dir)
    # filelist = os.listdir(data_dir)
    
    # filename = "STA_n*_step{}.csv".format(d)
    # list_file = file_filter(filename, data_dir)
    
    # list_filename = []
    # for list_file_temp in list_file:
    #     list_filename.append( list_file_temp.stem )

    u_dm_mean, u_dm_std = [], [] 
    u_kde_med_sc, u_kde_std_sc = [], []
    u_kde_med_sl, u_kde_std_sl = [], []
    u_kde_med_sj, u_kde_std_sj = [], []
    u_kde_med_pf, u_kde_std_pf = [], []
    
    u_kde_std_imp = []
    
    u_mle_med,    u_mle_std = [], []
    u_mle_med_ce, u_mle_std_ce = [], []
    u_mle_std_imp = []
    
    for n in n_list:
        filename = 'STA_n'+str(n)+'_d1.0.csv'
        filepath = data_dir+'/'+filename
        
        each_n_d = pd.read_csv(filepath, index_col=0)
        
        u_dm_mean_temp, u_dm_std_temp = [], [] 
        u_kde_med_sc_temp, u_kde_std_sc_temp = [], []
        u_kde_med_sl_temp, u_kde_std_sl_temp = [], []
        u_kde_med_sj_temp, u_kde_std_sj_temp = [], []
        u_kde_med_pf_temp, u_kde_std_pf_temp = [], []
        
        u_kde_std_imp_temp = []
        u_mle_std_imp_temp = []
        
        u_mle_med_temp, u_mle_std_temp = [], []
        u_mle_med_ce_temp, u_mle_std_ce_temp = [], []
        
        def u_matrix(colum_name, nor=truedis[0], inputDF = each_n_d):
            '''
            Parameters
            ----------
            inputDF : pd DateFrame
            colum_name : str
            nor : truedis[0] or truedis[1].

            '''
            u_temp = []
            u_temp.append( ( each_n_d[colum_name].mean() ) / nor )
            u_temp.append(  ( each_n_d[colum_name].std() )  )
            u_temp.append(  abs( each_n_d[colum_name].quantile(0.05) - each_n_d[colum_name].mean() ) / nor  )
            u_temp.append(  abs( each_n_d[colum_name].quantile(0.95) - each_n_d[colum_name].mean() ) / nor  )
            
            return u_temp
        
        u_dm_mean_temp = u_matrix('DM_mean', nor=truedis[0])
        u_dm_std_temp  = u_matrix('DM_std', nor=truedis[1])
        
        u_kde_med_sc_temp = u_matrix('KDE_med_sc', nor=truedis[0])
        u_kde_std_sc_temp = u_matrix('KDE_std_sc', nor=truedis[1])
        u_kde_med_sl_temp = u_matrix('KDE_med_sl', nor=truedis[0])
        u_kde_std_sl_temp = u_matrix('KDE_std_sl', nor=truedis[1])
        u_kde_med_sj_temp = u_matrix('KDE_med_sj', nor=truedis[0])
        u_kde_std_sj_temp = u_matrix('KDE_std_sj', nor=truedis[1])
        u_kde_med_pf_temp = u_matrix('KDE_med_pf', nor=truedis[0])
        u_kde_std_pf_temp = u_matrix('KDE_std_pf', nor=truedis[1])
        
        u_mle_med_temp = u_matrix('MLE_med', nor=truedis[0])
        u_mle_std_temp = u_matrix('MLE_std', nor=truedis[1])
        u_mle_med_ce_temp = u_matrix('MLE_med_ce', nor=truedis[0])
        u_mle_std_ce_temp = u_matrix('MLE_std_ce', nor=truedis[1])
        
        #%% correction to the MLE
                
        ### delimite the sigular value 
        imp = np.array( each_n_d['MLE_std_ce'] )
        imp3 = []
        for iii in range(0, len(imp)):
            if imp[iii] < 5*d :
                imp3.append(imp[iii])                
        imp3 = np.array(imp3)                
        imp4 = imp3 *(n-1)/(n-6.5)
        
        u_mle_std_ce_temp.append(  ( np.mean(imp3) ) / truedis[1]  )
        u_mle_std_ce_temp.append(  ( np.std(imp3) )  )
        u_mle_std_ce_temp.append(  abs( np.quantile(imp3, 0.05) - np.mean(imp3) ) / truedis[1]  )
        u_mle_std_ce_temp.append(  abs( np.quantile(imp3, 0.95) - np.mean(imp3) ) / truedis[1]  )

        u_mle_std_imp_temp.append(  ( np.mean(imp4) ) / truedis[1]  )
        u_mle_std_imp_temp.append(  ( np.std(imp4) )  )
        u_mle_std_imp_temp.append(  abs( np.quantile(imp4, 0.05) - np.mean(imp4) ) / truedis[1]  )
        u_mle_std_imp_temp.append(  abs( np.quantile(imp4, 0.95) - np.mean(imp4) ) / truedis[1]  )
        
        #%% correction to the KDE 
        imp1 = np.array( each_n_d['KDE_std_pf'] )  # standard deviation 
                
        ### improvment ----------
        s_imp2 = []
        for s_imp1 in imp1 :
            if s_imp1 > d :
                s_imp2.append( ((s_imp1/d) + 13*s_imp1) ** 0.51 )
            else:
                s_imp2.append( ((s_imp1/d) +  7*s_imp1) ** 0.51 )
        imp2 = s_imp2
        #### ----------------------

        u_kde_std_imp_temp.append(  ( np.mean(imp2) ) / truedis[1]  )
        u_kde_std_imp_temp.append(  ( np.std(imp2) )  )
        u_kde_std_imp_temp.append(  abs( np.quantile(imp2, 0.05) - np.mean(imp2) ) / truedis[1]  )
        u_kde_std_imp_temp.append(  abs( np.quantile(imp2, 0.95) - np.mean(imp2) ) / truedis[1]  )
        
        u_dm_mean.append(u_dm_mean_temp)
        u_dm_std.append(u_dm_std_temp)
        
        u_kde_med_sc.append(u_kde_med_sc_temp)
        u_kde_std_sc.append(u_kde_std_sc_temp)
        u_kde_med_sl.append(u_kde_med_sl_temp)
        u_kde_std_sl.append(u_kde_std_sl_temp)
        u_kde_med_sj.append(u_kde_med_sj_temp)
        u_kde_std_sj.append(u_kde_std_sj_temp)
        u_kde_med_pf.append(u_kde_med_pf_temp)
        u_kde_std_pf.append(u_kde_std_pf_temp)
        u_kde_std_imp.append(u_kde_std_imp_temp)
        
        u_mle_med.append(u_mle_med_temp)
        u_mle_std.append(u_mle_std_temp)
        u_mle_med_ce.append(u_mle_med_ce_temp)
        u_mle_std_ce.append(u_mle_std_ce_temp)
        u_mle_std_imp.append(u_mle_std_imp_temp)
        
    # uncertainty data type
    u_dm_mean, u_dm_std = np.array(u_dm_mean), np.array(u_dm_std)
    u_kde_med_sc, u_kde_std_sc = np.array(u_kde_med_sc), np.array(u_kde_std_sc)
    u_kde_med_sl, u_kde_std_sl = np.array(u_kde_med_sl), np.array(u_kde_std_sl)
    u_kde_med_sj, u_kde_std_sj = np.array(u_kde_med_sj), np.array(u_kde_std_sj)
    u_kde_med_pf, u_kde_std_pf = np.array(u_kde_med_pf), np.array(u_kde_std_pf)
    u_kde_std_imp =  np.array(u_kde_std_imp)
    u_mle_med, u_mle_std = np.array(u_mle_med), np.array(u_mle_std)
    u_mle_med_ce, u_mle_std_ce = np.array(u_mle_med_ce), np.array(u_mle_std_ce)
    u_mle_std_imp =  np.array(u_mle_std_imp)
    

    #%% Figures 
    x_plot = n_list
    dn = d / truedis[1]

    #%% KDE final
    fig, axis = plt.subplots()
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')

    axis.errorbar(x_plot, u_dm_std[:, 0].T, yerr=u_dm_std[:,[2,3]].T, alpha= 0.7, \
                  fmt='x', markersize=5, ecolor='green', color='darkgreen', elinewidth=1, capsize=5, label='DM')
    axis.plot(x_plot, u_dm_std[:, 0].T,  alpha= 0.7, color='green', ls='--')
    
    axis.errorbar(x_plot, u_mle_std_imp[:, 0].T, yerr=u_mle_std_imp[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=5, ecolor='blue', color='darkblue', elinewidth=1, capsize=4, label='MLE')
    axis.plot(x_plot, u_mle_std_imp[:, 0].T,  alpha= 0.7, color='blue', ls='--')
    
    axis.errorbar(x_plot, u_kde_std_imp[:, 0].T, yerr=u_kde_std_imp[:,[2,3]].T,  alpha= 0.7, \
              fmt='o', markersize=4, ecolor='red', color='darkred', elinewidth=1, capsize=6, label='KDE')
    axis.plot(x_plot, u_kde_std_imp[:, 0].T,  alpha= 0.7, color='red', ls='--')

    axis.set(xlabel='Number of specimens (Normalised d=' + str(dn) + ')',
             ylabel= 'Normalised standard deviation',
             xticks= [10, 20, 30, 40, 50, 70, 100]
             )
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_KDE_std_d{}.svg".format(dn)))
    
        
    #%% MLE compare
    fig, axis = plt.subplots()
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')

    axis.errorbar(x_plot, u_mle_std[:, 0].T, yerr=u_mle_std[:,[2,3]].T,  alpha= 0.7, \
                  fmt='x', markersize=3, ecolor='green', color='green', elinewidth=1, capsize=5, label='MLE')
    axis.plot(x_plot, u_mle_std[:, 0].T,  alpha= 0.7, color='green', ls='--')  
    axis.errorbar(x_plot, u_mle_std_ce[:, 0].T, yerr=u_mle_std_ce[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=4, ecolor='blue', color='darkblue', elinewidth=1, capsize=6, label='MLE-Cen')
    axis.plot(x_plot, u_mle_std_ce[:, 0].T,  alpha= 0.7, color='blue', ls='--') 
    axis.errorbar(x_plot, u_mle_std_imp[:, 0].T, yerr=u_mle_std_imp[:,[2,3]].T,  alpha= 0.7, \
                  fmt='o', markersize=5, ecolor='red', color='darkred', elinewidth=1, capsize=4, label='MLE-Cor')
    axis.plot(x_plot, u_mle_std_imp[:, 0].T,  alpha= 0.7, color='red', ls='--')   
    # axis.errorbar(x_plot, u_dm_std[:, 0].T, yerr=u_dm_std[:,[2,3]].T,  alpha= 0.7, \
    #                fmt='v', markersize=6, ecolor='orange', color='darkorange', elinewidth=1, capsize=3, label='DM')
    axis.set(xlabel='Number of specimens (Normalised d=' + str(dn) + ')',
              ylabel= 'Normalised standard deviation',
              xticks= [10, 20, 30, 40, 50, 70, 100]
              )
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_MLE_std_d{}.svg".format(dn)))
    
    #%% KDE compare different bandwidth
    fig, axis = plt.subplots()
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')

    axis.errorbar(x_plot, u_kde_std_sc[:, 0].T, yerr=u_kde_std_sc[:,[2,3]].T,  alpha= 0.7, \
                  fmt='x', markersize=3, ecolor='green', color='darkgreen', elinewidth=1, capsize=5, label='KDE-SC')
    axis.plot(x_plot, u_kde_std_sc[:, 0].T,  alpha= 0.7, color='green', ls='--')  
    axis.errorbar(x_plot, u_kde_std_sl[:, 0].T, yerr=u_kde_std_sl[:,[2,3]].T,  alpha= 0.7, \
              fmt='o', markersize=4, ecolor='blue', color='darkblue', elinewidth=1, capsize=6, label='KDE-SI')
    axis.plot(x_plot, u_kde_std_sl[:, 0].T,  alpha= 0.7, color='blue', ls='--')
    axis.errorbar(x_plot, u_kde_std_sj[:, 0].T, yerr=u_kde_std_sj[:,[2,3]].T,  alpha= 0.7, \
                    fmt='v', markersize=6, ecolor='orange', color='darkorange', elinewidth=1, capsize=3, label='KDE-SJ')
    axis.plot(x_plot, u_kde_std_sj[:, 0].T,  alpha= 0.7, color='orange', ls='--')
    axis.errorbar(x_plot, u_kde_std_pf[:, 0].T, yerr=u_kde_std_pf[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=6, ecolor='red', color='darkred', elinewidth=1, capsize=3, label='KDE-PF')
    axis.plot(x_plot, u_kde_std_pf[:, 0].T,  alpha= 0.7, color='red', ls='--')    
        
    # axis.errorbar(x_plot, u_kde_std_imp[:, 0].T, yerr=u_kde_std_imp[:,[2,3]].T,  alpha= 0.7, \
    #          fmt='^', markersize=5, ecolor='cyan', color='darkcyan', elinewidth=1, capsize=4, label='KDE-Cor')
    
    axis.set(xlabel='Number of specimens (Normalised d=' + str(dn) + ')',
             ylabel= 'Normalised standard deviation',
             xticks= [10, 20, 30, 40, 50, 70, 100]
             )
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_bw_std_d{}.svg".format(dn)))
    
 
    #%%   mean (median)    
    # KDE final
    fig, axis = plt.subplots()
    plt.subplots_adjust(left=0.14, right=0.98)
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')
    
    axis.errorbar(x_plot, u_dm_mean[:, 0].T, yerr=u_dm_mean[:,[2,3]].T, alpha= 0.7, \
                  fmt='x', markersize=6, ecolor='green', color='darkgreen', elinewidth=1, capsize=6, label='DM')
    axis.plot(x_plot, u_dm_mean[:, 0].T,  alpha= 0.7, color='green', ls='--')
    axis.errorbar(x_plot, u_mle_med[:, 0].T, yerr=u_mle_med[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=5, ecolor='blue', color='darkblue', elinewidth=1, capsize=4, label='MLE')
    axis.plot(x_plot, u_mle_med[:, 0].T,  alpha= 0.7, color='blue', ls='--')
    axis.errorbar(x_plot, u_kde_med_pf[:, 0].T, yerr=u_kde_med_pf[:,[2,3]].T,  alpha= 0.7, \
              fmt='o', markersize=4, ecolor='red', color='darkred', elinewidth=1, capsize=6, label='KDE')
    axis.plot(x_plot, u_kde_med_pf[:, 0].T,  alpha= 0.7, color='red', ls='--')
    
    axis.set(xlabel='Number of specimens (Normalised d=' + str(dn) + ')', 
             ylabel= 'Normalised median value' ,
             xticks= [10, 20, 30, 40, 50, 70, 100])
    # hb.set_axis_sci(axis, scilimits=(-1, -1))
    plt.legend( loc='best')
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_KDE_med_d{}.svg".format(dn)))
        

    #%% MLE compare mean 
    fig, axis = plt.subplots()
    plt.subplots_adjust(left=0.14, right=0.98)
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')

    axis.errorbar(x_plot, u_mle_med[:, 0].T, yerr=u_mle_med[:,[2,3]].T,  alpha= 0.7, \
                  fmt='x', markersize=3, ecolor='green', color='green', elinewidth=1, capsize=5, label='MLE')
    axis.plot(x_plot, u_mle_med[:, 0].T,  alpha= 0.7, color='green', ls='--')  
    axis.errorbar(x_plot, u_mle_med_ce[:, 0].T, yerr=u_mle_med_ce[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=4, ecolor='blue', color='darkblue', elinewidth=1, capsize=6, label='MLE-Cen')
    axis.plot(x_plot, u_mle_med_ce[:, 0].T,  alpha= 0.7, color='blue', ls='--') 
    # axis.errorbar(x_plot, u_mle_med_imp[:, 0].T, yerr=u_mle_med_imp[:,[2,3]].T,  alpha= 0.7, \
    #               fmt='o', markersize=5, ecolor='red', color='darkred', elinewidth=1, capsize=4, label='KDE-Cor')
    # axis.plot(x_plot, u_mle_med_imp[:, 0].T,  alpha= 0.7, color='red', ls='--')   
    # axis.errorbar(x_plot, u_dm_std[:, 0].T, yerr=u_dm_std[:,[2,3]].T,  alpha= 0.7, \
    #                fmt='v', markersize=6, ecolor='orange', color='darkorange', elinewidth=1, capsize=3, label='DM')
    axis.set(xlabel='Number of specimens (Normalised d=' + str(dn) + ')',
              ylabel= 'Normalised median value',
              xticks= [10, 20, 30, 40, 50, 70, 100]
              )
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_MLE_med_d{}.svg".format(dn)))
        
    
    #%% median : KDE compare different bandwidth
    fig, axis = plt.subplots()
    plt.subplots_adjust(left=0.13, right=0.98)
    axis.plot([min(x_plot), max(x_plot)], [1,1], color='black', ls='-', linewidth='0.5')

    axis.errorbar(x_plot, u_kde_med_sc[:, 0].T, yerr=u_kde_med_sc[:,[2,3]].T,  alpha= 0.7, \
                  fmt='x', markersize=3, ecolor='green', color='darkgreen', elinewidth=1, capsize=5, label='KDE-SC')
    axis.plot(x_plot, u_kde_med_sc[:, 0].T,  alpha= 0.7, color='green', ls='--')  
    axis.errorbar(x_plot, u_kde_med_sl[:, 0].T, yerr=u_kde_med_sl[:,[2,3]].T,  alpha= 0.7, \
              fmt='o', markersize=4, ecolor='blue', color='darkblue', elinewidth=1, capsize=6, label='KDE-SI')
    axis.plot(x_plot, u_kde_med_sl[:, 0].T,  alpha= 0.7, color='blue', ls='--')
    axis.errorbar(x_plot, u_kde_med_sj[:, 0].T, yerr=u_kde_med_sj[:,[2,3]].T,  alpha= 0.7, \
                    fmt='v', markersize=6, ecolor='orange', color='darkorange', elinewidth=1, capsize=3, label='KDE-SJ')
    axis.plot(x_plot, u_kde_med_sj[:, 0].T,  alpha= 0.7, color='orange', ls='--')
    axis.errorbar(x_plot, u_kde_med_pf[:, 0].T, yerr=u_kde_med_pf[:,[2,3]].T,  alpha= 0.7, \
                  fmt='^', markersize=6, ecolor='red', color='darkred', elinewidth=1, capsize=3, label='KDE-PF')
    axis.plot(x_plot, u_kde_med_pf[:, 0].T,  alpha= 0.7, color='red', ls='--')    
        
    # axis.errorbar(x_plot, u_kde_std_imp[:, 0].T, yerr=u_kde_std_imp[:,[2,3]].T,  alpha= 0.7, \
    #          fmt='^', markersize=5, ecolor='cyan', color='darkcyan', elinewidth=1, capsize=4, label='KDE-Cor')
    
    axis.set(xlabel= 'Number of specimens (Normalised d=' + str(dn) + ')',
             ylabel= 'Normalised median value',
             xticks= [10, 20, 30, 40, 50, 70, 100]
             )
    plt.legend( loc='best') 
    
    if save_to_disk == True:
        fig.savefig(output_dir.joinpath(disname+"_bw_med_d{}.svg".format(dn)))     
    

#%%------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Lets go")
    n_list = np.array([10, 15, 20, 25, 30, 35, 40, 50, 70, 100])
    # for n in n_list:
    #     plot_d(n)
    
    verbose = 1
    if verbose == 1: 
        print('Normal distribution')
        plot_d(30, verbose= verbose, truedist=[400, 10], data_dir="output/KDE_compare_normal", output_dir="figure/KDE_compare_normal")
        plot_n(n_list, d=10, verbose= verbose, data_dir="output/KDE_compare_normal", output_dir="figure/KDE_compare_normal")
    elif verbose ==2: 
        print('Lognormal distribution')
        plot_d(70, verbose= verbose, data_dir="output/KDE_compare_lognorm", output_dir="figure/KDE_compare_lognorm")
        plot_n(n_list, d=15, verbose= verbose, data_dir="output/KDE_compare_lognorm", output_dir="figure/KDE_compare_lognorm")
    elif verbose ==3: 
        print('Weibull distribution')
        plot_d(70, verbose= verbose, data_dir="output/KDE_compare_weibull", output_dir="figure/KDE_compare_weibull")
        plot_n(n_list, d=15, verbose= verbose, data_dir="output/KDE_compare_weibull", output_dir="figure/KDE_compare_weibull")
        
            
    
    # plot_n(n_list)
    

    
    

    
        
        
        
    