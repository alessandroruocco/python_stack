#!/usr/bin/env python
# coding=UTF-8

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
import pathos.multiprocessing as mp
from matplotlib import cm
import skimage.measure
from scipy.ndimage.filters import gaussian_filter1d
#import colormaps
import os
import sdf
from scipy import fftpack
from numpy.fft import fft,fftfreq,ifft,fftshift
from matplotlib import ticker, cm

import srsUtils
import sdfUtils

axis_font = {'fontname':'Arial', 'size':'30'}
title_font = {'fontname':'Arial', 'size':'25', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

mpl.style.use('classic')
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('figure', autolayout=True)

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000000

density_tick_uno = 0.25
#density_tick_due = 0.2
density_tick_tre = 0.19
density_tick_quattro = 0.10
density_tick_cinque = 0.05
density_tick_sei = 0.02

convert_k_to_kev = 1/11604*10**-3


def calculate_y_lineout_field(x_of_y_lineouts,index_x_of_y_lineouts,Legend,
                              type_of_field,files, IntervalInitial,
                                  IntervalFinal, Snapshots,y,Cropy,index_ycropping_1,index_ycropping_2,
                                  LaserIntensity,Te,logscale,type_of_profile,plotType,
                                  cumulate_plots,cumulate_plot_number):
    


    print(type_of_field)




    # last_file = int((IntervalFinal-IntervalInitial)/Snapshots)


    
    cumulate_plot_count = 0
    
    if IntervalInitial is None and IntervalFinal is None:
        
        for i in range(len(files)):
    
            print('initial',i)
    
            data_file =  sdf.read(files[i])
            print(files[i])
    
            time = np.array(data_file.Header['time'] )/1e-12
    
    
            energy = data_file.__dict__[type_of_field].data
            
            print('lineout field')
            
            if Cropy is None:
                energy = energy[index_x_of_y_lineouts,index_ycropping_1:len(energy[1,:])-index_ycropping_2]
            elif Cropy is not None:
                energy = energy[index_x_of_y_lineouts,index_ycropping_1:index_ycropping_2]

     
            
            
            if type_of_field.startswith('Derived_Poynting_Flux_x'):
                  l_intensity = LaserIntensity*1e4
                  norm   = l_intensity
                  plt.ylabel('$S_x$ ($I_0$)')
                  energy = np.abs(energy)
                  data_save_folder = 'sx'
    
            elif type_of_field.startswith('Derived_Poynting_Flux_y'):
                  l_intensity = LaserIntensity*1e4
                  norm   = l_intensity
                  plt.ylabel('$S_y$ ($I_0$)')
                  data_save_folder = 'sy'
    
            elif type_of_field.startswith('Derived_Temperature_Electron'):
                  norm    = convert_k_to_kev*Te
                  plt.ylabel('$T$ ($T_{e0}$)')
                  data_save_folder = 'e_temp'
    
            elif type_of_field.startswith('Electric_Field_Ex') :
                  # energy = smooth(energy**2,int(len(x)/100))**2
                  norm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
                  plt.ylabel('$e |E_x|^2/(m_e \omega_0|$)')
                  data_save_folder = 'ex'
    
            elif type_of_field.startswith('Electric_Field_Ey') :
                  # energy = smooth(energy**2,int(len(x)/100))**2
                  print('Sta entrnato qua')
                  norm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
                  plt.ylabel('$e |E_y|^2/(m_e \omega_0|$)')
                  data_save_folder = 'ey'
    
                   # variable    = np.abs(np.array(variable))
            elif type_of_field.startswith('Derived_Number_Density_Electron'):
                  norm    = critical_density_m
                  plt.ylabel('$n_e$')
                  data_save_folder = 'e_density'
    
    
            elif type_of_field.endswith('Density_a1_ion') or type_of_field.endswith('arbon') or type_of_field.endswith('ydrogen'):
                  #energy = smooth(energy,int(len(x)/100))
    
                  norm    = critical_density_m
                  data_save_folder = 'i_density'
                  plt.ylabel('$n_i$')
                 
            if args.plotType == 'normal':
                plt.figure()
                plt.xlabel('y ($\mu$m)')
                plt.xlim(y.min(),y.max())
                plt.grid()    
                if logscale == True:
                    plt.yscale('log')


    
           

            #  if Legend == True:
                plt.title('t='+str(round(time,3))+' ps')

                plt.plot(y,energy/norm,label ='t='+str(round(time,3)) )
    


           


                cumulate_plot_count = cumulate_plot_count +1

  
       
                plt.savefig('./pics/'+str(data_save_folder)+'_y_lineout/'+str(args.output)+str(round(time,3))+'.jpg')
    

                if cumulate_plots == True:
                    
# =============================================================================
#                     TO FIX!!!! - SAVING DIFFERENT PLOTS ON SAME GRAPHS
# =============================================================================
                    
                    plt.title('')

                    print(cumulate_plot_count)
    
                    if cumulate_plot_count%cumulate_plot_number == 0:
    
    
                        plt.legend(loc='best',prop={'size':18})
                        plt.savefig('./pics/'+str(data_save_folder)+'_y_lineout/'+str(args.output)+str(round(time,3))+'_cumulative_plot.jpg',dpi=600,pad='tight')
                plt.close()
    
    
    
            print('-------------------------------------')
            print('-------------------------------------')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataDir')
    parser.add_argument('field')


    parser.add_argument('--prefix',default='regular')
    parser.add_argument('--space',action='store_true')
    parser.add_argument('--averaging_over_y', type = bool, default=False)

    parser.add_argument('--parallel',action='store_true')
    parser.add_argument('--useCached',action='store_true')
    parser.add_argument('--cacheFile')



    parser.add_argument('--densProfFile',default='regular0000.sdf')
    parser.add_argument('--densitySpecies',default='electrons')
    parser.add_argument('--Te',type=float)

    parser.add_argument('--coloreMap',default = 'viridis')

    parser.add_argument('--ChooseInterval',type=bool, default = False)
    parser.add_argument('--IntervalInitial',type=int, default = None)
    parser.add_argument('--IntervalFinal',type=int, default = None)


    parser.add_argument('--Snapshots',type=int)
    parser.add_argument('--DSnapshots',type=int)

    parser.add_argument('--LaserIntensity',type=float)

    parser.add_argument('--Lambda',type=float,default = 0.35)


    parser.add_argument('--log',action='store_true')
    parser.add_argument('--density_plot_function_density',type = bool,default = False)
    parser.add_argument('--logscale',type=bool, default = False)

    parser.add_argument('--xLims',type=float,nargs=2)
    parser.add_argument('--kLims',type=float,nargs=2,default=[-3,3])
    parser.add_argument('--dens_lim',type=float,nargs=2)
    parser.add_argument('--yLims',type=float,nargs=2)
    parser.add_argument('--maxFPercentile',type=float,default=99.9)
    parser.add_argument('--minFPercentile',type=float,default=0.5)
    parser.add_argument('--maxF',type=float)
    parser.add_argument('--minF',type=float)

    parser.add_argument('--noMarkQC',action='store_true')
    parser.add_argument('--markTPDCutoff',action='store_true')
    parser.add_argument('--markSRSCutoff',action='store_true')
    parser.add_argument('--landauCutoff',type=float,default=0.30)
    parser.add_argument('--noCBar',action='store_true')
    parser.add_argument('--noCBarLabel',action='store_true')
    parser.add_argument('--minNeTick',type=float)
    parser.add_argument('--neTickInterval',type=float)

    parser.add_argument('--fontSize',type=float)
    parser.add_argument('-o','--output')
    parser.add_argument('--figSize',type=float,nargs=2)

    parser.add_argument('--Tempora_cut',type=bool,default=False)
    parser.add_argument('--which_field',type=str,default='sx')
    parser.add_argument('--smoothed_is_true', type = str, default = 'True')
    parser.add_argument('--cumulate_plots',type=bool,default=False)
    parser.add_argument('--cumulate_plot_number', type = int, default = 4)
    parser.add_argument('--Legend', default = False,type = bool)


    parser.add_argument('--type_of_profile', type = str, default = 'exponential')
    parser.add_argument('--l_nc_4', type = int, default = 600)
    parser.add_argument('--dens_init', type = float, default = 0.01)
    parser.add_argument('--dens_fin', type = float, default = 0.28)


    parser.add_argument('--density_normalized_to_nc',type = bool, default = True)
    parser.add_argument('--density_normalized_to_n0',type = bool, default = False)

######## CONTROLLING TDF TRANSFORMS
    parser.add_argument('--tdf_transform',type = bool,default=False)
    parser.add_argument('--k_transform',type = bool,default=False)
    parser.add_argument('--omega_transform',type = bool,default=False)
    parser.add_argument('--t_k_is_true',type = bool,default=False)
    parser.add_argument('--omega_k_is_true',type = bool,default=False)
    parser.add_argument('--omega_x_is_true',type = bool,default=False)
    parser.add_argument('--boundaries', type = int, default = 50)
    parser.add_argument('--writing_data_file', type = bool, default = False)

    parser.add_argument('--collision_frq_study', type = bool, default=False)
    parser.add_argument('--Z', type = float, default=3.5)

    parser.add_argument('--save_also_snapshots', type = bool, default=False)


    parser.add_argument('--norm_temp_to_init_temp', type = bool, default=False)
    
# =============================================================================
#     # NEED TO BE ASSESSED 
# =============================================================================
    
    parser.add_argument('--nCropy',type=int,nargs=2,default=[6,6])
    parser.add_argument('--Cropy',type=float,nargs=2,default=None)
    parser.add_argument('--nCropx',type=int,nargs=2,default=[6,6])
    parser.add_argument('--Cropx',type=float,nargs=2,default=None)
    
    parser.add_argument('--yLineout',type=bool,required=True)
    parser.add_argument('--xPositionLineout',type=float)
    
    parser.add_argument('--plotType',
                        choices=['normal','env','FT','FTy','wltx','meanDiffY'],
                                            default='normal')


    args = parser.parse_args()
    
    critical_density_m = 1.1 * 1e21 * (args.Lambda)**-2 * 1e6

    print(critical_density_m)
    print(args.dataDir,'arg data')
    
    DSnapshots = args.DSnapshots
    type_of_field = args.field
    prefix=args.prefix
    
    if args.ChooseInterval == True:
        files = sdfUtils.listFiles(args.dataDir,args.prefix)[args.IntervalInitial:args.IntervalFinal]
    else:
        print('entra qua')
        print(args.dataDir,args.prefix)
        files = sdfUtils.listFiles(args.dataDir,args.prefix)[::args.Snapshots]

  
    
    sdfProf = sdf.read(os.path.join(args.dataDir,args.densProfFile))

#            if args.acc_core_subset == '_Acc':
#                x  = sdfProf.__dict__['Grid_A'+str(args.positional_prefix)+'_'+str(args.prefix)].data[0]
#
#
#            else:
    field = args.field
    print(field)
    
    if field.startswith('Electric') or field.startswith('Magnetic'):
        

        xOrig  = sdfProf.Grid_Grid_mid.data[0]*1e6
        yOrig  = sdfProf.Grid_Grid_mid.data[1]*1e6
    
    else:
        
        xOrig  = sdfProf.Grid_Grid.data[0]*1e6
        yOrig  = sdfProf.Grid_Grid.data[1]*1e6
     
    
    if args.yLineout == True:
        
        x_of_y_lineouts,index_x_of_y_lineouts = find_nearest(xOrig,args.xPositionLineout)
    
    if args.Cropy is None:
        if args.nCropy is not None: 
            _nCropy = args.nCropy
            yOrig = yOrig[_nCropy[0]:len(yOrig)-_nCropy[1]]
            index_ycropping_1 = _nCropy[0]
            index_ycropping_2 = _nCropy[1]
            print(yOrig.min(),yOrig.max())


    else:
        ycropping_1,index_ycropping_1 = find_nearest(yOrig,args.Cropy[0])
        ycropping_2,index_ycropping_2 = find_nearest(yOrig,args.Cropy[1])
        yOrig = yOrig[index_ycropping_1:index_ycropping_2]
    

    
    print(yOrig.min())
    print('le',len(yOrig))
    if args.yLineout == True and args.xPositionLineout is not None:
        
        calculate_y_lineout_field(x_of_y_lineouts,index_x_of_y_lineouts,args.Legend,type_of_field,files,
                                          args.IntervalInitial,args.IntervalFinal,
                                          args.Snapshots,yOrig,args.Cropy,index_ycropping_1,index_ycropping_2,
                                          args.LaserIntensity,args.plotType,                                          
                                         args.Te,args.logscale,
                                      args.type_of_profile,args.cumulate_plots,args.cumulate_plot_number)
        
  