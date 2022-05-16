#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:51:02 2020

@author: alessandro_ruocco
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
import pathos.multiprocessing as mp
import skimage.measure
import os
import sdf
from matplotlib import cm,ticker
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.ndimage.filters import gaussian_filter1d
import srsUtils
import sdfUtils
from scipy import fftpack
from numpy.fft import fft,fftfreq,ifft,fftshift

axis_font = {'fontname':'Arial', 'size':'22'}


mpl.style.use('classic')
plt.switch_backend('agg')
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('figure', autolayout=True)


density_tick_uno = 0.25


convert_k_to_kev = 1/11604*10**-3

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def findDensity(x,ne,frac):
    xMid = 0.5*(x[:-1] + x[1:])

    if np.any(np.where(ne/srsUtils.nCritNIF >= frac)) \
      and np.any(np.where(ne/srsUtils.nCritNIF <= frac)):
        xQC = xMid[np.min(np.where(ne/srsUtils.nCritNIF >= frac))]
    else:
        xQC = None

    return xQC

def calcMeanEnergyVsSpaceTime(files,field,parallel=False):
    data = [ sdf.read(f) for f in files ]
    ts = np.array([ d.Header['time'] for d in data ])

    grid = data[0].Grid_Grid.data
    if len(grid) == 1:
        if field.startswith('Derived_'):
            energy = np.array([ d.__dict__[field].data for d in data ])


        else:
            energy = np.array([ d.__dict__[field].data**2 for d in data ])
        print('max val: {:}'.format(np.max(energy)))
    elif len(grid) == 2:
        # We have a 2D dataset, need to average over y
        y = data[0].Grid_Grid.data[1]

        if parallel:
            def aveFunc(f):
                data = sdf.read(f).__dict__[field].data
                energy = np.mean(data**2,axis=1)
                return energy

            pool = mp.Pool()
            result = pool.map(aveFunc,files)
            pool.close()
            pool.join()
            energy = np.array(result)
        else:
            energy = np.array([ np.mean(d.__dict__[field].data**2,axis=1) for d in data ])
    x = data[0].Grid_Grid.data[0]

    return energy,x,ts


def evaluation_noise(seed_amp,variable,intensita_pump,initial_boudary,final_boundary):
    noise = []
    # for i in range(len(time)-1):
    for i in range(25):
        noise.append(np.mean(variable[initial_boudary:final_boundary,i]))

    noise = np.array(noise)
    print('noise norm to seed',noise.mean()*1e-4/(1/seed_amp*float(intensita_pump)))
    return noise.mean()*1e-4

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


def Integrale_Trapezio(f,a,b,n,frequency_array):

    number_of_steps = int((b-a)/n)   #uniform grid
    S = []
    somma = 0
    for i in range(a,b,number_of_steps):
        S.append(somma+((f[i]+[i+1])*0.5*(-frequency_array[i-1]+frequency_array[i])))
    S = np.array(S)

def exp_fit(x,a,b,c):
    return a*np.exp(-b*x+c)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plasma_laser_conditions(Lambda,Zeta):
    eps_0=8.85*10**-12         #F/m2
    eel=1.60217662*10**-19
    converter_kelvin_to_eV=1/11604.52
    converter_kelvin_to_J = 7.24297166666667 * 10**22
    convert_eV_joule=1.6*10**-19   #eV to Joule
    c = 299792458. #m/s
    w_lenth = 0.35
    n_crit = 1.1*10**21/(w_lenth**2) *10**6  #m-3
    me = 9.31*10**-31
    Z = Zeta
    perm_mag_vacuum = 1.256*10**-6  #H/m
    convert_k_to_kev = 1/11604*10**-3

    lambda_mum = Lambda
    lambda_m = lambda_mum * 10**-6
    k0 = (2*3.14)/lambda_m

    omega0 = c*k0
    critical_density = 1.1 * (10**21) * lambda_mum**-2    #cm-3
    omega0=(2*3.14*c)/lambda_m
    omega0_fs = omega0*10**-15
    omega0_ps = omega0*10**-12
    return lambda_m


def x_y_temporal_cut(directory_output,Lambda,RatioPFlux,files,type_of_field,smoothed_is_true,
                      LaserIntensity,Te,
                      xLims,yLims,Log,critical_density_m,density_mark_2,
                      density_mark_3,
                      dens_lim,n_min,l_nc4,x_boundary_min,x_boundary_max,
                      y_boundary_min,y_boundary_max,IntervalFinal,
                      IntervalInitial,Snapshots,TimeAveraged,maxF,minF,
                      minFPercentile,maxFPercentile,noMarkQC,CBar=True,CBarLabel=True,markTPDCutoff=True,markSRSCutoff=True,
                              landauCutoff=0.3):


    last_file = int((IntervalFinal-IntervalInitial)/Snapshots)
    
    print(n_min*critical_density_m)
    print(n_min)
    print(n_min)
    print(n_min*critical_density_m)


    print('Saving '+str(last_file-1)+' files')
    print('Parameter '+str(type_of_field))

    looping_index = 0
    for i in range(0,last_file-1):
        if TimeAveraged == True and i ==0:
            time_averaged_field = 0

        print('initial',i)

        
        data =  sdf.read(files[i])
        time = np.array(data.Header['time'] )/1e-12
        
        x = data.Grid_Grid.data[0]*1e6
        y = data.Grid_Grid.data[1]*1e6
        k0 = 2*3.14/Lambda
        # k0 =1 /Lambda
        
        
        if xLims:
            x_min, index_x_min = find_nearest(x,xLims[0])
            x_max, index_x_max = find_nearest(x,xLims[1])
            
            x = x[index_x_min:index_x_max]
        
        if yLims:
            y_min, index_y_min = find_nearest(y,yLims[0])
            y_max, index_y_max = find_nearest(y,yLims[1])
            print(index_y_max)
            y = y[index_y_min:index_y_max]
  

        print('time ps',time)

        if RatioPFlux == False:


            energy = data.__dict__[type_of_field].data
            reduceArray = tuple([ max([1,s // 2000]) for s in energy.shape ])
            
            if xLims:
                energy = energy[index_x_min:index_x_max,:]
            elif yLims:
                energy = energy[:,index_y_min:index_y_max]
            elif xLims and yLims:
                energy = energy[index_x_min:index_x_max,index_y_min:index_y_max]


            field = type_of_field


            if type_of_field.endswith('Poynting_Flux_x'):
                l_intensity = LaserIntensity*1e4
                enorm   = l_intensity
                energy = skimage.measure.block_reduce(energy,reduceArray,np.mean)
                energy = energy/enorm
                colore_plot = cm.seismic
                quantity_title = '$Sx/I_0$'


            elif type_of_field.endswith('Poynting_Flux_y'):
                l_intensity = LaserIntensity*1e4
                enorm   = l_intensity
                colore_plot = 'PuBuGn'
                energy = skimage.measure.block_reduce(energy,reduceArray,np.mean)
                energy = energy/enorm
                quantity_title = '$Sy/I_0$'

            elif type_of_field.endswith('Poynting_Flux_z'):
                l_intensity = LaserIntensity*1e4
                enorm   = l_intensity
                colore_plot = 'PuBuGn'
                energy = skimage.measure.block_reduce(energy,reduceArray,np.mean)
                energy = energy/enorm
                quantity_title = '$Sz/I_0$'

            elif type_of_field.endswith('Field_Ex') or type_of_field.endswith('Field_Ez') or type_of_field.endswith('Field_Ey'):
                enorm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
                #energy = skimage.measure.block_reduce(energy,reduceArray,np.mean)
               # energy = np.abs(energy/enorm)**2
                energy = energy
                quantity_title = '$e |E_x|/( cm_e \omega_0)$'
                print(energy.max(),energy.min())


            elif type_of_field.endswith('Temperature_Electron'):
                enorm    = Te
                colore_plot = 'pink'
                energy = skimage.measure.block_reduce(energy,reduceArray,np.mean)/enorm




            # variable    = np.abs(np.array(variable))
            elif type_of_field.endswith('Density_Electron'):
                enorm    = critical_density_m
                if i == 0:
#                    energy_0 = skimage.measure.block_reduce(energy,reduceArray,np.mean)/enorm
                    energy_0 = energy
                colore_plot = 'goldenroad'
                quantity_title = '$n_e/n_c$'
 #               energy = skimage.measure.block_reduce(energy,reduceArray,np.mean)/enorm 
                energy = energy/energy_0




            elif type_of_field.endswith('Density_a1_ion') or type_of_field.endswith('carbon') or type_of_field.endswith('hydrogen'):

                enorm    = critical_density_m
                colore_plot = 'cool'
                energy = skimage.measure.block_reduce(energy,reduceArray,np.mean)/enorm

        elif RatioPFlux == True:


            type_of_field_x = 'Derived_Poynting_Flux_x'
            type_of_field_y = 'Derived_Poynting_Flux_y'

            sx = data.__dict__[type_of_field_x].data
            sy = data.__dict__[type_of_field_y].data

            reduceArray_x = tuple([ max([1,s // 2000]) for s in sx.shape ])
            reduceArray_y= tuple([ max([1,s // 2000]) for s in sy.shape ])

            sx = skimage.measure.block_reduce(sx,reduceArray_x,np.mean)
            sy = skimage.measure.block_reduce(sy,reduceArray_y,np.mean)
            l_intensity = LaserIntensity*1e4

            energy = np.abs(sy/sx)





        fig = plt.figure()
        ax = plt.subplot(111)

        if not maxF:
            maxF = np.percentile(energy,maxFPercentile)
        if not minF:
            if Log==True:
                minF = np.percentile(energy,minFPercentile)
            else:
                minF = 0.0
        if Log==False:
            norm = colors.Normalize(vmin=minF,vmax=maxF)
        else:
            
            norm = colors.LogNorm(vmin=minF,vmax=maxF)

        #energy[np.where(energy < 1e-5)] = np.nan

        # Downsample array if it is excessively large
        reduceArray = tuple([ max([1,s // 2000]) for s in energy.shape ])
        extent = srsUtils.misc.getExtent(x,y)
        energy_plot = skimage.measure.block_reduce(energy,reduceArray,np.mean)
        energy_plot = energy_plot.T
        
        if  RatioPFlux == False:
            if field.startswith('Electric_') and field.endswith('Ez'):
                im = ax.imshow(energy_plot/energy_plot.max(),interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='viridis',norm=norm)
            elif field.startswith('Electric_') and field.endswith('Ey'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='plasma',norm=norm)
            elif field.endswith('Bz') or field.endswith('Bz_Core_strip_y0'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='autumn',norm=norm)
            elif field.endswith('Ex') or field.endswith('Ex_Core_strip_y0') :
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Purples',norm=norm)
            elif field.endswith('Poynting_Flux_y') or field.endswith('Poynting_Flux_y_Subset_strip_y0'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='PuBuGn',norm=norm)
            elif field.endswith('Poynting_Flux_x') or field.endswith('Flux_x_Subset_strip_y0') :
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap=cm.seismic,norm=norm)
            elif field.endswith('Poynting_Flux_z') or field.endswith('Flux_z_Subset_strip_y0') :
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Greens',norm=norm)
            elif field.endswith('Density_a1_ion')  or type_of_field.endswith('carbon') or type_of_field.endswith('hydrogen'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='cool',norm=norm)
            elif field.startswith('Derived_') and field.endswith('Temperature_Electron'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Greys',norm=norm)
            elif field.endswith('Density_Electron'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Blues',norm=norm)
        elif RatioPFlux == True:
            im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='hot',norm=norm)

        if CBar:
            cb = fig.colorbar(im,ax=ax,orientation='vertical')
            if Log==False:
                cb.formatter.set_powerlimits((-4,4))
            cb.ax.yaxis.set_offset_position('left')
            cb.ax.tick_params(labelsize=20)
            
            cb.ax.set_ylabel(quantity_title, rotation=90,labelpad=10,**axis_font)

            cb.update_ticks()

        ax.tick_params(reset=True,axis='both',color='w',direction='in')
        print(noMarkQC)

        density_plot_twin = n_min*np.exp(x/l_nc4)

        xQC,index_xqc = find_nearest(density_plot_twin,density_tick_uno)
        if density_mark_2 is not None:
            x02,index_x02 = find_nearest(density_plot_twin,density_mark_2)
        if density_mark_3 is not None:
            x015,index_x015 = find_nearest(density_plot_twin,density_mark_3)

        if xQC is not None: ax.axvline(x[index_xqc],color='green',linestyle='--',linewidth = 6)
        if density_mark_2 is not None:
            if x02 is not None: ax.axvline(x[index_x02],color='dimgray',linestyle='--',linewidth = 6)

        if density_mark_3 is not None:
            if x015 is not None: ax.axvline(x[index_x015],color='darkgray',linestyle=':',linewidth = 4)

        

        density_plot = n_min*np.exp(x/l_nc4)
        ne = density_plot
        if Te is not None:
            bth = math.sqrt(const.k*Te/const.m_e)/const.c
            if markTPDCutoff:
               tpdCutoffNe = srsUtils.tpd.landauCutoffDens(bth,cutoff=landauCutoff)
#               print(ne,srsUtils.nCritNIF,tpdCutoffNe)
               diff = np.abs(ne/srsUtils.nCritNIF-tpdCutoffNe)

               dens_xCutoffTPD,index_tpd_cut_off = find_nearest(density_plot,tpdCutoffNe)
               xCutoffTPD = x[index_tpd_cut_off]
               ax.axvline(xCutoffTPD,linestyle=':',color='y',linewidth = 6)
               print(xCutoffTPD,'xCutoffTPD')
#
            if markSRSCutoff:
                srsCutoffNe = srsUtils.srs.landauCutoffDens(bth,math.pi,cutoff=0.3)
                diff = np.abs(ne/srsUtils.nCritNIF-srsCutoffNe)
                dens_xCutoffSRS,index_srs_cutoff =  find_nearest(density_plot,srsCutoffNe)
                xCutoffSRS = x[index_srs_cutoff]-45
                ax.axvline(xCutoffSRS,linestyle=':',color='cyan',linewidth = 6)
                print(xCutoffSRS,'xCutoffSRS')


 #       ax2 = ax.twiny()
 #       srsUtils.misc.addNeScale(ax,ax2,x,density_plot_twin,minVal=0.1,interval=0.05,minor=True)

        if xLims:
            plt.xlim(xLims)
        if yLims:
            plt.ylim(yLims)

        plt.tick_params(labelsize=20, length = 15, width=1,direction='inout')

        plt.xlabel(r'x $/\mu$m',**axis_font)
        plt.ylabel(r'y $/\mu$m',**axis_font)

        print(args.output)
        print(round(time,3))
        if RatioPFlux == True:
            plt.savefig('./pics/sy_sx_x_y/'+str(args.output)+'_'+str(round(time,3))+'.jpg',dpi=600,pad='tight')
        elif RatioPFlux == False:

            plt.savefig('./pics/'+args.output+'_x_y/'+str(args.output)+'_'+str(round(time,3))+'.jpg',dpi=600,pad='tight')
        plt.close()
        
        print('TimeAveraged',TimeAveraged)


        if TimeAveraged == True:
            looping_index = looping_index + 1
            print('it enters here')
            time_averaged_field = energy_plot + time_averaged_field

            

            if i == last_file-2:

                energy_plot = time_averaged_field/looping_index

                fig = plt.figure()
                ax = fig.add_subplot(111)

                if not maxF:
                    maxF = np.percentile(energy,maxFPercentile)
                if not minF:
                    if Log==True:
                        minF = np.percentile(energy,minFPercentile)
                    else:
                        minF = 0.0
                if Log==False:
                    norm = colors.Normalize(vmin=minF,vmax=maxF)
                else:

                    norm = colors.LogNorm(vmin=minF,vmax=maxF)


                
        if  RatioPFlux == False:
            if field.startswith('Electric_') and field.endswith('Ez'):
                im = ax.imshow(energy_plot/energy_plot.max(),interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='viridis',norm=norm)
            elif field.startswith('Electric_') and field.endswith('Ey'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='plasma',norm=norm)
            elif field.endswith('Bz') or field.endswith('Bz_Core_strip_y0'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='autumn',norm=norm)
            elif field.endswith('Ex') or field.endswith('Ex_Core_strip_y0') :
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Purples',norm=norm)
            elif field.endswith('Poynting_Flux_y') or field.endswith('Poynting_Flux_y_Subset_strip_y0'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='PuBuGn',norm=norm)
            elif field.endswith('Poynting_Flux_x') or field.endswith('Flux_x_Subset_strip_y0') :
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap=cm.seismic,norm=norm)
            elif field.endswith('Poynting_Flux_z') or field.endswith('Flux_z_Subset_strip_y0') :
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Greens',norm=norm)
            elif field.endswith('Density_a1_ion')  or type_of_field.endswith('carbon') or type_of_field.endswith('hydrogen'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='cool',norm=norm)
            elif field.startswith('Derived_') and field.endswith('Temperature_Electron'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Greys',norm=norm)
            elif field.endswith('Density_Electron'):
                im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Blues',norm=norm)
        elif RatioPFlux == True:
            im = ax.imshow(energy_plot,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='hot',norm=norm)

        if CBar:
            cb = fig.colorbar(im,ax=ax,orientation='vertical')
            if Log==False:
                cb.formatter.set_powerlimits((-4,4))
            cb.ax.yaxis.set_offset_position('left')
            cb.ax.tick_params(labelsize=20)
            
            cb.ax.set_ylabel(quantity_title, rotation=90,labelpad=10,**axis_font)

            cb.update_ticks()

        ax.tick_params(reset=True,axis='both',color='w',direction='in')
        print(noMarkQC)
        # Annotate location of important densities
#        if ne is not None:
#            if noMarkQC == False:

        density_plot_twin = n_min*np.exp(x/l_nc4)

        xQC,index_xqc = find_nearest(density_plot_twin,density_tick_uno)
        if density_mark_2 is not None:
            x02,index_x02 = find_nearest(density_plot_twin,density_mark_2)
        if density_mark_3 is not None:
            x015,index_x015 = find_nearest(density_plot_twin,density_mark_3)
     #   x01,index_x01 = find_nearest(density_plot_twin,density_tick_quattro)
    #    x005,index_x005 = find_nearest(density_plot_twin,density_tick_cinque)


        if xQC is not None: ax.axvline(x[index_xqc],color='green',linestyle='--',linewidth = 6)
        if density_mark_2 is not None:
            if x02 is not None: ax.axvline(x[index_x02],color='dimgray',linestyle='--',linewidth = 6)

        if density_mark_3 is not None:
            if x015 is not None: ax.axvline(x[index_x015],color='darkgray',linestyle=':',linewidth = 4)
#        if x01 is not None: ax.axvline(x[index_x01],color='lightgray',linestyle=':',linewidth = 4)
#        if x005 is not None: ax.axvline(x[index_x005],color='w',linestyle=':',linewidth = 4)

        

        density_plot = n_min*np.exp(x/l_nc4)
        ne = density_plot
        if Te is not None:
            bth = math.sqrt(const.k*Te/const.m_e)/const.c
            if markTPDCutoff:
               tpdCutoffNe = srsUtils.tpd.landauCutoffDens(bth,cutoff=landauCutoff)
#               print(ne,srsUtils.nCritNIF,tpdCutoffNe)
               diff = np.abs(ne/srsUtils.nCritNIF-tpdCutoffNe)

               dens_xCutoffTPD,index_tpd_cut_off = find_nearest(density_plot,tpdCutoffNe)
               xCutoffTPD = x[index_tpd_cut_off]
               ax.axvline(xCutoffTPD,linestyle=':',color='y',linewidth = 6)
               print(xCutoffTPD,'xCutoffTPD')
#
            if markSRSCutoff:
                srsCutoffNe = srsUtils.srs.landauCutoffDens(bth,math.pi,cutoff=0.3)
                diff = np.abs(ne/srsUtils.nCritNIF-srsCutoffNe)
                dens_xCutoffSRS,index_srs_cutoff =  find_nearest(density_plot,srsCutoffNe)
                xCutoffSRS = x[index_srs_cutoff]-45
                ax.axvline(xCutoffSRS,linestyle=':',color='cyan',linewidth = 6)
                print(xCutoffSRS,'xCutoffSRS')



                if xLims:
                    plt.xlim(xLims)

    
                if yLims:
                    plt.ylim(yLims)

                plt.tick_params(labelsize=15, length = 15, width=1,direction='inout')

                plt.xlabel(r'x $/\mu$m')
                plt.ylabel(r'y $/\mu$m')
    
    
                if RatioPFlux == True:
                    plt.savefig('./pics/sy_sx_x_y/'+str(args.output)+'_'+str(round(time,3))+'.jpg',dpi=600,pad='tight')
                elif RatioPFlux == False:

                    plt.savefig('./pics/'+args.output+'_x_y_time_averaged.jpg',dpi=600,pad='tight')
                plt.close()


    return fig,ax


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataDir')
    parser.add_argument('field')
    parser.add_argument('--prefix',default='regular_')
    parser.add_argument('--space',action='store_true')
    parser.add_argument('--parallel',action='store_true')
    parser.add_argument('--useCached',action='store_true')
    parser.add_argument('--cacheFile')

    parser.add_argument('--n_min',type = float,default = 0.1)
    parser.add_argument('--l_nc4',type = float,default =600)
    parser.add_argument('--densProfFile',default='regular_0000.sdf')
    parser.add_argument('--densitySpecies',default='electrons')
    parser.add_argument('--Te',type=float)

    parser.add_argument('--RatioPFlux',type=bool,default = False)


    parser.add_argument('--coloreMap',default = 'viridis')

    parser.add_argument('--Intervallo',default = 'False')
    parser.add_argument('--IntervalInitial',type=int)
    parser.add_argument('--IntervalFinal',type=int)


    parser.add_argument('--Snapshots',type=int)

    parser.add_argument('--LaserIntensity',type=float)

    parser.add_argument('--Lambda',type=float,default = 0.35)
    parser.add_argument('--Zeta',type=float,default = 3.5)




    parser.add_argument('--Log',type = bool, default = False)

    parser.add_argument('--xLims',type=float,nargs=2)
    parser.add_argument('--yLims',type=float,nargs=2)

    parser.add_argument('--densLims',type=float,nargs=2)
    parser.add_argument('--maxFPercentile',type=float,default=99.9)
    parser.add_argument('--minFPercentile',type=float,default=0.5)
    parser.add_argument('--maxF',type=float)
    parser.add_argument('--minF',type=float)


    parser.add_argument('--noMarkQC',type=bool,default = True)
    parser.add_argument('--density_mark_2',type=float,default = None)
    parser.add_argument('--density_mark_3',type=float,default = None)
    parser.add_argument('--markTPDCutoff',action='store_true')
    parser.add_argument('--markSRSCutoff',action='store_true')
    parser.add_argument('--landauCutoff',type=float,default=0.30)
    parser.add_argument('--noCBar',action='store_true')
    parser.add_argument('--noCBarLabel',action='store_true')
    parser.add_argument('--minNeTick',type=float)
    #parser.add_argument('--neTickInterval',type=float)

    parser.add_argument('--fontSize',type=float)
    parser.add_argument('-o','--output')
    parser.add_argument('--figSize',type=float,nargs=2)

    parser.add_argument('--Tempora_cut',type=bool,default=True)
    parser.add_argument('--which_field',type=str,default='sx')
    parser.add_argument('--smoothed_is_true', type = str, default = 'True')
    parser.add_argument('--TimeAveraged', type = bool, default = False)

    parser.add_argument('--x_boundary_min', type = int, default = 1)
    parser.add_argument('--x_boundary_max', type = int, default = 1)
    parser.add_argument('--y_boundary_min', type = int, default = 1)
    parser.add_argument('--y_boundary_max', type = int, default = 1)

    parser.add_argument('--directory_output',type=str,default = None)


    args = parser.parse_args()

    critical_density_m = 1.1 * 1e21 * (args.Lambda)**-2 * 1e6

    print(args.dataDir,'arg data')
    if args.Te is not None:
        Te = args.Te*srsUtils.TkeV
    else:
        Te = None
    if args.fontSize:
        import matplotlib as mpl
        mpl.rcParams.update({'font.size':args.fontSize})
    TimeAveraged = args.TimeAveraged
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if args.space:
        if args.useCached:
            print('It is using cachge')
            npCache = np.load(args.cacheFile)
            energy = npCache['energy']
            x      = npCache['x']
            t      = npCache['t']
            ne     = npCache['ne']
        else:
            print('Check on temporal cut')

            files = sdfUtils.listFiles(args.dataDir,args.prefix)[args.IntervalInitial:args.IntervalFinal]
            print(args.IntervalInitial,args.IntervalFinal)
            if len(files) == 0:
                raise IOError("Couldn't find any SDF files with prefix {:}".format(args.prefix))
            else:
                print("Found {:} SDF files with prefix {:}".format(len(files),args.prefix))


            # Find quarter critical surface
            sdfProf = sdf.read(os.path.join(args.dataDir,args.densProfFile))

            print(args.noMarkQC,args.noMarkQC,args.noMarkQC,args.noMarkQC)
            if args.noMarkQC == True:
                ne = None
            else:
                if args.densitySpecies == '':
                    ne = sdfProf.__dict__['Derived_Number_Density'].data
                else:
                    ne = sdfProf.__dict__['Derived_Number_Density_'+args.densitySpecies].data


                if len(ne.shape) == 2:
                    ne = np.mean(ne, axis = 1) #changed 1 to 0
                ne = gaussian_filter1d(ne,sigma=10)

            print(files)
            x_y_temporal_cut(args.directory_output,args.Lambda, args.RatioPFlux,
                             files,args.field,args.smoothed_is_true,
                             args.LaserIntensity,args.Te,args.xLims,args.yLims,
                             args.Log,critical_density_m,args.density_mark_2,
                             args.density_mark_3,
                             args.densLims,args.n_min,args.l_nc4,args.x_boundary_min,args.x_boundary_max,
                             args.y_boundary_min,
                             args.y_boundary_max,args.IntervalFinal,args.IntervalInitial,
                             args.Snapshots,args.TimeAveraged,
                             args.maxF,args.minF,
                             args.minFPercentile,
                             args.maxFPercentile,
                             args.noMarkQC)
