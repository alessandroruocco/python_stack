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


def findDensity(x,ne,frac):
    xMid = 0.5*(x[:-1] + x[1:])

    if np.any(np.where(ne/srsUtils.nCritNIF >= frac)) \
      and np.any(np.where(ne/srsUtils.nCritNIF <= frac)):
        xQC = xMid[np.min(np.where(ne/srsUtils.nCritNIF >= frac))]
    else:
        xQC = None

    return xQC


def density_profile(type_of_profile,l_nc_4,dens_init,dens_fin,space,critical_density):

    if type_of_profile == 'linear':

        delta_n = dens_fin - dens_init
        l_tot = l_nc_4*delta_n/0.25

        b = dens_init
        a = (dens_fin - b)/l_tot

        density_profile_x = critical_density*(a*space+b)

    elif type_of_profile == 'exponential':

        l_tot = l_nc_4*np.log(dens_fin/dens_init)

        b=0
        print(l_tot)
        a = dens_init
        density_profile_x = dens_init*np.exp(space/critical_density)


    return a,b,l_tot,density_profile_x



def plot_temporal_cut(different_from_usual,Legend,type_of_field,smoothed_is_true,
                                  files,IntervalInitial,IntervalFinal,Snapshots,
                                  x,
                                  LaserIntensity,Te,
                                  xLims,yLims,logscale,critical_density_m,type_of_profile,
                                  l_nc_4,dens_init,dens_fin,
                                  density_plot_function_density,dens_lim,
                                  cumulate_plots,cumulate_plot_number):


# =============================================================================
#     MAKE IT STRIP FRIENDLY - DIFFERENT COLOR
#     TDF
#     ALLOW TRANSVERSAL LINOUTS
# =============================================================================
    x = x*1e6

    a_liner,b_linear,l_tot,density_in_critical_density = density_profile(type_of_profile,l_nc_4,dens_init,dens_fin,x,critical_density_m)

    print(type_of_field)




    print(density_in_critical_density)
    last_file = int((IntervalFinal-IntervalInitial)/Snapshots)

    # print(xLims[1])
    # print(xLims)
    # print(xLims)
    # print(xLims)

    if xLims:
        x0,index_x0 = find_nearest(x,xLims[0])
        x1,index_x1 = find_nearest(x,xLims[1])

        print(x1,index_x1)
        print(x0,index_x0)

    if dens_lim:
        dens_i,index_di = find_nearest(density_in_critical_density,dens_lim[0])
        dens_f,index_df = find_nearest(density_in_critical_density,dens_lim[1])

    cumulate_plot_count = 0

    for i in range(0,last_file-1):

        print('initial',i)
        # print(files)
        # data = [ sdf.read(f) for f in  ]
        data_file =  sdf.read(files[i])
        print(files[i])

        # time = np.array([ d.Header['time'] for d in data ])/1e-12
        time = np.array(data_file.Header['time'] )/1e-12


        energy = data_file.__dict__[type_of_field].data


    #norm = 0.5*const.epsilon_0


        # if smoothed_is_true == 'True':
        #     energy = smooth(energy,int(len(space)/100))
        # elif smoothed_is_true == 'False':
        #     energy = energy

        # plt.title(''+str(round(time,3))+ ' ps')



        # if type_of_field.endswith('Derived_Poynting_Flux_x') or type_of_field.endswith('Poynting_Flux_y') or type_of_field.endswith('Poynting_Flux_y_Subset_+str(positional_prefix)') or type_of_field.endswith('Poynting_Flux_x_Subset_+str(positional_prefix)') :
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


        if density_plot_function_density == True:
            print('STO ENTRANDO IN X=DENSITY')
            plt.xlabel('$n_e$ ($n_c$)')

            if dens_lim:
                print('dens LIMITS')
                plt.xlim(dens_lim)
                plt.plot(density_in_critical_density[index_di:index_df],energy[index_di:index_df]/norm,label=str(round(time,3))+' ps')
                plt.legend(loc='best',prop={'size':18})
            else:
                print('no dens LIMITS')


                plt.plot(density_in_critical_density[0:len(energy)],energy/norm,label=str(round(time,3))+' ps')

        print(density_plot_function_density)
        print(energy)

        if density_plot_function_density ==False:

            if xLims:
                print('x LIMITS')

                plt.xlim(xLims)

                if Legend == True:
                    plt.plot(x[index_x0:index_x1],smooth(energy[index_x0:index_x1]/norm,150),label ='t='+str(round(time,3)) )
                elif Legend == False:
                    plt.plot(x[index_x0:index_x1],smooth(energy[index_x0:index_x1]/norm,150))

            else:
                print('no x LIMITS')
                if Legend == True:

                    plt.plot(x[0:len(energy)],energy/norm,label ='t='+str(round(time,3)) )
                elif Legend == False:

                    plt.plot(x[0:len(energy)],energy/norm)
            plt.xlabel('x ($\mu$m)')


        if logscale == True:
            plt.yscale('log')


       # ax.set_ylabel(r'$\frac{e^2\left\langle |E_x|^2 \right\rangle}{(m_e\omega_0c)^2}$')
        plt.grid()
        cumulate_plot_count = cumulate_plot_count +1

        print('Ylims',yLims)
        if yLims:
            plt.ylim(yLims)

        if cumulate_plots == False:
            plt.title('t = '+str(round(time,3))+' ps')
#            plt.savefig('./pics_scarf/'+str(data_save_folder)+'/'+str(args.output)+str(round(time,3))+'.jpg',dpi=600,pad='tight')
            plt.savefig('./pics/'+str(data_save_folder)+'/'+str(args.output)+str(round(time,3))+'.jpg')
            plt.close()


        elif cumulate_plots == True:

            if cumulate_plot_count%cumulate_plot_number == 0:


                plt.legend(loc='best',prop={'size':18})
                plt.savefig('./pics/'+str(data_save_folder)+'/'+str(args.output)+str(round(time,3))+'_cumulative_plot.jpg',dpi=600,pad='tight')



                plt.close()



        print('-------------------------------------')
        print('-------------------------------------')



def plot_tdf(minF,maxF,kLims,writing_data_file,field_as_function_t_k,field_as_function_omega_x,
                                 time_for_t_k_plot,looping_over_the_files,time,k_transform,omega_transform,save_also_snapshots,
             yLims, Legend,Lambda, type_of_field,
                                 smoothed_is_true,
                                  files,IntervalInitial,IntervalFinal,
                                  Snapshots,
                                  energy,x,
                                  field,LaserIntensity,Te,
                                  xLims,logscale,critical_density_m,type_of_profile,
                                  l_nc_4,dens_init,dens_fin,
                                  density_plot_function_density,dens_lim,boundaries,
                                  omega_k_is_true,t_k_is_true,
                                  omega_x_is_true,
                                  cumulate_plots,cumulate_plot_number,different_from_usual):


    c = 3*10**8
    lambda_m = Lambda * 10**-6
    k0 = (2*3.14)/lambda_m
    omega0 = c*k0
    omega0=(2*3.14*c)/lambda_m
    omega0_fs = omega0*10**-15
    delta_t_simulation = 0.1

    field_to_tdf= []
    time_for_tdf = []
    space_for_tdf = []

    if xLims:
        x0,index_x0 = find_nearest(x*1e6,xLims[0])
        x1,index_x1 = find_nearest(x*1e6,xLims[1])

        print(x1,index_x1)
        print(x0,index_x0)
        space_for_tdf = x[index_x0:index_x1]
    else:
        index_x0 = boundaries
        index_x1 = len(x)-boundaries
        space_for_tdf = x[index_x0:index_x1]

    print('k_transform',k_transform)
    print(index_x0,index_x1)
    if k_transform == True:

        print('Defining k plot')

        total_nodes = len(space_for_tdf)
        max_x = space_for_tdf.max()
        delta_fft = max_x/total_nodes
        k0 = 2*3.14/Lambda*1e6

        k_max = 2*3.14/delta_fft

        print('k_max')
        print(k_max)
        print('k_max')

        positive_k_axis = np.linspace(0,k_max/2,int(total_nodes/2))
        negative_k_axis = np.linspace(-k_max/2,0,int(total_nodes/2))

        k_axis = np.concatenate((negative_k_axis,0,positive_k_axis), axis=None)

        # plt.xlim(-2.2,2.2)

        k_axis = np.fft.fftshift(k_axis)
        kappa_plot = k_axis/k0



    cumulate_plot_count = 0





    time_for_t_k_plot.append(round(time,3))

    plt.title(''+str(round(time,3))+ ' ps')

    # for index_over_x in range(len(space_for_tdf)):
    #     field_as_function_omega_x[i,index_over_x ]  = energy[index_over_x]

    field_as_function_omega_x.append(energy)

    # if i ==0:
    #     if type_of_field.endswith('Density_Electron') or type_of_field.endswith('Density_a1_ion') :
    #         energy_0 = data_file.__dict__[type_of_field].data
    l_nc_4 = l_nc_4*1e-6
    if type_of_field.startswith('Derived_Number_Density'):
        print('warning!!! choose ln_nc_4')
        print(l_nc_4,space_for_tdf)




    density_0 = dens_init*critical_density_m*np.exp(space_for_tdf/l_nc_4)
        

    if k_transform == True:

        print('entra in k trans')

        fft_energy = energy[index_x0:index_x1]

        w = fftpack.fft(fft_energy)

        if type_of_field.startswith('Derived_Poynting_Flux_x'):
            l_intensity = LaserIntensity*1e4
            norm   = l_intensity
            plt.ylabel('$S_x$ ($I_0$)')
            spatial_quantity_tdf = np.abs(w)**2
            data_save_folder = 'sx'

        elif type_of_field.startswith('Derived_Poynting_Flux_y'):
            l_intensity = LaserIntensity*1e4
            norm   = l_intensity
            spatial_quantity_tdf = np.abs(w)**2

            plt.ylabel('$S_y$ ($I_0$)')
            data_save_folder = 'sy'


        elif type_of_field.startswith('Electric_Field_Ex'):


            # energy = smooth(energy**2,int(len(w)/1))
            spatial_quantity_tdf = np.abs(w)**2
            norm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
            plt.ylabel('$|E_x(k)|^2$ (a.u.)')
            data_save_folder = 'ex'



            plt.yscale('log')
            # plt.ylim(10e26,10e28)

        elif type_of_field.startswith('Electric_Field_Ez'):


            # energy = smooth(energy**2,int(len(w)/1))
            spatial_quantity_tdf = np.abs(w)**2
            norm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
            plt.ylabel('$|E_x(k)|^2$ (a.u.)')
            data_save_folder = 'ex'



            plt.yscale('log')


        elif type_of_field.startswith('Magnetic_Field_Bz'):


            # energy = smooth(energy**2,int(len(w)/1))
            spatial_quantity_tdf = np.abs(w)**2
            norm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
            plt.ylabel('$|E_x(k)|^2$ (a.u.)')
            data_save_folder = 'ex'



            plt.yscale('log')

        elif type_of_field.startswith('Electric_Field_Ey'):
            # energy = smooth(energy**2,int(len(w)/1))
            spatial_quantity_tdf = np.abs(w)**2

            norm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
            plt.ylabel('$|E_y(k)|^2$ (a.u.)')
            data_save_folder = 'ey'
            # plt.xlim(-1.1,1.1)
            plt.yscale('log')




            # field_to_plot = smooth(np.abs(w)**2, 10)

            plt.xlim(-1.5,1.5)

             # variable    = np.abs(np.array(variable))

        elif type_of_field.startswith('Derived_Number_Density'):
            # energy = smooth(energy**2,int(len(w)/1))
            print(fft_energy)
            spatial_quantity_tdf = np.abs(fftpack.fft(fft_energy/density_0-1))**2

            norm    = critical_density_m

            if type_of_field.endswith('Electron'):
                        
                plt.ylabel('$\delta n_e(k)$ ($n_0$)')
                data_save_folder = 'e_density'
            elif type_of_field.endswith('carbon'):
                
                plt.ylabel('$\delta n_c(k)$ ($n_0$)')
                data_save_folder = 'nc_density'

            elif type_of_field.endswith('proton'):
                
                plt.ylabel('$\delta n_p(k)$ ($n_0$)')

                data_save_folder = 'np_density'
            else:

                plt.ylabel('$\delta n_i(k)$ ($n_0$)')
                data_save_folder = 'i_density'


            


            print('fftpack.fft(fft_energy/density_0-1)')
            print(fft_energy.max())

            plt.yscale('log')
            plt.xlim(-4,4)




    
        spatial_tdf = []
        for index_kappa_tdf in range(len(kappa_plot)-1):
            spatial_tdf.append(spatial_quantity_tdf[index_kappa_tdf])
        spatial_quantity_tdf = np.array(spatial_tdf)


         # for subLst in spatial_quantity_tdf:
         #    lst.extend(subLst)
         # print(lst)
        print(save_also_snapshots)

        print('tdf')
        print(spatial_quantity_tdf.max())
        print(density_0.max())
        print('no_time_pics_snapshots')
        if save_also_snapshots == True:
            if not kLims:
                kLims = np.zeros(2)
                kLims[0] = -10
                kLims[1] = 10
            print(kappa_plot)
            k_k0_limit_right,index_limit_right = find_nearest(kappa_plot, kLims[0])
            k_k0_limit_left,index_limit_left = find_nearest(kappa_plot,kLims[1])

            index_1 = index_limit_right
            index_2 = index_limit_left
            print(index_1,index_2)
   #      plt.tick_params(labelsize=labelsize_tick, length = length_tick, width=1,direction='inout')
            plt.plot(kappa_plot[1:],smooth(spatial_quantity_tdf,int(len(spatial_quantity_tdf)/350))/spatial_quantity_tdf.max(),label ='t='+str(round(time,3)) )

            print(spatial_quantity_tdf)
            plt.xlabel('$k$ ($k_0$) ')
       #      # plt.xlim(-2.5,2.5)
       #      plt.xlim(-5.1,5,1)
            if kLims:
                plt.xlim(-5.5,5.5)
            plt.ylim(0.001,1.2)
       #      # plt.plot(k_axis*1e6/k0,smooth(field_in_k_space, 1))
            plt.grid()

            cumulate_plot_count = cumulate_plot_count +1
            if Legend == True:
                plt.legend(loc='best')


            if yLims:
                plt.ylim(yLims)
            if cumulate_plots == False:
                print(spatial_quantity_tdf)
                plt.title('t = '+str(round(time,3))+' ps')
                plt.savefig('./pics/'+str(data_save_folder)+'_k_t/'+str(args.output)+str(round(time,3))+'.jpg')
                plt.close()


            elif cumulate_plots == True:

                if cumulate_plot_count%cumulate_plot_number == 0:


                    plt.legend(loc='best',prop={'size':18})
                    plt.savefig('./pics/'+str(data_save_folder)+'_k_t/'+str(args.output)+str(round(time,3))+'_cumulative_plot.jpg',dpi=600,pad='tight')



                    plt.close()

        field_as_function_t_k.append(spatial_tdf)



    if t_k_is_true == True and looping_over_the_files == last_file-1:


        print('entering t k plot')
        field_as_function_t_k = np.array(field_as_function_t_k)
        time_for_t_k_plot = np.array(time_for_t_k_plot)

        field_as_function_t_k = np.array(field_as_function_t_k)
        
        if not kLims:
            kLims = np.zeros(2)
            kLims[0] = -10
            kLims[1] = 10
        k_k0_limit_right,index_limit_right = find_nearest(kappa_plot, kLims[0])
        k_k0_limit_left,index_limit_left = find_nearest(kappa_plot,kLims[1])

        
        levels = [0.01,1.5]
        kappa_plot_indexing_1 = 0 #len(kappa_plot)-4500
        kappa_plot_indexing_2 = len(kappa_plot)
        if minF:
            valore_minimo = minF
        else:
            valore_minimo = 0.0001
        if maxF:
            valore_massimo = maxF
        else:
            valore_massimo = 1
        levels = np.linspace(valore_minimo, valore_massimo, 7)
#        fig = plt.figure('t k plot',figsize=(10,10))
        fig, ax = plt.subplots()

        # cs = plt.contourf(time_for_t_k_plot ,limited_k_plot,np.log(field_as_function_t_k_limited_k.T),cmap=cm.seismic)
        print(len(kappa_plot[kappa_plot_indexing_1:kappa_plot_indexing_2]))
        cs = plt.contourf(time_for_t_k_plot ,kappa_plot[kappa_plot_indexing_1+1:kappa_plot_indexing_2],field_as_function_t_k[:,kappa_plot_indexing_1:kappa_plot_indexing_2].T/field_as_function_t_k.max(),locator=ticker.LogLocator(),levels = levels,cmap='viridis')

        
        cb = plt.colorbar(cs)
        plt.ylim(kLims[0],kLims[1])
        plt.xlabel(r"t (ps)",**axis_font);
        plt.ylabel('$k_x$ ($k_0$)',**axis_font)
        plt.tick_params(labelsize=20, length = 15, width=1,direction='inout')
        cb.ax.tick_params(labelsize = 20,length = 15)
        plt.savefig('./pics/'+str(args.output)+'_t_k.jpg')


    print(looping_over_the_files,'looping_over_the_files')
    print(last_file,'last_file')

    if omega_transform == True and looping_over_the_files == last_file-1:


        print('entering omega trans')

        time_for_t_k_plot = np.array(time_for_t_k_plot)

          #Prepare the frequencies
        freqs = 2*3.14*fftfreq(len(time_for_t_k_plot),d=delta_t_simulation)/omega0_fs

        #Mask array to be used in the power spectra, ignoring half of the values,
        # being complex conjucates
        mask = freqs > 0


        if omega_k_is_true == True:

            print('entering omega k plot')
            field_as_function_t_k = np.array(field_as_function_t_k)

            # # FFT and power spectra
            fft_theo = []

            for index_over_kappa in range(len(kappa_plot)):

            # fft_vals =

            # # true theoretical fft
                fft_theo.append(2*np.abs(fft(field_as_function_t_k[:,index_over_kappa])/len(time_for_t_k_plot))**2)


            fft_theo = np.array(fft_theo)




            k_k0_limit_r,index_limit_r = find_nearest(kappa_plot, 3.5)
            k_k0_limit_l,index_limit_l = find_nearest(kappa_plot, -3.5)


            kappa_plot_indexing_1 = 0 #4500
            kappa_plot_indexing_2 = len(kappa_plot)-4500





            fig = plt.figure('omega k plot',figsize=(10,10))

            cs = plt.contourf(kappa_plot,freqs[mask],fft_theo[:, mask].T, locator=ticker.LogLocator(),
                      cmap=cm.seismic)

            plt.tick_params(labelsize=15)
            cbr= plt.colorbar()
            cbr.ax.tick_params(labelsize=15)#        cbar = fig.colorbar(fig, ticks=[0,1,2,3])
            # plt.clim(1e-4,1.5)
            # plt.ylim(-3.5,3.5)
            plt.ylabel(r"$\omega/\omega_0$",**axis_font);
            plt.xlabel(r'$k$ ($k_0$)',**axis_font)
            plt.tick_params(labelsize=15, length = 10, width=1,direction='inout')
            plt.xlim(-2.5,2.5)
            plt.ylim(0.4,1.5)

            plt.savefig('./pics/'+str(args.output)+'_omega_k.jpg')




        if omega_x_is_true == True:

            print('entering omega x plot')
            space_for_tdf = space_for_tdf*1e6
            field_as_function_omega_x = np.array(field_as_function_omega_x)

            # # FFT and power spectra
            fft_theo = []

            for index_over_x in range(len(space_for_tdf)):

            # # true theoretical fft
                fft_theo.append(2*np.abs(fft(field_as_function_omega_x[:,index_over_x])/len(space_for_tdf))**2)


            fft_theo = np.array(fft_theo)



            fig = plt.figure('omega k plot',figsize=(10,10))

            cs = plt.contourf(space_for_tdf,freqs[mask],fft_theo[:, mask].T, locator=ticker.LogLocator(),
                      cmap=cm.seismic)

            plt.tick_params(labelsize=15)
            cbr= plt.colorbar()
            cbr.ax.tick_params(labelsize=15)#        cbar = fig.colorbar(fig, ticks=[0,1,2,3])
            # plt.clim(1e-4,1.5)
            # plt.ylim(-3.5,3.5)
            plt.ylabel(r"$\omega/\omega_0$",**axis_font);
            plt.xlabel(r'$x$ ($\mu$m)',**axis_font)
            plt.tick_params(labelsize=15, length = 10, width=1,direction='inout')
            # plt.xlim(-2.5,2.5)
            # plt.ylim(0.4,1.5)

            plt.savefig('./pics/'+str(args.output)+'_omega_x.jpg')
            if writing_data_file == True:
                datafile_path = "./files_value_box/data.txt"
                with open(datafile_path, 'w+') as datafile_id:
                     np.savetxt(datafile_id, fft_theo, fmt='%f')


def meanEnergyVsTime(ax,files,field,xLims=None,log=False):
    data = [ sdf.read(f) for f in files ]
    time = np.array([ d.Header['time'] for d in data ])
    energy = np.array([ np.mean(d.__dict__[field].data**2) for d in data ])
    norm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
    #norm = 0.5*const.epsilon_0
    energy = energy/norm

    ax.plot(time/1e-12,energy)

    if xLims:
        ax.set_xlim(xLims)
    if log:
        ax.set_yscale('log')
    ax.set_xlabel('time /ps')
    ax.set_ylabel(r'$\frac{e^2\left\langle |E_x|^2 \right\rangle}{(m_e\omega_0c)^2}$')
    ax.grid()



def calcMeanEnergyVsSpaceTime(prefix,averaging_over_y,files,field,different_from_usual,
                              acc_core_subset,max_accumulator_step,DSnapshots,parallel=False):

    data = [ sdf.read(f) for f in files ]
    ts = np.array([ d.Header['time'] for d in data ])

    

    if prefix.startswith('regular'):
        grid = data[0].Grid_Grid.data
        x= data[0].Grid_Grid.data[0]
    else:
        grid = data[0].__dict__['Grid_A'+str(args.positional_prefix)+'_'+str(prefix)].data
        x = data[0].__dict__['Grid_A'+str(args.positional_prefix)+'_'+str(prefix)].data[0]

    print('grid is')

    print(len(grid))
    if len(grid) == 1:
        print(field)

        if field.startswith('Derived_'):
            energy = np.array([ d.__dict__[field].data for d in data ])

        else:
            if acc_core_subset == '_Acc':
                if max_accumulator_step is None:
                    print('Error: YOU NEED TO Enter the accumulator steps')

                for index_accumulator in range(0,max_accumulator_step,DSnapshots):
                    energy = np.array([ d.__dict__[field].data[:,:index_accumulator]**2 for d in data ])

            else:

                energy = np.array([ d.__dict__[field].data**2 for d in data ])

        print('max val: {:}'.format(np.max(energy)))


    elif len(grid) == 2 or len(grid) == 3:

        if averaging_over_y == False:
        # We have a 2D dataset, need to average over y
            if field.startswith('Derived_'):
                energy = np.array([ d.__dict__[field].data for d in data ])

            else:
                if acc_core_subset == '_Acc':
                    if max_accumulator_step is None:
                        print('Error: YOU NEED TO Enter the accumulator steps')
                    
                    for index_accumulator in range(0,max_accumulator_step,DSnapshots):
                        print(index_accumulator)
                        #for d in data:
                         #   print(len(d.__dict__[field].data[:,0,index_accumulator]))

                        energy = np.array([ d.__dict__[field].data[:,0,index_accumulator]**2 for d in data ])
                else:
                    energy = np.array([ d.__dict__[field].data**2 for d in data ])

            print('max val: {:}'.format(np.max(energy)))


            if acc_core_subset == '_Acc':
                # y = data[0].__dict__['Grid'+str(args.positional_prefix)].data[1]
                y = data[0].__dict__['Grid_A'+str(args.positional_prefix)+'_'+str(prefix)].data[1]
            else:
                y = data[0].Grid_Grid.data[1]

        elif averaging_over_y == True:
             if field.startswith('Derived_'):
                energy = np.array([ np.mean(d.__dict__[field].data,axis=1) for d in data ])

             else:
                if acc_core_subset == '_Acc':

                    if max_accumulator_step is None:
                        print('Error: YOU NEED TO Enter the accumulator steps')

                    for index_accumulator in range(0,max_accumulator_step):
                        energy = np.array([ d.__dict__[field].data[:,:,index_accumulator]**2 for d in data ])
                else:
                    energy = np.array([ np.mean(d.__dict__[field].data**2,axis=1) for d in data ])



        #if acc_core_subset == '_Acc':

        # x = data[0].__dict__['Grid'+str(args.positional_prefix)].data[0]
         #   x = data[0].__dict__['Grid_A'+str(args.positional_prefix)+'_'+str(prefix)].data[0]
       # elif prefix.startswith('regular'):
        #    x= data[0].Grid_Grid.data[0]

        print('energy is')
        print(energy)
    return energy,x,ts

# 

def plotMeanEnergyVsSpaceTime(norm_temp_to_init_temp,density_normalized_to_n0,density_normalized_to_nc,critical_density_m,
                              density_profile_x,Z,collision_frq_study,fig,ax,energy,x,t,field,smoothLen=None,ne=None,
                              xLims=None,yLims=None,maxF=None,minF=None,
                              minFPercentile=0.5,maxFPercentile=99.9,
                              log=False,minNeTick=None,neTickInterval=None,
                              CBar=True,CBarLabel=True,Te=None,markQC=True,
                              markTPDCutoff=True,markSRSCutoff=True,
                              landauCutoff=0.3):

    extent = srsUtils.misc.getExtent(x/1e-6,t/1e-12)

    if field.startswith('Magnetic_Field_By'):
        smoothLen = 5*srsUtils.wlVacNIF
        dx = x[1]-x[0]
        energy = gaussian_filter1d(energy,sigma=smoothLen/dx)

        densityRange = np.array([0.194580059689,0.260175683371])*srsUtils.nCritNIF
        xMid = 0.5*(x[1:] + x[:-1])
        Ln = xMid/math.log(densityRange[1]/densityRange[0])
        ne = densityRange[0]*np.exp(xMid/Ln)
        op = np.sqrt(ne/(const.m_e*const.epsilon_0))*const.e
        kL = np.sqrt(srsUtils.omegaNIF**2 - op**2)/const.c
        vPhL = srsUtils.omegaNIF/kL

        eNorm = (const.m_e*srsUtils.omegaNIF/const.e)**2*(const.c/vPhL)**2
        w0 = srsUtils.speckle.speckleWidth(6.7,srsUtils.wlVacNIF)
        x0 = srsUtils.speckle.gaussianBeamLengthAtAmp(0.0,math.sqrt(0.5),srsUtils.wnVacNIF,w0)
        E0 = srsUtils.intensityToEField(3.75e15*1e4)*srsUtils.speckle.gaussianBeamAmplitude(x0,y,srsUtils.wnVacNIF,w0)
        E0 = 0.5*np.mean(E0**2)/(const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
        print(E0)
        levels = np.array([0.25,0.5,0.75,1.0])*E0
        ax.contour(energy,levels,extent=extent,origin='lower',linewidths=0.5,colors=['r','y','g','k'])
# =============================================================================
#  NORMALIZATIONS
# =============================================================================
    elif field.startswith('Magnetic_Field_Bz'):
        eNorm = (const.m_e*srsUtils.omegaNIF/const.e)**2
        energy = energy/eNorm
    elif field.startswith('Electric_Field_E'):
        eNorm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
        energy = energy/eNorm
    elif field.startswith('Derived_Poynting_Flux'):
        print('Entra qua?')
        eNorm = args.LaserIntensity*1e4
        energy = energy/eNorm

    elif field.startswith('Derived_Number_Density'):
        print(energy[0]/critical_density_m)
        if density_normalized_to_nc == True:
            print('len(energy[0])',len(energy[0]))
            eNorm = critical_density_m
        elif density_normalized_to_n0 == True:
            eNorm = energy[0]
        else:
            eNorm = 1
    elif field.startswith('Derived_Temperature'):
        if collision_frq_study == False:
            print(Te)
            print('Te')
            if norm_temp_to_init_temp == True:
                eNorm = Te
            else:
                eNorm = 1/(convert_k_to_kev)
            energy = energy/eNorm

        elif collision_frq_study == True:
            eps0 = 8.85418 * 10**(-12)

            tejoule = energy

            temporal_size = int(len(tejoule))

            costante_i_b = 4*np.sqrt(2)*np.pi * const.e**4 /(3*np.sqrt(const.m_e))
            accataglaito = 1.054 *1e-34
            lmin = Z*const.e**2/ tejoule
            # ldb = accataglaito/(2*electronmass*tejoule)**0.5


            temporal_density_profile_array = np.ones((temporal_size),len(density_profile_x))
            temporal_l_debye = np.ones((temporal_size),len(density_profile_x))

            for i in range(temporal_size):
                temporal_density_profile_array[i,:] = density_profile_x[:]


            temporal_l_debye = np.sqrt((eps0*tejoule)/(const.e**2*temporal_density_profile_array))

            # coulomb_lambda = 20
            coulomb_lambda = np.log(temporal_l_debye/lmin)
            collisional_frequency_e_i = 2*3.14*costante_i_b*np.log(coulomb_lambda)*temporal_density_profile_array*Z / (tejoule)**3/2
            eNorm = srsUtils.omegaNIF
            energy = collisional_frequency_e_i/eNorm

    #else:
    #    eNorm = (const.m_e*srsUtils.omegaNIF*const.c/const.e)**2
    #norm =0.5*const.epsilon_0
    if field.startswith('Derived_Number_Density') and density_normalized_to_nc == True:
        energy = (energy/eNorm)
    if field.startswith('Derived_Number_Density') and density_normalized_to_n0 == True:
        energy = (energy/eNorm-1)*100
    else:
        energy = energy


    if not maxF:
        maxF = np.percentile(energy,maxFPercentile)
    if not minF:
        if log:
            minF = np.percentile(energy,minFPercentile)
        else:
            minF = 0.0
    if not log:
        norm = colors.Normalize(vmin=minF,vmax=maxF)
    else:
        norm = colors.LogNorm(vmin=minF,vmax=maxF)

    #energy[np.where(energy < 1e-5)] = np.nan
    # Downsample array if it is excessively large
    reduceArray = tuple([ max([1,s // 2000]) for s in energy.shape ])
    energy = skimage.measure.block_reduce(energy,reduceArray,np.mean)

    print(energy.max())
    if field.startswith('Electric_Field_Ez'):
        im = ax.imshow(energy,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='plasma',norm=norm)
    elif field.startswith('Magnetic_Field_Bz'):
        im = ax.imshow(energy,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='autumn',norm=norm)
    elif field.startswith('Electric_Field_Ex'):
        im = ax.imshow(energy,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Purples',norm=norm)
    elif field.startswith('Derived_Poynting_Flux_y'):
        im = ax.imshow(energy,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='PuBuGn',norm=norm)
    elif field.startswith('Derived_Poynting_Flux_x'):
        im = ax.imshow(energy,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap=cm.seismic,norm=norm)
    elif field.endswith('Density_a1_ion') or field.endswith('arbon') or field.endswith('ydrogen'):
        im = ax.imshow((energy-energy[0,:])/energy[0,:],interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='cool',norm=norm)
    elif field.startswith('Derived_Temperature_Electron'):
        im = ax.imshow(energy,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='ocean',norm=norm)
    elif field.startswith('Derived_Number_Density_Electron'):
        im = ax.imshow((energy-energy[0,:])/energy[0,:],interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Blues',norm=norm)
    else:
        im = ax.imshow(energy,interpolation='none',origin='lower',aspect='auto',extent=extent,cmap='Oranges',norm=norm)

    if CBar:
        cb = fig.colorbar(im,ax=ax,orientation='vertical')
        if not log:
            cb.formatter.set_powerlimits((-4,4))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()

    ax.tick_params(reset=True,axis='both',color='w',direction='in')

    # Annotate location of important densities
    if ne is not None:
        if markQC:
            xQC,index_xqc = find_nearest(ne/critical_density_m,density_tick_uno)
            #x02,index_x02 = find_nearest(ne/critical_density_m,density_tick_due)
            x019,index_x019 = find_nearest(ne/critical_density_m,density_tick_tre)
            x01,index_x01 = find_nearest(ne/critical_density_m,density_tick_quattro)
            x005,index_x005 = find_nearest(ne/critical_density_m,density_tick_cinque)
            x002,index_x002 = find_nearest(ne/critical_density_m,density_tick_sei)

            if xQC is not None: ax.axvline(x[index_xqc]*1e6,color='g',linestyle=':')
            #if x02 is not None: ax.axvline(x[index_x02]*1e6,color='g',linestyle=':')
            if x019 is not None: ax.axvline(x[index_x019]*1e6,color='g',linestyle=':')
            if x01 is not None: ax.axvline(x[index_x01]*1e6,color='g',linestyle=':')
            if x005 is not None: ax.axvline(x[index_x005]*1e6,color='g',linestyle=':')
            if x002 is not None: ax.axvline(x[index_x002]*1e6,color='g',linestyle=':')



        if Te is not None:
            bth = math.sqrt(const.k*Te/const.m_e)/const.c
            if markTPDCutoff:
                tpdCutoffNe = srsUtils.tpd.landauCutoffDens(bth,cutoff=landauCutoff)
                diff = np.abs(ne/srsUtils.nCritNIF-tpdCutoffNe)
                xCutoffTPD = x[np.where(diff == np.min(diff))]
                print('xCutoffTPD',xCutoffTPD*1e6)
                ax.axvline(xCutoffTPD/1e-6,linestyle='--',color='w')

            if markSRSCutoff:
                srsCutoffNe = srsUtils.srs.landauCutoffDens(bth,math.pi,cutoff=landauCutoff)
                diff = np.abs(ne/srsUtils.nCritNIF-srsCutoffNe)
                xCutoffSRS = x[np.where(diff == np.min(diff))]
                print('xCutoffSRS',xCutoffSRS*1e6)
                ax.axvline(xCutoffSRS/1e-6,linestyle='--',color='k')




    if xLims:
        ax.set_xlim(xLims)
    if yLims:
        ax.set_ylim(yLims)

    ax.set_xlabel(r'x $/\mu$m')
    ax.set_ylabel('time /ps')

    return fig,ax

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataDir')
    parser.add_argument('field')

    parser.add_argument('--acc_core_subset',type=str,default='')
    parser.add_argument('--max_accumulator_step',type=int)
    parser.add_argument('--positional_prefix',type=str,default='')
    parser.add_argument('--different_from_usual', type = bool, default=False)


    parser.add_argument('--prefix',default='regular_')
    parser.add_argument('--space',action='store_true')
    parser.add_argument('--averaging_over_y', type = bool, default=False)

    parser.add_argument('--parallel',action='store_true')
    parser.add_argument('--useCached',action='store_true')
    parser.add_argument('--cacheFile')



    parser.add_argument('--densProfFile',default='regular_0000.sdf')
    parser.add_argument('--densitySpecies',default='electrons')
    parser.add_argument('--Te',type=float)

    parser.add_argument('--coloreMap',default = 'viridis')

    parser.add_argument('--Intervallo',default = 'False')
    parser.add_argument('--IntervalInitial',type=int)
    parser.add_argument('--IntervalFinal',type=int)
    parser.add_argument('--init_int_final',type=bool, default = False)
    parser.add_argument('--grid_is_missing_in_bulk',type=bool, default = False)


    parser.add_argument('--Snapshots',type=int,default = 1)
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


    args = parser.parse_args()

    critical_density_m = 1.1 * 1e21 * (args.Lambda)**-2 * 1e6

    print(critical_density_m)
    print('Tempora_cut',args.Tempora_cut)
    print(args.dataDir,'arg data')
    
    DSnapshots = args.DSnapshots
    type_of_field = args.field
    prefix=args.prefix
    #type_of_field = str(type_of_field)+str(args.acc_core_subset)+str(args.positional_prefix)
    if not  prefix.startswith('regular'):
        type_of_field = str(type_of_field)+str(args.acc_core_subset)+str(args.positional_prefix)+'_'+str(args.prefix)
    
    print('The quantity under  analysis is ')
    print(type_of_field)
    if args.Te is not None:
        Te = args.Te*srsUtils.TkeV
    else:
        Te = None

    if args.fontSize:
        import matplotlib as mpl
        mpl.rcParams.update({'font.size':args.fontSize})

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
            if args.Tempora_cut == False:


                if args.Intervallo == 'True':
                    files = sdfUtils.listFiles(args.dataDir,args.prefix)[args.IntervalInitial:args.IntervalFinal]

                elif args.Intervallo == 'False':

                    files = sdfUtils.listFiles(args.dataDir,args.prefix)[::args.Snapshots]

                    if args.init_int_final == True:
                        print('Initial and Final interval with delta snapshots')
                        files = sdfUtils.listFiles(args.dataDir,args.prefix)[args.IntervalInitial:args.IntervalFinal:args.Snapshots]


                if len(files) == 0:

                    raise IOError("Couldn't find any SDF files with prefix {:}".format(args.prefix))
                else:
                    print("Found {:} SDF files with prefix {:}".format(len(files),args.prefix))
                    print(files)
            elif args.Tempora_cut == True:

                print('Entra in Temporal cut')
                files = sdfUtils.listFiles(args.dataDir,args.prefix)[args.IntervalInitial:args.IntervalFinal:args.Snapshots]

                # print(files)
                print(args.IntervalInitial)
                print(args.IntervalFinal)
                print(files)

            # Find quarter critical surface
            sdfProf = sdf.read(os.path.join(args.dataDir,args.densProfFile))

#            if args.acc_core_subset == '_Acc':
#                x  = sdfProf.__dict__['Grid_A'+str(args.positional_prefix)+'_'+str(args.prefix)].data[0]
#
#
#            else:
            x  = sdfProf.Grid_Grid.data[0]


            print(args.densitySpecies)
            if args.noMarkQC:
                ne = None
            else:
                if args.densitySpecies == '':
                    ne = sdfProf.__dict__['Derived_Number_Density'+str(args.acc_core_subset)+str(args.positional_prefix)].data
                else:
                    ne = sdfProf.__dict__['Derived_Number_Density_Electron'].data


                if len(ne.shape) == 2:
                    ne = np.mean(ne, axis = 1) #changed 1 to 0
                ne = gaussian_filter1d(ne,sigma=10)


            if args.Tempora_cut == False:

                # energy,x,t = calcMeanEnergyVsSpaceTime(args.averaging_over_y,files,type_of_field ,args.different_from_usual,parallel=args.parallel)
                energy,x,t = calcMeanEnergyVsSpaceTime(args.prefix,args.averaging_over_y,files,type_of_field ,
                                                       args.different_from_usual, args.acc_core_subset,
                                                       args.max_accumulator_step,DSnapshots,parallel=False)

                if args.cacheFile is not None:
                        np.savez_compressed(args.cacheFile,energy=energy,x=x,t=t,ne=ne)


                a,b,l_tot,density_profile_x=density_profile(args.type_of_profile,args.l_nc_4,args.dens_init,args.dens_fin,x,critical_density_m)

                plotMeanEnergyVsSpaceTime(args.norm_temp_to_init_temp,args.density_normalized_to_n0,args.density_normalized_to_nc,critical_density_m,
                                          density_profile_x,args.Z,args.collision_frq_study,fig,ax,energy,x,t,type_of_field ,
                    ne=ne,xLims=args.xLims,yLims=args.yLims,maxF=args.maxF,
                    minF=args.minF,maxFPercentile=args.maxFPercentile,
                    minFPercentile=args.minFPercentile,log=args.log,
                    minNeTick=args.minNeTick,neTickInterval=args.neTickInterval,
                    markQC=not args.noMarkQC,CBar=not args.noCBar,
                    CBarLabel=not args.noCBarLabel,Te=Te,
                    markTPDCutoff=args.markTPDCutoff,markSRSCutoff=args.markSRSCutoff,
                    landauCutoff=args.landauCutoff)

            elif args.Tempora_cut == True:



                if args.tdf_transform == False:
                    #a,b,l_tot,density_profile_x=density_profile(args.type_of_profile,args.l_nc_4,args.dens_init,args.dens_fin,x,critical_density_m)
                    plot_temporal_cut(args.different_from_usual,args.Legend,
                                      type_of_field,args.smoothed_is_true,files,
                                      args.IntervalInitial,args.IntervalFinal,
                                      args.Snapshots,x,
                                      args.LaserIntensity,args.Te,args.xLims,
                                      args.yLims,args.logscale,critical_density_m,
                                  args.type_of_profile,args.l_nc_4,
                                  args.dens_init,args.dens_fin,
                                  args.density_plot_function_density,args.dens_lim,
                                  args.cumulate_plots,args.cumulate_plot_number)


                elif args.tdf_transform == True:
                    print('tdf_transform is true')
                    print(args.k_transform,'args.k_transform')
                    print(args.omega_transform,'args.omega_transform')

                    # last_file = int((args.IntervalFinal-args.IntervalInitial)/args.Snapshots)
                    last_file = len(files)

                    looping_over_the_files = 0


                    field_as_function_t_k = []
                    field_as_function_omega_x = []
                    # field_as_function_omega_x = np.zeros((last_file,len(space_for_tdf)))
                    time_for_t_k_plot = []

                    # print(files)
                    for i in range(args.IntervalInitial,args.IntervalFinal,args.Snapshots):

                        files_tdf = sdfUtils.listFiles(args.dataDir,args.prefix)[i]

                        data_file =  sdf.read(files_tdf)

                        time = np.array(data_file.Header['time'] )/1e-12

                        if args.acc_core_subset == '_Acc':

                            print('Enters in acc')
                            if args.max_accumulator_step is None:
                                print('Error: YOU NEED TO Enter the accumulator steps')

                            for index_accumulator in range(0,args.max_accumulator_step):
                                energy  =data_file.__dict__[type_of_field].data[:,:index_accumulator]


                                print(files_tdf)
                                print(time)

                                plot_tdf(args.minF,args.maxF,args.kLims,args.writing_data_file,field_as_function_t_k,field_as_function_omega_x,
                                         time_for_t_k_plot,
                                         looping_over_the_files,time,args.k_transform,args.omega_transform,args.save_also_snapshots,args.yLims,
                                         args.Legend,args.Lambda,type_of_field,
                                         args.smoothed_is_true,files,args.IntervalInitial,
                                         args.IntervalFinal,args.Snapshots,energy,x,
                                         type_of_field ,args.LaserIntensity,args.Te,args.xLims,args.logscale,critical_density_m,
                                              args.type_of_profile,args.l_nc_4,args.dens_init,
                                              args.dens_fin,
                                              args.density_plot_function_density,args.dens_lim,
                                              args.boundaries,args.omega_k_is_true,args.t_k_is_true,
                                              args.omega_x_is_true,
                                              args.cumulate_plots,args.cumulate_plot_number,
                                              args.different_from_usual)


                                looping_over_the_files = looping_over_the_files + 1
                            del files_tdf,data_file
                        else:
                            energy = data_file.__dict__[type_of_field].data
                            print(files_tdf)
                            print(time)

                            plot_tdf(args.minF,args.maxF,args.kLims,args.writing_data_file,field_as_function_t_k,field_as_function_omega_x,
                                     time_for_t_k_plot,
                                     looping_over_the_files,time,args.k_transform,args.omega_transform,args.save_also_snapshots,args.yLims,
                                     args.Legend,args.Lambda,type_of_field,
                                     args.smoothed_is_true,files,args.IntervalInitial,
                                     args.IntervalFinal,args.Snapshots,energy,x,
                                     type_of_field ,args.LaserIntensity,args.Te,args.xLims,args.logscale,critical_density_m,
                                          args.type_of_profile,args.l_nc_4,args.dens_init,
                                          args.dens_fin,
                                          args.density_plot_function_density,args.dens_lim,
                                          args.boundaries,args.omega_k_is_true,args.t_k_is_true,
                                          args.omega_x_is_true,
                                          args.cumulate_plots,args.cumulate_plot_number,
                                          args.different_from_usual)


                            looping_over_the_files = looping_over_the_files + 1
                            del files_tdf,data_file
                    # plotting_x_k(args.x_k_is_true,field_as_function_x_k,kappa_plot,ics_plot)





        # plot_temporal_cut(args.smoothed_is_true,energy,files,args.IntervalInitial,energy,x,type_of_field ,args.LaserIntensity,args.Te,xLims=None)



            else:

                files = sdfUtils.listFiles(args.dataDir,args.prefix)
                if len(files) == 0:
                    raise IOError("Couldn't find any SDF files with prefix {:}".format(args.prefix))
                else:
                    print("Found {:} SDF files with prefix {:}".format(len(files),args.prefix))

                meanEnergyVsTime(ax,files,type_of_field ,xLims=args.xLims,log=args.log)

            if args.output and args.Tempora_cut == False:
                if args.figSize:
                    fig.set_size_inches(args.figSize)
                fig.tight_layout(pad=0,w_pad=0,h_pad=0)
                fig.savefig('pics/'+args.output,dpi=600,pad='tight')
            # else:
            #     plt.show()



