#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 12:32:56 2022

@author: alessandro_ruocco
"""


import numpy as np
import sdf
import sdfUtils


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataDir')
    
    # Prefix of file to analyse
    parser.add_argument('--file',choices=['acc_field','regular'],default ='acc_field' )  
    
    # Field component
    parser.add_argument('--which_quantity',choices=['Ex','Bx','Ey','By','Ez','Bz'])     
    
    # Strips in 'acc_field' 
    parser.add_argument('--which_strip',choices=['y0','left','right','up','bottom','nc19']) 
    # 'y0':      y=0 longitudinal strip
    # 'left':    x=0 vertical strip
    # 'right':   x=Lmax vertical strip
    # 'up':      y=yMAX longitudinal strip 
    # 'bottom':  y=ymin longitudinal strip 
    # 'nc19':    x(n = 0.19nc) vertical strip 
    
    
    # this is not the case if 'regular'
    parser.add_argument('--is_accumulator',default ='Acc' )  
    
    # Must be True if one wants to select a given time interval
    parser.add_argument('--Interval',type= bool, default = False) 
    
    # Initial file of the time interval chosen
    parser.add_argument('-ff','--first_file',type=int)
    
    # Last file of the time interval chosen
    parser.add_argument('-lf','--last_file',type=int)   

    
  
    args = parser.parse_args()

    if args.which_quantity.startswith('E'):
        field_prefix = 'Electric_Field'
    elif args.which_quantity.startswith('B'):
        field_prefix = 'Magnetic_Field'
    
    strips_choices =['y0','left','right','up','bottom','nc19']
    
    if args.which_strip is not strips_choices:
        print('Please, choose one of this option',strips_choices)
    
    if args.which_strip == 'y0':
        strip_name = 'strip_y0'
    elif args.which_strip == 'left':
        strip_name = 'x_left'    
    elif args.which_strip == 'right':
        strip_name = 'x_right'
    elif args.which_strip == 'up':
        strip_name = 'y_up'
    elif args.which_strip == 'bottom':
        strip_name = 'y_b'
    elif args.which_strip == 'nc19':
        strip_name = 'x_nc19'
    
    
    
    field_name = str(field_prefix)+'_'+str(args.which_quantity)+'_'+str(args.is_accumulator)+'_'+str(strip_name)+'_'+str(args.file)

    print('The quantity we will analyze is: ')


    print(field_name)
    
    if args.Interval == False: # took all the availale files

        sdfFiles = sdfUtils.listFiles(args.dataDir,args.file)[:]
        
    if args.Interval == True: # took files in certain interval
    
        sdfFiles = sdfUtils.listFiles(args.dataDir,args.file)[args.first_file:args.last_file]


    print('The files we will analyze are: ')

    print(sdfFiles)
    


    time_simulated = []
    raw_field = []
    time_before = 0
    for index_files in range(len(sdfFiles)):
        
        data =  sdf.read(sdfFiles[index_files])
        
        print('Analysing ',sdfFiles[index_files])
        print(sdfFiles[index_files])

        
        # Here we define the time at we start our analysis
        
        if index_files == 0:
            initial_time =  np.array(data.Header['time'] )/1e-12


        
        time = np.array(data.Header['time'] )/1e-12
        
        
        # Here, we define the accumulator step for each field in each file
        
        accumulator_steps = data.__dict__[field_name].data[0,0,:].shape[0]
        
        print('accumulator_steps',accumulator_steps)
        accumulator_delta_time = (time-time_before)/accumulator_steps+1
        print('accumulator_delta_time',accumulator_delta_time)
        print('time',time)
        print('-------')
        print('')
        print('-------')
        time_before = time


        for index_accumulator_step in range(0,accumulator_steps):

            # Spatially averaged raw field vale at each accumulator time
            if args.which_strip == 'y0' or args.which_strip == 'up' or args.which_strip == 'bottom':
                
                raw_field.append(data.__dict__[field_name].data[0,:,index_accumulator_step].mean())  # .mean gives the x-integrated value of the field
            
            elif args.which_strip == 'left' or args.which_strip == 'right'  or args.which_strip == 'nc19':
            
                raw_field.append(data.__dict__[field_name].data[:,0,index_accumulator_step].mean())  # .mean gives the y-integrated value of the field

            # Time 
            time_simulated.append(time+index_accumulator_step*accumulator_delta_time)  # time in ps
            
raw_field = np.array(raw_field)
time_simulated = np.array(time_simulated)
           
for i in range(len(raw_field)):
    
    print('time (ps)' , time_simulated[i],'spatial integrated raw field (PIC units)', raw_field[i])
    
    
    
    
    
    
    
       