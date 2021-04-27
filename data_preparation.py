# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:51:19 2021

@author: Vien Che
"""

import numpy as np
import glob
import os
from scipy import integrate
import scipy.signal
# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname':'Arial', 'size':'26', 'color':'black', 'weight':'normal',
  'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'22'}

# Set the font properties (for use in legend)   
font_path = 'C:\Windows\Fonts\Arial.ttf'


#names=['PVC','LDPE','PS','PP','PET','PE','PC','PA','PMMA'] #BAM and BPI sample names
#names=['HDPE','LDPE','PA','PC','PET','PMMA','PP','PS'] #Brett samples name
#names=['PA','PC','PE300','PE500','PE1000','PET','PMMA','PP','PS','PVC'] #VG samples name
names=['N1','N2','N3','N4','N5inner','N5outer','N6inner','N6outer','N7','N8','N9','N10']
company='_Brett'
path='E:/destop/1st measurement PL/non plastic/'
for i in range(len(names)):
    name=names[i]
    samples=sorted(glob.glob(path+name+'/'+name+'*'),key = os.path.getmtime, reverse=False)
    background = sorted(glob.glob(path+name+'/background*'),key = os.path.getmtime, reverse=False)
    N= len(background) #the number of sample (since each sample has one background file)
    
    print('Total number of spectra: ',len(samples))
    bg=[]   #empty list for background
    for i in range(len(background)):
        data = np.loadtxt(background[i])
        taxis = data[:,0]
        bg1 = data[:,1]
        bg.append(bg1)            
    bg = np.asarray(bg)
    datafx=[] #empty list for spectra
    for i in range(len(samples)):
        data = np.loadtxt(samples[i])
        taxis = data[:,0]
        data = data[:,1] 
        datafx.append(data)
    
    #subtract background and smoothen the curve and perform nomorlization
    pl1 = (datafx-bg)
    
    #pl1= scipy.signal.savgol_filter(pl1, 51, 3) # window size 51, polynomial order 3    
    norm_1=[]
    pl_bg_subtract1=[]
    index=[]

    for j in range(len(pl1)):

        if np.max(pl1[j][712:1928]) > 2000 and np.max(pl1[j]) < 60000 and np.min(pl1[j])> -500 :

            pl1[j]=pl1[j]-np.mean(pl1[j][2683:]) #offset correction from 800nm
            #normalization based on the spectrum area
            norm1=pl1[j][658:2681]/np.abs(integrate.simps(pl1[j][658:2681], taxis[658:2681])) # spetrum interval from 410 to 800
            norm_1.append(norm1)
            pl_bg_subtract1.append(pl1[j][658:2681])
            index.append(j)
    
    
    pl_bg_subtract=np.array_split(pl_bg_subtract1,N)

    print(name)
    print(index)

    print('Total spectra after subtract: ',len(norm_1))

######## COLOR PLASTIC #################

plastic_types=['HDPE','LDPE','PET','PP']
color=['black','blue','grey','pink','purple','white','yellow','green','red','transparent']
path='E:/destop/1st measurement PL/spectra colored plastic sorted'
for i in range(len(plastic_types)):
    name=plastic_types[i]
    for k in color:
        for v in range (0,171):
            try:
                background = sorted(glob.glob(path+'/'+name+'/'+k+'/'+str(v)+'/background*'),key = os.path.getmtime, reverse=False)
                samples=sorted(glob.glob(path+'/'+name+'/'+k+'/'+str(v)+'/data/*'),key = os.path.getmtime, reverse=False)

    
    
                N= len(background) #the number of sample (since each sample has one background file)
        
                #print('Total number of spectra: ',len(samples))
                bg=[]   #empty list for background
                for i in range(len(background)):
                    data = np.loadtxt(background[i])
                    taxis = data[:,0]
                    A=data[0:613,1]
                    A1=scipy.signal.savgol_filter(data[613-18:613,1],17,3)#replacing 405 nm laser line
                    B=data[631:694,1]
                    B1=scipy.signal.savgol_filter(data[694-6:694,1],5,3)#replacing first stray peak from spectrometer
                    C=data[700:1881,1]
                    C1=scipy.signal.savgol_filter(data[1881-6:1881,1],5,3)#replacing second stray peak from spectrometer
                    D=data[1887:2127,1]
                    D1=scipy.signal.savgol_filter(data[2127-6:2127,1],5,3)#replacing third stray peak from spectrometer
                    E=data[2133:3653,1]
                    data1 = np.hstack((A,A1,B,B1,C,C1,D,D1,E)) 
    
                bg.append(data1)            
                bg = np.asarray(bg)
                '''
                if i == 0:
                    np.save(path+'taxis.npy',taxis)
                '''
                datafx=[] #empty list for spectra
                for i in range(len(samples)):
                    data = np.loadtxt(samples[i])
                    taxis = data[:,0]
                    A=data[0:613,1]
                    A1=scipy.signal.savgol_filter(data[613-18:613,1],17,3)#replacing 405 nm laser line
                    B=data[631:694,1]
                    B1=scipy.signal.savgol_filter(data[694-6:694,1],5,3)#replacing first stray peak from spectrometer
                    C=data[700:1881,1]
                    C1=scipy.signal.savgol_filter(data[1881-6:1881,1],5,3)#replacing second stray peak from spectrometer
                    D=data[1887:2127,1]
                    D1=scipy.signal.savgol_filter(data[2127-6:2127,1],5,3)#replacing third stray peak from spectrometer
                    E=data[2133:3653,1]
                    data1 = np.hstack((A,A1,B,B1,C,C1,D,D1,E))        
                    datafx.append(data1) 
        
        #subtract background and smoothen the curve and perform nomorlization
                pl1 = (datafx-bg)
        
        #pl1= scipy.signal.savgol_filter(pl1, 51, 3) # window size 51, polynomial order 3    
                norm_1=[]
                pl_bg_subtract1=[]
                index=[]
        #plt.figure()
                for j in range(len(pl1)):
        #for j in range(20,23):
            #plt.plot(taxis,pl1[j])
            #if np.max(pl1[j][712:1928]) > 2000 and np.max(pl1[j]) < 60000 and np.min(pl1[j])> -500 :
                #pl1[j][2128:2133]
                    '''
                    #normalization based on the highest peak 
                    norm1=pl1[j][658:2681]/np.max(pl1[j][658:2681]) # spetrum interval from 410 to 800
                    '''
                    pl1[j]=pl1[j]-np.mean(pl1[j][2683:]) #offset correction from 800nm
                #normalization based on the spectrum area
                    norm1=pl1[j][658:2681]/np.abs(integrate.simps(pl1[j][658:2681], taxis[658:2681])) # spetrum interval from 410 to 800
                    norm_1.append(norm1)
                    pl_bg_subtract1.append(pl1[j][658:2681])
                    index.append(j)
        
        
                pl_bg_subtract=np.array_split(pl_bg_subtract1,N)
                saved_name=name+'_'+k+'_'+str(v)
                print(saved_name)
                np.save('E:/destop/1st measurement PL/Spectra/norm/'+saved_name+'.npy',norm_1)
                np.save('E:/destop/1st measurement PL/Spectra/non_norm/'+saved_name+'.npy',pl_bg_subtract1)
            except:
                pass
    

