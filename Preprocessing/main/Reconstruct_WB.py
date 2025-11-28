
import sys
import os
import timeit
#check this dependencies
#from os import listdir
#from os.path import isfile, join

import warnings
import shutil
warnings.filterwarnings("ignore")


import pandas as pd
import SimpleITK as sitk
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from shutil import copyfile
from tqdm import tqdm
import glob as glob
import os
import pandas as pd
import SimpleITK as sitk
from multiprocessing import Pool


start = timeit.timeit()
#include new images and add rotation and locint
import Preprocessing
from Preprocessing import DICOM,Functional,Noise_Bias,ISIS,Registration,MergeWB,Standardisation,Utilitiess

# put in listdir no hidden
# put in kwargs check and help function to overwrite config.ini
# put in functinoality to save or delete old data
# add functionality to select which steps to include in the preprocessing, default to all.


#from Preprocessing.Utilities import *
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield path + '/'+f

            
            
            
reset_dir_bool=False 
Dicom_bool=False
ADC_bool=False
Noise_bool=False
ISIS_bool=False
sort_folders=False
rank_images=False
DWI_registration_bool=False
Merge_bool=True
resample_bool=False
ADC2T1_bool=False
T1dixon2T1_bool=False
T2dixon2T1_bool=False
Rasample_isotropic_bool=False
Background_bool=False
Standardise_bool=False
float16=False

# old data mhd_Ophelie not yet coregistered. 
Dicom_dir='/Users/joriswuts/Documents/Data/Metastatic_bone_disease/test2'
output_dir='/Users/joriswuts/Desktop/Data_Pedram'


if reset_dir_bool:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

# open config file and save new arguments

#convert DICOM images to MHD
if Dicom_bool:
    print('converting DICOMS')
    for path in listdir_nohidden(Dicom_dir):
        DICOM.Convert_Dicom(path,output_dir)

    
    for path in listdir_nohidden(output_dir):
        Utilitiess.Select_last_station(path)
    print()
    print()
## calculate ADC,cDWI and removing noise, bias
if ADC_bool:
    print('creating ADC files')
    for patient in tqdm(listdir_nohidden(output_dir)):
        Functional.ADC_folder2(patient)
    print()
    print()

if Noise_bool:
    print('Removing noise and bias')
    for patient in tqdm(listdir_nohidden(output_dir)):
        print(patient)
        Functional.noise_bias_folder(patient)
#         for station in tqdm(listdir_nohidden(patient)):
#             print(station)
#             Functional.noise_bias_folder(station)
    print()
    print()


# do ISIS and create folders per modality

print('Performing inter station intensity standardisation')
for patient in tqdm(listdir_nohidden(output_dir)):
    if sort_folders:
        ISIS.sort_folders(patient)
    
    for modality in listdir_nohidden(patient):
        if rank_images:
            ISIS.rank_images(modality)
        if ISIS_bool:
            ISIS.Propagate_From_Reference(modality)
    print()
    print()
# do registration
## DWI to DWI
check=10
if DWI_registration_bool:
    print('Registering functional images')
    
    for patient in tqdm(listdir_nohidden(output_dir)):
        check+=1
        if check>7:
            try:
                Registration.register_ADC_folder(patient)

            except:
                print(patient)

                pass

            

    
print()

if Merge_bool:
    print('cropping Images to the same size')

    # for patient in tqdm(listdir_nohidden(output_dir)):
    #     for modality in listdir_nohidden(patient):
    #         Utilitiess.crop_stations(modality)
            
    ##merge whole body
    print('Merging whole body images')
    for patient in tqdm(listdir_nohidden(output_dir)):
        print(patient)
        if True:
            anatomical_images=glob.glob(patient+'/Dixon*/*.mhd')+glob.glob(patient+'/T2*/*.mhd')+glob.glob(patient+'/T1*/*.mhd')
            #Utilitiess.crop_images(anatomical_images)
            for modality in listdir_nohidden(patient):
                try:
                    MergeWB.constructWholeBody(modality,False)
                    shutil.rmtree(modality)
                except:
                    pass
        else:
            pass

    print()
    print()

if resample_bool:
    print('Resampling images to target spacing')
    for patient in tqdm(listdir_nohidden(output_dir)):
        Utilitiess.resample_image(patient)
    print()
    print()
check=True
if ADC2T1_bool:
    print('Registering functional images to anatomical whole body images')
    for patient in tqdm(listdir_nohidden(output_dir)):
        if patient =='/Users/joriswuts/Desktop/dataset_03/S024-4':
            check=True
        if check:
            print(patient)
            Registration.register_DWI2T1_folder(patient)
        if patient == '/Users/joriswuts/Desktop/dataset_03/JSW-022a':
            check= False
            

    print()
    print()

if T1dixon2T1_bool:
    print('Registering T1dixon to T1 images')
    for patient in tqdm(listdir_nohidden(output_dir)):
        Registration.register_Dixon2T1_folder(patient)

    print()
    print()

if T2dixon2T1_bool:
    print('Registering T2dixon to T1 images')
    for patient in tqdm(listdir_nohidden(output_dir)):
        Registration.register_T22T1_folder(patient)

## 
##new stuff here, resapling itostropically, removing background, Nyul standardize, z-score and float16. 

if Rasample_isotropic_bool:
    print('Resampling to new spacing')
    Standardisation.Resample_folder(output_dir)
        
if Background_bool:
    print('Removing background')
    Standardisation.Remove_background(output_dir)
        
if Standardise_bool:
    print('standardizing images')
#     for patient in glob.glob(output_dir+'/*'):
#         lijst=glob.glob(patient+'/*T1.nii')+glob.glob(patient+'/*b1000.nii')+glob.glob(patient+'/*ADC.nii')
#         for image in lijst:
#             print(image)
#             Standardisation.transform_intensities_zscore(image,1)
            
 #   Standardisation.nyul_standardisation(output_dir)
        
if float16:
    print('Downsampling to float16')
    Standardisation.float16(output_dir)


