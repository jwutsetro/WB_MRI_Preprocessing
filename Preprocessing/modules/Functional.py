import SimpleITK as sitk

import numpy as np
import os
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob as glob
from multiprocessing import Pool
a = [15, 12, 8, 8, 7, 7, 7, 6, 5, 3]
b = [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]
from scipy.stats import linregress
linregress(a, b)


def ADC_numpy_list(lijst,b_values):
    mean=np.zeros(lijst[0].shape)
    for image in lijst:
        mean+=image
    mean=mean/len(lijst)
    diflist=[]
    for image in lijst:
        diflist.append(image-mean)
    mean_b=np.array(b_values).mean()
    diff_b_list=[]
    for b in b_values:
        diff_b_list.append(mean_b-b)
    denum=0
    for diff in diff_b_list:
        denum+=diff*diff
    ADC=np.zeros(lijst[0].shape)
    
    for dif_image,diff_b in zip(diflist,diff_b_list):
        ADC+=dif_image*diff_b
    ADC=np.nan_to_num(ADC)
    ADC=ADC.clip(min=0)
    return ADC/denum*1000000
def ADC_numpy(b1,b2,b3):

    mean=(b1+b2+b3)/3
    diff1=b1-mean
    diff2=b2-mean
    diff3=b3-mean
    ADC=(diff1*(350)+diff2*(250)+diff3*(-600))/(350^2+250^2+600^2)
    ADC=np.nan_to_num(ADC)
    #ADC=ADC.clip(min=0)
    ADC=ADC.clip(max=9999)*1000
    return ADC

def cDWI(ADC,b0):
    b1500=b0*np.exp(-1500*ADC)
    b2000=b0*np.exp(-2000*ADC)
    return b1500,b2000

def noise_bias_type(image):
    # cast to float32
    image = sitk.Cast(image,sitk.sitkFloat32)

    aniso = sitk.GradientAnisotropicDiffusionImageFilter()
    aniso.SetConductanceParameter(4)
    aniso.SetNumberOfIterations(10)
    aniso.SetTimeStep(0.01)
    imageFiltered = aniso.Execute(image)
    
    imageFiltered = sitk.Cast(imageFiltered, sitk.sitkFloat32)
    
    ### Create mask if needed
    threshold=5
    binarize = sitk.BinaryThresholdImageFilter()
    binarize.SetLowerThreshold(threshold)
    mask = binarize.Execute(image)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    
    bias = sitk.N4BiasFieldCorrectionImageFilter()
    
    imageBias = bias.Execute(image,mask)
    
    ### Get removed bias field
    sbtr = sitk.SubtractImageFilter()
    biasField = sbtr.Execute(image, imageBias)
    
    return imageBias, biasField

def noise_bias_type(image):
    # cast to float32
    image = sitk.Cast(image,sitk.sitkFloat32)

    aniso = sitk.GradientAnisotropicDiffusionImageFilter()
    aniso.SetConductanceParameter(4)
    aniso.SetNumberOfIterations(10)
    aniso.SetTimeStep(0.01)
    imageFiltered = aniso.Execute(image)
    
    imageFiltered = sitk.Cast(imageFiltered, sitk.sitkFloat32)
    
    ### Create mask if needed
    threshold=5
    binarize = sitk.BinaryThresholdImageFilter()
    binarize.SetLowerThreshold(threshold)
    mask = binarize.Execute(imageFiltered)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    
    bias = sitk.N4BiasFieldCorrectionImageFilter()
    
    imageBias = bias.Execute(imageFiltered,mask)

    
    return imageBias
def b_values(lijst):
    values=[]
    for name in lijst:
        value=int(name.split('/')[-1].split('_')[0][1:])
        values.append(value)
    return values    
    
    return b_values
def ADC_folder2(path_acquisition):
#     for im in glob.glob(path_acquisition+'/*head.mhd'):
#         image=sitk.ReadImage(im)
#         writer = sitk.ImageFileWriter()
#         writer.SetFileName(im[:-8]+'head.mhd')
#         writer.Execute(image)
#         os.remove(im)
#     for im in glob.glob(path_acquisition+'/*legs.mhd'):
#         image=sitk.ReadImage(im)
#         writer = sitk.ImageFileWriter()
#         writer.SetFileName(im[:-8]+'legs.mhd')
#         writer.Execute(image)
#         os.remove(im)
#     for im in glob.glob(path_acquisition+'/*torso.mhd'):
#         image=sitk.ReadImage(im)
#         writer = sitk.ImageFileWriter()
#         writer.SetFileName(im[:-9]+'torso.mhd')
#         writer.Execute(image)
#         os.remove(im)
#     for im in glob.glob(path_acquisition+'/*pelvis.mhd'):
#         image=sitk.ReadImage(im)
#         writer = sitk.ImageFileWriter()
#         writer.SetFileName(im[:-10]+'pelvis.mhd')
#         writer.Execute(image)
#         os.remove(im)
        
    stations=['1head','2torso','3pelvis','4legs','5llegs','6lllegs','7feet']
    used_stations=stations[:len(sorted(glob.glob(path_acquisition+'/b0*.mhd')))]
    
    for station_type in used_stations:
        b_stations_list=sorted(glob.glob(path_acquisition+'/b*'+station_type+'.mhd'))
        print(b_stations_list)
        b_values_list=b_values(b_stations_list)
        index=b_values_list.index(0)
        del b_stations_list[index]
        del b_values_list[index]
        np_list=[]
        for station in b_stations_list:
            np_list.append(np.log(sitk.GetArrayFromImage(sitk.ReadImage(station))))
        
        ADC=ADC_numpy_list(np_list,b_values_list)
        ADC_image=sitk.GetImageFromArray(ADC)
        ADC_image.CopyInformation(sitk.ReadImage(station))
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(path_acquisition+'/ADC_'+station_type+'.mhd')
        writer.Execute(ADC_image)
        
def ADC_folder(path_acquisition):
    ## add automation on stations and b values 
    stations=['1head','2torso','3pelvis','4legs','5llegs','6lllegs','7feet']
    used_stations=stations[:len(sorted(glob.glob(path_acquisition+'/b0*.mhd')))]
    for station in used_stations:
        b_stations_list=sorted(glob.glob(path_acquisition+'/b*'+station+'.mhd'))
        b0=sitk.GetArrayFromImage(sitk.ReadImage(b_stations_list[0]))
        template_image=sitk.ReadImage(b_stations_list[3])
        b50=np.log(sitk.GetArrayFromImage(template_image))
        b150=np.log(sitk.GetArrayFromImage(sitk.ReadImage(b_stations_list[2])))
        b1000=np.log(sitk.GetArrayFromImage(sitk.ReadImage(b_stations_list[1])))
        ADC=ADC_numpy(b50,b150,b1000)
        b1500,b2000=cDWI(ADC,b0)
        ADC_image=sitk.GetImageFromArray(ADC)
        ADC_image.CopyInformation(template_image)
        
        #b1500_image=sitk.GetImageFromArray(b1500)
        #b1500_image.CopyInformation(template_image)
        
        #b2000_image=sitk.GetImageFromArray(b2000)
        #b2000_image.CopyInformation(template_image)
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(path_acquisition+'/ADC_'+station+'.mhd')
        writer.Execute(ADC_image)
        
        #writer.SetFileName(path_acquisition+'/b1500_'+station+'.mhd')
        #writer.Execute(b1500_image)
        
        #writer.SetFileName(path_acquisition+'/b2000_'+station+'.mhd')
        #writer.Execute(b2000_image)
        
def anatomical_image(path):
    T1=glob.glob(path+'/T1*.mhd')
    if len(T1)>0:
        return path +'T1.mhd'
    else:
        return path+'DixonIP.mhd'

    
    return anatomic_image
    
def noise_bias_folder(path_acquisition):
    for imagename in glob.glob(path_acquisition+'/*.mhd'):
        image = sitk.ReadImage(imagename)
        image=noise_bias_type(image)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(imagename)
        writer.Execute(image)
#     lijst=['b1000','ADC']
#     lijst.append(anatomical_image(path_acquisition))
#     for imagename in glob.glob(path_acquisition+'/*.mhd'):
#         if imagename.split('/')[-1].split('_')[0] in lijst:
#             image = sitk.ReadImage(imagename)
#             image=noise_bias_type(image)
#             writer = sitk.ImageFileWriter()
#             writer.SetFileName(imagename)
#             writer.Execute(image)
        


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
