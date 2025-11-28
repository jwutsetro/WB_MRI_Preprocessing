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


def resample(image_1,image_2):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator = sitk.sitkNearestNeighbor

    resample.SetReferenceImage(image_1)
    
    newimage = resample.Execute(image_2)
    
    return newimage

Fixed_Image=sitk.ReadImage('/Users/joriswuts/Documents/Data/Multi_label_skeleton_segmentation/Source_T1/PICRIB_VUB_3/3B/Whole_Body_T1_gz.nii.gz')
Fixed_Mask=sitk.ReadImage('/Users/joriswuts/Documents/Data/Multi_label_skeleton_segmentation/Source_T1/PICRIB_VUB_3/3B/full_skeleton.nii.gz')
Moving_image=sitk.ReadImage('/Users/joriswuts/Documents/Data/Multi_label_skeleton_segmentation/PICRIB_VUB_3/T1.mhd')
Moving_mask=Moving_image*0
Moving_mask=sitk.Cast(Moving_mask,sitk.sitkUInt8)
Fixed_Mask.CopyInformation(Fixed_Image)

print('doing initial alignment')
intermediate_image,parammap_1=Registration.Register_initial(Fixed_Image,Moving_image)
#writer = sitk.ImageFileWriter()
#writer.SetFileName('/Users/joriswuts/Desktop/Source_T1/PICRIB_VUB_8/8A/intermediate.nii.gz')
#writer.Execute(intermediate_image)
#intermediate_image=sitk.ReadImage('/Users/joriswuts/Desktop/Source_T1/PICRIB_VUB_8/8A/intermediate.nii.gz')
#sitk.WriteParameterFile(parammap_1,'initial.txt')
#parammap_1=sitk.ReadParameterFile('initial.txt')
skel=[24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,25,26,27,28,29,
     30,31,32,33,35,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]
skel=[31,32,33,35,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]
#skel=[49,50,51,52,53,54,55,56,57,58,59,60,61]
for i in skel:
    print('aligning bone: '+str(i+1))
    Fixed_Mask_i=Fixed_Mask==(i+1)
    Fixed_Mask_i_dilated=Fixed_Mask_i
    filtertje=sitk.BinaryDilateImageFilter()
    filtertje.SetKernelRadius(15)
    Fixed_Mask_i_dilated=filtertje.Execute(Fixed_Mask_i)
    moving_im=intermediate_image
    if i<24:
        moving_im=result_image
    if i in [36,35]:
        moving_im=result_image
    result_image=Registration.resgiter_bone(Fixed_Image,Fixed_Mask_i_dilated,moving_im,i+1)
    # writer = sitk.ImageFileWriter()
    # writer.SetFileName('/Users/joriswuts/Desktop/Source_T1/PICRIB_VUB_8/8A/resultimage.nii.gz')
    # writer.Execute(resultImage)
    #resultImage=sitk.ReadImage('/Users/joriswuts/Desktop/Source_T1/PICRIB_VUB_8/8A/resultimage.nii.gz')
    inverse_transform = Registration.inverse_transform(Moving_image,result_image)
    sitk.WriteParameterFile(inverse_transform,'inverse_bone'+str(i+1)+'.txt')
    
    result_mask=Registration.transform(Fixed_Mask_i,inverse_transform)
    result_mask=result_mask>0.5
    result_mask.CopyInformation(Moving_mask)

    Moving_mask+=sitk.Cast(result_mask*(i+1),Moving_mask.GetPixelID())

    writer = sitk.ImageFileWriter()
    writer.SetFileName('/Users/joriswuts/Documents/Data/Multi_label_skeleton_segmentation/PICRIB_VUB_3/mask32.mhd')
    print('writing mask to hard drive')
    print()
    print()
    print()
    writer.Execute(Moving_mask)