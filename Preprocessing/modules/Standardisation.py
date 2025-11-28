"""
Created on Mon Mar 18 11:33:28 2019

@author: jakubc
"""

import SimpleITK as sitk
import numpy as np
from numpy import inf
from scipy import stats
import glob
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield path + '/'+f
            
def Resample_folder(folder):
    image_list=glob.glob(folder+'/*/*T1.mhd')+glob.glob(folder+'/*/*DixonIP.mhd')
    x=[]
    y=[]
    z=[]
    for image_name in image_list:
        image=sitk.ReadImage(image_name) 
        x.append(image.GetSpacing()[0])
        y.append(image.GetSpacing()[1])
        z.append(image.GetSpacing()[2])
    
    x_med=np.median(x)
    y_med=np.median(y)
    z_med=np.median(z)
    
    image_list=glob.glob(folder+'/*/*.mhd')
    for image_name in image_list:
        image=sitk.ReadImage(image_name)
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        new_spacing = [x_med,y_med,z_med]
        resample.SetOutputSpacing(new_spacing)

        orig_size = np.array(image.GetSize(), dtype=np.int)
        orig_spacing = image.GetSpacing()
        new_size = [osize*(ospac/newspac) for osize,ospac,newspac in zip(orig_size,orig_spacing,new_spacing)]
        new_size = [np.ceil(val).astype(np.int) for val in new_size] #  Image dimensions are in integers
        new_size = [int(s) for s in new_size]
        resample.SetSize(new_size)

        newimage = resample.Execute(image)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image_name)
        writer.Execute(newimage)
        
                         
           

                         
def Remove_background(folder):
    image_list=glob.glob(folder+'/*/*.mhd')
    for image_name in image_list:
        image=sitk.ReadImage(image_name)
        arr=sitk.GetArrayFromImage(image)
        new_image=(arr>1)*arr
        new_image=sitk.GetImageFromArray(new_image)
        new_image.CopyInformation(image)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image_name)
        writer.Execute(new_image)

def float16(folder):
    #nifti_list=glob.glob(folder+'/*/*.nii')
    #for image in nifti_list:
     #   os.remove(image)

    for patient in glob.glob(folder+'/*'):
        lijst=glob.glob(patient+'/*T1.nii')+glob.glob(patient+'/*b1000.nii')+glob.glob(patient+'/*ADC.nii')
        for image_name in lijst:
            image=sitk.ReadImage(image_name)
            new_image=sitk.Cast(image,sitk.sitkFloat32)
            im_arr=sitk.GetArrayFromImage(new_image).astype(np.float16)
            im_arr=im_arr.astype(np.float32)
            new16im=sitk.GetImageFromArray(im_arr)
            new16im.CopyInformation(new_image)
            writer = sitk.ImageFileWriter()
            print(image_name[:-3]+'nii')
            writer.SetFileName(image_name)
            writer.Execute(new16im)
                         
                         
def scale_histogram(hist_edges, minVal, maxVal):
  
    #scaledHistEdges = maxVal*(hist_edges - np.min(hist_edges))/np.ptp(hist_edges)
    scaledHistEdges = minVal+((hist_edges-np.min(hist_edges))/(np.max(hist_edges)-np.min(hist_edges)))*(maxVal-minVal)

    return scaledHistEdges

def calculate_avg_hist(filenames, bin_number, threshold, lowerOutlier=None, upperOutlier=None):

    histograms = []
    mins = []
    maxes = []
    
    for f in filenames:
        # Read Image
        image = sitk.ReadImage(f)
        image = sitk.GetArrayFromImage(image)
        image = image.flatten()

        imageZ = image[np.where(image > threshold)]
        
        # Z-Score Array
        # 
        #imageZ = stats.zscore(image)
        
        if upperOutlier:
        
            print('Stripping histogram from outliers')
#            print(np.percentile(imageZ, lowerOutlier))
#            print(imageZ.min())
#            print(np.percentile(imageZ, upperOutlier))
#            print(imageZ.max())

            imageZ = imageZ[np.where(imageZ>=np.percentile(imageZ, lowerOutlier))]
            imageZ = imageZ[np.where(imageZ<=np.percentile(imageZ, upperOutlier))]
        
#        print(np.percentile(imageZ, lowerOutlier))
#        print(imageZ.min())
#        print(np.percentile(imageZ, upperOutlier))
#        print(imageZ.max())
                   
        mins.append(imageZ.min())
        maxes.append(imageZ.max())
    
        histogram = np.histogram(imageZ, bins = bin_number)
        histograms.append(histogram)
        
    average_hist = np.mean(histograms, axis=0)
    avg_min = np.mean(mins, axis=0)
    avg_max = np.mean(maxes, axis=0)
    
    return average_hist[0], average_hist[1], avg_min, avg_max

def transform_intensities_zscore_mean_hist(mean_hist, std_hist, tar_filename, noBackground):
    # Load image to transform
    imageTar = sitk.ReadImage(tar_filename)
    arrayTar = sitk.GetArrayFromImage(imageTar)

    # Z-score
    arrayTar_z = np.where(arrayTar < 1, 0, (arrayTar - mean_hist)/std_hist)

    return arrayTar_z

def transform_intensities_zscore(filename, threshold):
    # load
    writer = sitk.ImageFileWriter()
    image = sitk.ReadImage(filename)
    array = sitk.GetArrayFromImage(image)

    arrayM = array[np.where(array > threshold)]
    meanArray = np.mean(arrayM)
    stdArray = np.std(arrayM)
        
    print('Mean and std of target image: {}, {}'.format(meanArray,stdArray))

    # Z-score
    array_z = np.where(array < threshold, 0, (array - meanArray)/stdArray)
    z_score_image=sitk.GetImageFromArray(array_z)
    z_score_image.CopyInformation(image)
    
    writer.SetFileName(filename)
    writer.Execute(z_score_image)
    

    
   

import SimpleITK as sitk
import numpy as np
import numpy.ma as ma

#%%

def get_landmarks_percentiles(image, numberOfLandmarks, upperOutlier, removeBG):
    ''' Function calculates positions (intensities) of all landmarks according to the number of landmarks
        specified by the user as in the implementation of Nyul method. 
        Minimum number of landmars is equal to 2: percentile 0 and percentile 99.8
        With 3 landmarks: percentile 0, percentile 50 and percentile 99.8
        with 5 landmarks: percentile 0, percentile 25, 50, 75, 99.8 
    '''
    image_temp = np.copy(image)
    
    if removeBG==True:
        #image_temp[image_temp == 0] = np.nan
        image_temp[image_temp < 5] = np.nan
        print('BG removed')
        
    upperPercentileValue = np.nanpercentile(image_temp, upperOutlier)
    lowerPercentileValue = np.nanpercentile(image_temp, 0)

#    if thresholdAtMeanIntensity==True:
#        image_temp[image_temp <= np.mean(image_temp)] = np.nan
#        print('Thresholding at mean intensity')
#    if thresholdAtIntensity!=False:
#        image_temp[image_temp <= thresholdAtIntensity] = np.nan
#        print('Thresholding at fixed intensity equal to {}'.format(thresholdAtIntensity))
    
    # this sets all pixels which were BG (so value of 0) to nan in order not to include them in percentile calculation with np.nanpercentile
    
    if numberOfLandmarks<2:
        print('Too few landmarks.')
        return

    elif numberOfLandmarks==2:
        landmarks = [lowerPercentileValue, upperPercentileValue]
        print(landmarks)
    else:
        percentiles=np.linspace(0,100,numberOfLandmarks)
        print('Calculating landmarks on the percentiles: {} and {}'.format(percentiles[:-1], upperOutlier))

        landmarks=[lowerPercentileValue]
        for percentile in percentiles[1:-1]:
            landmarks.append(np.nanpercentile(image_temp, percentile))
        landmarks.append(upperPercentileValue)
        print(landmarks)
        
    return landmarks

def standardize_image_nyul(imageStd, landmarks, standardScaleLandmarks):
    sizeX = imageStd.shape[0]
    sizeY = imageStd.shape[1]
    sizeZ = imageStd.shape[2]
    print(' ')
    print('Length of landmarks: {}'.format(len(landmarks)))
    
    imageCopy = np.copy(imageStd)
    imageStd = imageStd.flatten()
    imageStandardized = np.zeros(imageStd.shape)
    print('Number of voxels in the image to standardize: {}'.format(imageStd.size))
    print(' ')
    
    numberOfVoxels = 0
    
    imageIndices = []
    count = 0
    for i in range(0,len(landmarks)-1):
        
        if i==(len(landmarks)-2):
            imageIndices.append(np.argwhere(imageStd >= landmarks[i]))
        elif i==0:
            imageIndices.append(np.argwhere(imageStd < landmarks[i+1]))
        else:
            imageIndices.append(np.argwhere(np.logical_and(imageStd >= landmarks[i], imageStd < landmarks[i+1])))
            
        count = count + len(imageIndices[i])
        
        #print('Size of indeces array = {}'.format(len(imageIndices[i])))
        
    print('Total number of image voxels standardized= {}'.format(count))
    
    
    for i in range(0,len(landmarks)-1):
        if i==(len(landmarks)-2):  
            print('Intensity segment from {} to the maximum image intensity '.format(landmarks[i]))
        elif i==0:
            print('Intensity segment from image minimum intensity to {}'.format(landmarks[i+1]))
        else:
            print('Intensity segment from {} to {}'.format(landmarks[i],landmarks[i+1]))
        
        #print('Number of voxels in the segment: {}'.format(imageIndices.size))
              
        #imageStd_temp = standardScaleLandmarks[i] + ((imageStd[imageIndices[i]]-landmarks[i])/
        #                              (standardScaleLandmarks[i]-standardScaleLandmarks[i+1]))*(landmarks[i]-landmarks[i+1])
        
        a = (standardScaleLandmarks[i]-standardScaleLandmarks[i+1])/(landmarks[i]-landmarks[i+1])
        b = (standardScaleLandmarks[i]-(standardScaleLandmarks[i]-standardScaleLandmarks[i+1])/(landmarks[i]-landmarks[i+1])*landmarks[i])
        imageStd_temp = a*imageStd[imageIndices[i]]+b
        
        print('Linear function to standardize for segment {} equals: {:04.2f}x+{:04.2f}'.format(i+1, 
                                                                                    (standardScaleLandmarks[i]-standardScaleLandmarks[i+1])/(landmarks[i]-landmarks[i+1]),
                                                                                    standardScaleLandmarks[i]-(standardScaleLandmarks[i]-standardScaleLandmarks[i+1])/
                                                                                     (landmarks[i]-landmarks[i+1])*landmarks[i]))            
        numberOfVoxels = numberOfVoxels + imageStd_temp.size
        
        print('Filled number of voxels: {}'.format(numberOfVoxels))
        imageStandardized[imageIndices[i]] = imageStd_temp

        print('Intensity {} was mapped to intensity {}'.format(imageStd[imageIndices[i][0]], imageStandardized[imageIndices[i][0]]))
        print(' ')
        
    #np.savetxt('/Users/jakubc/Desktop/intensitiesFunctionUpper.txt',(imageStandardized[imageIndices[i]-31.92)/(imageStd[imageIndices]))
    imageStandardized = np.reshape(imageStandardized, (sizeX, sizeY, sizeZ))

    print(' ')
    print(' ')
    
    imageStandardized[imageCopy == 0] = 0

    return imageStandardized

def calculate_landmarks_from_hist(histogram, histogram_bins, numberOfLandmarks, upperOutlier):
  
    print(histogram_bins)
    percentiles=np.linspace(0,100,numberOfLandmarks)
    cum_hist = np.cumsum(histogram)/np.sum(histogram) * 100
  
    # Calculate first percentile at s1
    landmarks=[]
    print('Values of avarage histogram landmarks:')
    # Calculate the middle ones
    
    for p in percentiles[0:-1]:
        print('Percentile {} and value = {}'.format(p, histogram_bins[len(cum_hist[cum_hist <= p])]))
        landmarks.append(histogram_bins[len(cum_hist[cum_hist <= p])])
    
    # Calculate the last percentile at upperOutlier
    print('Percentile {} and value = {}'.format(upperOutlier, histogram_bins[len(cum_hist[cum_hist <= upperOutlier])]))
    landmarks.append(histogram_bins[len(cum_hist[cum_hist <= upperOutlier])])

    return landmarks




import SimpleITK as sitk
import os
import glob
import sys

numberOfBins = 240
upperOutlier = 95
numberOfLandmarks = 6



#glob.glob(folder+'/*/*T1.mhd')+glob.glob(folder+'/*/*DixonIP.mhd')
def nyul_standardisation(folder):
    modality_list =[glob.glob(folder+'/*/*rb1000.nii'),glob.glob(folder+'/*/*T1.nii')]
    for filenames in modality_list:

        avg_hist, avg_hist_x_label, avg_min,avg_hist_max = calculate_avg_hist(filenames, numberOfBins,1)
        bin_centres = 0.5*(avg_hist_x_label[1:] + avg_hist_x_label[:-1])

        for file in filenames:
            print(file)
            landmarks_avg = calculate_landmarks_from_hist(avg_hist, bin_centres, numberOfLandmarks, upperOutlier)
            print(landmarks_avg)# = [0, 164.06256, 414.0626, 882.8126, 2992.1877]

            print('Target image...')
            imageTar = sitk.ReadImage(file)
            arrayTar = sitk.GetArrayFromImage(imageTar)
            landmarksTar = get_landmarks_percentiles(arrayTar, numberOfLandmarks, upperOutlier,True)

            arrayT_avg = standardize_image_nyul(arrayTar, landmarksTar, landmarks_avg)

            writer = sitk.ImageFileWriter()
            imageT_avg = sitk.GetImageFromArray(arrayT_avg)
            imageT_avg.CopyInformation(imageTar)
            writer.SetFileName(file)
            writer.Execute(imageT_avg)
