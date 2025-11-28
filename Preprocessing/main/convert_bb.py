import shutil
import os
import glob
import SimpleITK as sitk
import numpy as np 


def resample_image(image,interpolator,new_size,ref_img):
    if ref_img==None:
        resample = sitk.ResampleImageFilter()
        if interpolator=='Neighbour':
            inte=sitk.sitkNearestNeighbor
        if interpolator=='Spline':
            inte=sitk.sitkBSpline
        resample.SetInterpolator(inte)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        new_spacing = [1, 1, 1]
        resample.SetOutputSpacing(new_spacing)

        orig_size = np.array(image.GetSize(), dtype=np.int)
        orig_spacing = image.GetSpacing()
        if new_size==0:
            new_size = orig_size*(orig_spacing)
            new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
            new_size = [int(s) for s in new_size]
        resample.SetSize(new_size)

        newimage = resample.Execute(image)
        #newimage=sitk.Cast(newimage,sitk.sitkFloat32)
        if interpolator=='Neighbour':
            newimage=sitk.Cast(newimage,sitk.sitkUInt8)
        return newimage ,new_size
    
    else:
        resample = sitk.ResampleImageFilter()
        if interpolator=='Neighbour':
            inte=sitk.sitkNearestNeighbor
        if interpolator=='Spline':
            inte=sitk.sitkBSpline
        resample.SetInterpolator(inte)
        resample.SetReferenceImage(ref_img)
        new_spacing = [1, 1, 1]
        resample.SetOutputSpacing(new_spacing)

        orig_size = np.array(image.GetSize(), dtype=np.int)
        orig_spacing = image.GetSpacing()
        if new_size==0:
            new_size = orig_size*(orig_spacing)
            new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
            new_size = [int(s) for s in new_size]
        resample.SetSize(new_size)

        newimage = resample.Execute(image)
        #newimage=sitk.Cast(newimage,sitk.sitkFloat32)
        if interpolator=='Neighbour':
            newimage=sitk.Cast(newimage,sitk.sitkUInt8)
        return newimage ,new_size


def create_bb(mask):

    
    closing = sitk.BinaryDilateImageFilter()
    closing.SetKernelRadius(5)
    mask = closing.Execute(mask)
    filtertje= sitk.LabelShapeStatisticsImageFilter()
    filtertje.Execute(mask)
    bb = np.array(filtertje.GetBoundingBox(1))
    return mask,bb

def crop_and_save_dir(patient,mask,bb):
    writer = sitk.ImageFileWriter()
    
    for image_name in glob.glob(patient+'/*.nii.gz'):
        image=sitk.ReadImage(image_name)
        image_np=sitk.GetArrayFromImage(image).astype(np.float16).astype(np.float32)
        
        #mask_np=sitk.GetArrayFromImage(mask)
        #new_image=image_np*mask_np
        new_image=sitk.GetImageFromArray(image_np)
        new_image.CopyInformation(image)
        new_image=new_image[bb[0]:(bb[0]+bb[3]),bb[1]:(bb[1]+bb[4]),bb[2]:(bb[2]+bb[5])]
        writer.SetFileName(image_name)
        writer.Execute(new_image)
        
        
input_dir='/Users/joriswuts/Desktop/dataset_01'
writer = sitk.ImageFileWriter()
for image in glob.glob(input_dir+'/*/*.mhd'):
    im=sitk.ReadImage(image)
    writer.SetFileName(image[:-4]+'.nii.gz')
    writer.Execute(im)
    os.remove(image)
    os.remove(image[:-4]+'.raw')
    
# reading a folder and
check=True
for patient in glob.glob(input_dir+'/*'):

    if check:
        T1=patient+'/T1.nii.gz'
        B1000=patient+'/b1000.nii.gz'
        ADC=patient+'/ADC.nii.gz'
        skel=patient+'/Skeleton_Mask_r.nii.gz'
        GT=patient+'/Whole_Body_GT_Man_Clean_r_smoothed.nii.gz'

        T1_im=sitk.ReadImage(T1)
        B1000_im=sitk.ReadImage(B1000)
        ADC_im=sitk.ReadImage(ADC)
        skel_im=sitk.ReadImage(skel)
        GT_im=sitk.ReadImage(GT)


    # cropping images to the same size and converting them to nii and removing nii.gz 
        T1_im,new_size=resample_image(T1_im,'Spline',0,None)
        B1000_im,new_size=resample_image(B1000_im,'Spline',new_size,T1_im)
        ADC_im,new_size=resample_image(ADC_im,'Spline',new_size,T1_im)
        GT_im,new_size=resample_image(GT_im,'Neighbour',new_size,T1_im)
        skel_im,new_size=resample_image(skel_im,'Neighbour',new_size,T1_im)
        skel_im=sitk.Cast(skel_im,sitk.sitkUInt8)
        GT_im=sitk.Cast(GT_im,sitk.sitkUInt8)
        writer = sitk.ImageFileWriter()

        writer.SetFileName(T1)
        writer.Execute(T1_im)

        writer.SetFileName(B1000)
        writer.Execute(B1000_im)

        writer.SetFileName(ADC)
        writer.Execute(ADC_im)

        writer.SetFileName(skel)
        writer.Execute(skel_im)

        writer.SetFileName(GT)
        writer.Execute(GT_im)


#         os.remove(T1)
#         os.remove(B1000)
#         os.remove(ADC)
#         os.remove(skel)
#         os.remove(GT)

        mask,bb=create_bb(skel_im)
        crop_and_save_dir(patient,mask,bb)
    if patient =='/home/jakub/U-Net/U-Net/data/dataset_02/Untitled.ipynb':
        check=True