# image common part
import SimpleITK as sitk 
import numpy as np
import glob
import os
from os import path
import shutil



def paste_white_mask(image,begin,end):
    # make a black volume

    slice_img = sitk.Image([image.GetSize()[0],image.GetSize()[1]],sitk.sitkUInt8)
    slice_img = slice_img + 1

# convert the 2d slice into a 3d volume
    slice_vol = sitk.JoinSeries(slice_img)
# paste the 3d white slice into the black volume
    for i in range(begin,end+1):
        image = sitk.Paste(image, slice_vol, slice_vol.GetSize(), destinationIndex=[0,0,i])
    return image

def anatomical_image(path):
    T1=glob.glob(path+'/T1*.mhd')
    if len(T1)>0:
        return path +'/T1.mhd'
    else:
        return path+'/DixonIP.mhd'

    
    return anatomic_image

def imageCommonPart(image1,image2):

    template=sitk.Image(image1.GetSize(),sitk.sitkUInt8)
    template.CopyInformation(image1)
    origin1=image1.GetOrigin()
    origin2=image2.GetOrigin()
    
    spacing1=image1.GetSpacing()
    spacing2=image2.GetSpacing()
    
    size1=image1.GetSize()
    size2=image2.GetSize()
    
    origin1=image1.GetOrigin()
    origin2=image2.GetOrigin()

    if origin1[2]>origin2[2]:
        
        overlay = int(np.ceil((size1[2]*spacing1[2]-(origin1[2]-origin2[2]))/spacing1[2]))
        template=paste_white_mask(template,0,overlay)

        
        
    else:
        overlay = int(np.ceil((size2[2]*spacing2[2]-(origin2[2]-origin1[2]))/spacing2[2]))
        template=paste_white_mask(template,template.GetSize()[2]-overlay,template.GetSize()[2])
        

    return template


def Find_Reference_Station(path):
    origins=[]
    images=[]
    print(path)
    for image in glob.glob(path+'/*.mhd'):
        origin= sitk.ReadImage(image).GetOrigin()
        origins.append(origin[2])
        images.append(image)
    origins, images = (list(t) for t in zip(*sorted(zip(origins, images))))
    reference_index=int(np.ceil(len(images)/2))-1
    return reference_index, images


        

def register_ADC_folder(path_acquisition):
    print(path_acquisition)
    stations = ['1head','2torso','3pelvis','4legs','5llegs','6lllegs', '7feet']
    adc=sorted(glob.glob(path_acquisition+'/ADC/*.mhd'))


    paramfiles=path_acquisition+'/paramfiles'
    os.mkdir(paramfiles)
    elastixImageFilter=sitk.ElastixImageFilter()
    writer = sitk.ImageFileWriter()
    reference_index, images=Find_Reference_Station(path_acquisition+'/ADC')
    uplist=images[reference_index:]
    downlist=images[:reference_index+1]
    downlist.reverse()

    # going up 
    for i in range(len(uplist)-1):
        print(uplist[i])
        print(uplist[i+1])
        fixedImage=sitk.ReadImage(uplist[i])
        movingImage=sitk.ReadImage(uplist[i+1])

        fixedImageMask=imageCommonPart(fixedImage,movingImage)
        elastixImageFilter.SetFixedImage(fixedImage)
        #elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.SetMovingImage(movingImage)
        elastixImageFilter.SetFixedMask(fixedImageMask)

        elastixImageFilter.SetParameterMap(sitk.ReadParameterFile(
                        '/Users/joriswuts/Code/Preprocessing_latest/01_PREPROCESSING/Euler_S2S_MSD.txt'))
        elastixImageFilter.Execute()

        parammap=elastixImageFilter.GetTransformParameterMap()[0]
        parammap['Origin']=[str(movingImage.GetOrigin()[0]),str(movingImage.GetOrigin()[1]),str(movingImage.GetOrigin()[2])]
        # split this based on station namen first letter
        sitk.WriteParameterFile(parammap,paramfiles+'/'+uplist[i].split('/')[-1][0]+'to'+uplist[i+1].split('/')[-1][0]+'.txt')
        if i>1:
            parammap['InitialTransformParametersFileName']=paramfiles+'/'+uplist[i].split('/')[-1][0]+'to'+uplist[i-1].split('/')[-1][0]+'.txt'
        transformixImageFilter=sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(parammap)

        #func_modalities=['ADC','b0','b50','b150','b1000']
        func_modalities=['ADC','b1000']
        for modality in func_modalities:

            if not path.exists(path_acquisition+'/r'+modality):
                reference_image_path=path_acquisition+'/'+modality+'/'+uplist[0].split('/')[-1]
                ref_im=sitk.ReadImage(reference_image_path)
                os.mkdir(path_acquisition+'/r'+modality)
                writer.SetFileName(path_acquisition+'/r'+modality+'/'+uplist[0].split('/')[-1])
                writer.Execute(ref_im)
            image_path=path_acquisition+'/'+modality+'/'+uplist[i+1].split('/')[-1]
            movingImage=sitk.ReadImage(image_path)
            transformixImageFilter.SetMovingImage(movingImage)
            transformixImageFilter.LogToConsoleOff()
            transformixImageFilter.Execute()
            resultImage=transformixImageFilter.GetResultImage()
            #resultImage.clip(0,max)
            writer.SetFileName(path_acquisition+'/r'+modality+'/'+uplist[i+1].split('/')[-1])
            writer.Execute(resultImage)

    #going down 
    for i in range(len(downlist)-1):
        print(downlist[i])
        print(downlist[i+1])
        fixedImage=sitk.ReadImage(downlist[i])
        movingImage=sitk.ReadImage(downlist[i+1])

        fixedImageMask=imageCommonPart(fixedImage,movingImage)
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(movingImage)
        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.SetFixedMask(fixedImageMask)

        elastixImageFilter.SetParameterMap(sitk.ReadParameterFile(
                        '/Users/joriswuts/Code/Preprocessing_latest/01_PREPROCESSING/Euler_S2S_MSD.txt'))
        elastixImageFilter.Execute()

        parammap=elastixImageFilter.GetTransformParameterMap()[0]
        parammap['Origin']=[str(movingImage.GetOrigin()[0]),str(movingImage.GetOrigin()[1]),str(movingImage.GetOrigin()[2])]
        # split this based on station namen first letter
        sitk.WriteParameterFile(parammap,paramfiles+'/'+downlist[i].split('/')[-1][0]+'to'+downlist[i+1].split('/')[-1][0]+'.txt')
        if i>1:
            parammap['InitialTransformParametersFileName']=paramfiles+'/'+downlist[i].split('/')[-1][0]+'to'+downlist[i-1].split('/')[-1][0]+'.txt'
        transformixImageFilter=sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(parammap)

        #func_modalities=['ADC','b0','b50','b150','b1000']
        func_modalities=['ADC','b1000']
        for modality in func_modalities:
            image_path=path_acquisition+'/'+modality+'/'+downlist[i+1].split('/')[-1]
            movingImage=sitk.ReadImage(image_path)
            transformixImageFilter.SetMovingImage(movingImage)
            transformixImageFilter.LogToConsoleOff()
            transformixImageFilter.Execute()
            resultImage=transformixImageFilter.GetResultImage()
            #resultImage.clip(0,max)
            writer.SetFileName(path_acquisition+'/r'+modality+'/'+downlist[i+1].split('/')[-1])
            writer.Execute(resultImage)


    # delete and rename the folders
    for modality in func_modalities:
        shutil.rmtree(path_acquisition+'/'+modality)
    shutil.rmtree(path_acquisition+'/paramfiles')


    
def register_DWI2T1_folder(patient):
    fixed_image=anatomical_image(patient)
    moving_image=patient+'/rb1000.mhd'
    parammap=register_image2image(moving_image,fixed_image)
    writer = sitk.ImageFileWriter()
    transformixImageFilter=sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(parammap)
    
    # all images including the original one 
    func_modalities=glob.glob(patient+'/*b*.mhd')+glob.glob(patient+'/*ADC*.mhd')
    for modality in func_modalities:
        
        image_path=modality
        movingImage=sitk.ReadImage(image_path)
        transformixImageFilter.SetMovingImage(movingImage)
        transformixImageFilter.Execute()
        resultImage=transformixImageFilter.GetResultImage()
        #resultImage.clip(0,max)
        writer.SetFileName(image_path)
        writer.Execute(resultImage)
            
def register_image2image(movingImage,fixedImage):
    
    fixedImage=sitk.ReadImage(fixedImage)
    movingImage=sitk.ReadImage(movingImage)
    elastixImageFilter=sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    writer = sitk.ImageFileWriter()
    #fixedImageMask=fixedImage>-100000
    fixedImage.SetDirection(movingImage.GetDirection())
    #fixedImageMask.SetSpacing(fixedImage.GetSpacing())
    parameterMapVector = sitk.VectorOfParameterMap()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    #elastixImageFilter.SetFixedMask(fixedImageMask)
    parameterMapVector.append(sitk.ReadParameterFile(
                        '/Users/joriswuts/Code/Preprocessing_latest/01_PREPROCESSING/S2A_Pair_Euler_WB.txt'))
    parameterMapVector.append(sitk.ReadParameterFile(
                        '/Users/joriswuts/Code/Preprocessing_latest/01_PREPROCESSING/S2A_Pair_BSpline_WB.txt'))    
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()

    parammap=elastixImageFilter.GetTransformParameterMap()[0]
    return parammap 

def Register_initial(Fixed_Image,Moving_image):
    elastixImageFilter=sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    parammap_1 = sitk.ReadParameterFile('/Users/joriswuts/Code/Preprocessing_latest/01_PREPROCESSING/Translation_Bone.txt')
    elastixImageFilter.SetFixedImage(Fixed_Image)
    elastixImageFilter.SetMovingImage(Moving_image)
    elastixImageFilter.SetParameterMap(parammap_1)
    elastixImageFilter.Execute()
    parammap_1=elastixImageFilter.GetTransformParameterMap()[0]
   
    
    #apply to moving image 
    transformixImageFilter=sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(parammap_1)
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetMovingImage(Moving_image)
    transformixImageFilter.Execute()
    resultImage=transformixImageFilter.GetResultImage()
    sitk.WriteParameterFile(parammap_1,'initial.txt')
    return resultImage,parammap_1

def register_T12T1(Fixed_Image,Fixed_Mask,Moving_image,i):
    
    #do second registration
    elastixImageFilter=sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    #if i<25:
     #   elastixImageFilter.SetInitialTransformParameterFileName('bone_'+str(i+1)+'.txt')
    parammap_2 = sitk.ReadParameterFile('/Users/joriswuts/Code/Preprocessing_latest/01_PREPROCESSING/Rigid_Bone.txt')
    elastixImageFilter.SetFixedImage(Fixed_Image)
    elastixImageFilter.SetMovingImage(Moving_image)
    elastixImageFilter.SetFixedMask(Fixed_Mask)
    elastixImageFilter.SetParameterMap(parammap_2)
    elastixImageFilter.Execute()

    parammap_2=elastixImageFilter.GetTransformParameterMap()[0]

    return parammap_2


def change_param_files(parammap):
    parammap['ResampleInterpolator']=['FinalNearestNeighborInterpolator']

    
    parammap['ResultImagePixelType']=['unsigned char']
    
    parammap['MovingInternalImagePixelType'] = ['unsigned char']


    parammap['TransformParameters'] = tuple(str(-float(x)) for x in parammap['TransformParameters'])

    
    return parammap

def transform(Moving_image,parammap):
    transformixImageFilter=sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(parammap)
    transformixImageFilter.SetMovingImage(Moving_image)
    
    transformixImageFilter.Execute()
    resultImage=transformixImageFilter.GetResultImage()

    

    return resultImage

def inverse_transform(original_image,transformed_image):
    elastixImageFilter=sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    parammap_2 = sitk.ReadParameterFile('/Users/joriswuts/Code/Preprocessing_latest/01_PREPROCESSING/Rigid_Bone2.txt')
    elastixImageFilter.SetFixedImage(original_image)
    elastixImageFilter.SetMovingImage(transformed_image)
    elastixImageFilter.SetParameterMap(parammap_2)
    elastixImageFilter.Execute()

    parammap_2=elastixImageFilter.GetTransformParameterMap()[0]
    sitk.WriteParameterFile(parammap_2,'bone_1_inverse_compute_aproach.txt')
    
    return parammap_2

def combine_param_files(parammap_1,parammap_2_files):
    parammap_1['InitialTransformParametersFileName']=[parammap_2_files]
    sitk.WriteParameterFile(parammap_1,'combined.txt')
    return parammap_1
def resgiter_bone(Fixed_Image,Fixed_Mask_i_dilated,Moving_Image,i):
    
    # euler register source to destination
    parammap_2=register_T12T1(Fixed_Image,Fixed_Mask_i_dilated,Moving_Image,i)
    sitk.WriteParameterFile(parammap_2,'bone_'+str(i)+'.txt')
    #parammap_2=sitk.ReadParameterFile('bone_'+str(i)+'.txt')
    
    result_image=transform(Moving_Image,parammap_2)
    
    return result_image


def register_Dixon2T1_folder(patient):
    fixed_image=anatomical_image(patient)
    moving_image=patient+'/DixonIP.mhd'
    if fixed_image !=moving_image:
        if os.path.isfile(moving_image):
            parammap=register_image2image(moving_image,fixed_image)
            writer = sitk.ImageFileWriter()
            transformixImageFilter=sitk.TransformixImageFilter()
            transformixImageFilter.SetTransformParameterMap(parammap)
            transformixImageFilter.LogToConsoleOff()
            # all images including the original one
            func_modalities=glob.glob(patient+'/*Dixon*.mhd')
            for modality in func_modalities:

                image_path=modality
                movingImage=sitk.ReadImage(image_path)
                transformixImageFilter.SetMovingImage(movingImage)
                transformixImageFilter.Execute()
                resultImage=transformixImageFilter.GetResultImage()
                #resultImage.clip(0,max)
                writer.SetFileName(image_path)
                writer.Execute(resultImage)

def register_T22T1_folder(patient):
    fixed_image=anatomical_image(patient)
    moving_image=patient+'/T2dixonIP.mhd'
    if os.path.isfile(moving_image):
        parammap=register_image2image(moving_image,fixed_image)
        writer = sitk.ImageFileWriter()
        transformixImageFilter=sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(parammap)
        transformixImageFilter.LogToConsoleOff()
        # all images including the original one 
        func_modalities=glob.glob(patient+'/*T2*.mhd')
        for modality in func_modalities:

            image_path=modality
            movingImage=sitk.ReadImage(image_path)
            transformixImageFilter.SetMovingImage(movingImage)
            transformixImageFilter.Execute()
            resultImage=transformixImageFilter.GetResultImage()
            #resultImage.clip(0,max)
            writer.SetFileName(image_path)
            writer.Execute(resultImage)
            
