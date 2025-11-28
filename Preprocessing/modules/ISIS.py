import SimpleITK as sitk
import numpy as np
import numpy.ma as ma
import sys
import glob
import shutil
from os import path 
import os

stations = ['1head','2torso','3pelvis','4legs','5llegs','6lllegs', '7feet']
def rank_images(directory):
    image_list=[]
    origins=[]
    image_names=[]
    writer = sitk.ImageFileWriter()
    for image_name in glob.glob(directory+'/*.mhd'):
        image=sitk.ReadImage(image_name)
        origin= image.GetOrigin()
        origins.append(origin[2])
        image_list.append(image)
        os.remove(image_name)
        os.remove(image_name.split('.')[0]+'.raw')
    origins, image_list = (list(t) for t in zip(*sorted(zip(origins, image_list))))
    image_list.reverse()

    for image,name in zip(image_list,stations[:len(image_list)]):

        writer.SetFileName(directory+'/'+name+'.mhd')
        writer.Execute(image)
        
def sort_folders(directory):
    images=glob.glob(directory+'/*.mhd')
    writer = sitk.ImageFileWriter()
    images.sort()
    
    for image in images:
        modality=image.split('/')[-1].split('_')[0]
        if not path.exists(directory+'/'+modality):
            os.mkdir(directory+'/'+modality)
        output_dir=os.path.dirname(image)+'/'+image.split('/')[-1].split('_')[0]+'/'+image.split('/')[-1].split('_')[1]
        image_object=sitk.ReadImage(image)
        writer.SetFileName(output_dir)
        writer.Execute(image_object)
    
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
def is_functional(image):
    func_dir=[ 1.,0.,-0.,-0.,1.,0.,0., -0.,  1.]

    if np.all(np.round(image.GetDirection())==func_dir):
        return True
    else:
        return False

def get_min_max(image_array):
    ### Return min and max value of an array.
    max = np.amax(image_array)
    min = np.amin(image_array)
    
    return min, max

def Find_Reference_Station(path):
    origins=[]
    images=[]
    for image in glob.glob(path+'/*.mhd'):
        origin= sitk.ReadImage(image).GetOrigin()
        origins.append(origin[2])
        images.append(image)
    origins, images = (list(t) for t in zip(*sorted(zip(origins, images))))
    reference_index=int(np.ceil(len(images)/2))-1
    return reference_index, images

def Propagate_From_Reference(path):
    writer = sitk.ImageFileWriter()
    reference_index,images=Find_Reference_Station(path)
    uplist=images[reference_index:]
    downlist=images[:reference_index+1]
    downlist.reverse()
    for i in range(len(uplist)-1):
        reference_image=sitk.ReadImage(uplist[i])
        target_image=sitk.ReadImage(uplist[i+1])
        target_ISIS=ISIS_up(reference_image,target_image)
        target_ISIS.CopyInformation(target_image)
        writer.SetFileName(uplist[i+1])
        writer.Execute(target_ISIS)
    for i in range(len(downlist)-1):
        reference_image=sitk.ReadImage(downlist[i])
        target_image=sitk.ReadImage(downlist[i+1])
        target_ISIS=ISIS_down(reference_image,target_image)
        target_ISIS.CopyInformation(target_image)
        writer.SetFileName(downlist[i+1])
        writer.Execute(target_ISIS)

        
        
def ISIS_up(referenceSegment, targetSegment):
    overlay=get_overlay(referenceSegment,targetSegment)
    arrayRef = sitk.GetArrayFromImage(referenceSegment)
    arrayTar = sitk.GetArrayFromImage(targetSegment)
    maskBG=True
    ### Define common region of interest
    if is_functional(referenceSegment):
        ref = arrayRef[-overlay:arrayRef.shape[0],:,:]
        target = arrayTar[:overlay,:,:]
    else:
        target = arrayTar[:,-overlay-3:arrayTar.shape[1]-3,:]
        ref = arrayRef[:,3:overlay-3,:]

    ref = ref.flatten()
    target = target.flatten()
      

    if maskBG==True:
        targetM = np.ma.array(target, mask = np.where(target>5, 0, 1))
        pointT = targetM.mean()
        refM = np.ma.array(ref, mask = np.where(ref>5, 0, 1))
        pointR = refM.mean()

    else:
        pointT = np.mean(target)
        pointR = np.mean(ref)


    (IMinT, IMaxT) = get_min_max(arrayTar)
    (IMinR, IMaxR) = get_min_max(arrayRef)
    scale = (arrayTar-IMinT)/(pointT-IMinT)
    arrayTarISIS = ((pointR-IMinR)*scale)+IMinR

    return sitk.GetImageFromArray(arrayTarISIS)

def ISIS_down(referenceSegment, targetSegment):
    overlay=get_overlay(targetSegment,referenceSegment)
    arrayRef = sitk.GetArrayFromImage(referenceSegment)
    arrayTar = sitk.GetArrayFromImage(targetSegment)
    maskBG=True
    
    ### Define common region of interest
    if is_functional(referenceSegment):
        ref = arrayRef[:overlay,:,:]
        target = arrayTar[-overlay:arrayTar.shape[0],:,:]
    else:
        target = arrayTar[:,3:overlay-3,:]
        ref = arrayRef[:,-overlay-3:arrayRef.shape[1]-3,:]

    ref = ref.flatten()
    target = target.flatten()
      

    if maskBG==True:
        targetM = np.ma.array(target, mask = np.where(target>5, 0, 1))
        pointT = targetM.mean()
        refM = np.ma.array(ref, mask = np.where(ref>5, 0, 1))
        pointR = refM.mean()

    else:
        pointT = np.mean(target)
        pointR = np.mean(ref)


    (IMinT, IMaxT) = get_min_max(arrayTar)
    (IMinR, IMaxR) = get_min_max(arrayRef)
    scale = (arrayTar-IMinT)/(pointT-IMinT)
    arrayTarISIS = ((pointR-IMinR)*scale)+IMinR

    return sitk.GetImageFromArray(arrayTarISIS)

def get_overlay(referenceSegment, targetSegment):
  

    if is_functional(referenceSegment):
        refOrgin = referenceSegment.GetOrigin()
        tarOrgin = targetSegment.GetOrigin()
        refSpacing = referenceSegment.GetSpacing()
        tarSpacing = targetSegment.GetSpacing()
        refSize = referenceSegment.GetSize()
        endRef = refOrgin[2] + refSize[2]*refSpacing[2]
        diff = tarOrgin[2] - endRef
        overlay = np.abs(np.floor(diff/tarSpacing[2]))
    else:
        tarOrgin = referenceSegment.GetOrigin()
        refOrgin = targetSegment.GetOrigin()
        tarSpacing = referenceSegment.GetSpacing()
        refSpacing = targetSegment.GetSpacing()
        refSize = referenceSegment.GetSize()
        endRef = refOrgin[2] - refSize[1]*refSpacing[1]
        diff = tarOrgin[2] - endRef
        overlay = np.floor(diff/tarSpacing[1])
        
    overlay = int(overlay)
    
    return overlay
    

