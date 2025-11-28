import SimpleITK as sitk
import numpy as np
import numpy.ma as ma
import sys
import os
from os import listdir
import glob
from os.path import isfile, join



def crop_stations(modality):
    images=glob.glob(modality+'/*.mhd')
    x=[]
    y=[]
    z=[]

    
    x_min=10000000
    y_min=10000000
    z_min=10000000
    for image in images:
        size=sitk.ReadImage(image).GetSize()
        x.append(size[0])
        y.append(size[1])
        z.append(size[2])

        
        if size[0]<x_min:
            x_min=size[0]
        if size[1]<y_min:
            y_min=size[1]
        if size[2]<z_min:
            z_min=size[2]

    if all(items == x[0] for items in x):
        if all(items == y[0] for items in y):
            if all(z == z[0] for items in z):
                return x
        
    for image,x_,y_,z_ in zip(images,x,y,z):
        original_image=sitk.ReadImage(image)
        print(original_image.GetSize())
        print(np.floor((x_-x_min)/2))
        print(np.floor((y_-y_min)/2))
        print(np.floor((z_-z_min)/2))
        ana_dir=[ 1. , 0.,  0., -0., -0.,  1.,  0., -1., -0.]
        if np.all(np.around(original_image.GetDirection())==ana_dir):
            new_image=original_image[int(np.floor((x_-x_min)/2)):int(x_-np.ceil((x_-x_min)/2))
                                 ,:,
                                     int(np.floor((z_-z_min)/2)):int(z_-np.ceil((z_-z_min)/2))
                                 ]
        else:
            new_image=original_image[int(np.floor((x_-x_min)/2)):int(x_-np.ceil((x_-x_min)/2))
                                 ,int(np.floor((y_-y_min)/2)):int(y_-np.ceil((y_-y_min)/2))
                                 ,:]
        print(new_image.GetSize())
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image)
        writer.Execute(new_image)
def distance(a,b):
    som=0
    for c,d in zip(a,b):
        som+= (c-d)*(c-d)
    return np.sqrt(som)
def Select_last_station2(path):
    modalities=[]
    for image in images:
        modality=image.split('/')[-1].split('_')[0]
        if modality not in modalities:
            modalities.append(modality)
            
    for modality in modalities:

        origins=[]
        images_mod=sorted(glob.glob(path+'/'+modality+'*.mhd'))
        for image in images_mod:
            origins.append(sitk.ReadImage(image).GetOrigin())
        
def Select_last_station(path):
    images=glob.glob(path+'/*.mhd')

    modalities=[]
    for image in images:
        modality=image.split('/')[-1].split('_')[0]
        if modality not in modalities:
            modalities.append(modality)
            
    for modality in modalities:

        origins=[]
        simular_aquis=[]
        images_mod=sorted(glob.glob(path+'/'+modality+'*.mhd'))
        for image in images_mod:
            origins.append(sitk.ReadImage(image).GetOrigin())
        
        for i,origin in enumerate(origins):
            distances=[]
            for origin2 in origins:
                distances.append(distance(origin,origin2))
            distances=np.array(distances) 

            second_aqui=np.where( distances<50)
            if len(second_aqui)>0:
                second_aqui=second_aqui[-1]
                simular_aquis.append([i,second_aqui[-1]])       
        for el in simular_aquis:

            el.sort()
        output = []
        for x in simular_aquis:
            if x not in output:
                if x[0] !=x[1]:
                    output.append(x)

        for image_index in output:
            name=images_mod[image_index[0]].split('.')[0]

            os.remove(name+'.mhd')
            os.remove(name+'.raw')
        

def get_stations(folder):
    file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    files=[]
    for station in stations:
        file=glob.glob(folder+'/*'+station+'*'+'.mhd')
        if file !=[]:
            files+=file
    WB_dict=dict()        
    
    for i, file in enumerate(files):
        station=sitk.ReadImage(file)
        origin = station.GetOrigin()
        spacing = station.GetSpacing()
        size = station.GetSize()
        direction = station.GetDirection()
        WB_dict[stations[i]]={'image':station,'origin':origin,'spacing':spacing,'size':size,'direction':direction}
        

    WB_dict=calculate_overlay(WB_dict)
    return WB_dict

def crop_images(path):
    if type(path)==str:
        images=glob.glob(path+'/*.mhd')
    else:
        images=path
    for image in images:
        im=sitk.ReadImage(image)
        #new_image=im[im>0]
        numpy_image=sitk.GetArrayFromImage(im)
        numpy_image1=numpy_image>1
        xs,ys,zs = np.where(numpy_image1!=0)

        result = numpy_image[:,min(ys):max(ys)+1,:]
        new_image=sitk.GetImageFromArray(result)
        new_image.SetSpacing(im.GetSpacing())
        new_image.SetOrigin([im.GetOrigin()[0],im.GetOrigin()[1],im.GetOrigin()[2]-min(ys)*im.GetSpacing()[1]])
        new_image.SetDirection(im.GetDirection())
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image)
        writer.Execute(new_image)


def change_WBT1(path):
    images=glob.glob(path+'/*Dixon*.mhd')+glob.glob(path+'/*T1*.mhd')+glob.glob(path+'/*T2*.mhd')
    b1000=glob.glob(path+'/*b1000*.mhd')[0]
    dwi=sitk.ReadImage(b1000)
    for image in images:
        im=sitk.ReadImage(image)
        imagenp=sitk.GetArrayFromImage(im)
        imagenp=np.moveaxis(imagenp,0,1)
        imagenp=np.flip(imagenp,0)
        im2=sitk.GetImageFromArray(imagenp)
        im2.SetDirection(dwi.GetDirection())
        
        im2.SetSpacing([im.GetSpacing()[0],im.GetSpacing()[2],im.GetSpacing()[1]])
        print(im.GetOrigin())
        im2.SetOrigin([im.GetOrigin()[0],im.GetOrigin()[1],dwi.GetOrigin()[2]+im.GetOrigin()[2]])
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image)
        writer.Execute(im2)     
    
def anatomical_image(path):
    T1=glob.glob(path+'/T1*.mhd')
    if len(T1)>0:
        return path +'/T1.mhd'
    else:
        return path+'/DixonIP.mhd'

    
    return anatomic_image
                                                    
def resample_image(path):
    fixed_image=anatomical_image(path)
    T1=sitk.ReadImage(fixed_image)
    #T1.SetOrigin([0,0,0,])

    func_modalities=glob.glob(path+'/*.mhd')
    for modality in func_modalities:
        image_path=modality
        dwi=sitk.ReadImage(image_path)
        #dwi.SetOrigin([0,0,0,])
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator = sitk.sitkBSpline

        resample.SetReferenceImage(T1)
        #resample.SetOutputDirection(T1.GetDirection())
        #resample.SetOutputOrigin(T1.GetOrigin())
        #new_spacing = np.array([T1.GetSpacing()[1],T1.GetSpacing()[2],T1.GetSpacing()[0]])
        #print(new_spacing)
        #resample.SetOutputSpacing(new_spacing)

        #orig_size = np.array([dwi.GetSize()[0],dwi.GetSize()[2],dwi.GetSize()[1]], dtype=np.int)
        #orig_spacing = np.array([dwi.GetSpacing()[0],dwi.GetSpacing()[2],dwi.GetSpacing()[1]])
        #print(orig_spacing)
        #new_size = orig_size*(orig_spacing/new_spacing)
        
        #new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
        #new_size = [int(s) for s in new_size]
        #resample.SetSize(new_size)
        #dwi_np=sitk.GetArrayFromImage(dwi)
        #dwi2=sitk.GetImageFromArray(dwi_np)
        #dwi2.CopyInformation(T1)
        newimage = resample.Execute(dwi)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image_path)
        writer.Execute(newimage)
        
        #T1.CopyInformation(dwi)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(fixed_image)
        writer.Execute(T1)
