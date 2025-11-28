#function to scan dicoms and extract mhd
from shutil import copyfile
from tqdm import tqdm
import glob as glob
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import shutil
from multiprocessing import Pool

print( 'I changed the dicom.py  file from vscode')

def Convert_Dicom(dicom_dir,output_dir):
    # check if dicom_dir is folder of patients or a folder of one patient
    patient_folders = [f.path for f in os.scandir(dicom_dir) if f.is_dir()]
    if patient_folders==[]:
        patient_folders=[dicom_dir]
    for patient in patient_folders:
        print(patient)
        output_dir=output_dir+'/'+os.path.basename(os.path.normpath(patient))
        os.mkdir(output_dir)
        fileframe=Dicom_DataFrame(patient)
        print()
        print('extracting mhd images for '+patient)
        create_DIXON_mhd(fileframe,output_dir)
        create_DWI_mhd(fileframe,output_dir)
        create_T1_mhd(fileframe,output_dir)
        create_T2Dixon_mhd(fileframe,output_dir)
        print()
        print()
def filter_file(data_dir,modalities,file):
    Filename=data_dir+'/'+file
    reader = sitk.ImageFileReader()
    reader.SetFileName(Filename)
    reader.LoadPrivateTagsOn()
    try:
        reader.ReadImageInformation()
        Modality=reader.GetMetaData('0008|103e')
        if Modality in modalities:
            return True
        else:
            return False
    except:
        return False
    


def Dicom_DataFrame(data_directory):
    print('sorting DICOMS for patient'+str(data_directory))
    fileframe=pd.DataFrame(columns=['Filename','Modality','serie_ID','instance_ID','loc','origin','type','b_value'])
    modalities=['DWIBS b0-50-150-1000 4MIN ','mDIXON 1.5abdo','3D_T1_TSE ts','CORO T2 DIX4mm']
    i=0
    for file in tqdm(os.listdir(data_directory)):
        Filename=data_directory+'/'+file
        reader = sitk.ImageFileReader()
        reader.SetFileName(Filename)
        reader.LoadPrivateTagsOn()
        try:
            reader.ReadImageInformation()
            Modality=reader.GetMetaData('0008|103e')
            if Modality[0:14]=='CORO T2 DIX4mm':
                Modality='CORO T2 DIX4mm'
                # here something weird happens, 4 for ophelie, 5 for vasiliki
            if 'DWIBS' in Modality[0:5]:
                Modality='DWIBS b0-50-150-1000 4MIN '
            if Modality in modalities:
                #put it in the dataframe
                serie_ID=reader.GetMetaData('0020|0011')
                instance_ID=reader.GetMetaData('0020|0013')
                loc=reader.GetMetaData('0020|0013')
                origin=reader.GetMetaData('0020|0032')
                types='no_type'
                b_value='no_dif'
                if Modality=='CORO T2 DIX4mm':
                    types=reader.GetMetaData('2005|1011')
                if Modality=='DWIBS b0-50-150-1000 4MIN ':
                    b_value=reader.GetMetaData('0018|9087')
                if Modality=='mDIXON 1.5abdo':
                    types=reader.GetMetaData('2005|1011')
                dicom={'Filename':Filename,'Modality':Modality,'serie_ID':serie_ID,'instance_ID':instance_ID,'loc':loc,'origin':origin,'type':types,'b_value':b_value}
                fileframe.loc[i]=pd.Series(dicom)
                i+=1

        except:
            pass
    return fileframe

## reading 3D T1 TSE 
def create_output_dir_T1(output_dir,modality,serie_ID,serie_IDs):
    #create upper files somewhere
    output_dir=output_dir+'/'+modality
    serie_IDs.sort()
    stations=['1head','2torso','3pelvis','4legs','5llegs','6lllegs','7feet']
    stations=stations[:len(serie_IDs)]
    n=serie_IDs.index(serie_ID)
    output_dir+='_'+stations[n]
    
    return output_dir
    
    
def create_T1_mhd(fileframe,output_dir):
    fileframe_T1=fileframe[fileframe['Modality']=='3D_T1_TSE ts']
    if fileframe_T1.shape[0]>0:
        path = os.path.dirname(fileframe_T1.Filename.unique()[0])
        os.mkdir(path+'/temp')
        for file in fileframe_T1.Filename.unique():
            shutil.copy(file,path+'/temp')
        reader = sitk.ImageSeriesReader()
        series_ids=reader.GetGDCMSeriesIDs(path+'/temp')
        i=0
        for ID in series_ids:

            dicom_names = reader.GetGDCMSeriesFileNames(path+'/temp',ID)
            reader.SetFileNames(dicom_names)

            image = reader.Execute()

            output_dir_S = create_output_dir_T1(output_dir,'T1',int(i),list(range(len(series_ids))))
            #original_image = sitk.ReadImage(files)

            # Write the image.
            output_file_name_3D = output_dir_S + '.mhd'
            sitk.WriteImage(image, output_file_name_3D)
            i+=1
        shutil.rmtree(path+'/temp')
            
#extracting DIXON images
def create_output_dir_DIXON(output_dir,modality,types,serie_ID,serie_IDs):
    #create upper files somewhere
    output_dir=output_dir+'/'+modality+types
    serie_IDs.sort()
    stations=['1head','2torso','3pelvis','4legs','5llegs','6lllegs','7feet']
    stations=stations[:len(serie_IDs)]
    stations.reverse()
    n=serie_IDs.index(serie_ID)
    output_dir+='_'+stations[n]
    
    return output_dir
    
    
def create_DIXON_mhd(fileframe,output_dir):
    fileframe_dixon=fileframe[fileframe['Modality']=='mDIXON 1.5abdo']
    possible_types=['OP','IP']
    serie_IDs = fileframe_dixon.serie_ID.unique()
    serie_IDs_np=list()
    for b in list(fileframe_dixon.serie_ID.unique()):
        serie_IDs_np.append(int(b))
        
    for types in possible_types:
        fileframe_dixon_ET=fileframe_dixon[fileframe_dixon['type']==types]
        for serie_ID in serie_IDs:
            fileframe_dixon_serie=fileframe_dixon_ET[fileframe_dixon_ET['serie_ID']==serie_ID]
            fileframe_dixon_serie.sort_values(by=['loc'],inplace=True)
            files=np.array(fileframe_dixon_serie.Filename.values)
            output_dir_S = create_output_dir_DIXON(output_dir,'Dixon',types,int(serie_ID),serie_IDs_np)
            original_image = sitk.ReadImage(files)

            # Write the image.
            output_file_name_3D = output_dir_S + '.mhd'
            sitk.WriteImage(original_image, output_file_name_3D)
    
#extracting DWI images
def create_output_dir_DWI(output_dir,name,serie_ID,serie_IDs):
    #create upper files somewhere
    b_values=['b0','b50','b150','b1000']
    output_dir=output_dir+'/'+name
    serie_IDs.sort()
    stations=['1head','2torso','3pelvis','4legs','5llegs','6lllegs','7feet']
    stations=stations[:len(serie_IDs)]
    n=serie_IDs.index(serie_ID)
    output_dir+='_'+stations[n]
    
    return output_dir

def create_DWI_mhd(fileframe,output_dir):
    fileframe_DWI=fileframe[fileframe['Modality']=='DWIBS b0-50-150-1000 4MIN ']
    for b_value in fileframe_DWI.b_value.unique():
        fileframe_DWI_b=fileframe_DWI[fileframe_DWI['b_value']==b_value]

        path = os.path.dirname(fileframe_DWI_b.Filename.unique()[0])
        os.mkdir(path+'/temp')
        for file in fileframe_DWI_b.Filename.unique():
            shutil.copy(file,path+'/temp')
        reader = sitk.ImageSeriesReader()
        series_ids=reader.GetGDCMSeriesIDs(path+'/temp')
        i=0
        for ID in series_ids:

            dicom_names = reader.GetGDCMSeriesFileNames(path+'/temp',ID)
            reader.SetFileNames(dicom_names)

            image = reader.Execute()

            output_dir_S = create_output_dir_DWI(output_dir,'b'+str(b_value),int(i),list(range(len(series_ids))))
            #original_image = sitk.ReadImage(files)

            # Write the image.
            output_file_name_3D = output_dir_S + '.mhd'
            sitk.WriteImage(image, output_file_name_3D)
            i+=1
        shutil.rmtree(path+'/temp')

            
## reading T2 DIXON images 
def create_output_dir_T2dix(output_dir,modality,serie_ID,serie_IDs):
    #create upper files somewhere
    output_dir=output_dir+'/'+modality
    serie_IDs.sort()
    stations=['1head','2torso','3pelvis','4legs','5llegs','6lllegs','7feet']
    stations=stations[:len(serie_IDs)]
    n=serie_IDs.index(serie_ID)
    output_dir+='_'+stations[n]
    
    return output_dir
    
    
def create_T2Dixon_mhd(fileframe,output_dir):
    fileframe_T2dix=fileframe[fileframe['Modality'].str.contains('CORO T2 DIX4mm')]
    if fileframe_T2dix.shape[0]>0:
        for contrast in fileframe_T2dix.type.unique():
            fileframe_T2dix_contrast=fileframe_T2dix[fileframe_T2dix['type']==contrast]
            path = os.path.dirname(fileframe_T2dix_contrast.Filename.unique()[0])
            os.mkdir(path+'/temp')
            for file in fileframe_T2dix_contrast.Filename.unique():
                shutil.copy(file,path+'/temp')
            reader = sitk.ImageSeriesReader()
            series_ids=reader.GetGDCMSeriesIDs(path+'/temp')
            i=0
            for ID in series_ids:

                dicom_names = reader.GetGDCMSeriesFileNames(path+'/temp',ID)
                reader.SetFileNames(dicom_names)

                image = reader.Execute()

                output_dir_S = create_output_dir_T2dix(output_dir,'T2dixon'+contrast,int(i),list(range(len(series_ids))))
                #original_image = sitk.ReadImage(files)

                # Write the image.
                output_file_name_3D = output_dir_S + '.mhd'
                sitk.WriteImage(image, output_file_name_3D)
                i+=1
            shutil.rmtree(path+'/temp')
        
def makeint(row):
    return int(row['loc'])



