import SimpleITK as sitk
import numpy as np
import numpy.ma as ma
import sys
from os import listdir
import glob
from os.path import isfile, join
# glob for the different aquisitions 
#
#input = sys.argv[1]
#patient = sys.argv[2]
#acquisition = sys.argv[3]
#axis = sys.argv[4]


stations = ['1head','2torso','3pelvis','4legs','5llegs','6lllegs', '7feet']
#stations.reverse()
modalities=['ADC','T1','b2000','b1000','b1500','b0']

def is_functional(WB_dict):
    func_dir=[ 1.,0.,-0.,-0.,1.,0.,0., -0.,  1.]
    ana_dir=[ 1. , 0.,  0., -0., -0.,  1.,  0., -1., -0.]
    if np.all(np.around(list(WB_dict.values())[0].get('direction'))==func_dir):
        return True
    elif np.all(np.around(list(WB_dict.values())[0].get('direction'))==ana_dir):
        return False
    else:
        return False

def calculate_overlay(WB_dict):
    if is_functional(WB_dict):
        axis=2
    else:
        axis=1
    for i, station in enumerate(list(WB_dict.keys())):
        if i ==0:
            WB_dict.get(station)['overlay']=0
        else:
            originL=WB_dict.get(station).get('origin')
            originH=WB_dict.get(stations[i-1]).get('origin')
            sizeH=WB_dict.get(stations[i-1]).get('size')
            spacingH=WB_dict.get(stations[i-1]).get('spacing')
            #overlay = int(np.floor((sizeH[2]*spacingH[2]-(originH[2]-originL[2]))/spacingH[2]))
            overlay = np.floor((originL[2]-(originH[2]-spacingH[axis]*sizeH[axis]))/spacingH[axis])
            WB_dict.get(station)['overlay']=overlay
    
    return WB_dict

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

def cumulative_size(WB_dict):
    if is_functional(WB_dict):
        axis=2
    else:
        axis=1
    WB_size=0
    for key , value in WB_dict.items():

        WB_size+=value.get('size')[axis]-value.get('overlay')
    return WB_size

def get_template_data(WB_dict):
    value=list(WB_dict.values())[0]
    size = value.get('size')
    spacing = value.get('spacing')
    origin = value.get('origin')
    direction = value.get('direction')
    
    return size,spacing,origin,direction

def get_index(i,WB_dict):
    if is_functional(WB_dict):
        axis=2
    else:
        axis=1
    size=0
    overlay=0
    for j in range(i):
        size+=list(WB_dict.values())[j].get('size')[axis]
        if j<i-1:
            overlay+=list(WB_dict.values())[j+1].get('overlay')
        else:
            overlay+=0.5*list(WB_dict.values())[i].get('overlay')
    return size-overlay

def stitch_ana(WB_dict):
    #create an empty template 
    size,spacing,origin,direction=get_template_data(WB_dict)
    cummulativeSize = cumulative_size(WB_dict)

    wholeBody = sitk.Image((size[0], int(cummulativeSize), size[2]), sitk.sitkFloat32)
    wholeBody.SetSpacing(spacing)
    wholeBody.SetOrigin(origin)
    wholeBody.SetDirection(direction)
    
    #pasting the stations in the empty template
    paste = sitk.PasteImageFilter()
    stations=list(WB_dict.keys())
    for i, station in enumerate(stations):
        image=WB_dict.get(stations[i]).get('image')
        image=sitk.Cast(image, sitk.sitkFloat32)
        if i ==0:
            paste.SetDestinationIndex([0,0,0])
            paste.SetSourceIndex([0,0,0])
            paste.SetSourceSize(size)
            wholeBody = paste.Execute(wholeBody, image)
        else:
            index=get_index(i,WB_dict)
            sizeL=WB_dict.get(stations[i]).get('size')
            overlay=WB_dict.get(stations[i]).get('overlay')
            
            paste.SetDestinationIndex([0,int(index),0])
            paste.SetSourceIndex([0,int(np.floor(0.5*overlay)),0])
            paste.SetSourceSize([sizeL[0], int(sizeL[1]-np.floor(0.5*overlay)), sizeL[2]])
            
            wholeBody = paste.Execute(wholeBody, image)
     
    return wholeBody

def stitch_func(WB_dict):
    #create an empty template 
    size,spacing,origin,direction=get_template_data(WB_dict)
    cummulativeSize = cumulative_size(WB_dict)
    wholeBody = sitk.Image(size[0], size[1], int(cummulativeSize), sitk.sitkFloat32)    
    wholeBody.SetSpacing(spacing)
    wholeBody.SetOrigin(origin)
    wholeBody.SetDirection(direction)
    
    #pasting the stations in the empty template
    paste = sitk.PasteImageFilter()
    stations=list(WB_dict.keys())
    for i, station in enumerate(reversed(stations)):
        if i==0:
            new_origin=WB_dict.get(station).get('image').GetOrigin()
        image=WB_dict.get(station).get('image')
        image=sitk.Cast(image, sitk.sitkFloat32)
        if i ==0:
            paste.SetDestinationIndex([0,0,0])
            paste.SetSourceIndex([0,0,0])
            paste.SetSourceSize(size)
            wholeBody = paste.Execute(wholeBody, image)
        else:
            index=get_index(i,WB_dict)
            print(index)
            sizeH=WB_dict.get(station).get('size')
            overlay=WB_dict.get(stations[i-1]).get('overlay')

            paste.SetDestinationIndex([0,0,int(index)])
            paste.SetSourceIndex([0,0, int(np.floor(0.5*overlay))])
            paste.SetSourceSize([sizeH[0], sizeH[1], int(sizeH[2]-np.floor(0.5*overlay))])
            wholeBody = paste.Execute(wholeBody, image)
            wholeBody.SetOrigin(new_origin)
    return wholeBody

    


def calc_start(i,WB_dict):
    if is_functional(WB_dict):
        axis=2
    else:
        axis=1
    size=0
    overlay=0
    for j in range(i):
        size+=list(WB_dict.values())[j].get('size')[axis]
        #try:
         #   size-=list(WB_dict.values())[j+1].get('overlay')
        #except:
         #   size-=0
        size-=list(WB_dict.values())[j].get('overlay')

    return size

def interpolate_ana(WB_dict):
#create an empty template 
    size,spacing,origin,direction=get_template_data(WB_dict)
    cummulativeSize = cumulative_size(WB_dict)
    
    wholeBody = sitk.Image(size[0], int(cummulativeSize), size[2], sitk.sitkFloat32)
    wholeBody.SetSpacing(spacing)
    wholeBody.SetOrigin(origin)
    wholeBody.SetDirection(direction)
    
    #pasting the stations in the empty template
    paste = sitk.PasteImageFilter()
    stations=list(WB_dict.keys())

    for i, station in enumerate(stations):
        image=WB_dict.get(stations[i]).get('image')
        image=sitk.Cast(image, sitk.sitkFloat32)
        index=calc_start(i,WB_dict)
        sizeL=WB_dict.get(stations[i]).get('size')
        overlay_prev=WB_dict.get(stations[i]).get('overlay')
        try:
            overlay_next=WB_dict.get(stations[i+1]).get('overlay')
        except:
            overlay_next=0
        paste.SetDestinationIndex([0,int(index),0])
        paste.SetSourceIndex([0,int(np.floor(overlay_prev)),0])

        paste.SetSourceSize([sizeL[0], int(sizeL[1]-np.floor(overlay_prev)-np.floor(overlay_next)), sizeL[2]])
        wholeBody = paste.Execute(wholeBody, image)
        
        if i < len(stations)-1:
            image2=WB_dict.get(stations[i+1]).get('image')

            image2=sitk.Cast(image2, sitk.sitkFloat32)
            prev_index=calc_start(i,WB_dict)
            index=calc_start(i+1,WB_dict)
            overlay=WB_dict.get(stations[i+1]).get('overlay')
            image2=image2[:,:int(overlay),:]
            image=image[:,int(-overlay):,:]
            image2.SetOrigin(image.GetOrigin())
            image2.SetSpacing(image.GetSpacing())

            image3=image+image2
            for k,j in enumerate(range(int(index-overlay),int(index))):
                percentage=float(1-k/overlay)

                image3=percentage*image+(1-percentage)*image2
                paste.SetDestinationIndex([0,j,0])
                paste.SetSourceIndex([0,k,0])
                paste.SetSourceSize([sizeL[0], 1, sizeL[2]])
                wholeBody = paste.Execute(wholeBody, image3)

    return wholeBody

def interpolate_func(WB_dict):
#create an empty template 
    size,spacing,origin,direction=get_template_data(WB_dict)
    cummulativeSize = cumulative_size(WB_dict)
    
    wholeBody = sitk.Image(size[0], size[1], int(cummulativeSize), sitk.sitkFloat32)
    wholeBody.SetSpacing(spacing)
    wholeBody.SetOrigin(origin)
    wholeBody.SetDirection(direction)
    
    #pasting the stations in the empty template
    paste = sitk.PasteImageFilter()
    stations=list(WB_dict.keys())
    

    for i, station in enumerate(reversed(stations)):
        if i==0:
            new_origin=WB_dict.get(station).get('image').GetOrigin()
        image=WB_dict.get(station).get('image')
        image=sitk.Cast(image, sitk.sitkFloat32)
        index=calc_start(i,WB_dict)
        sizeL=WB_dict.get(stations[i]).get('size')
        overlay_prev=WB_dict.get(stations[i]).get('overlay')
        try:
            overlay_next=WB_dict.get(stations[i+1]).get('overlay')
        except:
            overlay_next=0
        paste.SetDestinationIndex([0,0,int(index)])
        paste.SetSourceIndex([0,0,int(np.ceil(overlay_prev)+1)])
        
        paste.SetSourceSize([sizeL[0],  sizeL[1],int(sizeL[2]-np.ceil(overlay_prev)-np.ceil(overlay_next)),])
        if i == len(stations)-1:
                paste.SetSourceSize([sizeL[0],  sizeL[1],int(sizeL[2]-np.ceil(overlay_prev)-np.ceil(overlay_next)-1),])
        wholeBody = paste.Execute(wholeBody, image)
        
        if i < len(stations)-1:
            stations.reverse()
            image2=WB_dict.get(stations[i+1]).get('image')
            stations.reverse()
            image2=sitk.Cast(image2, sitk.sitkFloat32)
            prev_index=calc_start(i,WB_dict)
            index=calc_start(i+1,WB_dict)
            overlay=WB_dict.get(stations[i+1]).get('overlay')
            image2=image2[:,:,:int(overlay)]

            image=image[:,:,int(-overlay):]

            image2.SetSpacing(image.GetSpacing())
            image2.SetOrigin(image.GetOrigin())
            image3=image+image2
            for k,j in enumerate(range(int(index-overlay),int(index))):

                percentage=float(1-k/overlay)
                image3=percentage*image+(1-percentage)*image2
                paste.SetDestinationIndex([0,0,j])
                paste.SetSourceIndex([0,0,k])
                paste.SetSourceSize([sizeL[0], sizeL[1], 1])
                wholeBody = paste.Execute(wholeBody, image3)
    wholeBody.SetOrigin(new_origin)
    return wholeBody

def constructWholeBody(folder,interpolate):
    ### Load all stations and metadata
    WB_dict=get_stations(folder)

    # calculate final size
    if interpolate:
        if is_functional(WB_dict):
            wholeBody=interpolate_func(WB_dict)
        else:
            wholeBody=interpolate_ana(WB_dict)
     
    else :
        if is_functional(WB_dict):
            wholeBody=interpolate_func(WB_dict)
        else:
            wholeBody=interpolate_ana(WB_dict)
        

    writer = sitk.ImageFileWriter()
    writer.SetFileName(folder+'.mhd')
    writer.Execute(wholeBody)
    
    


