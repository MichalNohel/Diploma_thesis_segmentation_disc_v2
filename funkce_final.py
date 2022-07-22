# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:37:41 2022

@author: nohel
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import glob
from skimage.io import imread
from skimage.color import rgb2gray,rgb2hsv,rgb2xyz
from skimage.morphology import disk,remove_small_objects, binary_closing, binary_opening
from skimage.filters import gaussian

from scipy.ndimage import binary_erosion 
from scipy.ndimage.morphology import binary_fill_holes
import torchvision.transforms.functional as TF
from torch.nn import init
import matplotlib.pyplot as plt
from scipy.io import loadmat

## Dataloader
class DataLoader(torch.utils.data.Dataset):
    def __init__(self,split="Train",path_to_data="D:\Diploma_thesis_segmentation_disc/Data_640_640",color_preprocesing="RGB",segmentation_type="disc",output_size=(int(608),int(608),int(3))):
        self.split=split
        self.path_to_data=path_to_data+ '/' +split
        self.color_preprocesing=color_preprocesing
        self.segmentation_type=segmentation_type
        self.output_size=output_size
        
        
        if split=="Train":
            self.files_img=glob.glob(self.path_to_data+'/Images_crop/*.png')
            self.files_disc=glob.glob(self.path_to_data+'/Disc_crop/*.png')
            self.files_cup=glob.glob(self.path_to_data+'/Cup_crop/*.png')
            self.files_img.sort()
            self.files_disc.sort()
            self.files_cup.sort()
            self.num_of_imgs=len(self.files_img)
            
        if split=="Test":
            self.files_img=glob.glob(self.path_to_data+'/Images/*.png')
            self.files_disc=glob.glob(self.path_to_data+'/Disc/*.png')
            self.files_cup=glob.glob(self.path_to_data+'/Cup/*.png')
            self.files_fov=glob.glob(self.path_to_data+'/Fov/*.png')
            self.disc_centres_test=loadmat(self.path_to_data+'/Disc_centres_test.mat')          
            self.num_of_imgs=len(self.files_img)
            
            
    def __len__(self):
        return self.num_of_imgs
    
    def __getitem__(self,index):
        #Load of Train images
        if self.split=="Train":
            img=imread(self.files_img[index])
            disc=imread(self.files_disc[index]).astype(bool)
            cup=imread(self.files_cup[index]).astype(bool)
            output_size=self.output_size
            input_size=img.shape
            
            img,disc,cup=self.random_crop(input_size,output_size,img,disc,cup)
            img,disc,cup=self.random_rotflip(img,disc,cup)
            
            
                      
            #Preprocesing of img
            if(self.color_preprocesing=="RGB"):
                img=img.astype(np.float32)
                
            if(self.color_preprocesing=="gray"):
                img=rgb2gray(img).astype(np.float32)              
            
            if(self.color_preprocesing=="HSV"):
                img=rgb2hsv(img).astype(np.float32)
                
            if(self.color_preprocesing=="XYZ"):
                img=rgb2xyz(img).astype(np.float32)
                
            #Creation of labels masks: batch x width x height
            if(self.segmentation_type=="disc"):
                mask_output_size=(int(1),output_size[0],output_size[1]) # output size of image
                mask_output=np.zeros(mask_output_size)
                mask_output[0,:,:]=disc                
            elif(self.segmentation_type=="cup"):
                mask_output_size=(int(1),output_size[0],output_size[1]) # output size of image
                mask_output=np.zeros(mask_output_size)
                mask_output[0,:,:]=cup
            elif(self.segmentation_type=="disc_cup"):
                mask_output_size=(int(2),output_size[0],output_size[1]) # output size of image
                mask_output=np.zeros(mask_output_size)
                mask_output[0,:,:]=disc
                mask_output[1,:,:]=cup
            else:
                print("Wrong type of segmentation")
                
            mask_output=mask_output.astype(bool)
            img=TF.to_tensor(img)
            mask=torch.from_numpy(mask_output)
            return img,mask
        
        if self.split=="Test":
            img_orig=imread(self.files_img[index])
            disc_orig=imread(self.files_disc[index]).astype(bool)
            cup_orig=imread(self.files_cup[index]).astype(bool)
            fov_orig=imread(self.files_fov[index]).astype(bool)
            output_size=self.output_size
            
            output_crop_image, output_mask_disc,output_mask_cup=Crop_image(img_orig,disc_orig,cup_orig,output_size, self.disc_centres_test.get('Disc_centres_test')[index])
            
            #Preprocesing of img
            if(self.color_preprocesing=="RGB"):
                img_crop=output_crop_image.astype(np.float32)
                
            if(self.color_preprocesing=="gray"):
                img_crop=rgb2gray(output_crop_image).astype(np.float32)              
            
            if(self.color_preprocesing=="HSV"):
                img_crop=rgb2hsv(output_crop_image).astype(np.float32)
                
            if(self.color_preprocesing=="XYZ"):
                img_crop=rgb2xyz(output_crop_image).astype(np.float32)
            
            #Creation of labels masks: batch x width x height
            if(self.segmentation_type=="disc"):
                mask_output_size=(int(1),output_size[0],output_size[1]) # output size of image
                mask_output=np.zeros(mask_output_size)
                mask_output[0,:,:]=output_mask_disc                
            elif(self.segmentation_type=="cup"):
                mask_output_size=(int(1),output_size[0],output_size[1]) # output size of image
                mask_output=np.zeros(mask_output_size)
                mask_output[0,:,:]=output_mask_cup
            elif(self.segmentation_type=="disc_cup"):
                mask_output_size=(int(2),output_size[0],output_size[1]) # output size of image
                mask_output=np.zeros(mask_output_size)
                mask_output[0,:,:]=output_mask_disc
                mask_output[1,:,:]=output_mask_cup
            else:
                print("Wrong type of segmentation")
                
            mask_output=mask_output.astype(bool)
            img_crop=TF.to_tensor(img_crop)
            mask_output=torch.from_numpy(mask_output)
            coordinates=self.disc_centres_test.get('Disc_centres_test')[index].astype(np.int16)
            return img_crop,mask_output,img_orig,disc_orig,cup_orig,coordinates
        
        if self.split=="HRF":
            img_orig=imread(self.files_img[index])
            fov_orig=imread(self.files_fov[index]).astype(bool)
            output_size=self.output_size
            #sigma=60
            #size_of_erosion=80            
            #center_new=Detection_of_disc(img_orig,fov_orig[:,:,0],sigma,size_of_erosion)
            output_crop_image=Crop_image_HRF(img_orig,output_size,self.Disc_centres_HRF.get('center_new_HRF')[index].astype(np.int16))
            #coordinates=np.array(center_new)
            
            #Preprocesing of img
            if(self.color_preprocesing=="RGB"):
                img_crop=output_crop_image.astype(np.float32)
                
            if(self.color_preprocesing=="gray"):
                img_crop=rgb2gray(output_crop_image).astype(np.float32)              
            
            if(self.color_preprocesing=="HSV"):
                img_crop=rgb2hsv(output_crop_image).astype(np.float32)
                
            if(self.color_preprocesing=="XYZ"):
                img_crop=rgb2xyz(output_crop_image).astype(np.float32)
            
            img_crop=TF.to_tensor(img_crop)
            
            coordinates=self.Disc_centres_HRF.get('center_new_HRF')[index].astype(np.int16)
            return img_crop,img_orig,coordinates
            
        
        
    def random_crop(self,in_size,out_size,img,disc,cup):
        r=[int(torch.randint(in_size[0]-out_size[0],(1,1)).view(-1).numpy()),int(torch.randint(in_size[1]-out_size[1],(1,1)).view(-1).numpy())]
        img_crop=img[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1],:]
        disc_crop=disc[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]
        cup_crop=cup[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]
        return img_crop.copy(),disc_crop.copy(),cup_crop.copy()
    
    def random_rotflip(self,img,disc,cup):
            r=[torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(4,(1,1)).view(-1).numpy()]
            if r[0]:
                img=np.fliplr(img)
                disc=np.fliplr(disc)
                cup=np.fliplr(cup)
            if r[1]:
                img=np.flipud(img)
                disc=np.flipud(disc)
                cup=np.flipud(cup)
    
            img=np.rot90(img,k=r[2])
            disc=np.rot90(disc,k=r[2]) 
            cup=np.rot90(cup,k=r[2]) 
    
            return img.copy(),disc.copy(),cup.copy()
             
                  
        
## U-Net
class unetConv2(nn.Module):
    def __init__(self,in_size,out_size,filter_size=3,stride=1,pad=1,do_batch=1):
        super().__init__()
        self.do_batch=do_batch
        self.conv=nn.Conv2d(in_size,out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm2d(out_size,momentum=0.1)
        
    def forward(self,inputs):
        outputs=self.conv(inputs)    
        
        if self.do_batch:
            outputs=self.bn(outputs) 
        
        outputs=F.relu(outputs)
        return outputs
        
class unetConvT2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=2,pad=1,out_pad=1):
        super().__init__()
        self.conv=nn.ConvTranspose2d(in_size, out_size, filter_size,stride=stride, padding=pad, output_padding=out_pad)
        
    def forward(self,inputs):
        outputs=self.conv(inputs)
        outputs=F.relu(outputs)
        return outputs

        
class unetUp(nn.Module):
    def __init__(self,in_size,out_size):
        super(unetUp,self).__init__()
        self.up = unetConvT2(in_size,out_size)
        
    def forward(self,inputs1, inputs2):
        inputs2=self.up(inputs2)
        return torch.cat([inputs1,inputs2],1)
    
class Unet(nn.Module):
    def __init__(self, filters=(np.array([16, 32, 64, 128, 256])/2).astype(np.int32),in_size=3,out_size=1):
        super().__init__()
        self.out_size=out_size
        self.in_size=in_size
        self.filters=filters
        
        self.conv1 = nn.Sequential(unetConv2(in_size, filters[0]),unetConv2(filters[0], filters[0]),unetConv2(filters[0], filters[0]))
        self.conv2 = nn.Sequential(unetConv2(filters[0], filters[1] ),unetConv2(filters[1], filters[1] ),unetConv2(filters[1], filters[1] ))  
        self.conv3 = nn.Sequential(unetConv2(filters[1], filters[2] ),unetConv2(filters[2], filters[2] ),unetConv2(filters[2], filters[2] ))
        self.conv4 = nn.Sequential(unetConv2(filters[2], filters[3] ),unetConv2(filters[3], filters[3] ),unetConv2(filters[3], filters[3] ))

        self.center = nn.Sequential(unetConv2(filters[-2], filters[-1] ),unetConv2(filters[-1], filters[-1] ))                
        
        self.up_concat4 = unetUp(filters[4], filters[4] )        
        self.up_conv4=nn.Sequential(unetConv2(filters[3]+filters[4], filters[3] ),unetConv2(filters[3], filters[3] ))

        self.up_concat3 = unetUp(filters[3], filters[3] )
        self.up_conv3=nn.Sequential(unetConv2(filters[2]+filters[3], filters[2] ),unetConv2(filters[2], filters[2] ))

        self.up_concat2 = unetUp(filters[2], filters[2] )
        self.up_conv2=nn.Sequential(unetConv2(filters[1]+filters[2], filters[1] ),unetConv2(filters[1], filters[1] ))
    
        self.up_concat1 = unetUp(filters[1], filters[1] )
        self.up_conv1=nn.Sequential(unetConv2(filters[0]+filters[1], filters[0] ),unetConv2(filters[0], filters[0],do_batch=0 ))
            
        self.final = nn.Conv2d(filters[0], self.out_size, 1)
        
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
    
    
    def forward(self,inputs):
        conv1=self.conv1(inputs)
        x=F.max_pool2d(conv1,2,2)
        
        conv2=self.conv2(x)
        x=F.max_pool2d(conv2,2,2)
        
        conv3=self.conv3(x)
        x=F.max_pool2d(conv3,2,2)
        
        conv4=self.conv4(x)
        x=F.max_pool2d(conv4,2,2)
        
        x=self.center(x)
        
        x=self.up_concat4(conv4,x)
        x=self.up_conv4(x)
        
        x=self.up_concat3(conv3,x)
        x=self.up_conv3(x)
        
        x=self.up_concat2(conv2,x)
        x=self.up_conv2(x)
        
        x=self.up_concat1(conv1,x)
        x=self.up_conv1(x)
    
        x=self.final(x)
        
        return x
    


def dice_loss(X,Y):
    eps=1.
    dice=((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return 1-dice

def dice_coefficient(X,Y):
    # X-otuput, Y-Label
    TP=np.sum(np.logical_and(X,Y))
    FP=np.sum(np.logical_and(np.logical_not(X),Y))
    FN=np.sum(np.logical_and(X,np.logical_not(Y)))
    dice = 2*TP/(2*TP+FP+FN)
    return dice

def Sensitivity (X,Y):
    # X-otuput, Y-Label
    TP=np.sum(np.logical_and(X,Y))
    FN=np.sum(np.logical_and(X,np.logical_not(Y)))
    sensitivity = TP/(FN+TP)
    return sensitivity
    
def Specificity (X,Y):
    # X-otuput, Y-Label
    TN=np.sum(np.logical_and(np.logical_not(X),np.logical_not(Y)))
    FP=np.sum(np.logical_and(np.logical_not(X),Y))
    specificity = TN/(TN+FP)
    return specificity

def Detection_of_disc(image,fov,sigma,size_of_erosion):    
    img=rgb2xyz(image).astype(np.float32)
    img=rgb2gray(img).astype(np.float32)
    BW=binary_erosion(fov,disk(size_of_erosion))
    vertical_len=BW.shape[0]
    step=round(vertical_len/15);
    BW[0:step,:]=0;
    BW[vertical_len-step:vertical_len,:]=0;
    img[~BW]=0;
    img_filt=gaussian(img,sigma);
    img_filt[~BW]=0;
    max_xy = np.where(img_filt == img_filt.max() )
    r=max_xy[0][0]
    c=max_xy[1][0]
    center_new=[]
    center_new.append(c)
    center_new.append(r)
    return center_new
    
def Crop_image(image,mask_disc,mask_cup,output_image_size,center_new): 
    size_in_img=image.shape
    x_half=int(output_image_size[0]/2)
    y_half=int(output_image_size[1]/2)  
    
    if ((center_new[1]-x_half)<0):
        x_start=0
    elif ((center_new[1]+x_half)>size_in_img[0]):
        x_start=size_in_img[0]-output_image_size[0]
    else:
        x_start=center_new[1]-x_half           
    
    if ((center_new[0]-y_half)<0):
        y_start=0
    elif ((center_new[0]+y_half)>size_in_img[1]):
        y_start=size_in_img[1]-output_image_size[1]
    else:
        y_start=center_new[0]-y_half
    
    output_crop_image=image[x_start:x_start+output_image_size[0],y_start:y_start+output_image_size[1],:]
    output_mask_disc=mask_disc[x_start:x_start+output_image_size[0],y_start:y_start+output_image_size[1]]
    output_mask_cup=mask_cup[x_start:x_start+output_image_size[0],y_start:y_start+output_image_size[1]]
    return output_crop_image, output_mask_disc,output_mask_cup

def Crop_image_HRF(image,output_image_size,center_new): 
    size_in_img=image.shape
    x_half=int(output_image_size[0]/2)
    y_half=int(output_image_size[1]/2)  
    
    if ((center_new[1]-x_half)<0):
        x_start=0
    elif ((center_new[1]+x_half)>size_in_img[0]):
        x_start=size_in_img[0]-output_image_size[0]
    else:
        x_start=center_new[1]-x_half        
    
    if ((center_new[0]-y_half)<0):
        y_start=0
    elif ((center_new[0]+y_half)>size_in_img[1]):
        y_start=size_in_img[1]-output_image_size[1]
    else:
        y_start=center_new[0]-y_half
    
    output_crop_image=image[x_start:x_start+output_image_size[0],y_start:y_start+output_image_size[1],:]
    return output_crop_image


def Postprocesing(output,min_size,type_of_morphing,size_of_disk,ploting):
    
    output_final=binary_fill_holes(output)
    output_final=remove_small_objects(output_final,min_size=min_size)
    
    output_final=np.pad(output_final, pad_width=[(50, 50),(50, 50)], mode='constant')
    
    if (type_of_morphing=="closing"):
        output_final=binary_closing(output_final,disk(size_of_disk))
    elif (type_of_morphing=="openinig"):
        output_final=binary_opening(output_final,disk(size_of_disk))
    elif (type_of_morphing=="closing_opening"):
        output_final=binary_closing(output_final,disk(size_of_disk)) 
        output_final=binary_opening(output_final,disk(size_of_disk))
    elif (type_of_morphing=="opening_closing"):
        output_final=binary_opening(output_final,disk(size_of_disk)) 
        output_final=binary_closing(output_final,disk(size_of_disk))  
        
    output_final=output_final[50:338,50:338]
    
    if ploting:        
        plt.figure(figsize=[10,10])
        plt.subplot(1,2,1)
        plt.imshow(output)
        plt.title('Po vystupu sítě')    
        
        plt.subplot(1,2,2)
        plt.imshow(output_final)
        plt.title('Postprocesing')
        plt.show()
    
    return output_final
    




