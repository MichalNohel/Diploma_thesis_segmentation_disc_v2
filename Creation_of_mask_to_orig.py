# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:34:50 2022

@author: nohel
"""

from funkce_final import DataLoader,Unet,Postprocesing,zapis_kontury
import torch
import matplotlib.pyplot as plt
import numpy as np 
from skimage import measure, img_as_uint
from skimage.transform import resize
from skimage.io import imsave
import os
import pandas
import json
from json import JSONEncoder

if __name__ == "__main__": 
    
    ZAPISOVAT=True
    
    batch=1 
    threshold=0.5
    color_preprocesing="RGB"

    output_size=(int(288),int(288),int(3))
    
    #path_to_data="D:\Diploma_thesis_segmentation_disc_v2/Data_320_320_25px_preprocesing_all_database"
    
    #mereni UBMI
    path_to_data="D:\Diploma_thesis_segmentation_disc_v2/Data_320_320_25px_preprocesing_UBMI_mereni_verze_2"   
    
    
    # Cesta k naucenemu modelu
    path_to_model="D:\Diploma_thesis_segmentation_disc_v2/Data_320_320_25px_preprocesing_all_database/Naucene_modely/"
    
    # Disc and cup together
    segmentation_type="disc_cup"
    path_to_save_model = path_to_model + 'disc_and_cup_detection_25px_all_databases/'
    name_of_model='model_02_disc_cup_25px_all_modified_databases'
    net=Unet(out_size=2).cuda()  
    net.load_state_dict(torch.load(path_to_save_model+ name_of_model+ '.pth'))    
    
    
    net.eval()
    path_to_extracted_data=path_to_data+ "/Results/"  
    
    #Postsprocesing parameters
    min_size_of_optic_disk=1000
    size_of_disk_for_morphing=40
    #type_of_morphing='closing' 
    #type_of_morphing='openinig' 
    type_of_morphing='closing_opening' 
    #type_of_morphing='openinig_closing' 
    ploting=0    
    
      
    batch=1
    loader=DataLoader(split="UBMI",path_to_data=path_to_data,color_preprocesing=color_preprocesing,segmentation_type=segmentation_type,output_size=output_size)
    UBMI_loader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    test_files_name=UBMI_loader.dataset.files_img
    
    path_to_google='H:\Sdílené disky\Retina GAČR\Měření na UBMI\Sada01\Sada_01'
    files_google=os.listdir(path_to_google)
    
    
    
    for kk,(data,data_orig,lbl,img_full,img_orig_full, disc_orig,cup_orig,coordinates,img_orig_neprevzorkovany) in enumerate(UBMI_loader):
             data=data.cuda()
             lbl=lbl.cuda()
             
             output=net(data)
             output=torch.sigmoid(output)
             output=output.detach().cpu().numpy() > threshold
             
             lbl=lbl.detach().cpu().numpy()
             
             test_files_name_tmp=test_files_name[kk][97:]
             name_of_img=test_files_name_tmp[:-4]
             
             dir_pom=path_to_extracted_data+name_of_img
            
             isExist = os.path.exists(dir_pom)
             if not isExist:
                 os.makedirs(dir_pom)
                 
             
             pom_sourad=coordinates.detach().cpu().numpy()[0]  
                        
             output_mask_disc=np.zeros([disc_orig.shape[1],disc_orig.shape[2]]) 
             output_mask_disc_final=np.zeros([disc_orig.shape[1],disc_orig.shape[2]])  
                        
             output_mask_cup=np.zeros([disc_orig.shape[1],disc_orig.shape[2]]) 
             output_mask_cup_final=np.zeros([disc_orig.shape[1],disc_orig.shape[2]])  
                        
             if (pom_sourad[1]-int(output_size[0]/2)<0):
                 x_start=0
             elif((pom_sourad[1]+int(output_size[0]/2))>output_mask_disc.shape[0]):
                 x_start=output_mask_disc.shape[0]-output_size[0]
             else:
                 x_start=pom_sourad[1]-int(output_size[0]/2)
                            
             if (pom_sourad[0]-int(output_size[0]/2)<0):
                 y_start=0
             elif((pom_sourad[0]+int(output_size[0]/2))>output_mask_disc.shape[1]):
                 y_start=output_mask_disc.shape[1]-output_size[0]
             else:
                 y_start=pom_sourad[0]-int(output_size[0]/2)
             
        
        
             output_mask_disc[x_start:x_start+output_size[0],y_start:y_start+output_size[0]]=output[0,0,:,:]
             output_mask_disc=output_mask_disc.astype(bool)    
                        
             output_mask_cup[x_start:x_start+output_size[0],y_start:y_start+output_size[0]]=output[0,1,:,:]
             output_mask_cup=output_mask_cup.astype(bool)  
                        
             # Postprocesing 
             
             output_final=Postprocesing(output[0,0,:,:],min_size_of_optic_disk,type_of_morphing,size_of_disk_for_morphing,ploting)
                        
             
             output_mask_disc_final[x_start:x_start+output_size[0],y_start:y_start+output_size[0]]=output_final
             output_mask_disc_final=output_mask_disc_final.astype(bool)
                        
             disc_orig=disc_orig[0,:,:].detach().cpu().numpy() 
             cup_orig=cup_orig[0,:,:].detach().cpu().numpy() 
             
             plt.figure(figsize=[15,15])
             plt.subplot(2,3,1)    
             im_pom_orig=img_orig_full[0,:,:,:].numpy()/255   
             plt.imshow(im_pom_orig)   
             plt.title('Original_' + name_of_img)
             
                            
             plt.subplot(2,3,2)    
             plt.imshow(output_mask_disc)
             plt.title('Output of net - disc')
                            
             plt.subplot(2,3,3)    
             plt.imshow(output_mask_disc_final)
             plt.title('Postprocesing')
                            
             
             plt.subplot(2,3,4)                        
             im_pom_orig=img_orig_full[0,:,:,:].numpy()/255   
             plt.imshow(im_pom_orig)   
             plt.title('Original_' + name_of_img)
   
             plt.subplot(2,3,5) 
             plt.imshow(output_mask_cup) 
             plt.title('Output of net - cup')                      
                               
             plt.show() 
             
             # Převzorkování masek na orig shape
             
             output_mask_disc_final_orig_shape=resize(output_mask_disc_final,[img_orig_neprevzorkovany.shape[1],img_orig_neprevzorkovany.shape[2]]).astype(bool)
             output_mask_cup_final_orig_shape=resize(output_mask_cup,[img_orig_neprevzorkovany.shape[1],img_orig_neprevzorkovany.shape[2]]).astype(bool)
             
             plt.figure(figsize=[15,15])
             plt.subplot(1,3,1)    
             im_pom_orig=img_orig_neprevzorkovany[0,:,:,:].numpy()/255   
             plt.imshow(im_pom_orig)   
             plt.title('Original_' + name_of_img)
             
                            
             plt.subplot(1,3,2)    
             plt.imshow(output_mask_disc_final_orig_shape)
             plt.title('Output of net - disc')
                            
             plt.subplot(1,3,3)    
             plt.imshow(output_mask_cup_final_orig_shape)
             plt.title('Cup')  
             plt.show() 
             
             # Kontury
             
             disc_contours=measure.find_contours(output_mask_disc_final_orig_shape)
             cup_contours=measure.find_contours(output_mask_cup_final_orig_shape)
             
             if ZAPISOVAT:
                 # Uložení masek ze segmentace
                 # Disc
                 imsave(dir_pom+'/' + name_of_img +"_Disc.png",output_mask_disc_final_orig_shape)   
                 zapis_kontury(ZAPISOVAT, disc_contours, dir_pom, name_of_img, type_of_contur="_disc_contours" )
                 # Cup
                 imsave(dir_pom+'/' + name_of_img +"_Cup.png",output_mask_cup_final_orig_shape)
                 zapis_kontury(ZAPISOVAT, cup_contours, dir_pom, name_of_img, type_of_contur="_cup_contours" )
                 
                 
             
             plt.figure(figsize=[15,15])
             plt.subplot(1,2,1)
             im_pom_orig=img_orig_neprevzorkovany[0,:,:,:].numpy()/255   
             plt.imshow(im_pom_orig)  
             for contour in disc_contours:
                 plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
             plt.title('Original_' + name_of_img+'_disc')
             
             plt.subplot(1,2,2)
             im_pom_orig=img_orig_neprevzorkovany[0,:,:,:].numpy()/255   
             plt.imshow(im_pom_orig)  
             for contour in cup_contours:
                 plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
             plt.title('Original_' + name_of_img+'_cup')
             
             plt.savefig(dir_pom+'/' + name_of_img +"_kontury_in_orig.png")
             plt.show()
             
             
             for i in range(len(files_google)):
                 if files_google[i][0:14]==name_of_img[0:14]:
                     
                     if (len(files_google[i])!=len(files_google[0])):
                         continue
                     
                     file_tmp_disc=path_to_google + '/' + files_google[i]+'/ImageAnalysis/Disc_automated_segmentation_v1/'
                     file_tmp_cup=path_to_google + '/' + files_google[i]+'/ImageAnalysis/Cup_automated_segmentation_v1/'
                     isExist = os.path.exists(file_tmp_disc)
                     if not isExist:
                         os.makedirs(file_tmp_disc)
                         
                     isExist = os.path.exists(file_tmp_cup)
                     if not isExist:
                         os.makedirs(file_tmp_cup)
                     
                 
                     imsave(file_tmp_disc + name_of_img +"_Disc.png",output_mask_disc_final_orig_shape)  
                     imsave(file_tmp_cup + name_of_img +"_Cup.png",output_mask_cup_final_orig_shape)
                     
                 

             
             
             
             
             
             
             
             
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    