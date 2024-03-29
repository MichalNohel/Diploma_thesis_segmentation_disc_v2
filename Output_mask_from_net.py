# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:13:35 2023

@author: nohel
"""

from funkce_final import DataLoader,Unet,Postprocesing,zapis_kontury
import torch
import matplotlib.pyplot as plt
import numpy as np 
from skimage import measure, img_as_uint,img_as_ubyte
from skimage.io import imsave
import os


if __name__ == "__main__": 
    
    ZAPISOVAT=True
    ZAPISOVAT_KONTURY=True
    
    batch=1 
    threshold=0.5
    color_preprocesing="RGB"
    
    #Rozliseni_320_320_25px
    
    output_size=(int(288),int(288),int(3))    
    path_to_data="D:\Diploma_thesis_segmentation_disc_v2/Data_320_320_25px_preprocesing_all_database"
    #mereni UBMI
    #path_to_data="D:\Diploma_thesis_segmentation_disc_v2/Data_320_320_25px_preprocesing_UBMI_mereni"   
    
    # Cesta k naucenemu modelu
    path_to_model="D:\Diploma_thesis_segmentation_disc_v2/Data_320_320_25px_preprocesing_all_database/Naucene_modely/"
    
    # Disc and cup together
    segmentation_type="disc_cup"
    path_to_save_model = path_to_model + 'disc_and_cup_detection_25px_all_databases/'
    name_of_model='model_02_disc_cup_25px_all_modified_databases'
    net=Unet(out_size=2).cuda()  
    net.load_state_dict(torch.load(path_to_save_model+ name_of_model+ '.pth'))    
    
    path_to_extracted_data=path_to_data+ "/Vysledky_test_data_vystup_site/Data_320_320_25px/"  
    
    #%%
    #Rozliseni_480_480_35px
    '''
    output_size=(int(448),int(448),int(3))    
    path_to_data="D:\Diploma_thesis_segmentation_disc_v2/Data_480_480_35px_preprocesing_all_database"
    #mereni UBMI
    #path_to_data="D:\Diploma_thesis_segmentation_disc_v2/Data_480_480_35px_preprocesing_UBMI_mereni"   
    
    # Cesta k naucenemu modelu
    path_to_model="D:\Diploma_thesis_segmentation_disc_v2/Data_480_480_35px_preprocesing_all_database/Naucene_modely/"
    
    # Disc and cup together
    segmentation_type="disc_cup"
    path_to_save_model = path_to_model + 'disc_and_cup_detection_35px_all_databases/'
    name_of_model='model_01_disc_cup_35px_all_modified_databases'
    net=Unet(out_size=2).cuda()  
    net.load_state_dict(torch.load(path_to_save_model+ name_of_model+ '.pth')) 
    
    path_to_extracted_data=path_to_data+ "/Vysledky_test_data_vystup_site/Data_480_480_35px/"  
    
    '''
    #%%
    net.eval()    
    
    
    batch=1
    loader=DataLoader(split="Test",path_to_data=path_to_data,color_preprocesing=color_preprocesing,segmentation_type=segmentation_type,output_size=output_size)
    testloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    test_files_name=testloader.dataset.files_img
    #%%

    if (segmentation_type=="disc_cup"):
        
        for kk,(data,data_orig,lbl,img_full,img_orig_full, disc_orig,cup_orig,coordinates) in enumerate(testloader):
             data=data.cuda()
             lbl=lbl.cuda()
             
             output=net(data)
             output=torch.sigmoid(output)
             output=output.detach().cpu().numpy() > threshold
             
             lbl=lbl.detach().cpu().numpy()
             
             test_files_name_tmp=test_files_name[kk][95:]
             name_of_img=test_files_name_tmp[:-4]
             
             dir_pom=path_to_extracted_data+name_of_img
            
             isExist = os.path.exists(dir_pom)
             if not isExist:
                 os.makedirs(dir_pom)
                 
             
             pom_sourad=coordinates.detach().cpu().numpy()[0]  
                        
             output_mask_disc=np.zeros([disc_orig.shape[1],disc_orig.shape[2]])                        
             output_mask_cup=np.zeros([disc_orig.shape[1],disc_orig.shape[2]]) 
             
                        
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
                        
                
             disc_orig=disc_orig[0,:,:].detach().cpu().numpy() 
             cup_orig=cup_orig[0,:,:].detach().cpu().numpy() 
             
             orig_img=img_orig_full[0,:,:,:].detach().cpu().numpy()              
                                                    
             if ZAPISOVAT:
                 # Uložení masek ze segmentace
                 # Disc
                 imsave(dir_pom+'/' + name_of_img +"_Disc_output.png",img_as_ubyte(output_mask_disc))
                 imsave(dir_pom+'/' + name_of_img +"_Disc_orig.png",img_as_ubyte(disc_orig))
                                  
                 # Cup
                 imsave(dir_pom+'/' + name_of_img +"_Cup_output.png",img_as_ubyte(output_mask_cup))
                 imsave(dir_pom+'/' + name_of_img +"_Cup_orig.png",img_as_ubyte(cup_orig))      
                 
                 # Orig
                 imsave(dir_pom+'/' + name_of_img +".png",orig_img)     
    
    
    