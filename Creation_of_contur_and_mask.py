# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:02:47 2022

@author: nohel
"""

from funkce_final import DataLoader,Unet,Postprocesing
import torch
import matplotlib.pyplot as plt
import numpy as np 
from skimage import measure, img_as_uint
from skimage.io import imsave
import os
import pandas
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


if __name__ == "__main__": 
    
    ZAPISOVAT=False
    ZAPISOVAT_KONTURY=True
    
    batch=1 
    threshold=0.5
    color_preprocesing="RGB"
    
    segmentation_type="disc_cup"
    #segmentation_type="cup"
    #segmentation_type="disc"
    
    output_size=(int(288),int(288),int(3))
    #path_to_data="D:\Diploma_thesis_segmentation_disc/Data_500_500"
    #path_to_data="D:\Diploma_thesis_segmentation_disc/Data_320_320_25px/Data_320_320_25px_all_database"
    #path_to_data="D:\Diploma_thesis_segmentation_disc\Data_320_320_25px\Data_320_320_25px_UBMI_mereni"
    
    # Normalizace
    #path_to_data="D:\Diploma_thesis_segmentation_disc\Data_320_320_25px_preprocesing_UBMI_mereni"
    path_to_data="D:\Diploma_thesis_segmentation_disc\Data_320_320_25px_preprocesing_all_database"
    
    
    
    path_to_extracted_data=path_to_data+ "/Results_closing_opening_40/"
    
    
    #mereni UBMI
    #path_to_data="D:\Diploma_thesis_segmentation_disc\Data_320_320_25px\Data_320_320_25px_UBMI_mereni"
    
    #Postsprocesing parameters
    min_size_of_optic_disk=1000
    size_of_disk_for_morphing=40
    #type_of_morphing='closing' 
    #type_of_morphing='openinig' 
    type_of_morphing='closing_opening' 
    #type_of_morphing='openinig_closing' 
    ploting=0    
    
    loader=DataLoader(split="Train",path_to_data=path_to_data,color_preprocesing=color_preprocesing,segmentation_type=segmentation_type,output_size=output_size)
    trainloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    train_files_name=trainloader.dataset.files_img
    
    batch=1
    loader=DataLoader(split="Test",path_to_data=path_to_data,color_preprocesing=color_preprocesing,segmentation_type=segmentation_type,output_size=output_size)
    testloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    test_files_name=testloader.dataset.files_img

    
    #net = Unet().cuda()   
    net=Unet(out_size=2).cuda()  
    #net.load_state_dict(torch.load(path_to_data + '/Naucene_modely/disc_and_cup_detection_25px_all_databases/model_01_RGB_detection_disc_cup_25px_all_databases.pth'))
    net.load_state_dict(torch.load(path_to_data + '/Naucene_modely/disc_and_cup_detection_25px_all_databases/model_01_RGB_disc_cup_25px_all_modified_databases.pth'))
    
    net.eval()
    
    
    
    
    for k,(data,lbl) in enumerate(trainloader):
        data=data.cuda()
        lbl=lbl.cuda()
        
        output=net(data)
        output=torch.sigmoid(output)
        output=output.detach().cpu().numpy() > threshold
        
        lbl=lbl.detach().cpu().numpy()
        
        
        train_files_name_tmp=train_files_name[k][103:]
        
        name_of_img=train_files_name_tmp[:-4]
        
        dir_pom=path_to_extracted_data+"Train/"+name_of_img
        
        isExist = os.path.exists(dir_pom)
        if not isExist:
            os.makedirs(dir_pom)
        
        
        plt.figure(figsize=[15,15])
        plt.subplot(2,3,1)        
        data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
        data_pom=np.transpose(data_pom,(1,2,0))
        plt.imshow(data_pom)   
        plt.title(name_of_img)
                    
        plt.subplot(2,3,2)    
        plt.imshow(lbl[0,0,:,:])
        plt.title('Orig maska - disc')
                    
        plt.subplot(2,3,3)    
        plt.imshow(output[0,0,:,:])
        plt.title('Output of net - disc')
                    
        plt.subplot(2,3,4)        
        data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
        data_pom=np.transpose(data_pom,(1,2,0))
        plt.imshow(data_pom) 
        plt.title(name_of_img)
                    
        plt.subplot(2,3,5)    
        plt.imshow(lbl[0,1,:,:])
        plt.title('Orig maska - cup')
                    
        plt.subplot(2,3,6)    
        plt.imshow(output[0,1,:,:])
        plt.title('Output of net - cup')
        
        if ZAPISOVAT:
            plt.savefig(dir_pom+'/' + name_of_img +"_vysledek.png")
                    
        plt.show() 
        
        if ZAPISOVAT:
            # Uložení masek ze segmentace
            # Disc
            imsave(dir_pom+'/' + name_of_img +"_Disc.png",img_as_uint(output[0,0,:,:]))
            # Cup
            imsave(dir_pom+'/' + name_of_img +"_Cup.png",img_as_uint(output[0,1,:,:]))
        
        
        # Vytvoření a uložení kontur
        data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
        data_pom=np.transpose(data_pom,(1,2,0))
        
        # Disc
        # Kontura výstupu sítě    
        
        disc_pom_output=output[0,0,:,:]
        disc_contours_output = measure.find_contours(disc_pom_output)
        if ZAPISOVAT:
            numpyData = {"array": disc_contours_output}
            encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
            with open(dir_pom+'/' + name_of_img + "_disc_contours_output.json", "w") as fp:
                    json.dump(encodedNumpyData, fp)
                    print("Done writing JSON data into .json file")
        
        # Kontura GT dat
        disc_pom_lbl=lbl[0,0,:,:]
        disc_contours_lbl = measure.find_contours(disc_pom_lbl)
        if ZAPISOVAT:
            numpyData = {"array": disc_contours_lbl}
            encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
            with open(dir_pom+'/' + name_of_img + "_disc_contours_lbl.json", "w") as fp:
                    json.dump(encodedNumpyData, fp)
                    print("Done writing JSON data into .json file")
        
        # Cup
        # Kontura výstupu sítě    
        
        cup_pom_output=output[0,1,:,:]
        cup_contours_output = measure.find_contours(cup_pom_output)
        if ZAPISOVAT:
            numpyData = {"array": cup_contours_output}
            encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
            with open(dir_pom+'/' + name_of_img + "_cup_contours_output.json", "w") as fp:
                    json.dump(encodedNumpyData, fp)
                    print("Done writing JSON data into .json file")
        
        # Kontura GT dat
        cup_pom_lbl=lbl[0,1,:,:]
        cup_contours_lbl = measure.find_contours(cup_pom_lbl)
        if ZAPISOVAT:
            numpyData = {"array": cup_contours_lbl}
            encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
            with open(dir_pom+'/' + name_of_img + "_cup_contours_lbl.json", "w") as fp:
                    json.dump(encodedNumpyData, fp)
                    print("Done writing JSON data into .json file")
        
        
        
        plt.figure(figsize=[15,15])
        plt.subplot(2,2,1)  
        plt.imshow(data_pom)         
        for contour in disc_contours_lbl:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
            
        plt.title('Disc - GT data')
            
        plt.subplot(2,2,2)  
        plt.imshow(data_pom)         
        for contour in disc_contours_output:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2) 
        plt.title('Disc - Output of net')  
        
        plt.subplot(2,2,3)  
        plt.imshow(data_pom)         
        for contour in cup_contours_lbl:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
            
        plt.title('Cup - GT data')
            
        plt.subplot(2,2,4)  
        plt.imshow(data_pom)         
        for contour in cup_contours_output:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2) 
        plt.title('Cup - Output of net')  
        
        if ZAPISOVAT:
            plt.savefig(dir_pom+'/' + name_of_img +"_kontury.png")
            
        if ZAPISOVAT_KONTURY:
            plt.savefig(path_to_extracted_data+'/Train_kontury/' + name_of_img +"_kontury.png")        
        
        plt.show()
        
        
    for kk,(data,lbl,img_orig,disc_orig,cup_orig,coordinates) in enumerate(testloader):
         data=data.cuda()
         lbl=lbl.cuda()
         
         output=net(data)
         output=torch.sigmoid(output)
         output=output.detach().cpu().numpy() > threshold
         
         lbl=lbl.detach().cpu().numpy()
         
         test_files_name_tmp=test_files_name[kk][97:]
         name_of_img=test_files_name_tmp[:-4]
         
         dir_pom=path_to_extracted_data+"Test/"+name_of_img
        
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
         plt.subplot(3,4,1)        
         im_pom=img_orig[0,:,:,:].detach().cpu().numpy()/255   
         plt.imshow(im_pom)   
         plt.title(name_of_img)
                        
         plt.subplot(3,4,2)    
         plt.imshow(disc_orig)
         plt.title('Orig maska - disc')
                        
         plt.subplot(3,4,3)    
         plt.imshow(output_mask_disc)
         plt.title('Output of net - disc')
                        
         plt.subplot(3,4,4)    
         plt.imshow(output_mask_disc_final)
         plt.title('Postprocesing')
                        
         plt.subplot(3,4,5)        
         data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
         data_pom=np.transpose(data_pom,(1,2,0))
         #data_pom=hsv2rgb(data_pom)
         plt.imshow(data_pom) 
         plt.title(name_of_img + "_cut")
                        
         plt.subplot(3,4,6)                       
         plt.imshow(lbl[0,0,:,:])
         plt.title('Orig maska - disc ')
         
         plt.subplot(3,4,7)                       
         plt.imshow(output[0,0,:,:])
         plt.title('Output of net - disc')
                        
         plt.subplot(3,4,8)                       
         plt.imshow(output_final)
         plt.title('Postprocesing')
                        
         plt.subplot(3,4,9)        
         data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
         data_pom=np.transpose(data_pom,(1,2,0))
         #data_pom=hsv2rgb(data_pom)
         plt.imshow(data_pom)  
         plt.title(name_of_img + "_cut")
                        
         plt.subplot(3,4,10)                       
         plt.imshow(lbl[0,1,:,:])
         plt.title('Orig maska - cup ')
                    
         plt.subplot(3,4,11)                       
         plt.imshow(output[0,1,:,:]) 
         plt.title('Output of net - cup')    
         
         if ZAPISOVAT:
             plt.savefig(dir_pom+'/' + name_of_img +"_vysledek.png")                  
                           
         plt.show() 
         
         if ZAPISOVAT:
             # Uložení masek ze segmentace
             # Disc
             imsave(dir_pom+'/' + name_of_img +"_Disc_all.png",img_as_uint(output_mask_disc))
             imsave(dir_pom+'/' + name_of_img +"_Disc_all_postprocesing.png",img_as_uint(output_mask_disc_final))
             
             imsave(dir_pom+'/' + name_of_img +"_Disc_vyrez.png",img_as_uint(output[0,0,:,:]))
             imsave(dir_pom+'/' + name_of_img +"_Disc_vyrez_postprocesing.png",img_as_uint(output_final))
             
             # Cup
             imsave(dir_pom+'/' + name_of_img +"_Cup_vyrez.png",img_as_uint(output[0,1,:,:]))
         
         
         # Vytvoření a uložení kontur
         data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
         data_pom=np.transpose(data_pom,(1,2,0))
        
         # Disc
         # Kontura výstupu sítě  
         
         disc_pom_output=output[0,0,:,:]
         disc_contours_output = measure.find_contours(disc_pom_output)
         if ZAPISOVAT:
             numpyData = {"array": disc_contours_output}
             encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
             with open(dir_pom+'/' + name_of_img + "_disc_contours_output.json", "w") as fp:
                 json.dump(encodedNumpyData, fp)
                 print("Done writing JSON data into .json file")
             
         disc_pom_output_postprocesing=output_final
         disc_contours_output_postprocesing = measure.find_contours(disc_pom_output_postprocesing)
         if ZAPISOVAT:
             numpyData = {"array": disc_contours_output}
             encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
             with open(dir_pom+'/' + name_of_img + "_disc_pom_output_postprocesing.json", "w") as fp:
                 json.dump(encodedNumpyData, fp)
                 print("Done writing JSON data into .json file")
             
         # Kontura GT dat
         disc_pom_lbl=lbl[0,0,:,:]
         disc_contours_lbl = measure.find_contours(disc_pom_lbl)
         if ZAPISOVAT:
             numpyData = {"array": disc_contours_lbl}
             encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
             with open(dir_pom+'/' + name_of_img + "_disc_contours_lbl.json", "w") as fp:
                 json.dump(encodedNumpyData, fp)
                 print("Done writing JSON data into .json file")
    
         # Cup
         # Kontura výstupu sítě    
        
         cup_pom_output=output[0,1,:,:]
         cup_contours_output = measure.find_contours(cup_pom_output)
         if ZAPISOVAT:
             numpyData = {"array": cup_contours_output}
             encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
             with open(dir_pom+'/' + name_of_img + "_cup_contours_output.json", "w") as fp:
                 json.dump(encodedNumpyData, fp)
                 print("Done writing JSON data into .json file")
        
         # Kontura GT dat
         cup_pom_lbl=lbl[0,1,:,:]
         cup_contours_lbl = measure.find_contours(cup_pom_lbl)
         if ZAPISOVAT:
             numpyData = {"array": cup_contours_lbl}
             encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
             with open(dir_pom+'/' + name_of_img + "_cup_contours_lbl.json", "w") as fp:
                 json.dump(encodedNumpyData, fp)
                 print("Done writing JSON data into .json file")
         
            
         plt.figure(figsize=[15,15])
         plt.subplot(2,3,1)  
         plt.imshow(data_pom)         
         for contour in disc_contours_lbl:
             plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
            
         plt.title('Disc - GT data')
            
         plt.subplot(2,3,2)  
         plt.imshow(data_pom)         
         for contour in disc_contours_output:
             plt.plot(contour[:, 1], contour[:, 0], linewidth=2) 
         plt.title('Disc - Output of net')  
         
         plt.subplot(2,3,3)  
         plt.imshow(data_pom)         
         for contour in disc_contours_output_postprocesing:
             plt.plot(contour[:, 1], contour[:, 0], linewidth=2) 
         plt.title('Disc - Output of net postprocesing')  
        
         plt.subplot(2,3,4)  
         plt.imshow(data_pom)         
         for contour in cup_contours_lbl:
             plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
            
         plt.title('Cup - GT data')
            
         plt.subplot(2,3,5)  
         plt.imshow(data_pom)         
         for contour in cup_contours_output:
             plt.plot(contour[:, 1], contour[:, 0], linewidth=2) 
         plt.title('Cup - Output of net')  
         
         if ZAPISOVAT:
             plt.savefig(dir_pom+'/' + name_of_img +"_kontury.png")
             
         if ZAPISOVAT_KONTURY:
             plt.savefig(path_to_extracted_data+'/Test_kontury/' + name_of_img +"_kontury.png")
             
         plt.show()
    
    
    
    
    
    
    
    
    
    
    
    