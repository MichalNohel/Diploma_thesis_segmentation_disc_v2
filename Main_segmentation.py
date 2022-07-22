# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:39:07 2022

@author: nohel
"""



import numpy as np
from funkce_final import DataLoader, Unet, dice_loss, dice_coefficient,Postprocesing
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from IPython.display import clear_output
from skimage.color import hsv2rgb, xyz2rgb
import torchvision.transforms.functional as TF

if __name__ == "__main__": 
    #Parameters
    lr=0.001
    epochs=30
    batch=16
    threshold=0.5
    color_preprocesing="RGB"
    
    segmentation_type="disc_cup"
    #segmentation_type="cup"
    #segmentation_type="disc"
    
    output_size=(int(288),int(288),int(3))
    #path_to_data="D:\Diploma_thesis_segmentation_disc/Data_500_500"
    #path_to_data="D:\Diploma_thesis_segmentation_disc/Data_640_640_55px_all_database"
    #path_to_data="D:\Diploma_thesis_segmentation_disc/Data_320_320_25px/Data_320_320_25px_all_database"
    path_to_data="D:\Diploma_thesis_segmentation_disc\Data_320_320_25px_preprocesing_all_database"
    
    

    #Postsprocesing parameters
    min_size_of_optic_disk=1000
    size_of_disk_for_morphing=40
    #type_of_morphing='closing' 
    #type_of_morphing='openinig' 
    type_of_morphing='closing_opening' 
    #type_of_morphing='openinig_closing' 
    ploting=0    
    
    
    loader=DataLoader(split="Train",path_to_data=path_to_data,color_preprocesing=color_preprocesing,segmentation_type=segmentation_type,output_size=output_size)
    trainloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=True)
    
    batch=1
    loader=DataLoader(split="Test",path_to_data=path_to_data,color_preprocesing=color_preprocesing,segmentation_type=segmentation_type,output_size=output_size)
    testloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    
    if (segmentation_type=="disc_cup"):
        net=Unet(out_size=2).cuda()  
        
        optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=1e-8)
        sheduler=StepLR(optimizer,step_size=7, gamma=0.1) #decreasing of learning rate
        
        train_loss = []
        test_loss = []
        
        train_dice_disc = []
        test_dice_disc = []
       
        train_dice_cup = []
        test_dice_cup = []
        
        test_dice_final_disc = []
        test_dice_final_cup = []
        
        
        
        it_test=-1
        it_train=-1
        
        
        for epoch in range(epochs):
            loss_tmp = []
            
            dice_tmp_disc = []
            dice_tmp_final_disc = []
            
            dice_tmp_cup = []
            dice_tmp_final_cup = []
            
            print('epoch number ' + str(epoch+1))
            
            for k,(data,lbl) in enumerate(trainloader):
                it_train+=1
                data=data.cuda()
                lbl=lbl.cuda() 
                
                net.train()
                output=net(data)
                
                output=torch.sigmoid(output)
                
                #loss = -torch.mean(lbl*torch.log(output)+(1-lbl)*torch.log(1-output))
                #loss = -torch.mean(20*lbl*torch.log(output)+1*(1-lbl)*torch.log(1-output)) #vahovaní
                loss=dice_loss(lbl,output)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                lbl_mask=lbl.detach().cpu().numpy()
                output_mask=output.detach().cpu().numpy() > threshold

                loss_tmp.append(loss.cpu().detach().numpy()) 
                
                dice_tmp_disc.append(dice_coefficient(output_mask[:,0,:,:],lbl_mask[:,0,:,:]))
                dice_tmp_cup.append(dice_coefficient(output_mask[:,1,:,:],lbl_mask[:,1,:,:]))    
                
                
                if (it_train % 10==0):
                    clear_output()
                    plt.figure(figsize=[10,10])
                    plt.plot(loss_tmp,label='train loss')
                    plt.plot(dice_tmp_disc,label='dice_disk')
                    plt.plot(dice_tmp_cup,label='dice_cup')
                    plt.legend(loc="upper left")
                    plt.title('train')
                    plt.show()
                    
                    plt.figure(figsize=[10,10])
                    plt.subplot(2,3,1)        
                    data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
                    data_pom=np.transpose(data_pom,(1,2,0))
                    plt.imshow(data_pom)        
                    
                    plt.subplot(2,3,2)    
                    plt.imshow(lbl_mask[0,0,:,:])
                    
                    plt.subplot(2,3,3)    
                    plt.imshow(output_mask[0,0,:,:])
                    
                    plt.subplot(2,3,4)        
                    data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
                    data_pom=np.transpose(data_pom,(1,2,0))
                    plt.imshow(data_pom) 
                    
                    plt.subplot(2,3,5)    
                    plt.imshow(lbl_mask[0,1,:,:])
                    
                    plt.subplot(2,3,6)    
                    plt.imshow(output_mask[0,1,:,:])
                    
                    plt.show()                       
                    print('Train - iteration ' + str(it_train))
              
            train_loss.append(np.mean(loss_tmp))
            
            train_dice_disc.append(np.mean(dice_tmp_disc)) 
            train_dice_cup.append(np.mean(dice_tmp_cup)) 
            
            
            
            loss_tmp = []            
            dice_tmp_disc = []
            dice_tmp_final_disc = []            
            dice_tmp_cup = []
            dice_tmp_final_cup = []    
    
           
            for kk,(data,lbl,img_orig,disc_orig,cup_orig,coordinates) in enumerate(testloader):
                with torch.no_grad():
                    it_test+=1
                    net.eval()  
                    data=data.cuda()
                    lbl=lbl.cuda()
                    output=net(data)
                    output=torch.sigmoid(output)
                    
                    loss=dice_loss(lbl,output)                    
                    output=output.detach().cpu().numpy() > threshold
                    
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
                    
                    
                    output_final=Postprocesing(output[0,0,:,:],min_size_of_optic_disk,type_of_morphing,size_of_disk_for_morphing,ploting)
                    
                    output_mask_disc_final[x_start:x_start+output_size[0],y_start:y_start+output_size[0]]=output_final
                    output_mask_disc_final=output_mask_disc_final.astype(bool)
                    
                    disc_orig=disc_orig[0,:,:].detach().cpu().numpy() 
                    cup_orig=cup_orig[0,:,:].detach().cpu().numpy() 
                    
                    
                    loss_tmp.append(loss.cpu().detach().numpy())                          
                    
                    dice_tmp_disc.append(dice_coefficient(output_mask_disc,disc_orig))
                    dice_tmp_cup.append(dice_coefficient(output_mask_cup,cup_orig))
                    
                    dice_tmp_final_disc.append(dice_coefficient(output_mask_disc_final,disc_orig))
                    
                    
                    
                    if (it_test % 10==0):
                        clear_output()
                        plt.figure(figsize=[10,10])
                        #plt.plot(acc_tmp,label='test acc')
                        plt.plot(loss_tmp,label='test loss')
                        plt.plot(dice_tmp_disc,label='dice_disc')
                        plt.plot(dice_tmp_cup,label='dice_cup')
                        plt.plot(dice_tmp_final_disc,label='dice_disc_final')
                        plt.legend(loc="upper left")
                        plt.title('test')
                        plt.show()
                                                
                        plt.figure(figsize=[10,10])
                        plt.subplot(3,4,1)        
                        im_pom=img_orig[0,:,:,:].detach().cpu().numpy()/255   
                        plt.imshow(im_pom)   
                        plt.title(str(kk))
                        
                        plt.subplot(3,4,2)    
                        plt.imshow(disc_orig)
                        plt.title('Orig maska')
                        
                        plt.subplot(3,4,3)    
                        plt.imshow(output_mask_disc)
                        plt.title('Vystup sítě')
                        
                        plt.subplot(3,4,4)    
                        plt.imshow(output_mask_disc_final)
                        plt.title('Postprocesing')
                        
                        plt.subplot(3,4,5)        
                        data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
                        data_pom=np.transpose(data_pom,(1,2,0))
                        #data_pom=hsv2rgb(data_pom)
                        plt.imshow(data_pom)  
                        
                        plt.subplot(3,4,6)                       
                        plt.imshow(lbl[0,0,:,:].detach().cpu().numpy())
                    
                        plt.subplot(3,4,7)                       
                        plt.imshow(output[0,0,:,:])
                        
                        plt.subplot(3,4,8)                       
                        plt.imshow(output_final)
                        
                        plt.subplot(3,4,9)        
                        data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
                        data_pom=np.transpose(data_pom,(1,2,0))
                        #data_pom=hsv2rgb(data_pom)
                        plt.imshow(data_pom)  
                        
                        plt.subplot(3,4,10)                       
                        plt.imshow(lbl[0,1,:,:].detach().cpu().numpy())
                    
                        plt.subplot(3,4,11)                       
                        plt.imshow(output[0,1,:,:])                       
                        
                        plt.show() 
                        
                        print('Test - iteration ' + str(it_test))
                    
                 
            test_loss.append(np.mean(loss_tmp))
            
            test_dice_disc.append(np.mean(dice_tmp_disc)) 
            test_dice_cup.append(np.mean(dice_tmp_cup)) 

            test_dice_final_disc.append(np.mean(dice_tmp_final_disc)) 
            
                        
            sheduler.step()
            
            clear_output()
            plt.figure(figsize=[10,10])
            plt.plot(train_loss,label='train loss')
            plt.plot(train_dice_disc,label='dice_disc')
            plt.plot(train_dice_cup,label='dice_cup')
            plt.legend(loc="upper left")
            plt.title('train')
            plt.show()
            
            clear_output()
            plt.figure(figsize=[10,10])
            plt.plot(test_loss,label='train loss')
            plt.plot(test_dice_disc,label='dice_disc')
            plt.plot(test_dice_cup,label='dice_cup')
            plt.plot(test_dice_final_disc,label='dice_disc_final')
            plt.legend(loc="upper left")
            plt.title('test')
            plt.show()
            
            '''   
        
            plt.figure(figsize=[10,10])
            plt.subplot(1,3,1)        
            im_pom=img_orig[0,:,:,:].detach().cpu().numpy()   
            #im_pom=np.transpose(im_pom,(1,2,0))
            #im_pom=hsv2rgb(im_pom)
            plt.imshow(im_pom)        
            
            plt.subplot(1,3,2)    
            #plt.imshow(lbl[0,0,:,:].detach().cpu().numpy())
            plt.imshow(disc_orig)
            
            plt.subplot(1,3,3)    
            #plt.imshow(output[0,0,:,:].detach().cpu().numpy()>threshold)
            plt.imshow(output_mask_final)
            plt.show() 
            '''
        
    else:        
        net=Unet().cuda()    
        optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=1e-8)
        sheduler=StepLR(optimizer,step_size=7, gamma=0.1) #decreasing of learning rate
        
        train_loss = []
        test_loss = []
        train_dice = []
        test_dice = []
        train_dice_final= []
        test_dice_final = []
        
        it_test=-1
        it_train=-1
        
        
        for epoch in range(epochs):
            #acc_tmp = []
            loss_tmp = []
            dice_tmp = []
            dice_tmp_final = []
            print('epoch number ' + str(epoch+1))
            
            for k,(data,lbl) in enumerate(trainloader):
                it_train+=1
                data=data.cuda()
                lbl=lbl.cuda() 
                
                net.train()
                output=net(data)
                
                output=torch.sigmoid(output)
                
                #loss = -torch.mean(lbl*torch.log(output)+(1-lbl)*torch.log(1-output))
                #loss = -torch.mean(20*lbl*torch.log(output)+1*(1-lbl)*torch.log(1-output)) #vahovaní
                loss=dice_loss(lbl,output)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                lbl_mask=lbl.detach().cpu().numpy()
                output=output.detach().cpu().numpy() > threshold
                
                loss_tmp.append(loss.cpu().detach().numpy())            
                dice_tmp.append(dice_coefficient(output,lbl_mask))
                
                output_final=Postprocesing(output[0,0,:,:],min_size_of_optic_disk,type_of_morphing,size_of_disk_for_morphing,ploting) 
                dice_tmp_final.append(dice_coefficient(output_final,lbl_mask[0,0,:,:]))
                
                if (it_train % 10==0):
                    clear_output()
                    plt.figure(figsize=[10,10])
                    #plt.plot(acc_tmp,label='train acc')
                    plt.plot(loss_tmp,label='train loss')
                    plt.plot(dice_tmp,label='dice')
                    plt.plot(dice_tmp_final,label='dice_final')
                    plt.legend(loc="upper left")
                    plt.title('train')
                    plt.show()
                    
                    plt.figure(figsize=[10,10])
                    plt.subplot(1,4,1)        
                    data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
                    data_pom=np.transpose(data_pom,(1,2,0))
                    plt.imshow(data_pom)        
                    
                    plt.subplot(1,4,2)    
                    plt.imshow(lbl_mask[0,0,:,:])
                    
                    plt.subplot(1,4,3)    
                    plt.imshow(output[0,0,:,:])
                    
                    plt.subplot(1,4,4)    
                    plt.imshow(output_final)
                    plt.show()
                    
                    
                    print('Train - iteration ' + str(it_train))
                
            train_loss.append(np.mean(loss_tmp))
            #train_acc.append(np.mean(acc_tmp))
            train_dice.append(np.mean(dice_tmp)) 
            train_dice_final.append(np.mean(dice_tmp_final)) 
            
            
            #acc_tmp = []
            loss_tmp = []
            dice_tmp =  []
            dice_tmp_final = []
            for kk,(data,mask,img_orig,disc_orig,cup_orig,coordinates) in enumerate(testloader):
                with torch.no_grad():
                    it_test+=1
                    net.eval()  
                    data=data.cuda()
                    output=net(data)
                    output=torch.sigmoid(output)
                    output=output.detach().cpu().numpy() > threshold
                    
                    pom_sourad=coordinates.detach().cpu().numpy()[0]               
                    output_mask=np.zeros([disc_orig.shape[1],disc_orig.shape[2]]) 
                    output_mask_final=np.zeros([disc_orig.shape[1],disc_orig.shape[2]])  
                    
                    if (pom_sourad[1]-int(output_size[0]/2)<0):
                        x_start=0
                    elif((pom_sourad[1]+int(output_size[0]/2))>output_mask.shape[0]):
                        x_start=output_mask.shape[0]-output_size[0]
                    else:
                        x_start=pom_sourad[1]-int(output_size[0]/2)
                        
                    if (pom_sourad[0]-int(output_size[0]/2)<0):
                        y_start=0
                    elif((pom_sourad[0]+int(output_size[0]/2))>output_mask.shape[1]):
                        y_start=output_mask.shape[1]-output_size[0]
                    else:
                        y_start=pom_sourad[0]-int(output_size[0]/2)
                        
                        
    
                    output_mask[x_start:x_start+output_size[0],y_start:y_start+output_size[0]]=output[0,0,:,:]
                    output_mask=output_mask.astype(bool)      
                    if (segmentation_type=="disc"):
                        lbl=disc_orig
                    else:
                        lbl=cup_orig
                    
                    loss=dice_loss(lbl,TF.to_tensor(output_mask))
                    
                    output_final=Postprocesing(output[0,0,:,:],min_size_of_optic_disk,type_of_morphing,size_of_disk_for_morphing,ploting) 
                    output_mask_final[x_start:x_start+output_size[0],y_start:y_start+output_size[0]]=output_final
                    output_mask_final=output_mask_final.astype(bool)
                    
                    lbl=lbl[0,:,:].detach().cpu().numpy() 
                    
                    #acc=np.mean((output_mask==disc_orig))
                    #acc_tmp.append(acc)
                    loss_tmp.append(loss.cpu().detach().numpy())                
                    dice_tmp.append(dice_coefficient(output_mask,lbl))
                    dice_tmp_final.append(dice_coefficient(output_mask_final,lbl))
                    if (it_test % 10==0):
                        clear_output()
                        plt.figure(figsize=[10,10])
                        #plt.plot(acc_tmp,label='test acc')
                        plt.plot(loss_tmp,label='test loss')
                        plt.plot(dice_tmp,label='dice')
                        plt.plot(dice_tmp_final,label='dice_final')
                        plt.legend(loc="upper left")
                        plt.title('test')
                        plt.show()
                        
                        plt.figure(figsize=[10,10])
                        plt.subplot(2,4,1)        
                        im_pom=img_orig[0,:,:,:].detach().cpu().numpy()/255   
                        plt.imshow(im_pom)   
                        plt.title(str(kk))
                        
                        plt.subplot(2,4,2)    
                        plt.imshow(lbl)
                        plt.title('Orig maska')
                        
                        plt.subplot(2,4,3)    
                        plt.imshow(output_mask)
                        plt.title('Vystup sítě')
                        
                        plt.subplot(2,4,4)    
                        plt.imshow(output_mask_final)
                        plt.title('Postprocesing')
                        
                        plt.subplot(2,4,5)        
                        data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
                        data_pom=np.transpose(data_pom,(1,2,0))
                        #data_pom=hsv2rgb(data_pom)
                        plt.imshow(data_pom)  
                        
                        plt.subplot(2,4,6)                       
                        plt.imshow(mask[0,0,:,:].detach().cpu().numpy())
                    
                        plt.subplot(2,4,7)                       
                        plt.imshow(output[0,0,:,:])
                        
                        plt.subplot(2,4,8)                       
                        plt.imshow(output_final)
                        
                        plt.show() 
                        
                        print('Test - iteration ' + str(it_test))
                    
                    
            test_loss.append(np.mean(loss_tmp))
            #test_acc.append(np.mean(acc_tmp))
            test_dice.append(np.mean(dice_tmp)) 
            test_dice_final.append(np.mean(dice_tmp_final)) 
            
            
            
            sheduler.step()
            
            clear_output()
            
            plt.figure(figsize=[10,10])            
            plt.plot(train_loss,label='train loss')
            plt.plot(train_dice,label='dice')
            plt.plot(train_dice_final,label='dice_final')
            plt.legend(loc="upper left")
            plt.title('train')
            plt.show()
            
            clear_output()
            plt.figure(figsize=[10,10])            
            plt.plot(test_loss,label='test loss')
            plt.plot(test_dice,label='dice')
            plt.plot(test_dice_final,label='dice_final')
            plt.legend(loc="upper left")
            plt.title('test')
            plt.show()
        
            plt.figure(figsize=[10,10])
            plt.subplot(1,3,1)        
            im_pom=img_orig[0,:,:,:].detach().cpu().numpy()   
            #im_pom=np.transpose(im_pom,(1,2,0))
            #im_pom=hsv2rgb(im_pom)
            plt.imshow(im_pom)        
            
            plt.subplot(1,3,2)    
            #plt.imshow(lbl[0,0,:,:].detach().cpu().numpy())
            plt.imshow(lbl)
            
            plt.subplot(1,3,3)    
            #plt.imshow(output[0,0,:,:].detach().cpu().numpy()>threshold)
            plt.imshow(output_mask_final)
            plt.show() 
            
    #torch.save(net, 'model_01.pth')
    torch.save(net.state_dict(), 'model_01_RGB_disc_cup_25px_all_modified_databases.pth')
    
        
    
    
            
            
            
            
        
        
        
    
    
    