clear all
close all
clc
path_to_data=pwd;
percentage_number_test=0.2;
path_to_extract_image='D:\Diploma_thesis_segmentation_disc_v2\Data_35px_nn_unet/';

%% Dristi-GS train  - Expert 1
images_file = dir([path_to_data '\Drishti-GS\Images\*.png']);
disc_file = dir([path_to_data '\Drishti-GS\Disc\expert1\*.png']);
cup_file = dir([path_to_data '\Drishti-GS\Cup\expert1\*.png']);

pom=52; % split to test and train dataset

creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
sound(y/10,Fs)
%% REFUGE_train
images_file = dir([path_to_data '\REFUGE\Images\Train\*.png']);
disc_file = dir([path_to_data '\REFUGE\Disc\Train\*.png']);
cup_file = dir([path_to_data '\REFUGE\Cup\Train\*.png']);

pom=0; % split to test and train dataset
creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
sound(y/10,Fs)
%% REFUGE_Validation
images_file = dir([path_to_data '\REFUGE\Images\Validation\*.png']);
disc_file = dir([path_to_data '\REFUGE\Disc\Validation\*.png']);
cup_file = dir([path_to_data '\REFUGE\Cup\Validation\*.png']);

pom=0; % split to test and train dataset
creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
sound(y/10,Fs)
%% REFUGE_Test
images_file = dir([path_to_data '\REFUGE\Images\Test\*.png']);
disc_file = dir([path_to_data '\REFUGE\Disc\Test\*.png']);
cup_file = dir([path_to_data '\REFUGE\Cup\Test\*.png']);

pom=401; % split to test and train dataset
creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
sound(y/10,Fs)
%% Riga - Bin Rushed
images_file = dir([path_to_data '\RIGA\Images\BinRushed\*.png']);
disc_file = dir([path_to_data '\RIGA\Disc\BinRushed\expert1\*.png']);
cup_file = dir([path_to_data '\RIGA\Cup\BinRushed\expert1\*.png']);
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
sound(y/10,Fs)

%% Riga - Magrabia
images_file = dir([path_to_data '\RIGA\Images\Magrabia\*.png']);
disc_file = dir([path_to_data '\RIGA\Disc\Magrabia\expert1\*.png']);
cup_file = dir([path_to_data '\RIGA\Cup\Magrabia\expert1\*.png']);
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
sound(y/10,Fs)

%% Riga - MESSIDOS
images_file = dir([path_to_data '\RIGA\Images\MESSIDOS\*.png']);
disc_file = dir([path_to_data '\RIGA\Disc\MESSIDOS\expert1\*.png']);
cup_file = dir([path_to_data '\RIGA\Cup\MESSIDOS\expert1\*.png']);
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
sound(y/10,Fs)
%% RIM-ONE - Glaucoma
images_file = dir([path_to_data '\RIM-ONE\Images\Glaucoma\*.png']);
disc_file = dir([path_to_data '\RIM-ONE\Disc\Glaucoma\*.png']);
cup_file = dir([path_to_data '\RIM-ONE\Cup\Glaucoma\*.png']);

num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
load chirp
sound(y/10,Fs)

%% RIM-ONE -  Healthy
images_file = dir([path_to_data '\RIM-ONE\Images\Healthy\*.png']);
disc_file = dir([path_to_data '\RIM-ONE\Disc\Healthy\*.png']);
cup_file = dir([path_to_data '\RIM-ONE\Cup\Healthy\*.png']);

num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
load chirp
sound(y/10,Fs)

%% UoA_DR - Healthy
images_file = dir([path_to_data '\UoA_DR\Images\Healthy\*.png']);
disc_file = dir([path_to_data '\UoA_DR\Disc\Healthy\*.png']);
cup_file = dir([path_to_data '\UoA_DR\Cup\Healthy\*.png']);

num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
sound(y/10,Fs)

%% HRF - 
images_file = dir([path_to_data '\HRF\Images\*.png']);
disc_file = dir([path_to_data '\HRF\Disc\*.png']);
cup_file = dir([path_to_data '\HRF\Cup\*.png']);
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
load chirp
sound(y/10,Fs)

%%
function []= creation_of_nn_unet_dataset(images_file,disc_file,cup_file,pom,path_to_extract_image)
    num_of_img=length(images_file);
    for i=1:num_of_img
        %expert 1
        image=imread([images_file(i).folder '\' images_file(i).name ]); 
        mask_disc=logical(imread([disc_file(i).folder '\' disc_file(i).name ]));  
        mask_cup=logical(imread([cup_file(i).folder '\' cup_file(i).name ]));  

        label=uint8(zeros(size(mask_disc)));
        label(mask_disc)=1;
        label(mask_cup)=2;
                
        if i>=pom
            imwrite(image,[path_to_extract_image 'training\input\' images_file(i).name])
            imwrite(label,[path_to_extract_image 'training\output\' images_file(i).name])            
        else
            imwrite(image,[path_to_extract_image 'testing\input\' images_file(i).name])
            imwrite(label,[path_to_extract_image 'testing\output\' images_file(i).name]) 
        end
    end
end
