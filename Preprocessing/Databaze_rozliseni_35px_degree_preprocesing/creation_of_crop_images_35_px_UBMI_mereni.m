clear all
close all
clc
path_to_data=pwd;
output_image_size=[480,480];
sigma=30;
size_of_erosion=40;
% percentage_number_test=0.2;
percentage_number_test=1.1;
% percentage_number_test=0;

path_to_crop_image='D:\Diploma_thesis_segmentation_disc_v2\Data_480_480_35px_preprocesing_UBMI_mereni/';

%%

if ~exist([path_to_crop_image '\Test'], 'dir')
    mkdir([path_to_crop_image '\Test'])
end
if ~exist([path_to_crop_image '\Train'], 'dir')
    mkdir([path_to_crop_image '\Train'])
end
if ~exist([path_to_crop_image '\Test\Cup'], 'dir')
    mkdir([path_to_crop_image '\Test\Cup'])
end
if ~exist([path_to_crop_image '\Test\Disc'], 'dir')
    mkdir([path_to_crop_image '\Test\Disc'])
end
if ~exist([path_to_crop_image '\Test\Fov'], 'dir')
    mkdir([path_to_crop_image '\Test\Fov'])
end
if ~exist([path_to_crop_image '\Test\Images'], 'dir')
    mkdir([path_to_crop_image '\Test\Images'])
end
if ~exist([path_to_crop_image '\Test\Images_orig'], 'dir')
    mkdir([path_to_crop_image '\Test\Images_orig'])
end
if ~exist([path_to_crop_image '\Train\Cup_crop'], 'dir')
    mkdir([path_to_crop_image '\Train\Cup_crop'])
end
if ~exist([path_to_crop_image '\Train\Disc_crop'], 'dir')
    mkdir([path_to_crop_image '\Train\Disc_crop'])
end
if ~exist([path_to_crop_image '\Train\Images_crop'], 'dir')
    mkdir([path_to_crop_image '\Train\Images_crop'])
end
if ~exist([path_to_crop_image '\Train\Images_orig_crop'], 'dir')
    mkdir([path_to_crop_image '\Train\Images_orig_crop'])
end



%% UBMI_mereni
images_file = dir([path_to_data '\UBMI_mereni\Images\*.png']);
images_orig_file = dir([path_to_data '\UBMI_mereni\Images_orig\*.png']);
disc_file = dir([path_to_data '\UBMI_mereni\Disc\*.png']);
cup_file = dir([path_to_data '\UBMI_mereni\Cup\*.png']);
fov_file = dir([path_to_data '\UBMI_mereni\Fov\*.png']);

path_to_data_pom=[path_to_data '\UBMI_mereni\'];

coordinates_UBMI_mereni=load([path_to_data '\UBMI_mereni\coordinates_UBMI_mereni.mat']);
coordinates=coordinates_UBMI_mereni.coordinates_UBMI_mereni;
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
%%
creation_of_crop_images(output_image_size,images_orig_file,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom,path_to_data_pom)
load chirp
sound(y/10,Fs)



%% Detection of centres in Test datasets
clear all
close all
clc

path_to_crop_image='D:\Diploma_thesis_segmentation_disc_v2\Data_480_480_35px_preprocesing_UBMI_mereni/';

sigma=30;
size_of_erosion=40;
test_images_file = dir([path_to_crop_image 'Test\Images\*.png']);
test_fov_file = dir([path_to_crop_image 'Test\Fov\*.png']);
test_dics_file = dir([path_to_crop_image 'Test\Disc\*.png']);
num_of_img=length(test_images_file);
Disc_centres_test=[];
Accuracy_of_detec=[];
%% Detekce s chybami
for i=1:num_of_img
    image=imread([test_images_file(i).folder '\' test_images_file(i).name ]); 
    fov=imread([test_fov_file(i).folder '\' test_fov_file(i).name ]);
    mask_disc=logical(imread([test_dics_file(i).folder '\' test_dics_file(i).name ])); 
    [center_new] = Detection_of_disc(image,fov,sigma,size_of_erosion);
    Disc_centres_test(i,1)=center_new(1);
    Disc_centres_test(i,2)=center_new(2);
    if mask_disc(center_new(2),center_new(1))==1
        Accuracy_of_detec(i)=1;
    else
        Accuracy_of_detec(i)=0;
    end
end
accuracy=sum(Accuracy_of_detec)/length(Accuracy_of_detec)

%% save of test discs centers with mistakes
Disc_centres_test=Disc_centres_test-1;
save('Disc_centres_test_UBMI_mereni_with_mistakes.mat','Disc_centres_test')
%% Detekce bez chyb
for i=1:num_of_img
    image=imread([test_images_file(i).folder '\' test_images_file(i).name ]); 
    fov=imread([test_fov_file(i).folder '\' test_fov_file(i).name ]);
    mask_disc=logical(imread([test_dics_file(i).folder '\' test_dics_file(i).name ])); 
    [center_new] = Detection_of_disc(image,fov,sigma,size_of_erosion);
    Disc_centres_test(i,1)=center_new(1);
    Disc_centres_test(i,2)=center_new(2);

    if mask_disc(Disc_centres_test(i,2),Disc_centres_test(i,1))~=1
        s = regionprops(mask_disc,'centroid');
        Disc_centres_test(i,1)=round(s.Centroid(1));
        Disc_centres_test(i,2)=round(s.Centroid(2));
    end

    if mask_disc(Disc_centres_test(i,2),Disc_centres_test(i,1))==1
        Accuracy_of_detec(i)=1;
    else
        Accuracy_of_detec(i)=0;
        
    end
end
accuracy=sum(Accuracy_of_detec)/length(Accuracy_of_detec)

%% save of test discs centers without mistakes
Disc_centres_test=Disc_centres_test-1;
save('Disc_centres_test_UBMI_mereni_correct.mat','Disc_centres_test')
%% Functions
function[center_new] = Detection_of_disc(image,fov,sigma,velikost_erodovani)
image=rgb2xyz(im2double(image));
image=rgb2gray(image);
BW=imerode(fov,strel('disk',velikost_erodovani));
vertical_len=size(BW,1);
step=round(vertical_len/15);
BW(1:step,:)=0;
BW(vertical_len-step:vertical_len,:)=0;
image(~BW)=0;
img_filt=imgaussfilt(image,sigma);
img_filt(~BW)=0;
[r, c] = find(img_filt == max(img_filt(:)));
center_new(1)=c;
center_new(2)=r;
end

function [output_crop_image, output_crop_image_orig, output_mask_disc,output_mask_cup]=Crop_image(image,image_orig,mask_disc,mask_cup,output_image_size,center_new)
    size_in_img=size(image);
    x_half=round(output_image_size(1)/2);
    y_half=round(output_image_size(2)/2);
    if ((center_new(2)-x_half)<0)
        x_start=1;
    elseif ((center_new(2)+x_half)>size_in_img(1))
        x_start=size_in_img(1)-output_image_size(1);
    else
        x_start=center_new(2)-x_half;
        if x_start==0
            x_start=1;
        end
    end

    if ((center_new(1)-y_half)<0)
        y_start=1;
    elseif ((center_new(1)+y_half)>size_in_img(2))
        y_start=size_in_img(2)-output_image_size(2);
    else
        y_start=center_new(1)-y_half;
        if y_start==0
            y_start=1;
        end
    end

    output_crop_image=image(x_start:x_start+output_image_size(1)-1,y_start:y_start+output_image_size(2)-1,:);
    output_crop_image_orig=image_orig(x_start:x_start+output_image_size(1)-1,y_start:y_start+output_image_size(2)-1,:);
    output_mask_disc=mask_disc(x_start:x_start+output_image_size(1)-1,y_start:y_start+output_image_size(2)-1);
    output_mask_cup=mask_cup(x_start:x_start+output_image_size(1)-1,y_start:y_start+output_image_size(2)-1);
end

function []= creation_of_crop_images(output_image_size,images_orig_file,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom,path_to_data_pom)
    num_of_img=length(images_file);
    for i=1:num_of_img
        %expert 1
        image=imread([images_file(i).folder '\' images_file(i).name ]); 
        image_orig=imread([images_orig_file(i).folder '\' images_orig_file(i).name ]); 
        mask_disc=logical(imread([disc_file(i).folder '\' disc_file(i).name ]));  
        mask_cup=logical(imread([cup_file(i).folder '\' cup_file(i).name ]));  
        fov=imread([fov_file(i).folder '\' fov_file(i).name ]);

        [center_new] = Detection_of_disc(image,fov,sigma,size_of_erosion);
        if mask_disc(center_new(2),center_new(1))~=1
            center_new(1)=coordinates(i,1);
            center_new(2)=coordinates(i,2);
        end
        [output_crop_image, output_crop_image_orig, output_mask_disc,output_mask_cup]=Crop_image(image,image_orig,mask_disc,mask_cup,output_image_size,center_new);
        %zapis crop obrazků 
        imwrite(output_crop_image,[path_to_data_pom 'Images_crop\' images_file(i).name])
        imwrite(output_crop_image_orig,[path_to_data_pom 'Images_orig_crop\' images_file(i).name])
        imwrite(output_mask_disc,[path_to_data_pom 'Disc_crop\' disc_file(i).name])
        imwrite(output_mask_cup,[path_to_data_pom 'Cup_crop\' cup_file(i).name])

        if i>=pom
            imwrite(output_crop_image,[path_to_crop_image 'Train\Images_crop\' images_file(i).name])
            imwrite(output_crop_image_orig,[path_to_crop_image 'Train\Images_orig_crop\' images_file(i).name])
            imwrite(output_mask_disc,[path_to_crop_image 'Train\Disc_crop\' disc_file(i).name])
            imwrite(output_mask_cup,[path_to_crop_image 'Train\Cup_crop\' cup_file(i).name])
        else
            imwrite(image,[path_to_crop_image 'Test\Images\' images_file(i).name])
            imwrite(image_orig,[path_to_crop_image 'Test\Images_orig\' images_file(i).name])
            imwrite(mask_disc,[path_to_crop_image 'Test\Disc\' disc_file(i).name])
            imwrite(mask_cup,[path_to_crop_image 'Test\Cup\' cup_file(i).name])
            imwrite(fov,[path_to_crop_image 'Test\Fov\' fov_file(i).name])
        end
    end
end

    