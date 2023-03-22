clear all
close all
clc
path_to_data=[pwd '\UBMI_mereni_verze_2_sada2\']
path_to_crop_image='D:\Diploma_thesis_segmentation_disc_v2\Data_480_480_35px_preprocesing_UBMI_mereni_verze_2_sada02/';

%%
if ~exist([path_to_crop_image '\Images'], 'dir')
    mkdir([path_to_crop_image '\Images'])
end
if ~exist([path_to_crop_image '\Images_orig'], 'dir')
    mkdir([path_to_crop_image '\Images_orig'])
end
if ~exist([path_to_crop_image '\Disc'], 'dir')
    mkdir([path_to_crop_image '\Disc'])
end
if ~exist([path_to_crop_image '\Cup'], 'dir')
    mkdir([path_to_crop_image '\Cup'])
end
if ~exist([path_to_crop_image '\Fov'], 'dir')
    mkdir([path_to_crop_image '\Fov'])
end

if ~exist([path_to_crop_image '\Images_orig_full'], 'dir')
    mkdir([path_to_crop_image '\Images_orig_full'])
end

%%
%% UBMI_mereni
images_file = dir([path_to_data 'Images\*.png']);
images_orig_file = dir([path_to_data 'Images_orig\*.png']);
images_orig_full_file = dir([path_to_data 'UBMI_mereni_orig\Images\*.png']);
disc_file = dir([path_to_data 'Disc\*.png']);
cup_file = dir([path_to_data 'Cup\*.png']);
fov_file = dir([path_to_data 'Fov\*.png']);

coordinates_UBMI_mereni=load([path_to_data 'coordinates_UBMI_mereni.mat']);
coordinates=coordinates_UBMI_mereni.coordinates_UBMI_mereni;
num_of_img=length(images_file);

%%
for i=1:num_of_img
    image=imread([images_file(i).folder '\' images_file(i).name ]); 
    image_orig=imread([images_orig_file(i).folder '\' images_orig_file(i).name ]); 
    image_orig_full=imread([images_orig_full_file(i).folder '\' images_orig_full_file(i).name ]); 
    mask_disc=logical(imread([disc_file(i).folder '\' disc_file(i).name ]));  
    mask_cup=logical(imread([cup_file(i).folder '\' cup_file(i).name ]));  
    fov=imread([fov_file(i).folder '\' fov_file(i).name ]);

    imwrite(image,[path_to_crop_image 'Images\' images_file(i).name])
    imwrite(image_orig,[path_to_crop_image '\Images_orig\' images_file(i).name])
    imwrite(image_orig_full,[path_to_crop_image '\Images_orig_full\' images_file(i).name])
    imwrite(mask_disc,[path_to_crop_image '\Disc\' disc_file(i).name])
    imwrite(mask_cup,[path_to_crop_image '\Cup\' cup_file(i).name])
    imwrite(fov,[path_to_crop_image '\Fov\' fov_file(i).name])
end

%% Detection of centres in Test datasets
clear all
close all
clc
path_to_crop_image='D:\Diploma_thesis_segmentation_disc_v2\Data_480_480_35px_preprocesing_UBMI_mereni_verze_2_sada02/';
sigma=30;
size_of_erosion=40;
test_images_file = dir([path_to_crop_image 'Images\*.png']);
test_fov_file = dir([path_to_crop_image 'Fov\*.png']);
test_dics_file = dir([path_to_crop_image 'Disc\*.png']);
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


