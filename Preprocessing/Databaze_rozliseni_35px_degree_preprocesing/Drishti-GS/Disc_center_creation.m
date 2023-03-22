clear all
close all
clc
path = pwd;
path= [path '\'];
%% Dristi-GS - centre of disc
images_file = dir([path 'Disc\expert1\*.png']);
N_train=length(images_file);
coordinates_dristi_GS=[];
%train
for i=1:N_train
    %expert 1
    mask=imread([images_file(i).folder '\' images_file(i).name ]);
    s = regionprops(mask,'centroid');
    coordinates_dristi_GS(i,1)=round(s.Centroid(1));
    coordinates_dristi_GS(i,2)=round(s.Centroid(2));
end
% figure
% imshow(mask)
% hold on
% stem(coordinates_dristi_GS(end,1),coordinates_dristi_GS(end,2))
% close all
name=['coordinates_dristi_GS.mat'];
save(name,"coordinates_dristi_GS")
%%
drishti_train_na_101=imread([path 'Images\drishti_train_na_101.png' ]);
figure
imshow(drishti_train_na_101)
hold on
stem(coordinates_dristi_GS(end,1),coordinates_dristi_GS(end,2))

