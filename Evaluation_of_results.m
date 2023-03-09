clear all
close all
clc
%% Evaluace masek pro rozlišení 35px
path_to_data='D:\Diploma_thesis_segmentation_disc_v2\Data_480_480_35px_preprocesing_all_database\Vysledky_test_data_vystup_site\Data_480_480_35px/';
images_file = dir(path_to_data);
images_file(1:2)=[];
i=200;
image=logical(imread([images_file(i).folder '\' images_file(i).name '\' images_file(i).name '.png']));
disc_GT=logical(imread([images_file(i).folder '\' images_file(i).name '\' images_file(i).name '_Disc_orig.png']));
disc_output_net=logical(imread([images_file(i).folder '\' images_file(i).name '\' images_file(i).name '_Disc_output.png']));
cup_GT=logical(imread([images_file(i).folder '\' images_file(i).name '\' images_file(i).name '_Cup_orig.png']));
cup_output_net=logical(imread([images_file(i).folder '\' images_file(i).name '\' images_file(i).name '_Cup_output.png']));


%% Function for error of area (Chyba plochy)
[error_disc,error_cup]= Calculation_error_of_area(disc_GT,disc_output_net,cup_GT,cup_output_net)


%% Function for error of distance (Chyba délky)
[coordinates_disc_GT,coordinates_disc_net,coordinates_cup_GT,coordinates_cup_net]= Get_intersection(disc_GT,disc_output_net,cup_GT,cup_output_net) 
%%
[abs_error_disc,abs_error_cup,rel_error_disc,rel_error_cup]= Calculation_error_of_distance(disc_GT,disc_output_net,cup_GT,cup_output_net)
%%
disp(['pruměrna absolutni chyba disku je ' num2str(mean(abs_error_disc)) ' px'])
disp(['pruměrna absolutni chyba cupu je ' num2str(mean(abs_error_cup)) ' px'])
disp(['pruměrna relativní chyba disku je ' num2str(mean(rel_error_disc)) ' %'])
disp(['pruměrna relativní chyba cupu je ' num2str(mean(rel_error_cup)) ' %'])

%% Function for error of area (Chyba plochy)
function [error_disc,error_cup]= Calculation_error_of_area(disc_GT,disc_output_net,cup_GT,cup_output_net)
    area_disc_GT=regionprops(disc_GT,"Area").Area;
    area_disc_output_net=regionprops(disc_output_net,"Area").Area;
    area_cup_GT=regionprops(cup_GT,"Area").Area;
    area_cup_output_net=regionprops(cup_output_net,"Area").Area;
    error_disc=100*(abs(area_disc_GT-area_disc_output_net)/area_disc_GT);  
    error_cup=100*(abs(area_cup_GT-area_cup_output_net)/area_cup_GT);
end


%% Function for error of distance (Chyba délky)
function [abs_error_disc,abs_error_cup,rel_error_disc,rel_error_cup]= Calculation_error_of_distance(disc_GT,disc_output_net,cup_GT,cup_output_net)
    [coordinates_disc_GT,coordinates_disc_net,coordinates_cup_GT,coordinates_cup_net]= Get_intersection(disc_GT,disc_output_net,cup_GT,cup_output_net);
    
    center_disc=round(regionprops(disc_GT,"Centroid").Centroid);
    center_cup=round(regionprops(cup_GT,"Centroid").Centroid);
    
    for i=1:length(coordinates_disc_GT)
        abs_error_disc(i)=sqrt((coordinates_disc_GT(i,1)-coordinates_disc_net(i,1))^2+(coordinates_disc_GT(i,2)-coordinates_disc_net(i,2))^2);
        rel_error_disc(i)=100*(abs_error_disc(i)/sqrt((coordinates_disc_GT(i,1)-center_disc(1))^2+(coordinates_disc_GT(i,2)-center_disc(2))^2));      
    end

    for i=1:length(coordinates_cup_GT)
        abs_error_cup(i)=sqrt((coordinates_cup_GT(i,1)-coordinates_cup_net(i,1))^2+(coordinates_cup_GT(i,2)-coordinates_cup_net(i,2))^2);
        rel_error_cup(i)=100*(abs_error_cup(i)/sqrt((coordinates_cup_GT(i,1)-center_cup(1))^2+(coordinates_cup_GT(i,2)-center_cup(2))^2));
    end
end
%%
function [coordinates_disc_GT,coordinates_disc_net,coordinates_cup_GT,coordinates_cup_net]= Get_intersection(disc_GT,disc_output_net,cup_GT,cup_output_net)    
    center=round(regionprops(disc_GT,"Centroid").Centroid);
    c_x=center(1);
    c_y=center(2);
    contour_disc_GT = bwperim(disc_GT);
    contour_disc_output_net = bwperim(disc_output_net);
    contour_cup_GT = bwperim(cup_GT);
    contour_cup_output_net = bwperim(cup_output_net);

    [row_disc_GT, col_disc_GT] = find(contour_disc_GT);
    [row_disc_net, col_disc_net] = find(contour_disc_output_net);
    [row_cup_GT, col_cup_GT] = find(contour_cup_GT);
    [row_cup_net, col_cup_net] = find(contour_cup_output_net);

    coordinates_disc_GT=[];
    coordinates_disc_net=[];
    coordinates_cup_GT=[];
    coordinates_cup_net=[];
%%
    x0=c_x;
    y0=c_y;
    r = linspace(0,200,201);   
    for theta_indx=0:5:355
        theta = -theta_indx*pi/180; 
        x1=x0+ r* cos(theta);
        y1=y0+ r * sin(theta);
        y1=round(y1);
        x1=round(x1);
        matice_disc_GT=[];    
        for i=1:length(x1)
            for m=1:length(row_disc_GT)
                matice_disc_GT(i,m)=sqrt((y1(i)-row_disc_GT(m))^2 + (x1(i)-col_disc_GT(m))^2);
            end
        end
        matice_disc_net=[];
        for i=1:length(x1)
            for m=1:length(row_disc_net)
                matice_disc_net(i,m)=sqrt((y1(i)-row_disc_net(m))^2 + (x1(i)-col_disc_net(m))^2);
            end
        end
        matice_cup_GT=[];
        for i=1:length(x1)
            for m=1:length(row_cup_GT)
                matice_cup_GT(i,m)=sqrt((y1(i)-row_cup_GT(m))^2 + (x1(i)-col_cup_GT(m))^2);
            end
        end
        matice_cup_net=[];
        for i=1:length(x1)
            for m=1:length(row_cup_net)
                matice_cup_net(i,m)=sqrt((y1(i)-row_cup_net(m))^2 + (x1(i)-col_cup_net(m))^2);
            end
        end
        %%
        [value_GT,~]=min(matice_disc_GT(:));
        [Index_GT,~]=find(matice_disc_GT==value_GT);
        coordinates_disc_GT(end+1,:)=[x1(Index_GT(1)),y1(Index_GT(1))];
        %%
        [value_net,~]=min(matice_disc_net(:));
        [Index_net,~]=find(matice_disc_net==value_net);
        coordinates_disc_net(end+1,:)=[x1(Index_net(1)),y1(Index_net(1))];       
        %%
        [value_GT,~]=min(matice_cup_GT(:));
        [Index_GT,~]=find(matice_cup_GT==value_GT);
        coordinates_cup_GT(end+1,:)=[x1(Index_GT(1)),y1(Index_GT(1))];
        %%
        [value_net,~]=min(matice_cup_net(:));
        [Index_net,~]=find(matice_cup_net==value_net);
        coordinates_cup_net(end+1,:)=[x1(Index_net(1)),y1(Index_net(1))];
    end  
end
