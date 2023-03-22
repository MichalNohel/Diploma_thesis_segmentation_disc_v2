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
