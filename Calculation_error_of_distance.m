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