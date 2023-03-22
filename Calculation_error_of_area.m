function [error_disc,error_cup]= Calculation_error_of_area(disc_GT,disc_output_net,cup_GT,cup_output_net)
    area_disc_GT=regionprops(disc_GT,"Area").Area;
    area_disc_output_net=regionprops(disc_output_net,"Area").Area;
    area_cup_GT=regionprops(cup_GT,"Area").Area;
    area_cup_output_net=regionprops(cup_output_net,"Area").Area;
    error_disc=100*(abs(area_disc_GT-area_disc_output_net)/area_disc_GT);  
    error_cup=100*(abs(area_cup_GT-area_cup_output_net)/area_cup_GT);
end
