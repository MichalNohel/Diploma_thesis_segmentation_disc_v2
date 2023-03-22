%%
center=round(regionprops(disc_GT,"Centroid").Centroid);
c_x=center(1);
c_y=center(2);
contour_disc_GT = bwperim(disc_GT);
contour_disc_output_net = bwperim(disc_output_net);

[row1, col1] = find(contour_disc_GT)
[row2, col2] = find(contour_disc_output_net)
%%
r = linspace(0,200,201);             % line segment lenth max is 5 and min is 0
theta = -120*pi/180;             % angle of inclination
% starting point
x0=c_x;
y0=c_y;

x1=x0+ r* cos(theta);
y1=y0+ r * sin(theta);
y1=round(y1);
x1=round(x1);

ind= [25 75];
x1_h=x1(ind);
y1_h=y1(ind);
%%
pom=zeros(size(contour_disc_GT));
pom(contour_disc_GT)=1;
for i=1:length(x1)
    pom(y1(1,i),x1(1,i))=pom(y1(1,i),x1(1,i))+1;
end
imshow(pom,[])
%%
matice=[]
for i=1:length(x1)
    for m=1:length(row1)
        matice(i,m)=sqrt((y1(i)-row1(m))^2 + (x1(i)-col1(m))^2);
    end
end
%%
[u,v]=min(matice(:));
[I,M]=find(matice==u)

%%
x1(I)
y1(I)

%%
figure()
pom=zeros(898,1050,3);
pom(:,:,1)=contour_disc_output_net;
pom(:,:,2)=contour_disc_GT;
imshow(pom)
hold on
plot (x1,y1,x1_h,y1_h,'r+')

%%
radius1 = sqrt((row1 - c_x).^2 + (col1 - c_y).^2);
radius2 = sqrt((row2 - c_x).^2 + (col2 - c_y).^2);
% Define a range of angles to consider (e.g. every 5 degrees) 
angle_range = 0:5:360;
% Initialize an array to store the differences in distances for each angle 
dist_diff = zeros(size(angle_range));
% Loop over each angle 
for i = 1:length(angle_range)
    % Compute the x and y components of the vector for this angle 
    theta = angle_range(i) * pi / 180;
    % Convert to radians
    x = cos(theta);    
    y = sin(theta);
    % Compute the distance from the center for each circle along this vector 
    dist1 = x * radius1 + y * radius1;
    dist2 = x * radius2 + y * radius2;
    % Compute the difference in distances and store it in the dist_diff array 
    dist_diff(i) = mean(dist1) - mean(dist2);
end
% Plot the results 
plot(angle_range, dist_diff);
xlabel('Angle (degrees)');
ylabel('Distance difference');
%%
figure
subplot(1,2,1)
imshow(contour_disc_output_net)
subplot(1,2,2)
imshow(contour_disc_GT)
%%
pom=zeros(898,1050,3);
pom(:,:,1)=contour_disc_output_net;
pom(:,:,2)=contour_disc_GT;
imshow(pom)
