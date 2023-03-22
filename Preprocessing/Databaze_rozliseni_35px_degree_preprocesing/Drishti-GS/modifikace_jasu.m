function [img_interp]=modifikace_jasu(img,fov,sigma,Num_tiles_param,ClipLimit)
S=strel('disk',1);
mask=imerode(fov,S);

dt=double(bwdist(mask));
sobel = [1 2 1; 0 0 0; -1 -2 -1];
sx = conv2(dt,sobel,'same')/4;
sy = conv2(dt,sobel','same')/4;

[xx,yy]=meshgrid(1:size(dt,2),1:size(dt,1));
xxx = xx - sy .* dt;
yyy = yy - sx .* dt;

img_interp(:,:,1) = interp2(img(:,:,1), xxx, yyy, 'linear',0);
img_interp(:,:,2) = interp2(img(:,:,2), xxx, yyy, 'linear',0);
img_interp(:,:,3) = interp2(img(:,:,3), xxx, yyy, 'linear',0);

G=imgaussfilt(img_interp,sigma,"Padding","symmetric");

img_interp = (img_interp - G) ./ G + 0.5;
img_interp(img_interp < 0) = 0;
img_interp(img_interp > 1) = 1;

img_interp_hsv=rgb2hsv(img_interp);

Num_tiles=round(size(img_interp,1)/Num_tiles_param);
Num_tiles(2)=round(size(img_interp,2)/Num_tiles_param);

img_interp_hsv(:,:,3) = adapthisteq(img_interp_hsv(:,:,3),'NumTiles',Num_tiles,'ClipLimit',ClipLimit);
img_interp=hsv2rgb(img_interp_hsv);
img_interp(:,:,1)=img_interp(:,:,1).*mask;
img_interp(:,:,2)=img_interp(:,:,2).*mask;
img_interp(:,:,3)=img_interp(:,:,3).*mask;
end