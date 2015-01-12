addpath('mex');

% we provide two sequences "car" and "table"
example = '';
%example = 'car';

% load the two frames
im1 = im2double(imread([example '226.jpg']));
im2 = im2double(imread([example '286.jpg']));
im1 = im2double(uint8(imBicubic{2}));
im2 = im2double(uint8(imBicubic{1}));
% im1 = imresize(im1,0.5,'bicubic');
% im2 = imresize(im2,0.5,'bicubic');

% set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;

para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

% this is the core part of calling the mexed dll file for computing optical flow
% it also returns the time that is needed for two-frame estimation
tic;
[vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);
toc
[im_h, im_w, im_c] = size( im1 );
im = zeros(im_h, im_w*2, im_c);
for i = 1:im_c,
    im(:,:,i) = [im1(:,:,i), im2(:,:,i)];
end

xx = vx + repmat((1:im_w), [im_h, 1]);
yy = vy + repmat((1:im_h)', [1, im_w]);
xx =  round(xx); yy = round(yy);

figure;
hold on;
imshow( im );
for i = 1:20,
    x1 = randi(im_w);
    y1 = randi(im_h);
    x2 = vx(y1, x1)+x1+im_w;
    y2 = y1+vy(y1,x1);
    line([x1, x2], [y1, y2], 'Color', 'r', 'LineWidth', 2);
end
hold off;





