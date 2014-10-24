function [Cp, Cs] = Smp_patch_blur(patch_size, num_patch, par)


Cs = [];
Cp = [];


addpath('Data');
addpath('Utilities');
load('Data/train3.mat');

%img_path = 'Data\Face_Training1\';
%type = '*.jpg';

%img_dir = dir( fullfile(img_path, type) );
%img_num = length( img_dir ) / 10;
%nper_img = zeros(1, img_num);


hf1 = [-1,0,1];
vf1 = [-1,0,1]';

lf1 = zeros(3,3); lf1(1,1) = -1; lf1(3,3) = 1;
rf1 = zeros(3,3); rf1(1,3) = -1; rf1(3,1) = 1;
 
% second order gradient filters
hf2 = [1,0,-2,0,1];
vf2 = [1,0,-2,0,1]';

img_num = size(images_hr, 3)-1;
nper_img = zeros(1, img_num);
 

for i = 1 : img_num
   % imHR               =   imread(fullfile( img_path, img_dir(i).name)) ;
     imHR  =  images_hr(:,:,i);
    [im_h, im_w, ch]       =   size(imHR);
    if ch == 3,
        imHR = double( rgb2ycbcr( imHR ));
    end
    imHR = double(imHR(:,:,1));
    [im_h, im_w]       =   size(imHR);
    nper_img(i) = prod(size(imHR));
    
    [im_h, im_w,dummy] = size(imHR);
    im_h = floor((im_h )/par.nFactor)*par.nFactor ;
    im_w = floor((im_w )/par.nFactor)*par.nFactor ;
    imHR=imHR(1:im_h,1:im_w,:);
    
    psf                =     par.psf;             % The simulated PSF
    imLR = Blur('fwd', imHR, psf);
    imLR           =   imLR(1 : par.nFactor : im_h, 1 : par.nFactor : im_w);
  
    [CX CY] = meshgrid(1 : im_w, 1:im_h);
    [X Y] = meshgrid(1:par.nFactor:im_w, 1:par.nFactor:im_h);
    imBicubic  =   interp2(X, Y, imLR, CX, CY, 'spline');
    
    %fprintf('PSNR of Bicubic Training Image: %2.2f \n', csnr(imBicubic, imHR, 5, 5));
    HR_tr{i} = imHR;
    LR_Bicubic{i} = imBicubic;
      
end

nper_img = floor(nper_img*num_patch/sum(nper_img));


psf = fspecial('gauss', par.win+2, 2.2);
for i = 1 : img_num    
    n = nper_img(i);
   [v1, h2] = data2patch(conv2(double(LR_Bicubic{i}), vf1, 'same'), conv2(double( LR_Bicubic{i}), hf2, 'same'), par);
   [h1 , v2] = data2patch( conv2(double( LR_Bicubic{i}), hf1, 'same'), conv2(double( LR_Bicubic{i}), vf2, 'same'), par);
   Tl = [h1;v1;h2;v2];
   
 %   [h1 , v1] = data2patch( conv2(double( HR_tr{i}), hf1, 'same'), conv2(double( HR_tr{i}), vf1, 'same'), par);
 %   [l1 , r1] = data2patch( conv2(double( HR_tr{i}), lf1, 'same'), conv2(double( HR_tr{i}), rf1, 'same'), par);
 %   Th = [h1; v1; l1; r1];
    [Th, ~] = data2patch( double( HR_tr{i} - LR_Bicubic{i}), conv2(double( HR_tr{i}), vf1, 'same'), par);
 
    idx = randperm(size(Th, 2));
    if size(Th, 2) < n,
        n = size(Th, 2)
    end
    Th = Th(:, idx(1:n));
    Tl = Tl(:, idx(1:n)); 
    pvars = var(Th(1:patch_size*patch_size, :), 0, 1);
    idx = pvars > par.prunvar;
    Tl = Tl(:, idx);
    Th = Th(:, idx);
    
    
    Cs = [Cs, Th];
    Cp = [Cp, Tl];
end

Cp = double(Cp);
Cs = double(Cs);

