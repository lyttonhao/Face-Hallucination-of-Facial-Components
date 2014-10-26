function [Cp, Cs] = Smp_patch_blur_FC(patch_size, num_patch, par)

addpath('Data');
addpath('Utilities');

load('Data/train3.mat');

hf1 = [-1,0,1];
vf1 = [-1,0,1]';

lf1 = zeros(3,3); lf1(1,1) = -1; lf1(3,3) = 1;
rf1 = zeros(3,3); rf1(1,3) = -1; rf1(3,1) = 1;
 
% second order gradient filters
hf2 = [1,0,-2,0,1];
vf2 = [1,0,-2,0,1]';

comp{1} = 49:68;        %mouth
comp{2} = 18:27;        %eyebrows
comp{3} = 37:48;        %eyes
comp{4} = 28:36;        %nose
comp{5} = 1:17;         %face edge

img_num = size(images_hr, 3)-1;
nper_img = zeros(1, img_num);

for i = 1 : img_num
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
for i = 1:6,
    Cp{i} = [];
    Cs{i} = [];
end

Mid = createIdx( size(HR_tr{i},1), size(HR_tr{i},2), patch_size );

for i = 1 : img_num    
   lm = landmarks(:,:,i);
   n = nper_img(i);
   [v1, h2] = data2patch(conv2(double(LR_Bicubic{i}), vf1, 'same'), conv2(double( LR_Bicubic{i}), hf2, 'same'), par);
   [h1 , v2] = data2patch( conv2(double( LR_Bicubic{i}), hf1, 'same'), conv2(double( LR_Bicubic{i}), vf2, 'same'), par);
   Tl = [h1;v1;h2;v2];
   
   [Th, ~] = data2patch( double( HR_tr{i} - LR_Bicubic{i}), conv2(double( HR_tr{i}), vf1, 'same'), par);
 
    idx = randperm(size(Th, 2));
    if size(Th, 2) < n,
        n = size(Th, 2)
    end
    Th1 = Th(:, idx(1:n));
    Tl1 = Tl(:, idx(1:n)); 
    pvars = var(Th1(1:patch_size*patch_size, :), 0, 1);
    idx = pvars > par.prunvar;
    Tl1 = Tl1(:, idx);
    Th1 = Th1(:, idx);
    Cs{1} = [Cs{1}, Th1];
    Cp{1} = [Cp{1}, Tl1];
    [im_h, im_w] = size( LR_Bicubic{i} );
    for j = 1:5,
        if j == 5, 
            tmp = logical(zeros( im_h, im_w ));
            for iter = 1:numel(comp{j}),
                y1 = floor(max(1, lm(comp{j}(iter),1))-2*par.lg);
                y2 = ceil(min(im_h, lm(comp{j},1))+2*par.lg);
                x1 = floor(max(1, lm(comp{j},2))-2*par.lg);
                x2 = ceil(min(im_w, lm(comp{j},2))+2*par.lg);
                tmp(x1:x2, y1:y2) = 1;
            end
            t = Mid(:);
            idx = t(tmp(:));
        else
            y1 = floor(min(lm(comp{j},1))-par.lg);
            y2 = ceil(max(lm(comp{j},1))+par.lg);
            x1 = floor(min(lm(comp{j},2))-par.lg);
            x2 = ceil(max(lm(comp{j},2))+par.lg);
            idx = Mid(x1:x2, y1:y2);
        end
      %  fprintf('%d %d %d %d\n', x1, x2, y1, y2);
        Tl1 = Tl(:, idx(idx > 0));
        Th1 = Th(:, idx(idx > 0));
        Cs{j+1} = [Cs{j+1}, Th1];
        Cp{j+1} = [Cp{j+1}, Tl1];
    end
    
    
end

for i = 1:6,
    Cp{i} = double(Cp{i});
    Cs{i} = double(Cs{i});
end

