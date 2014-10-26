% LLE method

clear all;clc;

addpath('Flann')
addpath('Data');
addpath('Utilities');
addpath('lib');
addpath('spams-matlab');
addpath('spams-matlab/build');    
cc = 0;
re = [];

im_path = 'Data/Face_Testing1/';
im_dir = dir( fullfile(im_path, '*.png') );
im_num = length( im_dir );
%load('Data/train3.mat');

comp{1} = 49:68;        %mouth
comp{2} = 18:27;        %eyebrows
comp{3} = 37:48;        %eyes
comp{4} = 28:36;        %nose

for pp = [9],
    for ss = [100000],
     

%lambda      = 0.15;         % sparsity regularization
patch_size = pp;
nSmp        = ss;       % number of patches to sample

par.nFactor = 8;
par.win = patch_size;
par.step = 1;
par.prunvar = 5;
par.lg = 5;
par.psf =   fspecial('gauss', 7, 1.6);              % The simulated PSF


% randomly sample image patches
[Cp, Cs] = Smp_patch_blur_FC( patch_size, nSmp, par);
%[TU, BU, U, Cp, Cs] = Smp_patch_blur_NMF( patch_size, nSmp, par);
%[PL, PH, Cp, Cs] = Smp_patch_blur_PCA( patch_size, nSmp, par);
%[U, V, Ih, Cp, Cs] = Smp_patch_blur_NMF_GL( patch_size, nSmp, par);
%save(['Cps',num2str(par.nFactor)], 'Cp', 'Cs');
%load(['Cps',num2str(par.nFactor)]);
%load BU;

%[Cp, V_pca] = PCA(Cp);

for i = 1:5,
    dataset = Cp{i};  
    build_params.target_precision = 1;  
    build_params.build_weight = 0.5; 
    build_params.memory_weight = 0; 
    [index{i}, parameters{i}] = flann_build_index(dataset, build_params);
 %   index{i} = index1;
 %   parameters{i} = parameters1;
end

param.iter = 100;
param.L = 30;
for lambda = [ 0.15],
     param.lambda = lambda; 
for nnn = [9],
    tot = 0;
    for img = 1:3,

    imHR = imread( fullfile(im_path, im_dir(img).name) );
  %  imHR = images_hr(:,:,100);
   % imHR = imread('Data/Face_Testing1/1.png');
   % load('Data/lm1.mat', 'landmark_test'); 
 %   lm = landmark_test;
    
    [im_h, im_w,dummy] = size(imHR);
    im_h = floor((im_h )/par.nFactor)*par.nFactor ;
    im_w = floor((im_w )/par.nFactor)*par.nFactor ;
    imHR=imHR(1:im_h,1:im_w,:);
    
   
    ori_HR = imHR;
    imMid = zeros( size(imHR) );
  %  fprintf('%s\n', im_dir(img).name);
    if (size(imHR, 3) == 3)
        imHR = double(rgb2ycbcr( imHR ) );
        im_cb = imresize( imHR(:,:,2), 1/par.nFactor, 'Bicubic' );
        im_cr = imresize( imHR(:,:,3), 1/par.nFactor, 'Bicubic' );
        im_cb = imresize( im_cb, par.nFactor, 'Bicubic' );
        im_cr = imresize( im_cr, par.nFactor, 'Bicubic' );
        imMid(:,:,2) = im_cb;
        imMid(:,:,3) = im_cr;
    end
    
  %   im_cb = imresize( imHR(:,:,2), par.nFactor, 'Bicubic' );
  %   im_cr = imresize( imHR(:,:,3), par.nFactor, 'Bicubic' );
 
    imHR = double(imHR(:,:,1));
    
    psf                =     par.psf;            
    imLR = Blur('fwd', imHR, psf);
    imLR           =   imLR(1 : par.nFactor : im_h, 1 : par.nFactor : im_w);
    
    imMid(:,:,1) = imresize( imLR, par.nFactor, 'bicubic');
    try
        [bs, posemap] = F2_ReturnLandmarks( imMid, 'mi' );
    catch err
         disp('Error: The UCI algorithm does not detect a face.');
         continue;
    end
    lm = F4_ConvertBStoMultiPieLandmarks(bs(1));

  %  imLR = imHR;
    
  %  imBicubic = imresize(imHR, par.nFactor, 'bicubic');
  %  [im_h, im_w] = size(imBicubic);
 
  
    [CX CY] = meshgrid(1 : im_w, 1:im_h);
    [X Y] = meshgrid(1:par.nFactor:im_w, 1:par.nFactor:im_h);
    imBicubic  =   interp2(X, Y, imLR, CX, CY, 'spline');
    
%    imNMF = get_imNMF(TU, BU, U, imLR, im_h, im_w);
   % imNMF = get_imPCA(PL, PH, imLR, im_h, im_w);
 %   imNMF = get_imNMF_GL(U, V, Ih, imLR, im_h, im_w);
   % c = (BU) \ imLR(:);
   % imNMF = U * c;
   % imNMF = reshape(imNMF, [im_h, im_w]);
 %   imNMF = imBicubic;
 
    
    fprintf('Bicubic: %2.2f \n', csnr(imHR, imBicubic, 0, 0));
  %  fprintf('NMF: %2.2f \n', csnr(imHR, imNMF, 0, 0));
    
    hf1 = [-1,0,1];
    vf1 = [-1,0,1]';


    lf1 = zeros(3,3); lf1(1,1) = -1; lf1(3,3) = 1;
    rf1 = zeros(3,3); rf1(1,3) = -1; rf1(3,1) = 1;
 
    hf2 = [1,0,-2,0,1];
    vf2 = [1,0,-2,0,1]';
    
    %Mid = createIdx( size(imHR,1), size(imHR,2), patch_size );
    Type = ones(size(imHR));
      
    for j = 1:5,
        if j == 5, 
            tmp = logical(zeros( im_h, im_w ));
            for iter = 1:numel(comp{j}),
                y1 = floor(max(1, lm(comp{j}(iter),1))-2*par.lg);
                y2 = ceil(min(im_h, lm(comp{j},1))+2*par.lg);
                x1 = floor(max(1, lm(comp{j},2))-2*par.lg);
                x2 = ceil(min(im_w, lm(comp{j},2))+2*par.lg);
                Type(x1:x2, y1:y2) = j+1;
            end
        else
            y1 = floor(min(lm(comp{j},1))-par.lg);
            y2 = ceil(max(lm(comp{j},1))+par.lg);
            x1 = floor(min(lm(comp{j},2))-par.lg);
            x2 = ceil(max(lm(comp{j},2))+par.lg);
            Type(x1:x2, y1:y2) = j+1;
        end
    end
    Type = Type(1:size(imHR,1)-patch_size+1, 1:size(imHR,2)-patch_size+1);
    Type = Type(:);

   [v2 h1] = data2patch(conv2(double(imBicubic), vf2, 'same'), conv2(double(imBicubic), hf1, 'same'), par);
   [v1, h2] = data2patch(conv2(double(imBicubic), vf1, 'same'), conv2(double( imBicubic), hf2, 'same'), par);
   Tl = [h1;v1;h2;v2];
  
   vec_patches = Tl;
   nn = nnn;

   
   testset = double(vec_patches);
 %  testset = V_pca' * testset;

    for i = 1:5,
           [idx{i},dst{i}] = flann_search(index{i},testset,nn,parameters{i});
        %[idx1,dst] = flann_search(index{i},testset,nn,parameters{i});
        %idx{i} = idx1;
    end

   Q = zeros(nn,nn);
   output = zeros(patch_size*patch_size, size(testset,2));
   weight = zeros(size(testset,2), nn);
   for ii = 1:size(testset, 2),
        t = Type(ii);
                Ip = testset(:, ii);
                Ipk = zeros(size(testset,1), nn);
                Isk = zeros(size(Cs{t},1), nn);
            for i=1:nn
                Ipk(:, i) = Cp{t}(:,idx{t}(i,ii));
                Isk(:, i) = Cs{t}(:,idx{t}(i,ii));
            end
      
            Coeff = ( Ipk'*Ipk + lambda*eye(nn) ) \ Ipk' * Ip;
       %     Coeff = mexLasso(Ip, Ipk, param);
       
            Is = Isk * Coeff;          
            
            output(:, ii) = Is;    
        end

    %    [h1, v1] = patch2data1([output(1:patch_size*patch_size, :);output(patch_size*patch_size+1:patch_size*patch_size*2, :)], im_h, im_w, 1,par.win, par.step);
    %    [l1, r1] = patch2data1([output(patch_size*patch_size*2+1:patch_size*patch_size*3, :);output(patch_size*patch_size*3+1:patch_size*patch_size*4, :)], im_h, im_w, 1,par.win, par.step);
        [output, ~] = patch2data([output;output], im_h, im_w, 1,par.win, par.step);
 
        output = output +  imBicubic ;
        result = output;
     %   save ('Mat/cman4.mat', 'h1', 'v1', 'l1','r1');

        %save('Mat/out.mat', 'output');
      %  e_ori = conv2( double(imHR), hf1, 'same');
      %  fprintf('%d %d %d\nPSNR of Semi-Coupled DL: %2.2f \n',pp, ss, nnn, csnr(e_ori, h1, 5, 5));
     %   fprintf('%d %d %d\nPSNR of Semi-Coupled DL: %2.2f \n',pp, ss, nnn, csnr(imHR, output, pp, pp));

        %result = func_improve_NL_im(imLR, imHR, imBicubic, h1, v1, l1, r1 );
        
      

       fprintf('%d %d %d %s Result: %2.3f \n',pp, ss, nnn, im_dir(img).name, csnr(imHR, result, 0, 0));
       for j = 1:4,
            y1 = floor(min(lm(comp{j},1))-par.lg);
            y2 = ceil(max(lm(comp{j},1))+par.lg);
            x1 = floor(min(lm(comp{j},2))-par.lg);
            x2 = ceil(max(lm(comp{j},2))+par.lg);
            fprintf('%2.3f ', csnr(imHR(x1:x2,y1:y2), result(x1:x2,y1:y2), 0, 0));
       end
       fprintf('\n');
       
       tot = tot + csnr(imHR, result, 0, 0);
        
        im_rgb = zeros(size(imBicubic));
        im_rgb(:,:,1) = result;
        imB = zeros(size(imBicubic));
        imB(:,:,1) = imBicubic;
        if size(ori_HR, 3) == 3,
            im_rgb(:,:,2) = im_cb;
            im_rgb(:,:,3) = im_cr;
            im_rgb = ycbcr2rgb( uint8( im_rgb ) );
            imB(:,:,2) = im_cb;
            imB(:,:,3) = im_cr;
            imB = ycbcr2rgb( uint8( imB ) );
        end
    
      %  savefile( imLR, ori_HR, im_rgb, result, h1, v1, imB, im_dir(img).name);
        imwrite(uint8(im_rgb), ['Result/NEFC_s', num2str(par.nFactor), '_', num2str(pp),'-', num2str(nnn),'_',im_dir(img).name]);
        imwrite(uint8(imB), ['Result/s', num2str(par.nFactor), '_bicubic', im_dir(img).name]);
        imwrite(uint8(imLR), ['Result/s', num2str(par.nFactor), '_LR', im_dir(img).name]);
    end
   fprintf('%f %d %d  %d, average %2.2f\n',lambda, pp, ss, nnn, tot/im_num);
   
end
end
    for i = 1:5,
        flann_free_index(index{i});% free the memory      
    end

    end
end

%save('re.mat','re');
%imshow(int8(h1),[])
%output = uint8(output);
%imwrite(output, '1.jpg');


%improve(h1, v1);