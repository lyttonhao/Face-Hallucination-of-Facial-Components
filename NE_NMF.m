% LLE method

clear all;clc;

addpath('Flann')
addpath('Data');
addpath('Utilities');
addpath('spams-matlab');
addpath('spams-matlab/build');    
cc = 0;
re = [];

im_path = 'Data/Face_Testing1/';
im_dir = dir( fullfile(im_path, '*.jpg') );
im_num = length( im_dir );


for pp = [9],
    for ss = [100000],
     

%lambda      = 0.15;         % sparsity regularization
patch_size = pp;
nSmp        = ss;       % number of patches to sample

par.nFactor = 4;
par.win = patch_size;
par.step = 1;
par.prunvar = 5;
par.psf =   fspecial('gauss', 7, 1.6);              % The simulated PSF


% randomly sample image patches
%[Cp, Cs] = Smp_patch_blur( patch_size, nSmp, par);
%[TU, BU, U, Cp, Cs] = Smp_patch_blur_NMF( patch_size, nSmp, par);
[PL, PH, Cp, Cs] = Smp_patch_blur_PCA( patch_size, nSmp, par);
%[U, V, Ih, Cp, Cs] = Smp_patch_blur_NMF_GL( patch_size, nSmp, par);
save('Cps', 'Cp', 'Cs');
%load Cps;
%load BU;

%[Cp, V_pca] = PCA(Cp);

dataset = Cp;  
build_params.target_precision = 1;  
build_params.build_weight = 0.5; 
build_params.memory_weight = 0; 
[index, parameters] = flann_build_index(dataset, build_params);
param.iter = 100;
param.L = 30;
for lambda = [ 0.15],
     param.lambda = lambda; 
for nnn = [9],
    tot = 0;
    for img = 1:im_num,

    imHR = imread( fullfile(im_path, im_dir(img).name) );
    
    [im_h, im_w,dummy] = size(imHR);
    im_h = floor((im_h )/par.nFactor)*par.nFactor ;
    im_w = floor((im_w )/par.nFactor)*par.nFactor ;
    imHR=imHR(1:im_h,1:im_w,:);
    
   
    ori_HR = imHR;
  %  fprintf('%s\n', im_dir(img).name);
    if (size(imHR, 3) == 3)
        imHR = double(rgb2ycbcr( imHR ) );
        im_cb = imresize( imHR(:,:,2), 1/par.nFactor, 'Bicubic' );
        im_cr = imresize( imHR(:,:,3), 1/par.nFactor, 'Bicubic' );
        im_cb = imresize( im_cb, par.nFactor, 'Bicubic' );
        im_cr = imresize( im_cr, par.nFactor, 'Bicubic' );
    end
    
  %   im_cb = imresize( imHR(:,:,2), par.nFactor, 'Bicubic' );
  %   im_cr = imresize( imHR(:,:,3), par.nFactor, 'Bicubic' );
 
    imHR = double(imHR(:,:,1));
    
    psf                =     par.psf;            
    imLR = Blur('fwd', imHR, psf);
    imLR           =   imLR(1 : par.nFactor : im_h, 1 : par.nFactor : im_w);
  %  imLR = imHR;
    
  %  imBicubic = imresize(imHR, par.nFactor, 'bicubic');
  %  [im_h, im_w] = size(imBicubic);
 
  
    [CX CY] = meshgrid(1 : im_w, 1:im_h);
    [X Y] = meshgrid(1:par.nFactor:im_w, 1:par.nFactor:im_h);
    imBicubic  =   interp2(X, Y, imLR, CX, CY, 'spline');
    
%    imNMF = get_imNMF(TU, BU, U, imLR, im_h, im_w);
    imNMF = get_imPCA(PL, PH, imLR, im_h, im_w);
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
 
   [v2 h1] = data2patch(conv2(double(imNMF), vf2, 'same'), conv2(double(imNMF), hf1, 'same'), par);
   [v1, h2] = data2patch(conv2(double(imNMF), vf1, 'same'), conv2(double( imNMF), hf2, 'same'), par);
   Tl = [h1;v1;h2;v2];
  
   vec_patches = Tl;
   nn = nnn;

   testset = double(vec_patches);
 %  testset = V_pca' * testset;

   [idx,dst] = flann_search(index,testset,nn,parameters);

   Q = zeros(nn,nn);
   output = zeros(patch_size*patch_size, size(testset,2));
   weight = zeros(size(testset,2), nn);
   for ii = 1:size(testset, 2),

                Ip = testset(:, ii);
                Ipk = zeros(size(testset,1), nn);
                Isk = zeros(size(Cs,1), nn);
                err = zeros(size(Ipk));
            for i=1:nn
                Ipk(:, i) = Cp(:,idx(i,ii));
                Isk(:, i) = Cs(:,idx(i,ii));
            end
      
            Coeff = ( Ipk'*Ipk + lambda*eye(nn) ) \ Ipk' * Ip;
       %     Coeff = mexLasso(Ip, Ipk, param);
       
            Is = Isk * Coeff;          
            
            output(:, ii) = Is;    
        end

    %    [h1, v1] = patch2data1([output(1:patch_size*patch_size, :);output(patch_size*patch_size+1:patch_size*patch_size*2, :)], im_h, im_w, 1,par.win, par.step);
    %    [l1, r1] = patch2data1([output(patch_size*patch_size*2+1:patch_size*patch_size*3, :);output(patch_size*patch_size*3+1:patch_size*patch_size*4, :)], im_h, im_w, 1,par.win, par.step);
        [output, ~] = patch2data([output;output], im_h, im_w, 1,par.win, par.step);
 
        output = output +  imNMF ;
        result = output;
     %   save ('Mat/cman4.mat', 'h1', 'v1', 'l1','r1');

        %save('Mat/out.mat', 'output');
      %  e_ori = conv2( double(imHR), hf1, 'same');
      %  fprintf('%d %d %d\nPSNR of Semi-Coupled DL: %2.2f \n',pp, ss, nnn, csnr(e_ori, h1, 5, 5));
     %   fprintf('%d %d %d\nPSNR of Semi-Coupled DL: %2.2f \n',pp, ss, nnn, csnr(imHR, output, pp, pp));

        %result = func_improve_NL_im(imLR, imHR, imBicubic, h1, v1, l1, r1 );
        
      

       fprintf('%d %d %d %s Result: %2.2f \n',pp, ss, nnn, im_dir(img).name, csnr(imHR, result, 0, 0));
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
        imwrite(uint8(im_rgb), ['Result/s', num2str(par.nFactor), '_', im_dir(img).name]);
        imwrite(uint8(imB), ['Result/s', num2str(par.nFactor), '_bicubic', im_dir(img).name]);
        imwrite(uint8(imLR), ['Result/s', num2str(par.nFactor), '_LR', im_dir(img).name]);
    end
   fprintf('%f %d %d  %d, average %2.2f\n',lambda, pp, ss, nnn, tot/im_num);
   
end
end
    flann_free_index(index);% free the memory      

    end
end

%save('re.mat','re');
%imshow(int8(h1),[])
%output = uint8(output);
%imwrite(output, '1.jpg');


%improve(h1, v1);