
%clear all;clc;

addpath('Flann')
addpath('Data');
addpath('Utilities');
addpath('spams-matlab');
addpath('spams-matlab/build');    
addpath('lib');
addpath('lib/face1.0-basic');
addpath('lib/OpticalFlow/mex');
cc = 0;
re = [];

im_path = 'Data/Face_Testing8_v/';
im_dir = dir( fullfile(im_path, '*.png') );
im_num = length( im_dir );
%im_num = 10;
%im_num = 3;
%load('Data/train3.mat');
%load('Mat/lms.mat');
[compf, compp] = Comp_lm(); %components landmarks of frontal and profile faces

for pp = [9],
    for ss = [500000],
     

%lambda      = 0.15;         % sparsity regularization
patch_size = pp;
nSmp        = ss;       % number of patches to sample
par.lamB = 2;           % lamada in RefineBlur
par.niterB = 40;        % iterative number in RefineBlur
par.nFactor = 4;
par.patch_size = pp;
par.win = patch_size;
par.step = 1;
par.prunvar = 5;
par.lg = 5;
par.search_win = 2; %half of the search window size
par.margin = pp;
par.psf =   fspecial('gauss', 7, 1.6);              % The simulated PSF
par.method = 'LLC';
par.mode = 'Video'; %Video or Image
% set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;
par.OFpara = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];


par.training_set = '7';
% randomly sample image patches
%[Cp, Cs, Pose] = Smp_patch_blur_FC( patch_size, nSmp, par);
%[TU, BU, U, Cp, Cs] = Smp_patch_blur_NMF( patch_size, nSmp, par);
%[PL, PH, Cp, Cs] = Smp_patch_blur_PCA( patch_size, nSmp, par);
%[U, V, Ih, Cp, Cs] = Smp_patch_blur_NMF_GL( patch_size, nSmp, par);
%save(['Mat/Training',par.training_set,'_',num2str(ss), '_', num2str(par.nFactor)], 'Cp', 'Cs', '-v7.3');
%load(['Mat/Training',par.training_set,'_',num2str(ss), '_', num2str(par.nFactor)], 'Cp', 'Cs');
%load BU;

%[Cp, V_pca] = PCA(Cp);

for i = 1:5,
    dataset = Cp{i};  
    build_params.target_precision = 1;  
    build_params.build_weight = 0.5; 
    build_params.memory_weight = 0; 
    [index{i}, parameters{i}] = flann_build_index(dataset, build_params);
end

%param.iter = 100;
%param.L = 30;
for parameter = [ 0.15 ],
     par.param.lambda = parameter; 
     par.param.mode = 2;
     par.param.tu = 0.005;
for nnn = [9],
    nn = nnn;   
    tot = []; NLtot = []; re_bi = [];
    testsets = [];
    for img = 1:im_num+1,
        if img <= im_num,
            imHR1 = imread( fullfile(im_path, im_dir(img).name) );  
            [ imHR{img}, ori_HR{img}, im_rgb{img}, imB{img}, imBicubic{img}, Type{img}, preidx{img}, dst{img}, testset{img}  ]  = ...
                pre_process( imHR1, compf, compp, nn,  par, index, parameters, [] );
            [im_h, im_w, im_c] = size( imHR{img} );
        end
        if img > 1,
            if (img == 2 || img == im_num+1)
                idx{img-1} = preidx{img-1};
            else
                 idx{img-1} = LinkFrames(  img-1, imBicubic, im_h, im_w, testset, preidx, nn, dst, Type, Cp, par );
            end
            output = SR_reconstruct( Cp, Cs, testset{img-1}, idx{img-1}, Type{img-1}, nn, im_h, im_w, par );
            output = output +  imBicubic{img-1} ;
            result{img-1} = output;
           
          %  result = RefineBlur( imLR, imHR, result, par );       
           fprintf('%d %d %d %s Result: %2.3f \n',pp, ss, nnn, im_dir(img-1).name, csnr(imHR{img-1}, result{img-1}, 5, 5));
           tot = [tot, csnr(imHR{img-1}, result{img-1}, 5, 5)];
           if par.mode == 'Video',
               if (img > 2) 
                   result{img-1} = inter_inner_NLM( result{img-1}, result{img-2}, par );
               end
           else
               result{img-1} = inner_NLM( result{img-1}, par );
           end
           fprintf('%d %d %d %s NLResult: %2.3f \n',pp, ss, nnn, im_dir(img-1).name, csnr(imHR{img-1}, result{img-1}, 5, 5));
       %{
           for j = 1:4,
                y1 = max(1, floor(min(lms{img}(comp{j},1))-par.lg));
                y2 = min(im_w, ceil(max(lms{img}(comp{j},1))+par.lg));
                x1 = max(1, floor(min(lm{img}(comp{j},2))-par.lg));
                x2 = min(im_h, ceil(max(lm{img}(comp{j},2))+par.lg));
                fprintf('%2.3f ', csnr(imHR{img}(x1:x2,y1:y2), result(x1:x2,y1:y2), 0, 0));
           end
           fprintf('\n');
           %}
           NLtot = [NLtot, csnr(imHR{img-1}, result{img-1}, 5, 5)];

           im_rgb{img-1}(:,:,1) = result{img-1};
           im_rgb{img-1} = ycbcr2rgb( uint8( im_rgb{img-1} ) );

          %  savefile( imLR, ori_HR, im_rgb, result, h1, v1, imB, im_dir(img).name);
            imwrite(uint8(im_rgb{img-1}), ['Result8/v',par.method,'_training4_',num2str(ss),'_', 'NEFC_s', num2str(par.nFactor), '_', num2str(pp),'-', num2str(nnn),'_',im_dir(img-1).name]);
            imwrite(uint8(imB{img-1}), ['Result8/vbicubic_s', num2str(par.nFactor), im_dir(img-1).name]);
        %    info(img, :)
        %    imwrite(uint8(imLR), ['Result3/s', num2str(par.nFactor), '_LR', im_dir(img).name]);
            if (img > 2)
                testset{img-2} = [];
            end
        end
    end
   fprintf('%f %d %d  %d, average %2.2f ',parameter, pp, ss, nnn, sum(tot)/im_num);
   fprintf('\nbicubic average: %2.2f',sum(re_bi)/im_num);
%{
   for outi = 1:5,
        fprintf(', %2.2f ', sum(tot(outi:5:end))/(im_num/5) );
   end
   
   fprintf('\nbicubic average: %2.2f',sum(re_bi)/im_num);
   for outi = 1:5,
       fprintf(', %2.2f ',  sum(re_bi(outi:5:end))/(im_num/5));
   end
   %}
   fprintf('\n');
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