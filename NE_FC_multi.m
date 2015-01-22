%release 1.0
%by Li Yanghao
clear all;clc;

addpath('Flann')
addpath('Data');
addpath('Utilities');
addpath('lib');
addpath('lib/face1.0-basic');
addpath('lib/OpticalFlow/mex');

%testing images
im_path = 'Data/Face_Testing4/';
im_dir = dir( fullfile(im_path, '*.png') );
im_num = length( im_dir );

[compf, compp] = Comp_lm(); %components landmarks of frontal and profile faces

nSmp        = 500000;       % number of patches to sample
par.lamB = 2;           % lamada in RefineBlur
par.niterB = 40;        % iterative number in RefineBlur
par.nFactor = 4;
par.patch_size = 9;
par.win = par.patch_size;
par.step = 1;
par.prunvar = 5;
par.lg = 5;
par.search_win = 2; %half of the search window size
par.margin = par.patch_size;
par.nn = 9;   % number of nearest number
par.psf =   fspecial('gauss', 7, 1.6);              % The simulated PSF

par.method = 'LLC';
par.param.tu = 0.005;
par.mode = 'Image'; %Video or Image
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
[Cp, Cs] = Smp_patch_blur_FC( par.patch_size, nSmp, par);
%save(['Mat/Training',par.training_set,'_',num2str(ss), '_', num2str(par.nFactor)], 'Cp', 'Cs', '-v7.3');
%load(['Mat/Training',par.training_set,'_',num2str(ss), '_', num2str(par.nFactor)], 'Cp', 'Cs');

for i = 1:5,
    dataset = Cp{i};  
    build_params.target_precision = 1;  
    build_params.build_weight = 0.5; 
    build_params.memory_weight = 0; 
    [index{i}, parameters{i}] = flann_build_index(dataset, build_params);
end

nn = par.nn;   
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
        fprintf('%d %d %d %s Result: %2.3f \n',par.patch_size, nSmp, par.nn, im_dir(img-1).name, csnr(imHR{img-1}, result{img-1}, 0, 0));
        tot = [tot, csnr(imHR{img-1}, result{img-1}, 0, 0)];
        if par.mode == 'Video',
            if (img > 2) 
                result{img-1} = inter_inner_NLM( result{img-1}, result{img-2}, par );
            end
        else
            result{img-1} = inner_NLM( result{img-1}, par );
        end
           fprintf('%d %d %d %s NLResult: %2.3f \n',par.patch_size, nSmp, par.nn, im_dir(img-1).name, csnr(imHR{img-1}, result{img-1}, 0, 0));
           
           NLtot = [NLtot, csnr(imHR{img-1}, result{img-1}, 0, 0)];

           im_rgb{img-1}(:,:,1) = result{img-1};
           im_rgb{img-1} = ycbcr2rgb( uint8( im_rgb{img-1} ) );
            imwrite(uint8(im_rgb{img-1}), ['Result8/v',par.method,'_training4_',num2str(nSmp),'_', 'NEFC_s', num2str(par.nFactor), '_', num2str(par.patch_size),'-', num2str(par.nn),'_',im_dir(img-1).name]);
            imwrite(uint8(imB{img-1}), ['Result8/vbicubic_s', num2str(par.nFactor), im_dir(img-1).name]);
            if (img > 2)
                imHR{img-2} = []; ori_HR{img-2} = []; im_rgb{img-2} = []; imB{img-2} = [];
                imBicubic{img-2} = []; Type{img-2}= []; preidx{img-2} = []; dst{img-2} = []; testset{img-2} = [];
            end
    end
end

fprintf('%f %d %d  %d, average %2.2f ',parameter, par.patch_size, nSmp, par.nn, sum(tot)/im_num);
fprintf('\nbicubic average: %2.2f',sum(re_bi)/im_num);
fprintf('\n');
for i = 1:5,
    flann_free_index(index{i});% free the memory      
end