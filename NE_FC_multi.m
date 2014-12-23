
%clear all;clc;

addpath('Flann')
addpath('Data');
addpath('Utilities');
addpath('spams-matlab');
addpath('spams-matlab/build');    
addpath('lib');
addpath('lib/face1.0-basic');
cc = 0;
re = [];

im_path = 'Data/Face_Testing7/';
im_dir = dir( fullfile(im_path, '*.jpg') );
im_num = length( im_dir );
%im_num = 3;
%load('Data/train3.mat');
load('Mat/lms.mat');
[compf, compp] = Comp_lm(); %components landmarks of frontal and profile faces

for pp = [9],
    for ss = [100000],
     

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
par.method = 'Ridge';


% randomly sample image patches
%[Cp, Cs, Pose] = Smp_patch_blur_FC( patch_size, nSmp, par);
%[TU, BU, U, Cp, Cs] = Smp_patch_blur_NMF( patch_size, nSmp, par);
%[PL, PH, Cp, Cs] = Smp_patch_blur_PCA( patch_size, nSmp, par);
%[U, V, Ih, Cp, Cs] = Smp_patch_blur_NMF_GL( patch_size, nSmp, par);
%save(['Mat/Cps5',num2str(par.nFactor)], 'Cp', 'Cs', 'Pose');
load(['Mat/Cps5',num2str(par.nFactor)]);
%load BU;

%[Cp, V_pca] = PCA(Cp);

for i = 1:6,
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
     par.param.tu = parameter;
for nnn = [9],
    nn = nnn;   
    tot = []; re_bi = [];
    info = zeros(im_num, 3);
    for img = 1:im_num,
        imHR1 = imread( fullfile(im_path, im_dir(img).name) );  
        [ imHR{img}, ori_HR{img}, im_rgb{img}, imB{img}, imBicubic{img}, Type{img}, idx{img}, dst{img}, testset{img}  ]  = ...
            preprocess( imHR1, compf, compp, nn, lms{img}, par, index, parameters );
    end
    [im_h, im_w, im_c] = size( imHR{1} );
    [idx] = LinkFrames(  im_num, im_h, im_w, testset, idx, nn, dst, Type, Cp, par );
    for img = 1:im_num,
       Q = zeros(nn,nn);
       output = zeros(patch_size*patch_size, size(testset{img},2));
       weight = zeros(size(testset{img},2), nn);
       for ii = 1:size(testset{img}, 2),
                t = Type{img}(ii);
                Ip = testset{img}(:, ii);
                Ipk = zeros(size(testset{img},1), nn);
                Isk = zeros(size(Cs{t},1), nn);
            for i=1:nn
                Ipk(:, i) = Cp{t}(:,idx{img}{t}(i,ii));
                Isk(:, i) = Cs{t}(:,idx{img}{t}(i,ii));
           %     info(img,  Pose{t}(idx{img}{t}(i,ii)) ) = info(img, Pose{t}(idx{img}{t}(i,ii)) )+1;
            end
      
          %  Coeff = ( Ipk'*Ipk + lambda*eye(nn) ) \ Ipk' * Ip;
            Coeff = Cal_Coeff( Ipk, Ip, nn, par, par.method );
            weight(ii, :) = Coeff';
            
           
       %     Coeff = mexLasso(Ip, Ipk, param);
       
            Is = Isk * Coeff;          
            
            output(:, ii) = Is;    
        end

    %    [h1, v1] = patch2data1([output(1:patch_size*patch_size, :);output(patch_size*patch_size+1:patch_size*patch_size*2, :)], im_h, im_w, 1,par.win, par.step);
    %    [l1, r1] = patch2data1([output(patch_size*patch_size*2+1:patch_size*patch_size*3, :);output(patch_size*patch_size*3+1:patch_size*patch_size*4, :)], im_h, im_w, 1,par.win, par.step);
        [output, ~] = patch2data([output;output], im_h, im_w, 1,par.win, par.step);
 
        output = output +  imBicubic{img} ;
        result = output;
        
      %  result = RefineBlur( imLR, imHR, result, par );

       
       fprintf('%d %d %d %s Result: %2.3f \n',pp, ss, nnn, im_dir(img).name, csnr(imHR{img}, result, 5, 5));
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
       tot = [tot, csnr(imHR{img}, result, 5, 5)];
        
       im_rgb{img}(:,:,1) = result;
       im_rgb{img} = ycbcr2rgb( uint8( im_rgb{img} ) );
    
      %  savefile( imLR, ori_HR, im_rgb, result, h1, v1, imB, im_dir(img).name);
        imwrite(uint8(im_rgb{img}), ['Result7/multi', par.method, 'NEFC_s', num2str(par.nFactor), '_', num2str(pp),'-', num2str(nnn),'_',im_dir(img).name]);
        %imwrite(uint8(imB), ['Result7/s', num2str(par.nFactor), '_bicubic', im_dir(img).name]);
    %    info(img, :)
    %    imwrite(uint8(imLR), ['Result3/s', num2str(par.nFactor), '_LR', im_dir(img).name]);
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
    for i = 1:6,
      %  flann_free_index(index{i});% free the memory      
    end

    end
end

%save('re.mat','re');
%imshow(int8(h1),[])
%output = uint8(output);
%imwrite(output, '1.jpg');


%improve(h1, v1);