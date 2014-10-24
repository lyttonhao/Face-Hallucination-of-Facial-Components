function [U, V, Ih] = cal_NMF_LR( img_path, img_dir, par )
   
X = [];
img_num = length(img_dir);
for i = 1 : img_num
    imHR               =   imread(fullfile( img_path, img_dir(i).name)) ;
    [im_h, im_w, ch]       =   size(imHR);
    if ch == 3,
        imHR = double( rgb2ycbcr( imHR ));
    end
    imHR = double(imHR(:,:,1));
    
    [im_h, im_w, dummy] = size(imHR);
    im_h = floor((im_h )/par.nFactor)*par.nFactor ;
    im_w = floor((im_w )/par.nFactor)*par.nFactor ;
    imHR=imHR(1:im_h,1:im_w,:);
    
    psf                =    par.psf;              % The simulated PSF
    imLR = Blur('fwd', imHR, psf);
    imLR           =   imLR(1 : par.nFactor : im_h, 1 : par.nFactor : im_w);
    if (i == 1),
        X = zeros(prod(size(imLR)), img_num);
        Ih = zeros(prod(size(imHR)), img_num);
    end
    X(:, i) = imLR(:); 
    Ih(:, i) = imHR(:);
end

k = fix(prod(size(X))/(sum(size(X))));
options = statset('MaxIter', 1000);
[U, V] = nnmf(X, k, 'options', options);





end