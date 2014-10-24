function [TU, BU, U] = cal_NMF( img_path, img_dir, par )
   
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
    if (i == 1),
        X = zeros(im_h*im_w, img_num);
    end
    X(:, i) = imHR(:);     
end



k = fix(prod(size(X))/(sum(size(X))));
%[U0, V0] = nnmf(X, k, 'replicates', 5, 'algorithm', 'mult');
%[U, V]= nnmf(X, k, 'w0', U0, 'h0', V0, 'algorithm', 'als');
options = statset('MaxIter', 1000);
[U, V] = nnmf(X, k, 'options', options);

imLR = imHR(1:par.nFactor:im_h, 1:par.nFactor:im_w, :);
B = Set_blur_matrix( par.nFactor, imLR, par.psf, imHR );
BU = B * U;

p = [0,1,0;1,-4,1;0,1,0];
T = Set_matrix( 1, imLR, p, imHR);
TU = T * U;



end