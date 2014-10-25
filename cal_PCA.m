function [PL, PH, mY, mX] = cal_PCA( img_path, img_dir, par )
   
X = [];
Y = [];
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
        Y = zeros(prod(size(imLR)), img_num);
        X = zeros(prod(size(imHR)), img_num);
    end
    Y(:, i) = imLR(:); 
    X(:, i) = imHR(:);
end
mY = mean(Y, 2);
mX = mean(X, 2);
Y = Y - repmat(mY, [1, img_num]);
X = X - repmat(mX, [1, img_num]);

C = double( Y * Y' );
[V, D] = eig( C );
D = diag( D );
D = cumsum(D) / sum(D);
k = find( D >= 1e-3, 1);
PL = V(:, k:end);

%imLR = imHR(1:par.nFactor:im_h, 1:par.nFactor:im_w, :);
%B = Set_blur_matrix( par.nFactor, imLR, par.psf, imHR );
%PL = B*PH;

PH = X / (PL' * Y);
    
end