function [im_out] = scdl_interp(te_LR, im_h, im_w, nFactor)

if nFactor == 3
    cls_num = 1;
    nOuterLoop = 1;
    nInnerLoop = 1;
    load KMeans_5x5_32_Factor3_1 vec par param;
end
im_c = size(te_LR,3);
[CX CY] = meshgrid(1 : im_w, 1:im_h);
[X Y] = meshgrid(1:nFactor:im_w, 1:nFactor:im_h);
imSpline = zeros(im_h, im_w, im_c);
for i = 1 : im_c
    imSpline(:,:,i)  =   interp2(X, Y, te_LR(:,:,i), CX, CY, 'spline');
end
te_Out      = imSpline;

% Used for calculate CSNR
par.height = im_h;
par.width = im_w;
par.te_num = im_c;

imwrite(uint8(te_Out),'Result/GirlRGB_Bicubic.png');
if im_c == 3
    te_LR           =   rgb2ycbcr( uint8(te_LR) );
    te_LR           =   double( te_LR(:,:,1));
    tOut            =   rgb2ycbcr( uint8(te_Out) );
    te_Out           =   double( tOut(:,:,1));
end

psf = fspecial('gaussian', par.win+2, 2.2);

% Initial
im_lout = te_Out;
[YH ~] = data2patch(conv2(im_lout, psf, 'same') - im_lout, conv2(im_lout, psf, 'same') - im_lout, par);
[cls_idx]  = setPatchIdx(YH, vec');
clear YH;
[~, XL] = data2patch(te_Out, te_Out, par);  
meanX = repmat(mean(XL), [par.win^2 1]);
XL = XL - meanX;

im_out = te_Out;

load Dict_SR_Factor3_1 Dict;
AL = sparse(par.K, size(XL, 2));
AH = sparse(par.K, size(XL, 2));

for m = 1 : nOuterLoop
    fprintf('Iter: %d \n', m);
for t = 1 : nInnerLoop
    if t == 1
        [YH, ~] = data2patch(conv2(im_out, psf, 'same') - im_out, conv2(im_lout, psf, 'same') - im_lout, par);
        [cls_idx]  = setPatchIdx(YH, vec');
        clear YH;
    end
    [XH XL] = data2patch(im_out, im_lout, par);  
    meanX = repmat(mean(XH), [par.win^2 1]);
    XL = XL - meanX;
    XH = XH - meanX; 
    for i = 1 : cls_num
        fprintf('i:%d\n',i);
        idx_cluster   = find(cls_idx == i);
        length_idx = length(idx_cluster);
        start_idx = [1:10000:length_idx, length_idx];
        for j  = 1 : length(start_idx) - 1
        idx_temp = idx_cluster(start_idx(j):start_idx(j+1));
        Xh    = double(XH(:, idx_temp));
        Xl    = double(XL(:, idx_temp));
        Dh    = Dict.DH{i};
        Dl    = Dict.DL{i};
        Wl    = Dict.WL{i};
        Wh    = Dict.WH{i};  
        if (t == 1)
            Alphal = mexLasso(Xl, Dl, param);
            Alphah = Wl * Alphal;
            Xh = Dh * Alphah;
        else
            Alphah = AH(:, idx_temp);
        end
        D = [Dl; par.sqrtmu * Wl];
        Y = [Xl; par.sqrtmu * full(Alphah)];
        Alphal = mexLasso(Y, D,param);    
        clear Y D;
        D = [Dh; par.sqrtmu * Wh];
        Y = [Xh; par.sqrtmu * full(Alphal)];
        Alphah = full(mexLasso(Y, D,param));
        clear Y D;
        Xh = Dh * Alphah;
        XH(:, idx_temp) = Xh;
        AL(:, idx_temp) = Alphal;
        AH(:, idx_temp) = Alphah;
        end
    end
    [im_out, ~] = patch2data([XH+meanX;XL+meanX], im_h, im_w, 1,par.win, par.step);
    im_out(1 : nFactor : im_h, 1 : nFactor : im_w) = te_LR;
    [N, ~]       =   Compute_NLM_Matrix( im_out , 5, par);
    NTN          =   N'*N*0.05;
    
    im_f = sparse(double(im_out(:)));
    for i = 1 : fix(60 / t.^2)      
        im_f = im_f  - NTN*im_f;
    end
    im_out = reshape(full(im_f), im_h, im_w);
%    
end
end
tOut(:,:,1) = uint8(im_out);
if im_c == 3
    im_out = double(ycbcr2rgb(tOut));
end