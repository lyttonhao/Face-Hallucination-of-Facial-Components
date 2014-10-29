function [ output ] = RefineBlur( imLR, imHR, im_h, par )
    lamada = par.lamB;
    B = Set_blur_matrix( par.nFactor, imLR, par.psf, im_h );
    BTY = B' * imLR(:);
    BTB = B' * B;
    f = double(im_h(:));
    [im_h, im_w] = size( imHR );
    for i = 1:par.niterB,
        f = f + lamada * (BTY - BTB*f);
        fprintf('\nRefine%d: %2.2f \n', i, csnr(imHR, reshape(f, [im_h, im_w]), 0, 0));
    end
    output = reshape(f, [im_h, im_w]);
end