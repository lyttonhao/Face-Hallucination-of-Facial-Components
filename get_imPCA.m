function [imPCA] = get_imPCA(PL, PH, imLR, im_h, im_w)

imPCA = PH * (PL \ imLR(:));

imPCA = reshape( imPCA, [im_h, im_w] );

end