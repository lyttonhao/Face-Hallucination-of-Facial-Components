function [imPCA] = get_imPCA(PL, PH, imLR, mY, mX, im_h, im_w)

imPCA = PH * (PL \ (imLR(:)-mY));

imPCA = reshape( imPCA + mX, [im_h, im_w] );

end