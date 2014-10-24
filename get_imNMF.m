function [imNMF] = get_imNMF(TU, BU, U, imLR, im_h, im_w)

beta = 0.1;

c = ([BU;beta*TU]) \ [imLR(:);zeros(size(TU, 1),1)];
imNMF = U * c;
imNMF = reshape( imNMF, [im_h, im_w] );

end