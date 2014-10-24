function [imNMF] = get_imNMF_GL(U, V, Ih, imLR, im_h, im_w)

h = U \ imLR(:);
alpha = V \ h;

imNMF = Ih * alpha;

imNMF = reshape( imNMF, [im_h, im_w] );

end