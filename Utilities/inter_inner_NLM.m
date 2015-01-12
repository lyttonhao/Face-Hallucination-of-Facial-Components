function [im1, im2] = inter_inner_NLM( im1, im2, par)
    [vx, vy] = Coarse2FineTwoFrames(im2double(uint8(im1)), im2double(uint8(im2)), par.OFpara);
    N = compute_NLM_Matrix_inter( im1, im2, 5, vx, vy );
    par.lamNL = 0.5;
    par.NLiter = 1;
    NTN = N'*N*par.lamNL;
    f = double( [im1(:); im2(:)] );
    for i = 1:par.NLiter,
        f = f - NTN*f;
    end
    [im_h, im_w] = size(im1);
    im1 = reshape( f(1:im_h*im_w), [im_h, im_w] );
    [im_h, im_w] = size(im2);
    im2 = reshape( f(1:im_h*im_w), [im_h, im_w] );
end