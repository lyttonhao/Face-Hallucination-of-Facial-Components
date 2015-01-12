function [im] = inner_NLM( im, par)
    N = Compute_NLM_Matrix( im, 5, par );
    par.lamNL = 0.3;
    par.NLiter = 1;
    NTN = N'*N*par.lamNL;
    f = double( im(:) );
    for i = 1:par.NLiter,
        f = f - NTN*f;
    end
    [im_h, im_w] = size(im);
    im = reshape( f, [im_h, im_w] );
end