function [ idx ] = LinkFrames(  img, im_h, im_w, testset, preidx, nn, dst, Type, Cp, par )

idx = preidx;

window = Make_searchWin( im_h, im_w, par.search_win, par.patch_size);
for i = 2:img-1,
    for j = 1:size(testset{i}, 2),
        [fid] = getSimilar( testset{i-1}, testset{i}(:, j), window(j, :) );
        [pid] = getSimilar( testset{i+1}, testset{i}(:, j), window(j, :) );
        t = Type{i}(j);
        cdt = [ preidx{i-1}{t}(:, fid); preidx{i}{t}(:, j); preidx{i+1}{t}(:, pid) ]';
        cdt = unique( cdt );
        dst = distance( Cp{t}(:, cdt), testset{i-1}(:, fid) );
        dst = dst + distance( Cp{t}(:, cdt), testset{i}(:, j) );
        dst = dst + distance( Cp{t}(:, cdt), testset{i+1}(:, pid) );
        [dst, id] = sort( dst );
        idx{i}{t}(:, j) = cdt( id(1:nn) );
    end
end
end

function [window] = Make_searchWin( im_h, im_w, win, patch_size )
    im_h = im_h - patch_size + 1;
    im_w = im_w - patch_size + 1;
    id = reshape(1:im_h*im_w, [im_h, im_w]);
    window = zeros( im_h, im_w, (2*win+1)^2 );
    for i = 1:im_h,
        for j = 1:im_w,
            x1 = max( 1, i - win ); x2 = min( im_h, i + win );
            y1 = max( 1, j - win ); y2 = min( im_w, j + win );
            k = id(x1:x2, y1:y2);
            window(i, j, 1:numel(k)) = k(:)';
        end
    end
    window = reshape( window, [im_h*im_w, (2*win+1)^2] );
end

function [id] = getSimilar( A, y, cdt )
    A = A(:, cdt(cdt > 0));
    d = A - repmat( y, [1, size(A, 2)] );
    d = sum(d.^2, 1);
    [~, id] = min( d );
    id = cdt( id );
end

function [dst] = distance( A, y )
    d = A - repmat( y, [1, size(A, 2)] );
    d = sum(d.^2, 1);
    dst = sqrt( d );
end

