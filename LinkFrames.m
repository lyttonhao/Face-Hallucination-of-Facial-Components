function [ idx ] = LinkFrames(  i, imBicubic, im_h, im_w, testset, preidx, nn, dst, Type, Cp, par )
%find similar patches among neighbor frames for current frame
%then get new nearest neighbors

idx = preidx{i};
if par.mode == 'Image',
    return ;
end
%window = Make_searchWin( im_h, im_w, par.search_win, par.patch_size);
imi = im2double( uint8(imBicubic{i}) );
im1 = im2double( uint8(imBicubic{i-1}) );
im2 = im2double( uint8(imBicubic{i+1}) );
[vx1,vy1,~] = Coarse2FineTwoFrames(imi,im1,par.OFpara);
[vx2,vy2,~] = Coarse2FineTwoFrames(imi,im2,par.OFpara);
win1 = Make_searchWin1( im_h, im_w, par.search_win, par.patch_size, vx1, vy1 );
win2 = Make_searchWin1( im_h, im_w, par.search_win, par.patch_size, vx2, vy2 );
%win1 = Make_searchWin( im_h, im_w, par.search_win, par.patch_size);
%win2 = Make_searchWin( im_h, im_w, par.search_win, par.patch_size);

for j = 1:size(testset{i}, 2),
        [fid] = getSimilar( testset{i-1}, testset{i}(:, j), win1(j, :) );
        [pid] = getSimilar( testset{i+1}, testset{i}(:, j), win2(j, :) );
        t = Type{i}(j);
        cdt = [ preidx{i-1}{t}(:, fid); preidx{i}{t}(:, j); preidx{i+1}{t}(:, pid) ]';
        cdt = unique( cdt );
        dst = distance( Cp{t}(:, cdt), testset{i-1}(:, fid) );
        dst = dst + distance( Cp{t}(:, cdt), testset{i}(:, j) );
        dst = dst + distance( Cp{t}(:, cdt), testset{i+1}(:, pid) );
        [dst, id] = sort( dst );
        idx{t}(:, j) = cdt( id(1:nn) );
    end
end


function [window] = Make_searchWin1( im_h, im_w, win, patch_size, vx, vy )
    im_h = im_h - patch_size + 1;
    im_w = im_w - patch_size + 1;
    id = reshape(1:im_h*im_w, [im_h, im_w]);
    window = zeros( im_h, im_w, (2*win+1)^2 );
    for i = 1:im_h,
        for j = 1:im_w,
            x1 = round(max( 1, j+vx(i,j) - win ));  x1 = min(im_w, x1);
            x2 = round(min( im_w, j+vx(i,j) + win )); x2 = max(1, x2);
            y1 = round(max( 1, i+vy(i,j) - win ));   y1 = min(im_h, y1);
            y2 = round(min( im_h, i+vy(i,j) + win )); y2 = max(1, y2);
            k = id(y1:y2, x1:x2);
            window(i, j, 1:numel(k)) = k(:)';
        end
    end
    window = reshape( window, [im_h*im_w, (2*win+1)^2] );
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
    
    d = (A(1, :) - y(1)).^2;
    for k = 2:size(A, 2)
        d = d + (A(k, :) - y(k)).^2;
    end
    
    d = A - repmat( y, [1, size(A, 2)] );
    d = sum(d.^2, 1);
    [~, id] = min( d );
    id = cdt( id );
end

function [dst] = distance( A, y )
    d = (A(1, :) - y(1)).^2;
    for k = 2:size(A, 2)
        d = d + (A(k, :) - y(k)).^2;
    end
    d = A - repmat( y, [1, size(A, 2)] );
    d = sum(d.^2, 1);
    dst = sqrt( d );
end

