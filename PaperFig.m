function [] =paperfig() 
addpath('Data');
addpath('Utilities');
addpath('lib');
addpath('lib/face1.0-basic');

% load('Data/Face_Training4.mat');
% 
% im = imread('E:\LiYanghao\Face Hallucination\NEFC\Result8\bicubic_s3049.png');
% %imshow( im );
% 
% modelname = 'mi';
% try
%  %     [bs, posemap] = F2_ReturnLandmarks( im, modelname );
% catch err
%          disp('Error: The UCI algorithm does not detect a face.');
% end
% lm = F4_ConvertBStoMultiPieLandmarks(bs(1));
% 
% imwrite(im, ['Fig/fc_input.png']);
% save_landmark_fig(bs, posemap, im, lm, ['Fig/fc_landmarks.png'] )
% 
% comp{1} = 49:68;        %mouth
% comp{2} = 18:27;        %eyebrows
% comp{3} = 37:48;        %eyes
% comp{4} = 28:36;        %nose
% comp{5} = 1:17;         %face edge
% 
% par.margin = 5;
% par.lg = 6;
% [im_h, im_w, ~] = size( im );
% im1 = im;
% for j = 1:4,  %±ﬂ‘µ”√1
%             y1 = max(1+par.margin, floor(min(lm(comp{j},1))-par.lg));
%             y2 = min(im_w-par.margin, ceil(max(lm(comp{j},1))+par.lg));
%             x1 = max(1+par.margin, floor(min(lm(comp{j},2))-par.lg));
%             x2 = min(im_h-par.margin, ceil(max(lm(comp{j},2))+par.lg));
%      imwrite( uint8(im(x1:x2, y1:y2, :)), ['Fig/fc_comp', num2str(j), '.png'] );
%      im1(x1:x2, y1:y2, :) = 255;
% end
% imwrite( im1, ['Fig/fc_comp_other.png'] );



%%make NLM rect
im = imread('E:\LiYanghao\Face Hallucination\NEFC\Result8\LLC_training4_100000_NEFC_s4_9-9_049.png');
im2 = imread('E:\LiYanghao\Face Hallucination\NEFC\Result8\LLC_training4_100000_NEFC_s4_9-9_048.png');
img = rgb2gray( im );
img2 = rgb2gray( im2 );
par.win = 13;
par.step = 1;
par.width = 2;
px = 65; py = 90;

im1 = addrec( im, px, py, par.win, par.width, [255,0,0] );
%figure, imshow(im1);

[V, ~] = data2patch( double(img2),  img,  par );
%y = V(:, xy2idx(px, py,  size(im,1)-par.win+1, size(im,2)-par.win+1));
y = img(px:px+par.win-1, py:py+par.win-1)';
y = double(y(:));
dis = zeros( 1, size(V, 2) );
for i = 1:size(V,1),
    dis = dis + (V(i, :) - y(i)).^2;
end
[dis, idx] = sort( dis );

for i = [1, 15, 20],
[x1, y1] = idx2xy( idx(i), size(im,1)-par.win+1, size(im,2)-par.win+1 );
im2 = addrec(im2, x1, y1, par.win, par.width, [0,255,0]);
end
imwrite( im2, 'Fig/NLM_48.png');
imshow( im2 );
end

function [idx] = xy2idx( x, y, h, w )
    idx = (y - 1) * h + x ;
end
function [x,y] = idx2xy( idx, h, w )
    idx = idx -1;
    x = mod(idx, h) + 1;
    y = fix(idx / h ) + 1;
end

function [ret] = addrec( im, cx, cy, br, w, color )
    ret = im;
    for i = cx:cx+br-1,
        for j = cy:cy+br-1,
            for k = 1:3, 
                ret(i,j,k) = color(k);
            end
        end
    end
    
    for i = cx+w:cx+br-w-1,
        for j = cy+w:cy+br-w-1,
            for k = 1:3, 
                ret(i,j,k) = im(i,j,k);
            end
        end
    end
end
