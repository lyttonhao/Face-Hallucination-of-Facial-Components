%Generate training and test dataset from CAS-PEAL-R1 dataset
addpath('lib');
addpath('lib/face1.0-basic');
addpath('lib/OpticalFlow/mex');

Train_path = 'Data/Face_Training6/';
sav_mat = 'Data/Face_Training6.mat';
Test_path = 'Data/Face_Testing4/';
dataset_path = 'E:\Database\CAS-PEAL-R1\CAS-PEAL-R1\POSE\';

if exist( Train_path, 'dir' ) == 0,
    mkdir( Train_path );
end
if exist( Test_path, 'dir') == 0,
    mkdir( Test_path );
end
%sample images

% for i = [11:310],
%     dirpath = [dataset_path, sprintf('%06d', i),'\'];
%     fileID = fopen([dirpath, 'FaceFP_2.txt'],'r');
%     facedata  = {}; n = 0;
%     tline = fgetl(fileID);
%     while ischar(tline)
%         n = n+1;
%         [facedata{n}.name, facedata{n}.x1, facedata{n}.y1, facedata{n}.x2, facedata{n}.y2] = strread(tline, '%s %d %d %d %d'); 
%         tline =  fgetl(fileID);
%     end
%     
%     p=[8,10,13];
%     
%   %  for j = [8,10,11, 13,14],
%         j = p(mod(i, 3)+1);
%     %   j = 8;
%         im_name = facedata{j}.name{1};
%         im = imread([dirpath, im_name, '.tif']);
%         k = fix((230-(facedata{j}.x2-facedata{j}.x1))/2);
%         im = im( facedata{j}.y1-150: min(facedata{j}.y1+170,size(im,1)),40:320,:);
%         [im_h, im_w] = size(im);
%         im2 = zeros(im_h, im_w, 3);
%         im2(:,:,1) = im; im2(:,:,2) = im; im2(:,:,3) = im;
%         imwrite(uint8(im2), [Train_path, im_name, '.png']);
%   %  end
%   fclose( fileID );
% end


im_dir = dir( fullfile(Train_path, '*.png') );
n = length( im_dir );
pose = zeros(n, 1);
for i = 1:n,
    t =  im_dir(i).name(20:22)
    if (strcmp(t,'+00')),
        pose(i) = 1;
    elseif (t(1) == '-'),
        pose(i) = 2;
    else
        pose(i) = 3;
    end
end
im = imread( [Train_path, im_dir(1).name] );
[im_h, im_w, im_c] = size(im);
images_hr = uint8( zeros(im_h, im_w, n) );
landmarks = zeros( 68, 2, n);
lmnum = zeros(1, n);
for i = 1:n,
    i
    im = imread( [Train_path, im_dir(i).name] );
    try
        [bs, posemap] = F2_ReturnLandmarks( im, 'mi' );
    catch err
         disp('Error: The UCI algorithm does not detect a face.');
         continue;
    end
    lm = F4_ConvertBStoMultiPieLandmarks(bs(1));
    landmarks(1:size(lm, 1),:,i) = lm;
    lmnum(i) = size(lm, 1);
    images_hr(1:size(im,1),1:size(im,2),i) = im(:,:,1);
end
save( sav_mat, 'images_hr', 'landmarks', 'lmnum', 'pose');

