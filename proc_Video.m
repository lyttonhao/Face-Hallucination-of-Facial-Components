
%readVideo
addpath('Utilities/YUV2Image');
workingDir = 'Data/Face_Testing8/';
VideoName = ('Data/FOREMAN_cif.yuv');
width = 352; Height = 288;
Frames = 1:90;
 %[mov, imgRgb] = loadFileYuv( VideoName, width, Height, Frames);

for ii = 45:55,
   %   filename = [sprintf('%03d',ii) '.png'];
  % img = imread( fullfile(workingDir, 'Origin/', filename) ); %read(Video, ii);
   %img = readVideo( Video, ii );
   img = mov(ii).cdata;
   im = zeros( size(img, 1), size(img, 2), 3);
   for j = 1:3,
    im(:,:,j) = img(:,:,j);
   end
   im = im(30:245, 80:270,:);
   filename = [sprintf('%03d',ii) '.png'];
   fullname = fullfile(workingDir,filename);
 %  im = imresize(im, 0.4, 'bicubic');
  % im = im(1:3:end, 1:3:end, :);
   imwrite(uint8(im),fullname)    % Write out to a JPEG file (img1.jpg, img2.jpg, etc.)
end


%writeVideo

% workingDir = 'Result7/';
% writer = VideoWriter(fullfile(workingDir, 'two.avi'));
% writer.FrameRate = 3;
% open(writer);
% 
% imdir = dir( fullfile(workingDir, 'RidgeNEFC_s4_*.jpg' ));
% imdir1 = dir( fullfile(workingDir, 'multiRidge*.jpg' ));
% for i = 1:length(imdir),
%     im1 = imread( fullfile(workingDir, imdir(i).name(1:end)) );
%     im2 = imread( fullfile(workingDir, imdir1(i).name(1:end)) );
%     im = uint8(zeros( [size(im1, 1), size(im2, 2)*2, 3] ));
%     for j = 1:3,
%         im(:,:, j) = [im1(:, :, j), im2(:,:,j) ];
%     end
%     writeVideo( writer, im );
% end
% close( writer );




