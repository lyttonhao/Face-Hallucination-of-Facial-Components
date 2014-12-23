
%readVideo
%{
workingDir = 'Data/Face_Testing7/';
%Video = VideoReader('Data/VIDEO0012.mp4');
%nFrames = Video.NumberOfFrames;

for ii = 226:3:415,
      filename = [sprintf('%03d',ii) '.jpg'];
   img = imread( fullfile(workingDir, 'Origin/', filename) ); %read(Video, ii);
   im = zeros( size(img, 2), size(img, 1), 3);
   for j = 1:3,
    im(:,:,j) = img(:,:,j)';
   end
   im = im(300:910, 260:950,:);
   filename = [sprintf('%03d',ii) '.jpg'];
   fullname = fullfile(workingDir,filename);
   im = imresize(im, 0.4, 'bicubic');
  % im = im(1:3:end, 1:3:end, :);
   imwrite(uint8(im),fullname)    % Write out to a JPEG file (img1.jpg, img2.jpg, etc.)
end
%}
%writeVideo

workingDir = 'Result7/';
writer = VideoWriter(fullfile(workingDir, 'multi.avi'));
writer.FrameRate = 3;
open(writer);

imdir = dir( fullfile(workingDir, 'multiRidgeNEFC_s4_*.jpg' ));
for i = 1:length(imdir),
    im = imread( fullfile(workingDir, imdir(i).name(1:end)) );
    writeVideo( writer, im );
end
close( writer );




