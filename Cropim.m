addpath('lib');
addpath('lib/face1.0-basic');
im_path = 'Data/Face_Testing6/';
im_dir = dir( fullfile(im_path, '*.jpg') );
im_num = length( im_dir );

%ll = [];
for i = 1:20,
    im = imread( fullfile( im_path, im_dir(i).name ) );
    try
  %      [bs, posemap] = F2_ReturnLandmarks( im, 'mi' );
    catch err
         disp('Error: The UCI algorithm does not detect a face.');
         continue;
    end
 %   lm = F4_ConvertBStoMultiPieLandmarks(bs(1));
 %   save_landmark_fig(bs, posemap, uint8(im), lm, ['tmp/Landmark',im_dir(i).name,'.png'] );
 %   ll{i} = lm;
    lm = ll{i};
    [compf, compp] = Comp_lm(); %components landmarks of frontal and profile faces
    if (size(lm, 1)==68),
        l = 37; r = 48;
    else
        l = 7; r = 11;
    end
    y1 = min( lm(:, 1) ); y2 = max( lm(:,1) );
    x1 = min( lm(l:r, 2) ); x2 =max( lm(l:r,2) );
    xx = fix((x1+x2)/2); yy = fix((y1+y2)/2);
    fprintf('%d %d %d\n', i, xx, yy);
    x1 = xx -130; x2 = xx+180;
    y1 = yy - 130; y2 = yy+130;
    im = im(x1:x2,y1:y2,:);
    imshow( im );
    imwrite( im, [im_path,'cropped', im_dir(i).name]);
end