%load('Data/ExampleDataForLoad_ScalingFactor3.mat');
addpath('lib');
addpath('lib/face1.0-basic');

im = imread('Data/Face_Testing1/1.png');
im = imread('Result/s16_bicubic1.png');
%im = rgb2ycbcr( im);
modelname = 'mi';
bs = [];
imshow(im);
try
        [bs, posemap] = F2_ReturnLandmarks( im, modelname );
catch err
         disp('Error: The UCI algorithm does not detect a face.');
end
landmark_test = F4_ConvertBStoMultiPieLandmarks(bs(1));

 bshownumber = true;
        bdrawpose = true;
        bvisible = false;
        c = bs(1).c;    %we assume there is only one face in the image.
        val_pose = posemap(c);
        str_pose = sprintf('%d',val_pose);        
        hfig = U21b_DrawLandmarks_Points_ReturnHandle(im,landmark_test,str_pose,bshownumber,bdrawpose,bvisible);
        fn_save = sprintf('re_UCI.png');
        saveas(hfig, fn_save);
        close(hfig);


[n,m,k] = size(exampleimages_hr);
H = double(reshape(exampleimages_hr, [n*m, k]));
[nl,ml,k] = size(exampleimages_lr);
L = double(reshape(exampleimages_lr, [nl*ml, k]));
%R = H/L;
csnr(R*L(:,1), H(:,1), 0,0)


%{
comp{1} = 49:68;        %mouth
comp{2} = 18:27;        %eyebrows
comp{3} = 37:48;        %eyes
comp{4} = 28:36;        %nose
setname{1} = 'mouth';
setname{2} = 'eyebrows';
setname{3} = 'eyes';
setname{4} = 'nose';

HR = exampleimages_hr(:,:,22);
lm = landmarks(:,:,22);
for i = 1:1,
    y1 = min(lm(comp{i},1));
    y2 = max(lm(comp{i},1));
    x1 = min(lm(comp{i},2));
    x2 = max(lm(comp{i},2));
end
figure, imshow(HR);
for i = fix(x1):fix(x2),
    for j = fix(y1):fix(y2),
        HR(i,j) = 0;
    end
end
im = HR(fix(x1):fix(x2),fix(y1):fix(y2));
figure, imshow(im);
figure, imshow(HR);
%}