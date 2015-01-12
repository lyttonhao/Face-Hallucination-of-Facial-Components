
for i = 1:6,
    im = imread(['Resulttmp1/',num2str(i),'.jpg']);
    imHR = imread(['Resulttmp1/g',num2str(i),'.jpg']);
    im = im(:,:,1);
    imHR = imHR(:,:,1);
    imHR = imHR(1:size(im,1), 1:size(im,2) );
    diff = abs(im-imHR);
    diff(diff > 50) = 50;
    figure, imshow(diff, []);
end