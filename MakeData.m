load('Data/Face_Training6.mat');

par.nFactor = 4;
par.psf =   fspecial('gauss', 7, 1.6);              % The simulated PSF
images_lr = uint8(zeros(80,70,100));
for i = 1:300,
    imHR = images_hr(:,:,i);
    [imHR, imLR] = get_LR( imHR, par );
    images_lr(:,:,i) = imLR;
end
save('Data/Face_Training6_addLR.mat', 'images_hr', 'images_lr', 'landmarks');