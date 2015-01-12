input = 'Data/Face_Testing4/';
genfold =  'Data/Face_Testing4_input/';


im_dir = dir( fullfile(input, '*.png') );
im_num = length( im_dir );

par.nFactor = 4;
par.psf =   fspecial('gauss', 7, 1.6);              % The simulated PSF
fid = fopen([genfold, 'test.txt'], 'w');
for img = 1:im_num,
    imHR = imread( fullfile(input, im_dir(img).name) );  

    
    [im_h, im_w, ~] = size(imHR);
    im_h = floor((im_h )/par.nFactor)*par.nFactor ;
    im_w = floor((im_w )/par.nFactor)*par.nFactor ;
    imHR=double(imHR(1:im_h,1:im_w,:));
    
    psf                =    par.psf;              % The simulated PSF
    imLR = Blur('fwd', imHR, psf);
    imLR           =   imLR(1 : par.nFactor : im_h, 1 : par.nFactor : im_w, :);

    imwrite(uint8(imLR), [genfold, im_dir(img).name]);
    fwrite( fid, [num2str(img), ' ', im_dir(img).name, char(10)] );
end
fclose(fid);