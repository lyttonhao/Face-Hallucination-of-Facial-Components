function [ output ] = SR_reconstruct( Cp, Cs, testset, idx, Type, nn, im_h, im_w, par  )
 
       output = zeros(par.patch_size*par.patch_size, size(testset,2));
       weight = zeros(size(testset,2), nn);
       for ii = 1:size(testset, 2),
                t = Type(ii);
                Ip = testset(:, ii);
                Ipk = zeros(size(testset,1), nn);
                Isk = zeros(size(Cs{t},1), nn);
            for i=1:nn
                Ipk(:, i) = Cp{t}(:,idx{t}(i,ii));
                Isk(:, i) = Cs{t}(:,idx{t}(i,ii));
           %     info(img,  Pose{t}(idx{img}{t}(i,ii)) ) = info(img, Pose{t}(idx{img}{t}(i,ii)) )+1;
            end
      
          %  Coeff = ( Ipk'*Ipk + lambda*eye(nn) ) \ Ipk' * Ip;
            Coeff = Cal_Coeff( Ipk, Ip, nn, par, par.method );
            weight(ii, :) = Coeff';
            
           
       %     Coeff = mexLasso(Ip, Ipk, param);
       
            Is = Isk * Coeff;          
            
            output(:, ii) = Is;    
        end

    %    [h1, v1] = patch2data1([output(1:patch_size*patch_size, :);output(patch_size*patch_size+1:patch_size*patch_size*2, :)], im_h, im_w, 1,par.win, par.step);
    %    [l1, r1] = patch2data1([output(patch_size*patch_size*2+1:patch_size*patch_size*3, :);output(patch_size*patch_size*3+1:patch_size*patch_size*4, :)], im_h, im_w, 1,par.win, par.step);
        [output, ~] = patch2data([output;output], im_h, im_w, 1,par.win, par.step);


end

