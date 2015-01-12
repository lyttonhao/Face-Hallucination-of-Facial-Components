function  [mW, W1] =  Compute_NLM_Matrix_inter( im1, im2, ws, vx, vy )
S       =  12;
S2      =  12;
f       =  ws;
t       =  floor(f/2);
nv      =  10;  %par.nblk;
hp      =  65;

e_im1    =  padarray( im1, [t t], 'symmetric' );
e_im2    =  padarray( im2, [t t], 'symmetric' );
[h w]   =  size( im1 );
nt      =  (nv)*2*h*w;
R       =  zeros(nt,1);
C       =  zeros(nt,1);
V       =  zeros(nt,1);

L       =  h*w;
X       =  zeros(f*f, 2*L, 'single');

% For the Y component
k    =  0;
for i  = 1:f
    for j  = 1:f
        k    =  k+1;
        blk1  =  e_im1(i:end-f+i,j:end-f+j);
        blk2  =  e_im2(i:end-f+i,j:end-f+j);
        X(k,:) =  [blk1(:)', blk2(:)'];
    end
end

% Index image
I1    =   reshape((1:L), h, w);
I2    =   reshape((L+1:2*L), h, w);
X    =   X'; 
f2   =   f^2;

cnt     =  1;
for  row  =  1 : h
    for  col  =  1 : w
        
        off_cen  =  (col-1)*h + row;
        
        rmin    =   max( row-S, 1 );
        rmax    =   min( row+S, h );
        cmin    =   max( col-S, 1 );
        cmax    =   min( col+S, w );
         
        idx1     =   I1(rmin:rmax, cmin:cmax);
        
        x1 = round(max( 1, col + vx(row, col) - S2 ));  x1 = min(w, x1);
        x2 = round(min( w, col + vx(row, col) + S2 )); x2 = max(1, x2);
        y1 = round(max( 1, row + vy(row, col) - S2 ));   y1 = min(h, y1);
        y2 = round(min( h, row + vy(row, col) + S2 )); y2 = max(1, y2);
        idx2 = I2(y1:y2, x1:x2);
        idx = [idx1(:); idx2(:)];
        
        B       =   X(idx, :);        
        v       =   X(off_cen, :);
        
        
        dis     =   (B(:,1) - v(1)).^2;
        for k = 2:f2
            dis   =  dis + (B(:,k) - v(k)).^2;
        end
        dis   =  dis./f2;
        [val,ind]   =  sort(dis);        
        dis(ind(1))  =  dis(ind(2));        
        wei         =  exp( -dis(ind(1:nv))./hp );
        
        R(cnt:cnt+nv-1)     =  off_cen;
        C(cnt:cnt+nv-1)     =  idx( ind(1:nv) );
        V(cnt:cnt+nv-1)     =  wei./(sum(wei)+eps);
        cnt                 =  cnt + nv;
        
    end
end
R    =  R(1:cnt-1);
C    =  C(1:cnt-1);
V    =  V(1:cnt-1);
W1   =  sparse(R, C, V, 2*h*w, 2*h*w);

R    =  zeros(h*w,1);
C    =  zeros(h*w,1);
V    =  zeros(h*w,1);

R(1:end)  =  1:h*w;
C(1:end)  =  1:h*w;
V(1:end)  =  1;
mI        =  sparse(R,C,V,2*h*w,2*h*w);
mW        =  mI - W1;




