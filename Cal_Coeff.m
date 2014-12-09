function [ Coeff ] = Cal_Coeff( Ipk, Ip, nn, par, method )

param = par.param;
switch method
    case 'Ridge'   %L2
        Coeff = ( Ipk'*Ipk + param.lambda*eye(nn) ) \ (Ipk') * Ip;
    case 'LLE'
        A = repmat( Ip, [1, nn] ) - Ipk;
        Q = A' * A;
        R = pinv(Q);
        Coeff = sum(R, 2) / (sum(sum(R))+eps);
    case 'LS'    %Least Square
        Coeff = Ipk \ Ip;
    case 'LASSO'    %L1
        Coeff = mexLasso( Ip, Ipk, param );
    case 'LLC'      %:Locality-constrainted Linear Codeing (LLC)
        A = repmat( Ip, [1, nn] ) - Ipk;
        C = A' * A;
        C = C + eye(nn, nn) * param.tu * trace( C );
        Coeff = C \ ones(nn, 1);   
        Coeff = Coeff / sum(Coeff);
end


end