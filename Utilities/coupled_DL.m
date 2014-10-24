function [Alphap, Alphas, Xp, Xs, Dp, Ds, Wp, Ws,f] = coupled_DL(Alphap, Alphas, Xp, Xs, Dp, Ds, Wp, Ws, par)
% Semi-Coupled Dictionary Learning
% Shenlong Wang
% Reference: S Wang, L Zhang, Y Liang and Q. Pan, "Semi-coupled Dictionary Learning with Applications in Super-resolution and Photo-sketch Synthesis", CVPR 2012

[dimX, numX]        =       size(Xp);
dimY                =       size(Alphap, 1);
numD                =       size(Dp, 2);
rho                 =       par.rho;
lambda1             =       par.lambda1;
lambda2             =       par.lambda2;
mu                  =       par.mu;
sqrtmu              =       sqrt(mu);
nu                  =       par.nu;
nIter               =       par.nIter;
t0                  =       par.t0;
epsilon             =       par.epsilon;
param.lambda       = lambda1; % not more than 20 non-zeros coefficients
param.lambda2       = lambda2;
param.mode          = 2;       % penalized formulation
param.approx=0;
param.K = par.K;
param.L = par.L;
f = 0;
for t = 1 : nIter
    % Alphat = mexLasso(Xt,D,param);
    f_prev = f;
    Alphas = mexLasso([Xs;sqrtmu * full(Alphap)], [Ds; sqrtmu * Ws],param);
    Alphap = mexLasso([Xp;sqrtmu * full(Alphas)], [Dp; sqrtmu * Wp],param);
    dictSize = par.K;
    % Update D
    for i=1:dictSize
       ai        =    Alphas(i,:);
       Y         =    Xs-Ds*Alphas+Ds(:,i)*ai;
       di        =    Y*ai';
       di        =    di./(norm(di,2) + eps);
       Ds(:,i)    =    di;
    end
    for i=1:dictSize
       ai        =    Alphap(i,:);
       Y         =    Xp-Dp*Alphap+Dp(:,i)*ai;
       di        =    Y*ai';
       di        =    di./(norm(di,2) + eps);
       Dp(:,i)    =    di;
    end
    % Update W
%     Ws = Alphap * Alphas' * inv(Alphas * Alphas' + par.nu * eye(size(Alphas, 1))) ;
%     Wp = Alphas * Alphap' * inv(Alphap * Alphap' + par.nu * eye(size(Alphap, 1))) ;    
    Ws = (1 - rho) * Ws  + rho * Alphap * Alphas' * inv(Alphas * Alphas' + par.nu * eye(size(Alphas, 1))) ;
    Wp = (1 - rho) * Wp  + rho * Alphas * Alphap' * inv(Alphap * Alphap' + par.nu * eye(size(Alphap, 1))) ;
    % Alpha = pinv(D' * D + lambda2 * eye(numD)) * D' * X;
    P1 = Xp - Dp * Alphap;
    P1 = P1(:)'*P1(:) / 2;
    P2 = lambda1 *  norm(Alphap, 1);    
    P3 = Alphas - Wp * Alphap;
    P3 = P3(:)'*P3(:) / 2;
    P4 = nu * norm(Wp, 'fro');
    fp = 1 / 2 * P1 + P2 + mu * (P3 + P4);
    
    P1 = Xs - Ds * Alphas;
    P1 = P1(:)'*P1(:) / 2;
    P2 = lambda1 *  norm(Alphas, 1);    
    P3 = Alphap - Ws * Alphas;
    P3 = P3(:)'*P3(:) / 2;
    P4 = nu * norm(Ws, 'fro'); 
    fs = 1 / 2 * P1 + P2 + mu * (P3 + P4);
    f = fp + fs;
    if (abs(f_prev - f) / f < epsilon)
        break;
    end
    fprintf('Energy: %d\n',f);
    save tempDict_SR_NL Ds Dp Ws Wp par param i;
    % fprintf('Iter: %d, E1 : %d, E2 : %d, E : %d\n', t, mu * (P1 + P2), (1 - mu) * (P3 + P4), E);
end
    