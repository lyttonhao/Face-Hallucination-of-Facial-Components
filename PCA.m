function [Cp, V_pca] = PCA( Cp )
    C = double( Cp * Cp' );
    [V, D] = eig( C );
    D = diag( D );
    D = cumsum(D) / sum(D);
    k = find( D >= 1e-3, 1);
    V_pca = V(:, k:end);
    
    
    
    Cp = V_pca' * Cp;
end