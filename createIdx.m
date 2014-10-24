function [Mid] = createIdx( n, m, s )
    Mid = zeros( n, m );
    Mid(1:n-s+1, 1:m-s+1) = reshape(1:(n-s+1)*(m-s+1), [n-s+1, m-s+1]);
end