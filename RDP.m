function [WW] = RDP(W, K)

n = size(W, 1);          %%% number of data points
I = eye(n);              %%%% identity matrix of size n
W = W ./ repmat(sum(W, 2)+eps, 1, n);
S = knnSparse(W, K);

WW = S; 
maxIter = 50;
epsilon = 1e-2;        %%% convergence threshold
alpha = 0.8;
for t = 1:maxIter
    temp = alpha*S*WW*S' + (1-alpha)*S;
    if norm(temp-WW,'fro') < epsilon, break; end  
    WW = temp;   
end

%% Post-processing, useful for spectral clustering
WW = WW - diag(diag(WW));
WW = (WW + WW')/2;