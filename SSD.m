function A = SSD(W, K, class_num)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This code use a label information guided diffusion process to propagate
%%% label, with a learned affinity matrix as the by-product.
%%% Input:
%%%       --- W:              intitial weight matrix (symmetric and non-negtive)
%%%       --- K:              size of k-nearest neighbor
%%%       --- class_num:      number of cluster
%%% Output:
%%%       --- A:              learned affinity matrix
%%% By QILIN LI (qilin.li@curtin.edu.au)
%%% Last Update 28/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = size(W, 1);          %%% number of data points
I = eye(n);              %%%% identity matrix of size n

% Pre-processing of weight matrix W
d = sum(W, 2);
D = diag(d + eps);
W = W - diag(diag(W)) + I;   %%% use node degree as self-affinity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Normalization  %%%%%%%%%%%%%%%%%
% S = W ./ repmat(sum(W, 2)+eps, 1, n);

d = sum(W,2);
D = diag(d + eps);
S = D^(-1/2)*W*D^(-1/2);      %%% Symmetric normalization is better
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% KNN sparse
S = knnSparse(S, K);

%%% Main iteration
maxIter_out = 20;
maxIter_in = 50;
alpha = 0.99;
epsilon = 5e-2;

%%% Initialization
Z = zeros(n,n);
A = S;

for t = 1:maxIter_out
     
    %%% Update A
    A_old = A;
    for ii = 1:maxIter_in
        temp = alpha*S*(A + Z)*S' + (1-alpha)*I;
        if norm(temp - A, 'fro') < epsilon, break; end
        A = temp;
    end
%    

    err = norm(A - A_old, 'fro');
%     fprintf("%.2f...", err);
    if err < epsilon, break; end
    
    % update Z
    Z = label_similarity(A, class_num); 
    d = sum(Z,2);
    D = diag(d + eps);
    Z = D^(-1/2)*Z*D^(-1/2);
    Z = knnSparse(Z, K); 
    a = 10;
end


%% Post-processing, useful for spectral clustering
A = A - diag(diag(A));
A = (A + A')/2;
