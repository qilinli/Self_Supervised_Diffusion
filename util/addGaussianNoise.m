function x=addGaussianNoise(x,p)

% add p percentage noise to x%

[m,n]=size(x);
N=ceil(n*p);
idx=ceil(rand(1,N)*n);
if (length(idx)~=0)
    for i=1:length(idx)
        x(:,idx(i))=x(:,idx(i))+normrnd(0,0.3*norm(x(:,idx(i))),m,1);
    end
end
return