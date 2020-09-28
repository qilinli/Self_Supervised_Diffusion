% This illustrates the computation of entropic affinities (EAs) in two
% examples. Note that, when using sparse affinities, the runtime is mostly
% due to computing the nearest neighbors, which is O(N²), rather than to
% computing the EAs, which is O(N). If you have precomputed the nearest
% neighbors, you should pass them as input argument to ea.m as in example 2.


% Example 1: 2D dataset. We draw the bandwidth values as circles centered
% on each data point.
rng(2); X = rand(5,2); X = [X; rand(5,2)*.1+0.5];
K = 5; [P,s] = ea(X,K);

% Plot resulting bandwidths s per pixel
figure(1); clf; hold on; box on;
plot(X(:,1),X(:,2),'k.','MarkerSize',20);
viscircles(X(:,1:2),s,'LineStyle','-','LineWidth',1);
title(['Bandwidth \sigma for K=' num2str(K) ' for each data point']);
daspect([1 1 1]);


% Example 2: image (pixels = data points). We plot the bandwidth values as
% an image.
filename = 'cameraman.png';
[X,sd,nn] = imgsqd2(filename,[],5);	% Image features
K = 25;					% Desired perplexity
tic; [W,s] = ea(X,K,{sd,nn}); t = toc;	% Entropic affinities

% Plot resulting bandwidths s per pixel
figure(2); clf;
imagesc(reshape(X(:,3),128,128)); colormap(gray(256)); axis image; axis off; 
title('Original image of 128 \times 128');
figure(3); clf;
imagesc(reshape(s,128,128)); colormap jet; colorbar; axis image; axis off;
title(['Bandwidth \sigma for K=' num2str(K) '. Runtime: ' num2str(t,3) ' s']);

