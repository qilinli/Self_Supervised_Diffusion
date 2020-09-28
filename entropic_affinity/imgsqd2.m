% [X,sqd,N,S,r,V,H] = imgsqd2(imgfile[,S,r])
% Compute feature vector and pairwise square distances between image pixels,
% ensuring each pixel has exactly the same number of neighbours.
%
% Unlike in imgsqd.m, where the pixels at the boundaries have a smaller
% neighbourhood (because it is clipped), in imgsqd2.m each pixel has exactly
% (2*r+1)^2-1 neighbours, because for pixels at the boundaries we shift the
% neighbourhood so it fits inside the image. We use a square neighbourhood
% shape to simplify the calculations.
%
% "imgfile" contains an image of N = VxH pixels with intensities or RGB
% colours in 0..255.
%
% In:
%   imgfile: image file name.
%   S: 1xK array, the relative scales of the features. Default: [1 1 0.01]
%      for greyscale images and [1 1 1 1 1] for colour images.
%   r: window size (in pixels). If <= 0, then sqd won't be computed.
%      Default: 5.
% Out:
%   X: Nx? array of feature vectors for each pixel 1:N:
%    . For greyscale images, Nx3 array where each row is
%      (x,y,I) = (positionx in [1,H], positiony in [1,V], intensity in [0,1]).
%      Each feature is further divided by (S1,S2,S3) for relative scaling.
%    . For colour images, Nx5 array with columns 3-5 being the CIE L*u*v*
%      values (uniform colour space), divided by (S1,...,S5).
%   D2: N x k matrix of sorted square distances to the k nearest neighbors.
%   nn: N x k matrix of indices of the corresponding nearest neighbors.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2013 by Max Vladymyrov and Miguel A. Carreira-Perpinan

function [X,D2,nn] = imgsqd2(imgfile,S,r)

[F,map] = imread(imgfile);
iscolour = (ndims(F) > 2) | (size(map,2) == 3);

% ---------- Argument defaults ----------
if ~exist('S','var') | isempty(S)
  if iscolour
    S = [1 1 1 1 1];					% Colour image
  else
    S = [1 1 0.01];					% Greyscale image
  end
end;
if ~exist('r','var') | isempty(r) r = 5; end;
% ---------- End of "argument defaults" ----------


% Image features
V = size(F,1); H = size(F,2); [FH,FV] = meshgrid(1:H,1:V);
if iscolour				% Colour image
  if ndims(F) > 2			% RGB
    X = [FV(:) FH(:) rgb2luv(double(reshape(F,H*V,3))/255)]*diag(S.^(-1));
  else					% Indexed
    X = [FV(:) FH(:) rgb2luv(map(1+F(:),:))]*diag(S.^(-1));
  end
else					% Greyscale image
  X = [FV(:) FH(:) double(F(:))/255]*diag(S.^(-1));		% Intensity
  % $$$   F = rgb2luv(repmat(double(F(:))/255,1,3));
  % $$$   X = [FV(:) FH(:) F(:,1)]*diag(S.^(-1));		% L* value
end
N = size(X,1);
if nargout==1 return; end;

% Matrix of square distances
if r > 0
  % Build list of pixel indices inside the window
  rr = floor(r);
  K = (2*rr+1)^2;
  D2 = zeros(V*H,K-1);
  nn = zeros(V*H,K-1);
  [wH,wV] = meshgrid(-rr:rr,-rr:rr); wH = wH(1:end); wV = wV(1:end);
  w = [wV' wH'];
  
  % Pairwise distances
  x2 = sum(X.^2,2);
  for v=1:V
    for h=1:H
      tmpH = w(:,1)+h; tmpV = w(:,2)+v;
      
      % move along the boundaries
      if h-rr<=0 tmpH = tmpH-h+rr+1; end
      if h+rr>=H tmpH = tmpH+H-h-rr; end
      if v-rr<=0 tmpV = tmpV-v+rr+1; end
      if v+rr>=V tmpV = tmpV+V-v-rr; end
      
      hh = v + (h-1).*V;	% Faster than: hh = sub2ind([V H],v,h);
      vv = tmpV + (tmpH-1).*V;	% Faster than: vv = sub2ind([V H],tmpV,tmpH);
      ss = x2(hh) + x2(vv) - 2*X(vv,:)*X(hh,:)';
      [ss,ind] = sort(ss);
      D2(hh,:) = ss(2:K); nn(hh,:) = vv(ind(2:K));
    end
  end
end

