Matlab code for fast computation of Gaussian entropic affinities

(C) 2013 by Max Vladymyrov and Miguel A. Carreira-Perpinan
    Electrical Engineering and Computer Science
    University of California, Merced
    http://eecs.ucmerced.edu

References:
- The paper introducing the entropic affinities (EAs):
    G. E. Hinton & S. Roweis: "Stochastic Neighbor Embedding", NIPS 2002.
- The paper introducing fast algorithms to compute EAs:
    M. Vladymyrov & M. A. Carreira-Perpinan: "Entropic affinities:
    properties and efficient numerical computation", ICML 2013.
  This paper presents several algorithms to compute EAs. This version of
  the code implements just one of them, which is simple and typically
  pretty fast. Specifically, it uses Newton's method for root finding and
  a sequential order of points based on a K-nearest-neighbor density
  estimate.

Given a dataset X and desired perplexity K, a typical use of this code is:
  [W,s] = ea(X,K);
where W is the EAs matrix and s the per-point bandwidths. If you want W to
be sparse, specify the number of neighbors k:
  [W,s] = ea(X,K,k);
If you have precomputed squared distances run the code as
  [W,s] = ea(X,K,{D2,nn});
where D2 is an N x k matrix of sorted square distances to the k nearest
neighbors and nn is an N x k matrix containing the indices of the
corresponding nearest neighbors.

Entry W(n,m) in the matrix W of EAs is a Gaussian affinity using the
bandwidth for the nth point, and normalized by the row sum of W. Thus, W
is a stochastic matrix. It is the random-walk matrix commonly used in
machine learning, but where each row (data point) has its own bandwidth in
the Gaussian kernel.

See demo.m for detailed examples of usage.

List of functions:
- ea.m: computes entropic affinities.
- demo.m: examples using a 2D dataset and an image.
See each function for detailed usage instructions.

The following are used internally by other functions:
- eabounds.m: computes bounds on the bandwidth for every data point.
- eabeta.m: computes log(beta) value for a data point (root finding).
- nnsqdist.m: computes nearest-neighbor square distances.
- imgsqd2.m: computes feature vector and pairwise square distances between
  image pixels.

