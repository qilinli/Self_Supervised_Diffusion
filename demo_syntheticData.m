%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This code is a demo for the experiments in the paper:
%%% "Affinity Learning via Self-Supervised Diffusion for Spectral Clustering"
%%% By QILIN LI (qilin.li@curtin.edu.au)
%%% Last Update 28/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
clear;

addpath('util/');
addpath('L1_ADMM/');


%%% Hyperparameters to be set
affine = 0;
alpha = 10;
k = 10;
trail = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:11
    corruption = (i-1)*0.1;
    fprintf("\n===Synthetic-corruption(%.1f): ===\n", corruption);
    
    for j = 1:trail
        fprintf("Trial %d: ", j);
        [data, Y] = dataGenerator_subspaceData(500, 50, 5);
        X = addGaussianNoise(data', corruption)';
        
        %%% Data processing
        X = NormalizeFea(X, 1);    %%% Normalization
        data_num = size(X, 1);
        class_num = length(unique(Y));
        
        %%% Affinity learning
        CMat = admmOutlier_mat_func(X', affine, alpha);
        C = CMat(1:data_num,:);
        W_SSC = BuildAdjacency(thrC(C,1));
        
        W_DSSC = IterativeDiffusionTPGKNN(W_SSC, k);
        W_SSD = ADP(W_SSC, k, class_num);
        
        %%% Spectral clustering
        Y_SSC = SpectralClustering(W_SSC, class_num);
        Y_DSSC = SpectralClustering(W_DSSC, class_num);
        Y_SSD = SpectralClustering(W_SSD, class_num);
        
        %%% Check accuracy
        acc_SSC(j) = clusteringAcc(Y_SSC, Y);
        acc_DSSC(j) = clusteringAcc(Y_DSSC, Y);
        acc_SSD(j) = clusteringAcc(Y_SSD, Y);
        
        fprintf("ACC-SSC(%.3f), DSSC(%.3f), SSD(%.3f)\n",...
            acc_SSC(j), acc_DSSC(j), acc_SSD(j));

    end
    fprintf("=======================================================================================\n")
    fprintf("Corruption(%.1f) ACC-average: SSC(%.3f),DSSC(%.3f), SSD(%.3f)\n",...
        corruption, mean(acc_SSC),mean(acc_DSSC),mean(acc_SSD));
    fprintf("Corruption(%.1f) STD: Gaussian(%.3f), TPG(%.3f), SSC(%.3f),DSSC(%.3f), SSD(%.3f)\n",...
        corruption, std(acc_SSC),std(acc_DSSC),std(acc_SSD));
end


