%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This code is a demo for the experiments in the paper:
%%% "Affinity Learning via Self-Supervised Diffusion for Spectral Clustering"
%%% By QILIN LI (qilin.li@curtin.edu.au)
%%% Last Update 28/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
clear;

load('data/PIE_N1340_D1024.mat');
addpath('L1_ADMM/');
addpath('util/');

%%% Hyperparameters to be set
affine = 0;
alpha = 20;
k = 5;
trials = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Data processing
dataSetName = 'PIE';
data_num = size(X, 1);
class_num = length(unique(Y));
data = X;
gnd = Y;

%%% 3 test cases of face images
for i = 5:5:65
    for t = 1:trials
        X = [];
        Y = [];
        class_idx = randperm(class_num);
        for j = 1:i
            X = [X; data(gnd == class_idx(j),:)];
            Y = [Y; ones(sum(gnd == class_idx(j)),1)*j];
        end
        X = NormalizeFea(X, 1);    %%% Normalization
        
        %%% sparse representation
        CMat = admmOutlier_mat_func(X', affine, alpha);
        C = CMat(1:size(X,1),:);
        C1 = abs(C) + abs(C');   %%% L1 affinity matrix
        W_SSC = BuildAdjacency(C);
        
        %%% Affinity learning
        W_DSSC = TPG(C1, k);
        W_RDP = RDP(C1, k);
        W_SSD = SSD(C1, k, i);
        
        %%% Spectral clustering
        Y_SSC = SpectralClustering(W_SSC, i);
        Y_DSSC = SpectralClustering(W_DSSC, i);
        Y_RDP = SpectralClustering(W_RDP, i);
        Y_SSD = SpectralClustering(W_SSD, i);
        
        %%% Check accuracy
        acc_RDP(t) = clusteringAcc(Y_RDP, Y);
        acc_SSC(t) = clusteringAcc(Y_SSC, Y);
        acc_DSSC(t) = clusteringAcc(Y_DSSC, Y);
        acc_SSD(t) = clusteringAcc(Y_SSD, Y);
        fprintf("===%s, %d classes, trial %d===: ", dataSetName, i, t);
        fprintf("SSC(%.3f), DSSC(%.3f), RDP(%.3f), SSD(%.3f)===\n",...
            acc_SSC(t), acc_DSSC(t), acc_RDP(t), acc_SSD(t));
    end
    fprintf("=======================================================================================\n")
    fprintf("%s, %d classes, average: SSC(%.3f),DSSC(%.3f), RDP(%.3f), SSD(%.3f)\n",...
        dataSetName, i, mean(acc_SSC),mean(acc_DSSC),mean(acc_RDP),mean(acc_SSD));
    fprintf("=======================================================================================\n")
end