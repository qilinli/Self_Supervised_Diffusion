%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This code is a demo for the experiments in the paper:
%%% "Affinity Learning via Self-Supervised Diffusion for Spectral Clustering"
%%% By QILIN LI (qilin.li@curtin.edu.au)
%%% Last Update 28/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
clear;

load('data/COIL20Resized_N1440_D1024.mat');
addpath('L1_ADMM/');
addpath('util/');
addpath('entropic_affinity/');

%%% Hyperparameters to be set
perplexity = 20;
k = 5;
trials = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Data processing
dataSetName = 'COIL20';
data_num = size(X, 1);
class_num = length(unique(Y));
data = X;
gnd = Y;

%%% 3 test cases of face images
result = [];
for i = 2:class_num
    for t = 1:trials
        X = [];
        Y = [];
        class_idx = randperm(class_num);
        for j = 1:i
            X = [X; data(gnd == class_idx(j),:)];
            Y = [Y; ones(sum(gnd == class_idx(j)),1)*j];
        end
        X = NormalizeFea(X, 1);    %%% Normalization
        
        %%% Affinity learning
        [W_G,W_KNN] = adaptiveGaussian(X, 27);  %%% adaptive Gaussian affinity matrix
        [W_EA,~] = ea(X, perplexity); 
        W_EA = knnSparse(W_EA, k);
        W_KNN = knnSparse(W_KNN, k);
        W_TPG = TPG(W_G, k);
        W_RDP = RDP(W_G, k);
        W_SSD = SSD(W_G, k, i);
        
        %%% Spectral clustering
        Y_G = SpectralClustering(W_KNN, i);
        Y_EA = SpectralClustering(W_EA, i);
        Y_TPG = SpectralClustering(W_TPG, i);
        Y_RDP = SpectralClustering(W_RDP, i);
        Y_SSD = SpectralClustering(W_SSD, i);
        
        %%% Check accuracy
        acc_G(t) = clusteringAcc(Y_G, Y);
        acc_EA(t) = clusteringAcc(Y_EA, Y);
        acc_TPG(t) = clusteringAcc(Y_TPG, Y);
        acc_RDP(t) = clusteringAcc(Y_RDP, Y);
        acc_SSD(t) = clusteringAcc(Y_SSD, Y);
        
        fprintf("%s, %d classes, trial %d: ", dataSetName, i, t);
        fprintf("Gaussian(%.3f), Entropic(%.3f), TPG(%.3f), RDP(%.3f), SSD(%.3f)\n",...
            acc_G(t), acc_EA(t), acc_TPG(t), acc_RDP(t), acc_SSD(t));
    end
    fprintf("=======================================================================================\n")
    fprintf("%s, %d classes, average: Gaussian(%.3f), Entropic(%.3f),TPG(%.3f), RDP(%.3f), SSD(%.3f)\n",...
        dataSetName, i, mean(acc_G), mean(acc_EA), mean(acc_TPG),mean(acc_RDP),mean(acc_SSD));
    fprintf("=======================================================================================\n")
    result = [result; i, mean(acc_G), mean(acc_EA), mean(acc_TPG),mean(acc_RDP),mean(acc_SSD)];
end