function [initMeans initCovariances initPriors] = initGMM(img, numClusters)

% Run KMeans to pre-cluster the data

[m n] = size(img);
   
[initMeans, assignments] = vl_kmeans(img, numClusters, 'Algorithm','Lloyd', 'MaxNumIterations',5);

initCovariances = zeros(m,numClusters);
initPriors = zeros(1,numClusters);

% Find the initial means, covariances and priors
for i=1:numClusters
    data_k = img(:,assignments==i);
    initPriors(i) = size(data_k,2) / numClusters;

    if size(data_k,1) == 0 || size(data_k,2) == 0
        initCovariances(:,i) = diag(cov(img'));
    else
        initCovariances(:,i) = diag(cov(data_k'));
    end
end





end