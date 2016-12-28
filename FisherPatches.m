function [FV] = FisherPatches(img, blockSize, blockStride)

% Get ImageSize
[X Y] = size(img);
eps = 0.01;

for i = 1:blockStride:X-blockSize
    for j = 1:blockStride:Y-blockSize
        
        % Zero Mean & Illumination Invariance
        imgBlock = img(i:i+blockSize-1, j:j+blockSize-1);
        imgBlock = double(imgBlock) - double(mean(imgBlock(:)));
        
        % Avoid division through 0
        imgBlock = double(abs(imgBlock)) / (double(std(double(abs(imgBlock(:)))))+eps);
       
        
        
%         for k = 1:roiStride:blockSize-roiSize
%             for l = 1:roiStride:blockSize-roiSize
%                 
%                 roiPatch = imgBlock(k:k+roiSize-1, l:l+roiSize-1);
%                 
%                 if(numClusters > roiSize) error('Number of Clusters bigger than roiSize!'); return; end
%                 
%                 [meanVal covariance prior] = vl_gmm(roiPatch, numClusters);
%                 
%                 cell_FV{k, l} = vl_fisher(roiPatch, meanVal, covariance, prior,'Improved', 'Fast')';
%             end
%         end
        
        
        block_FV{i, j} = cell2mat(cell_FV);
        
    end
end

FV = cell2mat(block_FV);

FV = FV(:)';
end