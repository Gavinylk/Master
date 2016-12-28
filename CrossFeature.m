function [FV] = CrossFeature(img, roiSize, stride)

%get image size
X = size(img, 1);
Y = size(img, 2);

eps = 0.01;

img = gpuArray(img);

% get roi & corr
maskTL = gpuArray(img(1:roiSize, 1:roiSize));    
maskTR = gpuArray(img(1:roiSize, X-roiSize:X-1));    
maskM = gpuArray(img(floor(X/2)-floor(roiSize/2):floor(X/2)+floor(roiSize/2), floor(Y/2)-floor(roiSize/2):floor(Y/2)+floor(roiSize/2)));    
maskBL = gpuArray(img(Y+1-roiSize:Y, 1:roiSize));    
maskBR = gpuArray(img(X+1-roiSize:X, Y+1-roiSize:Y));    

% Zero mean
maskTL = maskTL - mean(maskTL(:));
maskTR = maskTR - mean(maskTR(:));
maskM = maskM - mean(maskM(:));
maskBL = maskBL - mean(maskBL(:));
maskBR = maskBR - mean(maskBR(:));
img = img - mean(img(:));

% Illumination Invariance through STD division
maskTL = double(abs(maskTL)) / double(std(double(abs(maskTL(:)))) + eps);
maskTR = double(abs(maskTR)) / double(std(double(abs(maskTR(:)))) + eps);
maskM = double(abs(maskM)) / double(std(double(abs(maskM(:)))) + eps);
maskBL =double(abs(maskBL)) / double(std(double(abs(maskBL(:)))) + eps);
maskBR = double(abs(maskBR)) / double(std(double(abs(maskBR(:)))) + eps);
img = double(abs(img)) / double(std(double(abs(img(:)))) + eps);


FV1 = xcorr2(maskTL, img);               
FV2 = xcorr2(maskTR, img);   
FV3 = xcorr2(maskM, img);   
FV4 = xcorr2(maskBL, img);   
FV5 = xcorr2(maskBR, img);  

FVCat = cat(3, FV1, FV2, FV3, FV4, FV5);

FV = gather(median(FVCat, 3));
% for i = 1:stride:X-roiSize
%     for j = 1:stride:Y-roiSize
%         
% 
%         
%         % calc L2-Norm of the 5 values above
%         FV(i, j) = norm([abs(outTL(i:i+roiSize-1, j:j+roiSize-1)), abs(outTR(i:i+roiSize-1, j:j+roiSize-1)), abs(outM(i:i+roiSize-1, j:j+roiSize-1)), abs(outBL(i:i+roiSize-1, j:j+roiSize-1)), abs(outBR(i:i+roiSize-1, j:j+roiSize-1))], 2);
%             
%       
%         
%         
%     end
% end

% % Generate Histogram
% [FV,edges] = histcounts(FV, 'Normalization', 'probability');
% % Dispatch zeros
% FV(FV==0) = [];
end

