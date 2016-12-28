function [F] = CrossFeature_sparse(img, roiSize, stride)

%get image size hh
X = size(img, 1);
Y = size(img, 2);

% get roi & corr
maskTL = img(1:roiSize, 1:roiSize);    
maskTR = img(1:roiSize, X-roiSize:X-1);    
maskM = img(floor(X/2)-floor(roiSize/2):floor(X/2)+floor(roiSize/2)-1, floor(Y/2)-floor(roiSize/2):floor(Y/2)+floor(roiSize/2)-1);    
maskBL = img(Y+1-roiSize:Y, 1:roiSize);    
maskBR = img(X+1-roiSize:X, Y+1-roiSize:Y);    

% Zero mean
maskTL = maskTL - mean(maskTL(:));
maskTR = maskTR - mean(maskTR(:));
maskM = maskM - mean(maskM(:));
maskBL = maskBL - mean(maskBL(:));
maskBR = maskBR - mean(maskBR(:));
img = img - mean(img(:));

% Illumination Invariance through STD division
maskTL = maskTL / std(double(maskTL(:)));
maskTR = maskTR / std(double(maskTR(:)));
maskM = maskM / std(double(maskM(:)));
maskBL = maskBL / std(double(maskBL(:)));
maskBR = maskBR / std(double(maskBR(:)));
img = img / std(double(img(:)));

outTL = xcorr2_fft(maskTL, img);               
outTR = xcorr2_fft(maskTR, img);   
outM = xcorr2_fft(maskM, img);   
outBL = xcorr2_fft(maskBL, img);   
outBR = xcorr2_fft(maskBR, img);   



        
% calc mean of the 5 values above and scale it to [0 1]
%FV = (imresize(outTL, [512 512]) + imresize(outTR, [512 512]) + imresize(outM, [512 512]) + imresize(outBL, [512 512]) + imresize(outBR, [512 512]))./5;

% Normalize patches
outTL = abs(outTL)./max(abs(outTL(:)));
outTR = abs(outTR)./max(abs(outTR(:)));
outTL = abs(outM)./max(abs(outM(:)));
outTL = abs(outBL)./max(abs(outBL(:)));
outTL = abs(outBR)./max(abs(outBR(:)));

% Extract feature vector from input image
F = [norm(outTL, 1), norm(outTR, 1), norm(outM, 1), norm(outBL, 1), norm(outBR, 1), std(outTL(:)), std(outTR(:)), std(outM(:)), std(outBL(:)), std(outBR(:))];
      
        
     

end