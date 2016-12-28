function [F] = MultiFeature(img, roiSize, stride, patchSize, binSize)

[X Y] = size(img);

[Gx, Gy] = imgradientxy(img);
[Gmag, Gdir] = imgradient(Gx, Gy);

four = abs(fft2(img));

sd = zeros(size(Gmag));

for i = 1:stride:X-roiSize
    for j = 1:stride:Y-roiSize
        
        std_temp = img(i:i+roiSize-1, j:j+roiSize-1);
        
        sd(i, j) = std(std_temp(:));
        
        
    end
end

Gmag = mat2gray(Gmag);
Gdir = mat2gray(Gdir);
four = mat2gray(four);
sd = mat2gray(sd);

imgGmag = zeros(patchSize^2, Y*X/patchSize^2);
imgGdir = zeros(patchSize^2, Y*X/patchSize^2);
imgfour = zeros(patchSize^2, Y*X/patchSize^2);
imgsd = zeros(patchSize^2, Y*X/patchSize^2);

ind = 1;
for i=1:patchSize:Y
    for j=1:patchSize:X
       patch_Gmag = Gmag(i:i+patchSize-1,j:j+patchSize-1);
       patch_Gdir = Gdir(i:i+patchSize-1,j:j+patchSize-1);
       patch_four = four(i:i+patchSize-1,j:j+patchSize-1);
       patch_sd = sd(i:i+patchSize-1,j:j+patchSize-1);
       
       imgGmag(:,ind) = patch_Gmag(:);
       imgGdir(:,ind) = patch_Gdir(:);
       imgfour(:,ind) = patch_four(:);
       imgsd(:,ind) = patch_sd(:);
       
       ind = ind + 1;
    end
end

[F_Gmag,~] = histcounts(imgGmag, binSize,'Normalization', 'probability');
[F_Gdir,~] = histcounts(imgGdir, binSize,'Normalization', 'probability');
[F_four,~] = histcounts(imgfour, binSize,'Normalization', 'probability');
[F_sd,~] = histcounts(imgsd, binSize,'Normalization', 'probability');




F = horzcat(F_Gmag, F_Gdir, F_four, F_sd);

end