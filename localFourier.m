function [F] = localFourier(img, roiSize, stride)

[X Y] = size(img);
% [FX,FY] = gradient(double(img));
% S = sqrt(FX^2+FY^2);

for i = 1:stride:X-roiSize
    for j = 1:stride:Y-roiSize

        temp = abs(fft2(img(i:i+roiSize-1, j:j+roiSize-1)));
        F(i, j) = mean(temp(:));

    end
end

F = mat2gray(F);
F = reshape(F, 1, size(F,1)*size(F,2));
F(F==0) = [];
end