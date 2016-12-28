function [F] = stdClassify(img, roiSize, stride)


[X Y] = size(img);

img = mat2gray(img);

for i = 1:stride:X-roiSize
    for j = 1:stride:Y-roiSize
        
        
           temp =  img(i:i+roiSize-1, j:j+roiSize-1);
           F(i, j) = std(temp(:));
        
    end
end

F = reshape(F, 1, size(F,1)*size(F,2));
F(F==0) = [];


end