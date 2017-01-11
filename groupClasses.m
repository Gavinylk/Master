function [classes] = groupClasses(img, size_train, numImgClass)

iter_s = 1:numImgClass:size_train;
iter_e = numImgClass:numImgClass:size_train;
classes = cell(size(iter_s, 2), 1);
k = 1;
set = cell(size_train, 1);

for n = 1:size_train
   
   set{n} = double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', img{n}))), [512 512]));  
   
   if(n == iter_e(k))
      
      classes{k} = cat(3, set{iter_s(k):iter_e(k)});
      k = k + 1;
      
   end
   
end


end