function G = preSelectionGMM(sizeDataset, numImgClass, train_name, train_label)

% Create IMG histogramm and calculate kurtosis
iter_s = 1:numImgClass:sizeDataset;
iter_e = numImgClass:numImgClass:sizeDataset;
k = 1;
j = 1;

for i = 1:sizeDataset
   
   %[N{i},~] = histcounts(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/Kylberg', '/', train_name{i}, '/',train_label{i}))), [512 512]), 256, 'Normalization', 'probability');
   [N{i},~] = histcounts(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', train_name{i}))), [512 512]), 256, 'Normalization', 'probability');
   E(j) = kurtosis(N{i});
   if(i == iter_e(k))   
      
      [~,G(k)] = max(abs(E(:)));
      if(k~=1)G(k) = G(k)+iter_s(k);end
      k = k + 1;
      j = 0;
      E = 0;
   end
   j = j + 1;
end

end



