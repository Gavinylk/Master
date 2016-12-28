function [FV] = localHist(img, patchSize)

% Get ImageSize
[X Y] = size(img);

k = 1;
for i = 1:patchSize:X-patchSize
    for j = 1:patchSize:Y-patchSize
        
        
        [counts,~] = histcounts(img(i:i+patchSize-1, j:j+patchSize-1), 9, 'Normalization', 'probability');
        FV{k} = counts;
        k = k + 1;
    end
end
FV = cell2mat(FV);
%FV = cell2mat(FV);
%FV = FV(:)';

end