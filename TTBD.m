function [I_b] = TTBD(img, T)

% Two-Thresolded-Binary-Decomposition

range = getrangefromclass(img);
range = range(2);

I_b = cell(numel(T)-2, 1);

parfor i = 1:numel(T)-2

    lowerThresh = T(i);
    upperThresh = T(i+1);
    
    I_b{i} = img > (lowerThresh * range) & img < (upperThresh * range);
    
end


end

