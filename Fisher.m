function [FV] = Fisher(img, m, c, p)
   
    
    
    FV = vl_fisher(double(img), m, c, p, 'Improved');
    FV = abs(FV);
    [FV, ~] = histcounts(FV, 256);
    FV(FV==0) = [];
    FV(1) = [];
    FV = abs(FV)/norm(FV, 1);

end