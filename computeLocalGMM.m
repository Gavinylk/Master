function [m c p] = computeLocalGMM(gmm, sizeData, type)


iter_s = 1:40:sizeData;
iter_e = 40:40:sizeData;
m = cell(47, 1);
c = cell(47, 1);
p = cell(47, 1);

for n = 1:47
    
    if strcmp(type, 'mean')
        m{n} = mean(cat(3, gmm{iter_s(n):iter_e(n), 1}), 3);
        c{n} = mean(cat(3, gmm{iter_s(n):iter_e(n), 2}), 3);
        p{n} = mean(cat(3, gmm{iter_s(n):iter_e(n), 3}), 3);
    end
    
    if strcmp(type, 'median')
        
        m{n} = median(cat(3, gmm{iter_s(n):iter_e(n), 1}), 3);
        c{n} = median(cat(3, gmm{iter_s(n):iter_e(n), 2}), 3);
        p{n} = median(cat(3, gmm{iter_s(n):iter_e(n), 3}), 3);
          
        
    end
    
    if strcmp(type, 'STD')
        
        
        m{n} = std(cat(3, gmm{iter_s(n):iter_e(n), 1}), [], 3);
        c{n} = std(cat(3, gmm{iter_s(n):iter_e(n), 2}), [], 3);
        p{n} = std(cat(3, gmm{iter_s(n):iter_e(n), 3}), [], 3);
        
    end
    
        
end

end