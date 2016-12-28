function [m c p] = computeLocalGMM(gmm, size, type)


iter_s = 1:40:size;
iter_e = 40:40:size;
m = cell(47, 1);
c = cell(47, 1);
p = cell(47, 1);

for n = 1:47
    
    m{n} = cat(3, gmm{iter_s(n):iter_e(n), 1});
    c{n} = cat(3, gmm{iter_s(n):iter_e(n), 2});
    p{n} = cat(3, gmm{iter_s(n):iter_e(n), 3});
    
end


if strcmp(type, 'L1')
        
        m{n} = arrayfun(@(c) norm(c{1}, 1), m);
        c{n} = cellfun(@(n) norm(c(:,:,n), 1), 1:size(c,3));
        p{n} = cellfun(@(n) norm(p(:,:,n), 1), 1:size(p,3));
        %m{n} = norm(cat(3, gmm{iter_s(n):iter_e(n), 1}), 3);
        %c{n} = norm(cat(3, gmm{iter_s(n):iter_e(n), 2}), 3);
        %p{n} = norm(cat(3, gmm{iter_s(n):iter_e(n), 3}), 3);
        
end

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
    
    if strcmp(type, 'L1')
        
        
        m{n} = norm(cat(3, gmm{iter_s(n):iter_e(n), 1}), 3);
        c{n} = norm(cat(3, gmm{iter_s(n):iter_e(n), 2}), 3);
        p{n} = norm(cat(3, gmm{iter_s(n):iter_e(n), 3}), 3);
        
    end
    %if(n == iter_s(k) && k ~= 47) k = k + 1; end
        
end

end