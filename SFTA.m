function [D] = SFTA(I, n)

    T = otsurec(I, n);
	
    dSize = (numel(T) * 6) - 3;
    D = zeros(1, dSize);
    pos = 1;
	
    for t = 1 : numel(T)
        thresh = T(t);
        
        Ib = im2bw(I, thresh); 
        Ib = findBorders(Ib);
        
        vals = double(I(Ib));
        
        D(pos) = hausDim(Ib);
        pos = pos + 1;
        
        D(pos) = mean(vals);
        pos = pos + 1;

        D(pos) = numel(vals);
        pos = pos + 1;
    end
    
    T = [T; 1.0];
    range = getrangefromclass(I);
    range = range(2);
    
    for t = 1 : (numel(T) - 2)
        lowerThresh = T(t);
        upperThresh = T(t + 1);
            
        Ib = I > (lowerThresh * range) & I < (upperThresh * range);
        Ib = findBorders(Ib);
        
        vals = double(I(Ib));
        
        D(pos) = hausDim(Ib);
        pos = pos + 1;
        
        D(pos) = mean(vals);
        pos = pos + 1;

        D(pos) = numel(vals);
        pos = pos + 1;
    end
	D(isnan(D)) = 0;
    
end

