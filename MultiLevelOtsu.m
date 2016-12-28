function [T] = MultiLevelOtsu(img, n, k1, k2)
    
    % Calculate Histogram of Image
    H = imhist(uint8(img));
    
    a = size(H, 2);
    b = size(H, 1);
    T = zeros(1, n);
    
    % Case if number of threshs < 4
    R = (a:b)';
    mu = mean(R(a:b));
    sigma = std(R(a:b));
    
    % Step 1 : Repeat Step 2-5 n/2-1 times
    for i = 1:ceil(n/2-1)
        
        % Step 2 : Find mean & std of all pixels in R
        mu = mean(R(a:b));
        sigma = std(R(a:b));
        
        % Step 3 : Sub-ranges T1 & T2
        T1 = floor(mu - k1*sigma);
        T2 = floor(mu + k1*sigma);
        
        % Step 4 : Threshold Calculation
        if( T(i) ~= 0) i = i+1; end
        
        if ( T(i) ~= 0) i = i+1;end
        
        T(i) = sum(H(a:T1).*R(a:T1))/sum(H(a:T1));
        if( T(i) ~= 0 ) i = i+1; end
        
        if ( T(i) ~= 0) i = i+1;end
       
        T(i) = sum(H(T2:b).*R(T2:b))/sum(H(T2:b));
        
        % Step 5 : Update a & b
        a = floor(T1+1);
        b = floor(T2-1); 
      
    end
    
    % Step 6 : Step 4
    T1 = mu;
    T2 = mu+1;
    T(end-1) = sum(H(a:T1).*R(a:T1))/sum(H(a:T1));
    T(end) = sum(H(T2:b).*R(T2:b))/sum(H(T2:b));
    T = sort(floor(T(1:end)));
    T = T/size(H, 1); 
  
end
