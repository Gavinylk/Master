function [A] = zernikeFeature(img, n, m)

    %get image size
    X = size(img, 1);
    Y = size(img, 2);

    binary = cell(7, 1);
    A = cell(7, 1);
    normedImg = cell(7, 1);
    
    % Normalize images for invaraince of translation and scaling
    normedImg = imnorm(img);
        
    
    
    % Multithreshold with different thresholds
    thresh = double(multithresh(normedImg, 7));
    
    for i = 1:7
        
        binary{i} = im2bw(img, thresh(i));
        imshow(binary{i});
    end
     
    
    % Zernike moments for ROIs of different binaries and average them
    for k = 1:7
        
        
        [~,A{k},~] = Zernikmoment(binary{k}, n, m);
        
      
    end
    
    
    % Average all zernike moments from the cell array
    A = cat(3, A{:});
    A = mean(A, 3);
    A(A==0) = [];
end