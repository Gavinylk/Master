function [LBP] = LBP(img, radius, P, noHist)

% Pattern Structure
[m, n] = size(img);
pattern = 1:P; 

 parfor i = radius+1:m-radius
    for j = radius+1:n-radius
        
        g_p_x = floor(radius*cos((2*pi/P)*(pattern-1))+i);
        g_p_y = ceil(radius*sin((2*pi/P)*(pattern-1))+j);
        a = mod(radius*sin((2*pi/P)*(pattern-1)), 1);
        b = mod(radius*cos((2*pi/P)*(pattern-1)), 1);
        
        % Interpolation Bilinear
        pA = double(img(sub2ind(size(img), g_p_x, g_p_y)));
        pB = double(img(sub2ind(size(img), g_p_x, g_p_y)));
        pC = double(img(sub2ind(size(img), g_p_x, g_p_y)));
        pD = double(img(sub2ind(size(img), g_p_x, g_p_y)));
        
        vp = pA.*(1-a).*(1-b)+pB.*a.*(1-b)+pC.*(1-a).*b+pD.*a.*b;
        
        % Compare interpolated value gp with center value gc
        binaryLBP = vp >= img(i, j); 
        
        % Only calculate uniform pixel pattern else set to 0
        
        if(sum((abs(diff(binaryLBP)))) < 3)
        
            LBPImage(i, j) = sum((2.^(pattern-1)).*binaryLBP);
            
        else
            LBPImage(i, j) = 0;
        end
        
    end
 end

if (noHist == 0)
    % Normalized Histogram & Mapping
    [LBP_Pre, ~] = histcounts(LBPImage, 2^P);%imhist(uint8(LBPImage)); 
    LBP_Pre(LBP_Pre==0) = []; % Delete Entries with 0 as element & the first entry with GV 0
    LBP_Pre(1) = [];
    LBP = abs(LBP_Pre)/norm(LBP_Pre, 1);%abs(LBP_Pre)./max(abs(LBP_Pre(:))); % Feature Vector
    
   % LBP = LBP_Pre';
    
elseif (noHist == 1)
  
    LBP = LBPImage; 
    LBP = abs(LBP)./max(abs(LBP(:)));
    LBP = localHist(LBP, 9);
end

end

