function [LBP] = LBP(img, radius, P, noHist)

% Pattern Structure
[m, n] = size(img);
pattern = 1:P; 

LBPImage = zeros(m-radius, n-radius);

a = mod(radius*sin((2*pi/P)*(pattern-1)), 1);
b = mod(radius*cos((2*pi/P)*(pattern-1)), 1);

p_x = radius*cos((2*pi/P)*(pattern-1));
p_y = radius*sin((2*pi/P)*(pattern-1));

 parfor i = radius+1:m-radius
    for j = radius+1:n-radius
        % Check normal rectangular ROI with LogPolar and FV and without FV
        g_p_x = floor(p_x + i);
        g_p_y = ceil(p_y + j);
        
        
        idx = double(img(sub2ind(size(img), g_p_x, g_p_y)));%img(g_p_x+(g_p_y-1)*m)
        
        vp = idx.*(1-a).*(1-b)+idx.*a.*(1-b)+idx.*(1-a).*b+idx.*a.*b;
        
        % Compare interpolated value gp with center value gc
        binaryLBP = vp >= img(i, j); 
        
        % Only calculate uniform pixel pattern else set to 0
        
        if(sum((abs(diff(binaryLBP)))) < 3)

           LBPImage(i, j) = sum((2.^(pattern-1)).*binaryLBP);
            
        
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
    %LBP = abs(LBP)./max(abs(LBP(:)));
    %LBP = localHist(LBP, 9);
end

end

