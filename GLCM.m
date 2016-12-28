function [F] = GLCM(img, level, delta, theta)

% Quantisation of the input Image to desired value
imgQ = ImageQuantisation(img, level);
[m n] = size(imgQ);

% Get the number of gray levels
minGV = min(imgQ(:));

% Remapp to range (n bit)
parfor k = 1:m
    for l = 1:n
        
        imgQ(k, l) = imgQ(k, l)/minGV; 
    end
end

% Create GLCM initial Matrix
maxGV = max(imgQ(:));
C = zeros(maxGV, maxGV);

% Positions
delta_x = ceil(delta*cos(theta));
delta_y = ceil(delta*sin(theta));

% Padd Array
imgQ = padarray(imgQ, [abs(delta_x) abs(delta_y)]);

% Find Occurences
for i = 1:m-delta_x
    for j = 1:n-delta_y
        parfor o = 1:maxGV
            for p = 1:maxGV
                
                if(imgQ(i, j) == o & imgQ(i+delta_x, j+delta_y) == p)
            
                    C(o, p) = C(o, p) + 1;
                end
            end
        end
        
    end
end


% Generate the Coocurence probability Matrix + Histogram + average
P = C./sum(C(:));
F = reshape(P.',1,[]);         

end

