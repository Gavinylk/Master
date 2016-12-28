function [HOGF] = HOG(img, blockSize, cellSize, numBins, Dx, Dy)

%Histogram of Orientated Gradients

% Get Image Size
[x y] = size(img);
HOGF = [];
Mag = zeros(x, y);
Orient = zeros(x, y);

%% Step 1: Gradient Computation
% Gradient Kernel 
Dx = Dx;
Dy = Dy';

% Convolve with img to get Derivates Ix and Iy
Ix = conv2(double(img), double(Dx), 'same');
Iy = conv2(double(img), double(Dy), 'same');


% Compute Magnitude and Orientation
parfor i = 1:x
    
    for j = 1:y
        
        Mag(i, j) = sqrt(Ix(i, j)^2+Iy(i, j)^2);
        Orient(i, j) = atan2(Iy(i, j),Ix(i, j));
        
    end
    
end



%% Step 2: Orientation Binning 0-180° into 9 bins

%Calculate the number of Horizonal and Vertical Cells, round down 
numHCells = floor(y/cellSize);
numVCells = floor(x/cellSize);
histograms = zeros(numVCells, numHCells, numBins);


% For each cell in the y-direction
for i = 0:(numVCells-1)
    
    rowOffset = (i * cellSize) + 1;
    
    % For each cell in the x-direction
    for j = 0:(numHCells - 1)
        
        
        colOffset = (j * cellSize) + 1;
        
        
        % Compute the indices of the pixels within this cell.
        rowIndeces = rowOffset : (rowOffset + cellSize - 1);
        colIndeces = colOffset : (colOffset + cellSize - 1);
        
        % Select the angles and magnitudes for the pixels in this cell.
        cellOrient = Orient(rowIndeces, colIndeces); 
        cellMag = Mag(rowIndeces, colIndeces);
        
        
        % Compute the histogram for this cell.
        % Convert the cells to column vectors before passing them in.
        histograms(i + 1, j + 1, :) = getHistogram(cellMag(:), cellOrient(:), numBins);
    end
end
   
%% Step 3: Block Normalization L2-Norm
% For each cell in the y-direction...
parfor i = 1:(numVCells - 1)    
    % For each cell in the x-direction...
    for j = 1:(numHCells - 1)
    
        % Get the histograms for the cells in this block.
        blockHists = histograms(i : i + 1, j : j + 1, :);
        
        % Put all the histogram values into a single vector
        % Add a small amount to the magnitude to ensure that it's never 0.
        magnitude = norm(blockHists(:)) + 0.01;
    
        % Divide all of the histogram values by the magnitude to normalize 
        % them.
        normalized = blockHists / magnitude;
        
        % Append the normalized histograms to our descriptor vector.
        HOGF = [HOGF; normalized(:)];
        
    end


    
end

HOGF = HOGF(HOGF~=0)';

end