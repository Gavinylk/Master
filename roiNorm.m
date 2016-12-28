function [FV] = roiNorm(outTL, outTR, outM, outBL, outBR, roiSize, stride, X, Y)


for i = 1:stride:X-roiSize
    for j = 1:stride:Y-roiSize
        

        
        % calc L2-Norm of the 5 values above
        FV(i, j) = norm([abs(outTL(i:i+roiSize-1, j:j+roiSize-1)), abs(outTR(i:i+roiSize-1, j:j+roiSize-1)), abs(outM(i:i+roiSize-1, j:j+roiSize-1)), abs(outBL(i:i+roiSize-1, j:j+roiSize-1)), abs(outBR(i:i+roiSize-1, j:j+roiSize-1))], 2);
            
      
        
        
    end
end




end