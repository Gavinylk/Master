function [borderImg] = BorderImage(binaryImg)


borderImg = false(size(binaryImg));
    
    binaryImg = padarray(binaryImg, [1, 1], 1);
    [h w] = size(borderImg);
    
    bkgFound = false;
    parfor row = 1 : h
        for col = 1 : w
            if binaryImg(row + 1, col + 1)
                
                bkgFound = false;
                for i = 0:2
                    for j = 0:2
                        if ~binaryImg(row + i, col + j)
                            borderImg(row, col) = 1;
                            bkgFound = true;
                            break;% Iterations for Blocks
for i = 1: floor(rows/blockSize)
    for j= 1: floor(cols/blockSize)
        
      
       
       % Iteration for the cells
       for k = 1:blockSize
            for l= 1:blockSize
            
                
            
            end
       end
       
    end
end
                        end;
                    end;
                    
                    if bkgFound
                        break;
                    end;
                end;
            end;
        end;
    end;

end

