function [M,I] = mymedian(A,dim)
  if nargin==1
    dim = 1; 
  end
M = median(A,dim); 
I = find(A==M); 
end