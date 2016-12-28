function [normim, normtform, xdata, ydata] = imnorm(im)
%
% Implementation of the image normalization part of:
%   1. P. Dong, J.G. Brankov, N.P. Galatsanos, Y. Yang, and F. Davoine,
%      "Digital Watermarking Robust to Geometric Distortions," IEEE Trans.
%      Image Processing, Vol. 14, No. 12, pp. 2140-2150, 2005.
%
% Input:
%   im: input grayscale image
%
% Output:
%   normim: normalized image of class double
%   normtform: tform of the normalization
%   xdata, ydata: spatial coordinates of the normalized image
%
% Copyright (c), Yuan-Liang Tang
% Associate Professor
% Department of Information Management
% Chaoyang University of Technology
% Taiwan, R.O.C.
% Email: yltang@cyut.edu.tw
% http://www.cyut.edu.tw/~yltang
% 
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software without restriction, subject to the following
% conditions:
% The above copyright notice and this permission notice should be included
% in all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
%
% Created: May 2, 2007
% Last updated: Jul. 30, 2009
%

if ~isa(im, 'double')
  im = double(im);
end

% Normalization steps: 
% 1. Translation invariance: translate coordinate to the image centroid
[cx cy] = imcentroid(im);
tmat = [1 0 0; 0 1 0; -cx -cy 1];    % Translation matrix
mat = tmat;
tform = maketform('affine', mat);
[imt xdata ydata] = imtransform(im, tform, 'XYScale', 1);
%showim(imt, 'Translation', xdata, ydata);




function [cx, cy] = imcentroid(im)
% Compute image centroid
m00 = immoment(im, 0, 0);
cx = immoment(im, 1, 0)/(m00+eps);
cy = immoment(im, 0, 1)/(m00+eps);





