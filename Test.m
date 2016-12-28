%% Read Image Data
clear;
IMG = imread('DotGrid1_PTV2-7_z=-55.7mm.PNG');
IMG = double(IMG(1:2000,1:2000));
%% Calculate Zernike Coefficients

coeff = ZernikeCalc([1 1; 1 -1; 2 0; 2 -2; 2 2; 3 -1; 3 1]', IMG, [], 'standard'); 

%% Plot Aberrations
ZernikeCalc([2, 3, 4, 5, 6, 7, 8], coeff);

%% Zernike Fit

ZernikeCalc([], IMG);

% First of all thanks for providing this great code, with excellent explanation. 
%But I have a question considering the normalization of the Zernike polynomials. 
%When using ZernikeDef ‘STANDARD’ it seems that their is no normalization. 
%This came up after I wanted to retrieve some Zernike coefficients of a known surface. 
%The surface data was made with normalized Zernike’s polynomials. 
%But the output of you code was without the normalizations factors.
% 
% I fixed the issue by placing ‘result=theFactor.*result; ‘ before ‘switch ZernikeDef…..’. 
%In the function result (row 789). 
%Now my question is, whether this fix is correct and either of this is a problem in the first place? 