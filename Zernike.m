%% Read Image Data
clear;
IMG = double(imread('blanket.png'));
%IMG = double(IMG(1:2000,1:2000));
%% Calculate Zernike Coefficients

coeff = ZernikeCalc([],IMG, [], 'standard'); 

%% Plot Aberrations
ZernikeCalc([],coeff);

%% Zernike Fit

ZernikeCalc([], IMG);



