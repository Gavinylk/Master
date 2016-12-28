function [FV] = CLBPHF(img, p, r)


fv1= []; fv2 = [];fv3= []; fv4 = [];

patternMapping1u2 = getmapping(p);

[CLBP_S,CLBP_M,CLBP_C] = clbp(img,r,p,patternMapping1u2,'x');

% Generate histogram of CLBP_S
CLBP_SH = hist(CLBP_S(:),0:patternMapping1u2.num-1);
               
% Generate histogram of CLBP_M
CLBP_MH = hist(CLBP_M(:),0:patternMapping1u2.num-1);

% Generate LBPHF_S
fv1 = constructhf(CLBP_SH,patternMapping1u2);

% Generate LBPHF_M
fv2 = constructhf(CLBP_MH,patternMapping1u2);
                
                
FV = [fv1 fv2 fv3 fv4];
                
                 
                 
end