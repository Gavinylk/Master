function [R] = RadonTransform(img)

theta = 0:180;
[R, ~] = radon(img, theta);

end