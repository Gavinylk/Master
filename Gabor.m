function [DCFreeReal, GaborFilterImag] = Gabor(Scale, f, theta, sigma, psi)

for i= -Scale:Scale
   for j = -Scale:Scale
      
       x_t = i*cos(theta)+j*sin(theta);
       y_t = j*cos(theta)-i*sin(theta);
       
       GaborFilterReal(Scale+i+1, Scale+j+1) = exp(-.5*(((x_t)^2+(y_t)^2)/(sigma^2)))*cos(2*pi*f*x_t+psi);
       GaborFilterImag(Scale+i+1, Scale+j+1) = exp(-.5*(((x_t)^2+(y_t)^2)/(sigma^2)))*sin(2*pi*f*x_t+psi);
   end
end
%
% Remove in Real Part of the kernel the DC Component
DCVal = mean(GaborFilterReal(:));
DCFreeReal = GaborFilterReal-DCVal;
end

