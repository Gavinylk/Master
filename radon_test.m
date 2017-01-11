iptsetpref('ImshowAxesVisible','on')
theta = 0:180;
for i = 1:80
   [R,xp] = radon(FV_train_gmm{i,1},theta);
   %figure, imagesc(R/norm(R, 1));
   R = abs(R)/norm(R, 1);
   meanR(i) = std(R(:)); 
end


