iptsetpref('ImshowAxesVisible','on')
theta = 0:180;
[R,xp] = radon(F,theta);
[R2,xp] = radon(F2,theta);

figure, imagesc(R), figure, imagesc(R2);
