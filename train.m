xx = [];
yy = [];

for z = 0:2000
    h = theta * x.';
    erro = h-y.';
    
    der = erro * x;
    theta = theta - (alpha/m)*der;
   
    J = (erro * erro.')/(2*m);
    xx = [xx, z];
    yy = [yy, J];
    
    plot(xx,yy);
    refresh();
    printf("[%d] Erro: %d\n", z, J);
    fflush(stdout);
endfor