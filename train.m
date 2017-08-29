% Vetor para plotar o grafico
xx = [];
yy = [];

% Iteração
for z = 0:2000
    % Itera
    h = theta * x.';
    erro = h-y.';
    
    der = erro * x;
    theta = theta - (alpha/m)*der;
   
    % Plota o erro
    J = (erro * erro.')/(2*m);
    xx = [xx, z];
    yy = [yy, J];
    
    plot(xx,yy);
    refresh();
    printf("[%d] Erro: %d\n", z, J);
    fflush(stdout);
endfor