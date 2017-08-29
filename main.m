%file_train = csvread("year-prediction-msd-train.txt");
%file_train = csvread("teste1.csv");
%m = rows(file_train);
%n = rows(file_train.');

%Chute inicial dos 91 thetas
%theta(1:n) = 0;

%y = file_train(1:m, 1);
%x = file_train;
%x(1:m, 1) = 1;

%xMax = max(x);
%xMean = mean(x);
%xt = bsxfun(@minus, x, xMean);
%xtt = bsxfun(@rdivide, xt, xMax);
%xtt(1:m, 1) = 1;

%x = xtt;
%thetaDer = (inv(x.' * x) * x.' * y).';

xx = [];
yy = [];
alpha = 0.2;
for z = 0:10000
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