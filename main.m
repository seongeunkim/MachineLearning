file_train = csvread("year-prediction-msd-test.txt");
m = rows(file_train);

% Chute inicial do Theta
theta(1) = 1;
theta(2:90) = 0;
alpha = 0.00000005;
xx = [];
yy = [];

for z = 1:100
  % Para cada dado
  for i = 1:m
    y = file_train(i,1);
    x = file_train(i, 2:91);
    h = theta * x.';

    erro(i) = h-y;
    var(i, 1:90) = erro(i) * x(1:90);
  endfor

  % Calcula o erro
  J = (erro * erro.')/(2*m);
  printf("Vez %d: %d\n",z ,J);
  xx = [xx, z];
  yy = [yy, J];
  plot(xx,yy);
  refresh();
  fflush(stdout);
  fflush(stderr);

  % Para cada theta
  for j = 1:90
    soma(j) = sum(var(1:m, j));
    thetaTemp(j) = theta(j) - (alpha/m) * soma(j);
  endfor

  for j = 1:90
      theta(j) = thetaTemp(j);
  endfor
endfor



