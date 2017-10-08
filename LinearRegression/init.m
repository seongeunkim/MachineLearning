% Abre o arquivo
%file_train = csvread("year-prediction-msd-train.txt");
%file_test = csvread("year-prediction-msd-test.txt");

% X
x = file_train;

% Linhas
m = rows(x);
x2 = x(:, 14:91).*x(:, 14:91);
x3 = x(:, 14:91).*x2;
mult2 = zeros(m, 78);

k=1;
for i=2:13
  for j=i:13
    mult2(:, k) = x(:,i) .* x(:,j);
    k++;
  endfor
endfor

mult3 = zeros(m, 364);

l=1;
for i=2:13
  for j=i:13
    for k=j:13
      mult3(:, l) = x(:,i) .* x(:,j) .* x(:,k);
      l++;
    endfor
  endfor
endfor


x = [x, mult2, x2, mult3, x3];

% Colunas
n = rows(x.');

% Y
y = file_train(:, 1);

% Normalizando X
xMax = max(x);
xMean = mean(x);
xt = bsxfun(@minus, x, xMean);
x = bsxfun(@rdivide, xt, xMax);
x(:, 1) = 1;

% Alpha
alpha = 0.2;

% Chute inicial
theta= zeros(1, n);

% Equação Normal
thetaEN = (inv(x.' * x) * x.' * y).';
hEN = thetaEN * x.';
erroEN = hEN-y.';
JEN = (erroEN * erroEN.')/(2*m);