% Abre a pasta do Dataset de treino
treino = dir("dataset/train");

% Remove o labels e as pastas . e ..
treino = treino( 3:50002, 1);

% Abro os labels
labels = csvread("dataset/train/labels");

% Cria a matriz Y
y = zeros(50000, 10, "boolean");
for i=1:length(treino)
  y(i, labels(i)+1) = 1;
endfor

% Crio a matriz X
x = zeros(50000, 3073);

tmp0 = zeros(32, 32, 3);
tmp1 = zeros(1, 3072);
tmp2 = zeros(1, 3073);

for i = 1:50000
  tmp0 = imread(sprintf('dataset/train/%05d.png', i-1));
  tmp1 = reshape(tmp0, 1, []);
  tmp2 = [1, tmp1];
  x(i, :) = tmp2;
endfor

% Normalizando X
xMax = max(x);
xMean = mean(x);
xt = bsxfun(@minus, x, xMean);
x = bsxfun(@rdivide, xt, xMax);
x(:, 1) = 1;

% Alpha
%alpha = 0.2;

% Chute inicial
%theta= zeros(1, 3073);

% Equação Normal
tEN = zeros(10, 3073);

for i=1:10
  tEN(i, :) = (inv(x.' * x) * x.' * y(:, i)).'; 
endfor

hEN = tEN * x.';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


HH = zeros(50000, 1);
resp = zeros(50000, 1);

for i = 1:50000
  HH(i) = 1/(1+exp(-hEN(i)));
endfor

for i = 1:50000
  if (HH(i) >= 0.55)
    resp(i) = 1;
  else 
    resp(i) = 0;
  endif
endfor

dif = abs(resp - y);

errroo = mean(dif);
