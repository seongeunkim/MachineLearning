% Abre o arquivo
file_train = csvread("year-prediction-msd-train.txt");

% Linhas
m = rows(file_train);

% Colunas
n = rows(file_train.');

% Chute inicial
theta(1:n) = 0;

% Y
y = file_train(1:m, 1);

% X
x = file_train;

% Normalizando X
xMax = max(x);
xMean = mean(x);
xt = bsxfun(@minus, x, xMean);
x = bsxfun(@rdivide, xt, xMax);
x(1:m, 1) = 1

% Alpha
alpha = 0.2;