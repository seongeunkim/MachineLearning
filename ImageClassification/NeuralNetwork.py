import numpy
import pickle
from PIL import Image, ImageFilter
from sklearn import svm
from sklearn import datasets
from numpy import genfromtxt
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

splits = 2
classe = 0
nTestes = 10000
numInputs = 3072 + 3072  + 1
numImages = 500

X = numpy.ones((numImages, numInputs))
for num in range(0, numImages):
    im = Image.open('../dataset/train/' + '{:05d}'.format(num) + '.png')
    edge = im.filter( ImageFilter.FIND_EDGES )
    # sharp = im.filter( ImageFilter.SHARPEN )

    im = numpy.array(im).flatten()
    edge = numpy.array(edge).flatten()
    # sharp = numpy.array(sharp).flatten()

    X[num, 1:3073] = (im-127)/255
    X[num, 3073:6145] = (edge-127)/255
    # X[num, 6145:numInputs] = (sharp-127)/255

labelsT = genfromtxt('../dataset/train/labels', delimiter=',')
y = numpy.zeros((numImages))

for num in range(0,numImages):
    if labelsT[num] == classe:
        y[num] = 1

XX_test = numpy.ones((nTestes, numInputs))
for num in range(0, nTestes):
    im = Image.open('../dataset/test/' + '{:05d}'.format(num) + '.png')
    edge = im.filter( ImageFilter.FIND_EDGES )
    # sharp = im.filter( ImageFilter.SHARPEN )

    im = numpy.array(im).flatten()
    edge = numpy.array(edge).flatten()
    # sharp = numpy.array(sharp).flatten()

    XX_test[num, 1:3073] = (im-127)/255
    XX_test[num, 3073:6145] = (edge-127)/255
    # XX_test[num, 6145:numInputs] = (sharp-127)/255


labelsTest = genfromtxt('../dataset/test/labels', delimiter=',')
yy_test = numpy.zeros((nTestes))

for num in range(0,nTestes):
    if labelsTest[num] == classe:
        yy_test[num] = 1

############################################

kf = KFold(n_splits = splits, shuffle = True)
kf.get_n_splits(X)

it = 0
mlp = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mlp += [MLPClassifier(
            hidden_layer_sizes = (1500, 1000),
            solver = 'sgd',
            learning_rate_init = 0.01,
            max_iter = 300,
            verbose = True
        )]
    mlp[it].fit(X, y)
    it += 1

for it in range(0, splits):
    yyp = mlp[it].predict(XX_test)
    print ('[', it, '] Treino: ', mlp[it].score(X, y), '\t Teste :', mlp[it].score(XX_test, yy_test))
    print(confusion_matrix(yy_test,yyp))
    joblib.dump(mlp[it], '{:03d}'.format(it) + '.pkl')
