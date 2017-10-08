import numpy
from PIL import Image
from numpy import genfromtxt
import bigfloat
bigfloat.exp(5000,bigfloat.precision(100))

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+numpy.exp(-x))


classe = 0
numImages = 1000
nInputs = 3073

X = numpy.ones((numImages, 3073))


for num in range(0, numImages):
    im = Image.open('../dataset/train/' + '{:05d}'.format(num) + '.png')
    im = numpy.array(im).flatten()
    X[num, 1:3073] = im

X = (X-127)/255


labels = genfromtxt('../dataset/train/labels', delimiter=',')

Y = numpy.zeros((numImages, 1))

for num in range(0,numImages):
    if labels[num] == classe:
        Y[num][0] = 1


# seed random numbers to make calculation
# deterministic (just a good practice)
numpy.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*numpy.random.random((nInputs,1)) - 1

for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(numpy.dot(l0,syn0))

    # how much did we miss?
    l1_error = Y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1

    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += numpy.dot(l0.T,l1_delta)

print ("Output After Training:")

l1_error = Y - l1
for i in range(0, numImages):
    print ('{:2f}'.format(l1_error[i][0]))
