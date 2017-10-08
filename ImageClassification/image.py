import numpy
from PIL import Image

im = Image.open('../dataset/test/00098.png')
pixelMap = im.load()

img = Image.new( im.mode, (im.size[0], im.size[1]-1))
pixelsNew = img.load()

for i in range(img.size[0]):
    for j in range(img.size[1]-1):
        pixelsNew[i,j] = tuple(numpy.subtract(pixelMap[i,j], pixelMap[i,j+1]))
img.show()
