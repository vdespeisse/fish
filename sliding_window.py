import pandas as pd
import numpy as np

class Rect():
	def __init__(self,pos,size):
		self.x = pos[0]
		self.y = pos[1]
		self.width = size[0]
		self.height = size[1]
	def snap(image):
		return image[self.y:self.y+self.height,self.x:self.x+self.width]
def sliding_window(image, stepSize, windowSize):
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			if (x+windowSize[0]<image.shape[1] & y+windowSize[1]<image.shape[0])
				yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def find_best(windowScores, imageSize): #windowScores = {Rect : score}
	maxScore = 0.0
	bestWindow = (0,0)
	for k,v in scores.iteritems():
		if v["score"] > maxScore & (k.x+k.width<=imageSize[0] & k.y+k.height<=imageSize[1]):
			maxScore = v["score"]
			bestWindow = k
			bestNames = v["names"]
	return bestWindow,maxScore

def optimal_window(image,start):
	w = image.shape[1]
	h = image.shape[0]

	topCorner = (max(0,start[0]-210),max(0,start[1]-210))
	newImgSize = (min(w,start[0]+420),min(h,start[1]+420))
	newImg = image[topCorner[1]:topCorner[1]+newImgSize[1],topCorner[0]:topCorner[0]+newImgSize[0]]

	windowSizes = [(110,210),(210,110),(150,300),(300,150),(300,300),(250,350),(350,250),(350,350)]
	stepSize = 10
	return window_iterator(newImg,stepSize,windowSizes)


def window_iterator(img,stepSize,windowSizes):
	for windowSize in windowSizes:
		yield (img,stepSize,windowSize)
