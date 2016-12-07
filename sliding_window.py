import pandas as pd
import numpy as np
from classify_image import run_inference_on_image


def sliding_window(image, stepSize, windowSize):
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def run_on_windows(image,stepSize,windowSize,fishDict = "./data/fishnames.csv"):
	fishnames = pd.read_csv(fishDict).fishname.values
	windowScores = {}
	for (x,y,window) in sliding_window(image,stepSize,windowSize):
		results = run_inference_on_image(window)
		fishscore = 0.0
		for res in results :
			names = res[0].replace(', ',',').lower().split(',')
			if len(np.intersect1d(names,fishnames)) > 0:
				fishscore += res[1]
		windowScores[(x,y)] = fishscore

	return windowScores
