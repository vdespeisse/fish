from classify_image import *
import time
import glob
import os
import cv2
import numpy as np
import preprocess as pre



def run(stepSize=20, windowSize=(200,200), fishDict ="./data/fishnames.csv"):
    maybe_download_and_extract()
    # folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    folders = ['ALB']
    for fld in folders:
        path = os.path.join('.', 'testrun', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = pre.readimg(fl, width= 1280,height=720)
            with tf.Graph().as_default():
                windowScores = run_inference_on_image(img,stepSize, windowSize, fishDict ="./data/fishnames.csv")
            maxScore = 0.0
            bestWindow = (0,0)
            for k,v in windowScores.iteritems():
                if v > maxScore:
                    maxScore = v
                    bestWindow = k
            flname = os.path.basename(fl)
            output_path = os.path.join('.', 'output',fld,flname)
            fish = img[bestWindow[1]:bestWindow[1]+stepSize,bestWindow[0]:bestWindow[0]+stepSize]
            cv2.imwrite(output_path,fish)
