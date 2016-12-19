from fish_detection import *
import time
import glob
import os
import cv2
import numpy as np
import preprocess as pre



def run(stepSize=20, windowSize=(350,350), fishDict ="./data/fishnames.csv",node_lookup = NodeLookup()):
    maybe_download_and_extract()
    # folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    folders = ['LAG']
    print(folders,"ezae")
    for fld in folders:
        path = os.path.join('.', 'testrun', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flname = os.path.basename(fl)
            print("running on %s ..." % (flname))
            img = pre.readimg(fl, width= 1280,height=720)

            bestWindow,bestScore = run_inference_on_image(img,stepSize, windowSize, node_lookup, fishDict ="./data/fishnames.csv")
            print(bestWindow.x,bestWindow.y)
            output_path = os.path.join('.', 'output',fld,flname)
            fish = bestWindow.snap(img)
            cv2.imwrite(output_path,fish)
