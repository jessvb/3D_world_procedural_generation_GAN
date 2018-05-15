############################################################
###### Run this script within the source image folder ######
############################################################

import PIL
import glob
import os
import numpy as np
from PIL import Image

# first delete duplicate images:
# (note that this deletes the duplicates of the images provided in the folder that follows)
imgsToDelFolder = 'duplicates/'
imgsToDel = glob.glob(imgsToDelFolder + '*.png')

# loop through all images to delete:
for imgNameToDel in imgsToDel:
    # open the image to delete:
    imgToDel = Image.open(imgNameToDel)
    # update image list every iteration (b/c we deleted images):
    imgList = glob.glob('*.png')
    
    # loop through all images, deleting as we go
    for imgName in imgList:
        img = Image.open(imgName)
        if list(img.getdata()) == list(imgToDel.getdata()):
            print("deleting duplicate image: " + imgName)
            os.remove(imgName)
        # else:
        #     print('different')

print('done duplicates!')


# next, delete dark and/or unvaried images
imgList = glob.glob('*.png')
stdDevThresh = 6000             # good = 8000-13000, dark = 200-1800
blackThresh = 9000

for imgName in imgList:
    img = Image.open(imgName)
    pixels = img.getdata()      # get the pixels as a vector

    ## check if image is un-varied
    stdDev = np.sqrt(np.var(pixels)) # get variance

    if stdDev < stdDevThresh:       # not very varied picture
        print("deleting unvaried image:" + imgName)
        os.remove(imgName)
    else: # it's not un-varied, but it might still be too dark
        ## check if image is dark:
        numBlack = 0
        for pixel in pixels:
            if pixel < blackThresh:
                numBlack += 1
        # get percentage of black pixels:
        n = len(pixels)
        if (numBlack / float(n)) > 0.5:       # more than 50% of pixels are black
            print("deleting dark image:" + imgName)
            os.remove(imgName)

print('done unvaried!')

