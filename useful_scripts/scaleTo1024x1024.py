############################################################
###### Run this script within the source image folder ######
############################################################

import PIL
import glob
from PIL import Image

imgList = glob.glob('*.png')

fileOutputPath = 'C:/Users/jessv/Dropbox (MIT)/1st Year EECS/6.S198/6.S198 Final Project/Test PIL/ResizedImg/'

basewidth = 1024

for imgName in imgList:
    img = Image.open(imgName)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save(fileOutputPath + str(basewidth) + imgName)
    print('done resizing ' + imgName)

print('done!')
