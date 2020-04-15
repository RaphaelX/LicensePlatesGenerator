from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageFilter

import matplotlib.pyplot as plt
import numpy as np
import random as r
import cv2
import os
import copy
import argparse as ap

#? important variables
letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Z']
numbers = ['0','1','2','3','4','5','6','7','8','9']
PATH = os.getcwd()
baseplate_path = PATH +'\\FrenchLicensePlate_base.png'
baseplate = Image.open(baseplate_path)

def GenerateRandomLicensePlate(baseplate, letters, numbers, gray=240, blue=245):
    baseplate_cp = copy.deepcopy(baseplate)

    #? choosing the letters and numbers
    lp_letters = [] 
    lp_numbers = []
    for i in range(4):
        lp_letters.append(letters[r.randrange(0,len(letters),1)])
    for i in range(5):
        lp_numbers.append(numbers[r.randrange(0,len(numbers),1)])

    lp = lp_letters[0] + lp_letters[1] + '-' + lp_numbers[0] + lp_numbers[1] + lp_numbers[2] + '-' + lp_letters[2] + lp_letters[3]
    departement = lp_numbers[-2] + lp_numbers[-1]

    #? set font and draw over the baseplate
    font = ImageFont.truetype('./fonts/cargo2.ttf', 200, encoding='unic')

    #license plate's middle part
    canvas = Image.new('RGB', (1080, 250), (gray,gray,gray))
    draw = ImageDraw.Draw(canvas, 'RGBA')
    draw.text((0, 0), lp, 'black', font)

    factor = 1.95
    canvas_cp = copy.deepcopy(canvas)
    canvas_cp.thumbnail((180*factor, 65*factor))

    baseplate_cp.paste(canvas_cp, (53, 5))
    # baseplate_cp.show()

    #license plate's middle part
    canvas = Image.new('RGB', (300, 300), (0,0,blue))
    draw = ImageDraw.Draw(canvas, 'RGBA')
    draw.text((0, 0), departement, 'white', font)

    factor = .5
    canvas_cp = copy.deepcopy(canvas)
    canvas_cp.thumbnail((180*factor, 65*factor))

    baseplate_cp.paste(canvas_cp, (416, 53))
    return baseplate_cp, lp, departement

def GenerateMultipleRandomPlates(number_of_plates=100, gray=False, PATH=PATH, baseplate=baseplate, letters=letters, numbers=numbers):
    for i in range(number_of_plates):
        img, lp, departement= GenerateRandomLicensePlate(baseplate, letters, numbers)
        if gray:
            img = img.convert('1')
        
        # lp[2] = '_'
        # lp[6] = '_'
        img.save(PATH+'\\data\\'+lp+'_'+departement+'.png')

if __name__=="__main__":
    if not os.path.exists("data"):
        os.mkdir('data')

    parser = ap.ArgumentParser(
            description='Generate random French license plates.')
    parser.add_argument("-n", "--number_of_plates", nargs=1, type=int,
                        help='')
    parser.add_argument("-g", "--gray", nargs=1, type=bool,
                        help='choose if grayscale or not')
    args = parser.parse_args()

    number_of_plates = args.number_of_plates[0] if args.number_of_plates else 100
    gray = args.gray[0] if args.gray else False
    
    GenerateMultipleRandomPlates(number_of_plates=number_of_plates, gray=False)




