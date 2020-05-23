import cv2
import numpy as np
import math
import imutils
from msvcrt import getch
from matplotlib import pyplot as plt
import sys
from os import path

args = sys.argv[1:]

if len(args) == 0:
    print("No file given. Exiting...")
    sys.exit()
else:
    filename = args[0]

    if not(path.exists(filename)):
        print("The given file \"" + filename + "\" does not exist. Exiting...")
        sys.exit()

# Set to True if you want to see every key being pressed in stdout.
DEBUG = False

font = cv2.FONT_HERSHEY_DUPLEX


window_name = filename.split("\\")[-1]


img = cv2.imread(filename, 1)

cv2.imshow(window_name, img)


k_code = 0

while (k_code != 113) & (k_code != 81) & (k_code != 27):
    k_code = cv2.waitKeyEx()

    try:
        char = chr(k_code)

        if char.lower() == 's':
            print("Saving image...")
            cv2.imwrite(filename, img)
            print("Save successful!")
        elif char.lower() == 't':
            text = input("Please enter the image caption: ")

            place_y = math.floor(img.shape[0] * .9)
            place_x = math.floor(img.shape[1] * .1)

            # print(str(place_y) + " : " + str(place_x))

            cv2.putText(img, text, (place_x, place_y), font, 1, (255, 255, 255))

            cv2.imshow(window_name, img)
        elif char.lower() == 'p':
            plt.imshow(img[:, :, ::-1])
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
        elif char.lower() == 'd':
            pass

        if DEBUG:
            print("'" + char + "' : " + str(k_code))
    except ValueError:
        if k_code == 2490368:
            if DEBUG:
                print("Up")
        elif k_code == 2621440:
            if DEBUG:
                print("Down")
        elif k_code == 2424832:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow(window_name, img)

            if DEBUG:
                print("Left")
        elif k_code == 2555904:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow(window_name, img)

            if DEBUG:
                print("Right")
        else:
            if DEBUG:
                print(k_code)

print("Exiting...")