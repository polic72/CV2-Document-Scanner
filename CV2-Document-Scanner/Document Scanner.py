import cv2
import numpy as np
import math
import imutils
from msvcrt import getch
from matplotlib import pyplot as plt
import sys
from os import path
from skimage.filters import threshold_local


# Set to True if you want to see every key being pressed in stdout.
DEBUG = False

STEPBY = False

WARP = False


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


# Yes = True, No = False
def yes_no_cv2():
    temp_char = 0
    while (temp_char != 'y') & (temp_char != 'n'):
        temp_char = chr(cv2.waitKey()).lower()

    if temp_char == 'y':
        return True
    else:
        return False


args = sys.argv[1:]

if len(args) == 0:
    print("No file given. Exiting...")
    sys.exit()
else:
    filename = args[0]

    if not(path.exists(filename)):
        print("The given file \"" + filename + "\" does not exist. Exiting...")
        sys.exit()

font = cv2.FONT_HERSHEY_DUPLEX


window_name = filename.split("\\")[-1]


img = cv2.imread(filename, 1)

cv2.imshow(window_name, img)


k_code = 0

while (k_code != 113) & (k_code != 81) & (k_code != 27):
    k_code = cv2.waitKeyEx()

    try:
        char = chr(k_code)

        if char.lower() == 'h':
            print("'h' - show this help screen")
            print("'q' - quit")
            print("'ESC' - quit")
            print("'s' - save image")
            print("'t' - add entered text")
            print("'d' - read document")
            print("'p' - show image as matplot")
            print("'Left' - Rotate 90° counter-clockwise")
            print("'Right' - Rotate 90° clockwise")
        elif char.lower() == 's':
            print("Overwrite existing file? [y/n]")

            if yes_no_cv2():
                print("Saving image...")
                cv2.imwrite(filename, img)
                print("Save successful!")
            else:
                splits = filename.split("\\")
                pth = "\\".join(splits[:-1])
                splits_name = splits[-1].split(".")
                name_name = ".".join(splits_name[:-1])

                temp_filename = filename

                counter = 1
                while path.exists(temp_filename):
                    temp_filename = pth + "\\" + name_name + " (" + str(counter) + ")." + splits_name[-1]
                    counter += 1

                print("Saving image...")
                cv2.imwrite(temp_filename, img)
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
            ratio = img.shape[0] / 500.0
            orig = img.copy()
            img_resized = imutils.resize(img, height=500)

            screenCnt = None
            filter_size = 1
            break_me = False
            failed = False

            while True:
                while screenCnt is None:
                    filter_size = filter_size + 2

                    if filter_size == 21:
                        print("Failed to find the document. Cancelling operation...")
                        failed = True
                        cv2.imshow(window_name, img)
                        break

                    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    if STEPBY:
                        cv2.imshow(window_name, gray)
                        cv2.waitKeyEx()
                    gray = cv2.GaussianBlur(gray, (filter_size, filter_size), 0)
                    if STEPBY:
                        cv2.imshow(window_name, gray)
                        cv2.waitKeyEx()
                    edged = cv2.Canny(gray, 75, 200)
                    if STEPBY:
                        cv2.imshow(window_name, edged)
                        cv2.waitKeyEx()

                    (contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    # contours = imutils.grab_contours(contours)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

                    if STEPBY:
                        temp = img_resized.copy()
                        cv2.drawContours(temp, contours, -1, (0, 255, 0), 2)  # [screenCnt]
                        cv2.imshow(window_name, temp)
                        cv2.waitKeyEx()

                    for c in contours:
                        # approximate the contour
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        # if our approximated contour has four points, then we
                        # can assume that we have found our screen
                        if len(approx) == 4:
                            screenCnt = approx
                            break

                    if screenCnt is None:
                        continue

                    look_at_me = img_resized.copy()

                    cv2.drawContours(look_at_me, [screenCnt], -1, (0, 255, 0), 2)  # [screenCnt]
                    cv2.imshow(window_name, look_at_me)

                    print("Is this the document? [y/n]")

                    if yes_no_cv2():
                        break_me = True
                    else:
                        screenCnt = None
                if break_me or failed:
                    break

            if not failed:
                if DEBUG:
                    print("Filter Size: " + str(filter_size))

                # Draw contours onto the image:
                # cv2.drawContours(img_resized, [screenCnt], -1, (0, 255, 0), 2)  # [screenCnt]
                # cv2.imshow(window_name, img_resized)

                # apply the four point transform to obtain a top-down
                # view of the original image
                warped = four_point_transform(img, screenCnt.reshape(4, 2) * ratio)

                # convert the warped image to grayscale, then threshold it
                # to give it that 'black and white' paper effect
                if WARP:
                    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    T = threshold_local(warped, 11, offset=10, method="gaussian")
                    warped = (warped > T).astype("uint8") * 255
                img = warped

                cv2.imshow(window_name, img)

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