import cv2
import math
import numpy as np
import random
from utils import *
from process_img import *
# from pdf2image import convert_from_path

# dpi = 500 # dots per inch
# pdf_file = 'ttn_0001.pdf'
# pages = convert_from_path(pdf_file ,dpi , poppler_path=r'D:/python-demo/Auto-Scores-National-Multiple-Choice-Test-master/poppler-0.68.0/bin')
# for i in range(len(pages)):
#    page = pages[i]
#    page.save('output/output_{}.jpg'.format(i), 'JPEG')
# exit()

image = cv2.imread("nguyen_hue_01.JPEG")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray,  50, 100, )
blurred = cv2.GaussianBlur(canny, (5, 5), 0)

# 2. Threshold de
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY_INV, 31, 3)
# cv2.imshow("Anh tai buoc 2", thresh)
# cv2.waitKey()


# 3. Tim khung ben ngoai de tach van ban khoi nen
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
approx = cv2.approxPolyDP(
    contours[1], 0.01 * cv2.arcLength(contours[1], True), True)
rect = cv2.minAreaRect(contours[1])
box = cv2.boxPoints(rect)

# 4. Thuc hien transform de xoay van ban
corner = find_corner_by_rotated_rect(box, approx)
image = four_point_transform(image, corner)
wrap = four_point_transform(canny, corner)
cv2.imshow("Anh sau buoc 4", wrap)
cv2.waitKey()

# 5. Tim cac o tick trong hinh
contours, _ = cv2.findContours(
    wrap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x_old, y_old, w_old, h_old = 0, 0, 0, 0
ans_blocks = []
# loop over the contours
# ensure that at least one contour was found
if len(contours) > 0:
    # sort the contours according to their size in descending order
    cnts = sorted(contours, key=get_x_ver1)

    # loop over the sorted contours
    for i, c in enumerate(cnts):
        x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

        if w_curr * h_curr > 300000 and  w_curr * h_curr < 600000:
            # check overlap contours
            check_xy_min = x_curr * y_curr - x_old * y_old
            check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - \
                (x_old + w_old) * (y_old + h_old)

            # if list answer box is empty
            if len(ans_blocks) == 0:
                ans_blocks.append(
                    (gray[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                # update coordinates (x, y) and (height, width) of added contours
                x_old = x_curr
                y_old = y_curr
                w_old = w_curr
                h_old = h_curr
            elif check_xy_min > 20000 and check_xy_max > 20000:
                ans_blocks.append(
                    (gray[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                # update coordinates (x, y) and (height, width) of added contours
                x_old = x_curr
                y_old = y_curr
                w_old = w_curr
                h_old = h_curr
            cv2.imshow("Anh sau buoc 4", wrap[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr])
            cv2.waitKey()


    # sort ans_blocks according to x coordinate
    sorted_ans_blocks = sorted(ans_blocks, key=get_x)
    print(len(ans_blocks))
    
