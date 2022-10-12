from math import ceil
import cv2
import numpy as np
import imutils
from utils import *
# read input image
imageFrame = cv2.imread('output/output_0.jpg')
height, width, channels = imageFrame.shape
# imageFrame = imageFrame[1949:5848, 0:4139]
img_grey = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
img_blur = cv2.medianBlur(img_grey, 5).astype('uint8')
img_thresh = cv2.adaptiveThreshold(
    img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cnts = cv2.findContours(img_thresh.copy(), cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

ans_blocks = []
if len(cnts) > 0:
    x_old, y_old, w_old, h_old = 0, 0, 0, 0
    # sort the contours according to their size in descending order
    cnts = sorted(cnts, key=get_x_ver1)

    for i, c in enumerate(cnts):
        x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

        # check overlap contours
        check_xy_min = x_curr * y_curr - x_old * y_old
        check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - \
            (x_old + w_old) * (y_old + h_old)

        if w_curr > 1000 and w_curr * h_curr > 100000 and w_curr * h_curr < height * width / 2 and w_curr < h_curr:
            # if list answer box is empty
            if len(ans_blocks) == 0 or (check_xy_min > 20000 and check_xy_max > 20000):
                ans_blocks.append(
                    (imageFrame[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                # update coordinates (x, y) and (height, width) of added contours
                x_old = x_curr
                y_old = y_curr
                w_old = w_curr
                h_old = h_curr
                img_contour = imageFrame[y_curr:y_curr +
                                         h_curr, x_curr:x_curr + w_curr]
                fileName = "./exports/output_i_{i}w_curr{w_curr}_h_curr{h_curr}.jpg".format(
                    i=i, w_curr=w_curr, h_curr=h_curr)
                cv2.imwrite(fileName, img_contour)
sorted_ans_blocks = sorted(ans_blocks, key=get_x)
# process_ans_blocks
list_box_answers = []
block_count = 5
answer_count_each_row = 5
padding_vertical = 28

right_anser = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1, 5: 1, 6: 0, 7: 4,8 :2,9:2,10: 1, 11: 4, 12: 0, 13: 3, 14: 1, 15: 1, 16: 0, 17: 4,18 :2,19:2,
20: 1, 21: 4, 22: 0, 23: 3, 24: 1, 25: 1, 26: 0, 27: 4,28 :2,29:2,30: 1, 31: 4, 32: 0, 33: 3, 34: 1, 35: 1, 36: 0, 37: 4,38 :2,39:2,
40: 1, 41: 4, 42: 0, 43: 3, 44: 1, 45: 1, 46: 0, 47: 4,48 :2,49:2,}

for ans_block in sorted_ans_blocks:
    print("ans_block")
    ans_block_img = np.array(ans_block[0])
    offset1 = ceil(ans_block_img.shape[0] / block_count)
    # Loop over each box in answer block
    for i in range(block_count):
        print("block_count", i)

        box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
        height_box = box_img.shape[0]
        weight_box = box_img.shape[1]
        # box_img = box_img[14:height_box - 14, 200:]
        offset2 = ceil(box_img.shape[0] / 5)
        # loop over each line in a box
        box_gray_img =  cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
        box_blurred = cv2.GaussianBlur(box_gray_img, (5, 5), 0)
        box_canny = cv2.Canny(box_blurred, 50, 170)
        box_cnts = cv2.findContours(box_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        box_cnts = imutils.grab_contours(box_cnts)
        print("box_cnts",len(box_cnts))
        cv2.imwrite("./exports/output_box_img_i_{i}_x_{x}_y_{y}.jpg".format(
                i=i, x=i * offset1, y=(i + 1) * offset1), box_canny)
        # for j in range(5):
        #     cv2.imwrite("./exports/output_row_img_i_{i}_j_{j}_x_{x}_y_{y}.jpg".format(
        #         i=i,j=j, x=i * offset1, y=(i + 1) * offset1), box_img[j * offset2:(j + 1) * offset2, :])
        # contours, _ = cv2.findContours(
        #     box_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        x_old, y_old, w_old, h_old = 0, 0, 0, 0
        list_choices = []
        sorted_box_cnts = sorted(box_cnts, key=sort_by_x)
        for box_cnt in sorted_box_cnts:
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(box_cnt)
            ar = w_curr / float(h_curr)
            if w_curr >= 50 and h_curr >= 50 and w_curr * h_curr >= 365 and w_curr < 100 and h_curr < 100 and 0.8 <= ar <= 1.2:
                print("x_curr", x_curr,"x_old" ,x_old,"y_curr", y_curr,"y_old" ,y_old,"x_curr - x_old", x_curr - x_old, "y_curr- y_old" , y_curr- y_old)
               
                if abs(x_curr - x_old) > 200 or abs(y_curr - y_old) > 200:
                    list_choices.append(
                        (box_gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr

                    bubble_choice = cv2.threshold(box_gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
                    bubble_choice = bubble_choice.reshape((28, 28, 1))
                        # cv2.imwrite("./exports/output_answer_img_{i}_{len}_x_{x}_y_{y}.jpg".format(
                        #  i=i,len=len(list_choices),x=x_curr, y=y_curr), bubble_choice)
        print("list_choices",len(list_choices))
