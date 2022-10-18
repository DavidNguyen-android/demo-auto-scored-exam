from math import ceil
import cv2
import numpy as np
import imutils
from utils import *
import glob
import os
import datetime
# read input image


def getAnswerFromImage(imageFrame):
    height, width, channels = imageFrame.shape
    answer_start_y = 2200
    answer_start_left_x = 0
    answer_start_right_x = int(width / 2)
    imageCroppedLeft = imageFrame[answer_start_y:height,
                                  answer_start_left_x:answer_start_right_x]
    cntsLeft = findAnswerContainerWithDrawLines(
        imageCroppedLeft,"imageCroppedLeft") + findAnswerContainer(imageCroppedLeft)
    imageCroppedRight = imageFrame[answer_start_y:height,
                                   answer_start_right_x:width]
    cntsRight = findAnswerContainerWithDrawLines(
        imageCroppedRight,"imageCroppedRight") + findAnswerContainer(imageCroppedRight)
    ans_blocks_left = []
    if len(cntsLeft) > 0:
        x_old, y_old, w_old, h_old = 0, 0, 0, 0
        # sort the contours according to their size in descending order
        cntsLeft = sorted(cntsLeft, key=get_x_ver1)

        for i, c in enumerate(cntsLeft):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)
            x_curr = x_curr + answer_start_left_x
            y_curr = y_curr + answer_start_y
            # check overlap contours
            check_xy_min = x_curr * y_curr - x_old * y_old
            check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - \
                (x_old + w_old) * (y_old + h_old)

            if x_curr < x_old + w_old:
                continue
            min_w_curr = 1100
            max_w_curr = 1200
            min_h_curr = 2900
            max_h_curr = 3000
            if w_curr > min_w_curr and h_curr > min_h_curr and w_curr < max_w_curr and h_curr < max_h_curr:
                # if list answer box is empty
                if len(ans_blocks_left) == 0 or (check_xy_min > 20000 and check_xy_max > 20000):
                    ans_blocks_left.append(
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

    ans_blocks_right = []
    if len(cntsRight) > 0:
        x_old, y_old, w_old, h_old = 0, 0, 0, 0
        # sort the contours according to their size in descending order
        cntsRight = sorted(cntsRight, key=get_x_ver1)

        for i, c in enumerate(cntsRight):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)
            x_curr = x_curr + answer_start_right_x
            y_curr = y_curr + answer_start_y
            # check overlap contours
            check_xy_min = x_curr * y_curr - x_old * y_old
            check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - \
                (x_old + w_old) * (y_old + h_old)

            if x_curr < x_old + w_old:
                continue
            min_w_curr = 1100
            max_w_curr = 1200
            min_h_curr = 2900
            max_h_curr = 3000
            if w_curr > min_w_curr and w_curr < max_w_curr and h_curr > min_h_curr and h_curr < max_h_curr:
                # if list answer box is empty
                if len(ans_blocks_right) == 0 or (check_xy_min > 20000 and check_xy_max > 20000):
                    ans_blocks_right.append(
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

    sorted_ans_blocks = sorted(ans_blocks_left + ans_blocks_right, key=get_x)
    # process_ans_blocks
    block_count = 5

    print("sorted_ans_blocks", len(sorted_ans_blocks))
    for ans_block in sorted_ans_blocks:
        ans_block_img = np.array(ans_block[0])
        offset1 = ceil(ans_block_img.shape[0] / block_count)
        # Loop over each box in answer block
        for i in range(block_count):
            # print("block_count", i)
            box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
            # loop over each line in a box
            box_gray_img = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
            box_blurred = cv2.GaussianBlur(box_gray_img, (5, 5), 0)
            box_canny = cv2.Canny(box_blurred.copy(), 100, 200)
            box_cnts = cv2.findContours(
                box_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            box_canny_2 = cv2.Canny(box_blurred.copy(), 10, 20)
            box_cnts_2 = cv2.findContours(
                box_canny_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            _, box_threshold = cv2.threshold(
                box_gray_img, 170, 255, cv2.THRESH_BINARY_INV)
            box_check_cnts = cv2.findContours(
                box_threshold.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            box_canny_3 = cv2.Canny(box_blurred.copy(), 0, 50)
            box_cnts_3 = cv2.findContours(
                box_canny_3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            box_cnts_5 = cv2.findContours(
                box_canny_3.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            box_threshold_4 = cv2.adaptiveThreshold(
                box_blurred.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            box_cnts_4 = cv2.findContours(
                box_threshold_4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # box_cnts = box_cnts + box_check_cnts
            box_cnts = imutils.grab_contours(box_cnts)
            box_cnts_2 = imutils.grab_contours(box_cnts_2)
            box_cnts_3 = imutils.grab_contours(box_cnts_3)
            box_cnts_4 = imutils.grab_contours(box_cnts_4)
            box_cnts_5 = imutils.grab_contours(box_cnts_5)
            box_check_cnts = imutils.grab_contours(box_check_cnts)
            box_cnts = box_cnts + box_cnts_2 + box_check_cnts + \
                box_cnts_3 + box_cnts_4 + box_cnts_5
            # print("box_cnts", len(box_cnts))
            # cv2.imwrite("./exports/output_box_canny_i_{i}_x_{x}_y_{y}.jpg".format(
            #     i=i, x=i * offset1, y=(i + 1) * offset1), box_canny)
            # cv2.imwrite("./exports/output_box_canny2_i_{i}_x_{x}_y_{y}.jpg".format(
            #     i=i, x=i * offset1, y=(i + 1) * offset1), box_canny_2)
            # cv2.imwrite("./exports/output_box_canny_3_i_{i}_x_{x}_y_{y}.jpg".format(
            # i=i, x=i * offset1, y=(i + 1) * offset1), box_canny_3)
            # cv2.imwrite("./exports/output_box_threshold_i_{i}_x_{x}_y_{y}.jpg".format(
            #     i=i, x=i * offset1, y=(i + 1) * offset1), box_threshold)
            # loop over the contours
            x_old, y_old, w_old, h_old = 0, 0, 0, 0
            list_choices = []
            box_cnts = sorted(box_cnts, key=sort_by_y)
            box_cnts_group = []
            box_cnts_each_group = []
            y_start = 0
            # print("box_cnts", len(box_cnts))
            for box_cnt in box_cnts:
                x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(box_cnt)
                ar = w_curr / float(h_curr)
                # cv2.imwrite("./exports/output_answer_img_{i}_y_{y}_x_{x}.jpg".format(
                #             i=i, x=x_curr, y=y_curr), box_gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr])
                # print("x_curr", x_curr, "y_curr", y_curr,"w_curr", w_curr, "h_curr", h_curr,)
                # print(w_curr >= 10, h_curr >= 10,  w_curr * h_curr >= 365, w_curr < 100, h_curr < 100,0.8 <= ar <= 1.2)
                # if w_curr * h_curr > 3500 and 0.8 <= ar <= 1.2:
                if w_curr * h_curr > 3500 and 0.7 <= ar <= 1.3:
                    # cv2.imwrite("./exports/output_answer_img_{i}_y_{y}_x_{x}.jpg".format(
                    #         i=i, x=x_curr, y=y_curr), box_gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr])
                    # print( "w_curr", w_curr, "h_curr", h_curr)

                    if len(box_cnts_each_group) < 4 or y_curr - y_start < 80:
                        box_cnts_each_group.append(box_cnt)
                        if y_start == 0:
                            y_start = y_curr
                            x_old = x_curr
                        # print("x_curr", x_curr, "y_curr", y_curr,)
                    else:
                        box_cnts_group.append(
                            sorted(box_cnts_each_group, key=sort_by_x))
                        box_cnts_each_group = [box_cnt]
                        y_start = 0
                        x_old = 0
                        # print("--------------------------------")
                        # print("x_curr", x_curr, "y_curr", y_curr,)
            if len(box_cnts_each_group) > 0:
                box_cnts_group.append(
                    sorted(box_cnts_each_group, key=sort_by_x))
                box_cnts_each_group = []
                y_start = 0
                x_old = 0
                # print("--------------------------------")

            # cv2.imwrite("./exports/output_box_draw_contour_img_i_{i}_x_{x}_y_{y}.jpg".format(
            #     i=i, x=i * offset1, y=(i + 1) * offset1), box_canny_draw_cnt)
            # sorted_box_cnts = []
            for box_cnt_each_group in box_cnts_group:
                each_answers = []
                # print("box_cnt_each_group", len(box_cnt_each_group),)
                for box_cnt in box_cnt_each_group:
                    x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(box_cnt)
                    # print("x_curr", x_curr, "y_curr", y_curr,)
                    if abs(x_curr - x_old) > 200:
                        each_answers.append(
                            (box_gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                        list_choices.append(
                            (box_gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                        # update coordinates (x, y) and (height, width) of added contours
                        x_old = x_curr
                        y_old = y_curr
                        w_old = w_curr
                        h_old = h_curr
                        bubble_choice = cv2.threshold(
                            box_gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                        bubble_choice = cv2.resize(
                            bubble_choice, (28, 28), cv2.INTER_AREA)
                        bubble_choice = bubble_choice.reshape((28, 28, 1))
                        cv2.drawContours(box_img, box_cnt, -1, (0, 255, 0), 3)

                        # cv2.imwrite("./exports/output_answer_img_{i}_{len}_y_{y}_x_{x}.jpg".format(
                        # i=i, len=len(list_choices), x=x_curr, y=y_curr), bubble_choice)
                # print("each_answers len" , len(each_answers),"\n---------------------------------\n")
            # for box_cnt in sorted_box_cnts:
            #     x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(box_cnt)

            #     print("x_curr", x_curr, "y_curr", y_curr,)

            #     if (len(list_choices_each_row) < 4 and abs(x_curr - x_old) > 150 and x_curr > x_old) or (len(list_choices_each_row) == 0 and abs(y_curr - y_old) > 70):
            #             list_choices_each_row.append((box_gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
            #             list_choices.append(
            #                 (box_gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
            #             # update coordinates (x, y) and (height, width) of added contours
            #             x_old = x_curr
            #             y_old = y_curr
            #             w_old = w_curr
            #             h_old = h_curr

            #             bubble_choice = cv2.threshold(
            #                 box_gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            #             bubble_choice = cv2.resize(
            #                 bubble_choice, (28, 28), cv2.INTER_AREA)
            #             bubble_choice = bubble_choice.reshape((28, 28, 1))
            #             # if len(box_cnts) == 154:
            #             cv2.imwrite("./exports/output_answer_img_{i}_{len}_x_{x}_y_{y}.jpg".format(
            #              i=i,len=len(list_choices),x=x_curr, y=y_curr), bubble_choice)
            #     if (len(list_choices_each_row) >= 4):
            #             list_choices_each_row = []
            # cv2.imwrite("./exports/output_box_img_i_{i}_x_{x}_y_{y}.jpg".format(
            #     i=i, x=i * offset1, y=(i + 1) * offset1), box_img)
            if len(list_choices) != 20:
                print("list_choices ----", len(list_choices))
            else:
                print("list_choices --------------", len(list_choices))


def findAnswerContainerWithDrawLines(imageFrame, name):
    # Convert image to grayscale
    image = imageFrame.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use canny edge detection
    edges = cv2.Canny(gray, 0, 200, apertureSize=3)

    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list = []
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi/180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=10  # Max allowed gap between line for joining them
    )

    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Maintain a simples lookup list for points
        lines_list.append([(x1, y1), (x2, y2)])

    # Save the result image
    cv2.imwrite('./exports/{name}_detectedLines.png'.format(name = name), image)
    imageFrameDrawLine = image.copy()
    cnts = findAnswerContainer(imageFrameDrawLine)
    return cnts


def findAnswerContainer(imageFrame):
    img_grey = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_grey, 5).astype('uint8')

    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cnts = cv2.findContours(img_thresh.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)

    img_grey_2 = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
    img_blur_2 = cv2.GaussianBlur(img_grey_2, (5, 5), 0)
    img_canny_2 = cv2.Canny(img_blur_2,  10, 20, )
    cnts7 = cv2.findContours(img_canny_2.copy(), cv2.RETR_LIST,
                             cv2.CHAIN_APPROX_NONE)
    img_canny_3 = cv2.Canny(img_blur_2,  0, 50, )
    cnts8 = cv2.findContours(img_canny_3.copy(), cv2.RETR_LIST,
                             cv2.CHAIN_APPROX_NONE)
    _, img_thresh_2 = cv2.threshold(
        img_canny_2, 45, 255, cv2.THRESH_BINARY_INV)

    img_thresh_2 = cv2.erode(img_thresh_2, None, iterations=2)
    img_thresh_2 = cv2.dilate(img_thresh_2, None, iterations=2)
    # cv2.imwrite("./exports/img_canny_2.jpg", img_canny_2)
    # cv2.imwrite("./exports/img_thresh.jpg", img_thresh)
    # cv2.imwrite("./exports/img_thresh_2.jpg", img_thresh_2)

    _, thresh3 = cv2.threshold(
        img_blur_2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts6 = cv2.findContours(
        thresh3.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imwrite("./exports/img_thresh_3.jpg", thresh3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh3 = cv2.morphologyEx(thresh3.copy(), cv2.MORPH_OPEN, kernel)
    thresh3 = cv2.morphologyEx(thresh3.copy(), cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite("./exports/img_thresh_3_morphologyEx.jpg", thresh3)
    cnts2 = cv2.findContours(img_thresh_2.copy(), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)

    cnts3 = cv2.findContours(img_canny_2.copy(), cv2.RETR_LIST,
                             cv2.CHAIN_APPROX_NONE)

    cnts4 = cv2.findContours(img_canny_2.copy(), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)

    cnts5 = cv2.findContours(
        thresh3.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts9 = cv2.findContours(
        img_grey.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts10 = cv2.findContours(
        img_grey.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts2 = imutils.grab_contours(cnts2)
    cnts3 = imutils.grab_contours(cnts3)
    cnts4 = imutils.grab_contours(cnts4)
    cnts5 = imutils.grab_contours(cnts5)
    cnts6 = imutils.grab_contours(cnts6)
    cnts7 = imutils.grab_contours(cnts7)
    cnts8 = imutils.grab_contours(cnts8)
    cnts9 = imutils.grab_contours(cnts9)
    cnts10 = imutils.grab_contours(cnts10)
    cnts = cnts + cnts2 + cnts3 + cnts4 + cnts5 + \
        cnts6 + cnts7 + cnts8 + cnts9 + cnts10
    return cnts


if __name__ == '__main__':
    os.chdir("./output")
    print("start------------------",
          datetime.datetime.now(), '------------------",')
    for file in glob.glob("*.jpg"):
        print("start------------------", file,
              datetime.datetime.now(), '------------------",')
        imageFrame = cv2.imread(file)
        getAnswerFromImage(imageFrame)
        print("end------------------", file,
              datetime.datetime.now(), '------------------",')
    print("end------------------", datetime.datetime.now(), '------------------",')
    # print("start------------------",
    #       datetime.datetime.now(), '------------------",')
    # imageFrame = cv2.imread('output/output_11.jpg')
    # getAnswerFromImage(imageFrame)
    # print("end------------------", datetime.datetime.now(), '------------------",')
