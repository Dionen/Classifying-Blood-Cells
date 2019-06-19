import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np
import xml.etree.ElementTree as ET
from utils import *
from draw import *
import cv2

HOUGH_MASK_FACTOR = 85


def platelets(rgb_img, hsv_img, wbc_mask):
    wbc_mask[wbc_mask < 2] = 0
    wbc_mask[wbc_mask > 0] = 255
    wbc_mask = wbc_mask.astype(np.uint8)

    # Treat WBC Mask
    kernel_wbc = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10))
    wbc_mask_dilated = cv2.dilate(wbc_mask, kernel_wbc, iterations=5)

    # HSV SEGMENTATION
    # Color range
    light = (170, 60, 0)
    dark = (180, 255, 255)

    light2 = (0, 20, 0)
    dark2 = (10, 255, 255)

    mask1 = cv2.inRange(hsv_img, light, dark)
    mask2 = cv2.inRange(hsv_img, light2, dark2)
    mask = mask1 + mask2 - wbc_mask_dilated
    mask[mask == 1] = 0

    # MORPHOLOGICAL OPERATIONS
    # Filter Kernels
    kernel_close2 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(20, 20))
    kernel_open = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10))
    kernel_dilate = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))

    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    dilation = cv2.dilate(opening, kernel_dilate, iterations=3)
    sure_bg = dilation

    result = cv2.bitwise_and(rgb_img, rgb_img, mask=sure_bg)

    # SPLIT MASK IF NEEDED
    # Finding certain foreground
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.70 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_cells, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(result, markers)

    return markers, n_cells-1


def wbc(rgb_img, hsv_img):
    # HSV SEGMENTATION
    # Color range
    light = (150, 50, 0)
    dark = (180, 255, 255)

    light2 = (0, 50, 0)
    dark2 = (10, 255, 255)

    mask1 = cv2.inRange(hsv_img, light, dark)
    mask2 = cv2.inRange(hsv_img, light2, dark2)
    mask = mask1 + mask2

    # MORPHOLOGICAL OPERATIONS
    # Filter Kernels
    kernel_close2 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(14, 14))
    kernel_denoise = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(25, 35))
    kernel_close = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(40, 40))
    kernel_close3 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10))
    kernel_dilate = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))

    closing_test = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close2)
    opening = cv2.morphologyEx(closing_test, cv2.MORPH_OPEN, kernel_denoise)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)
    dilation = cv2.dilate(closing, kernel_dilate, iterations=1)
    sure_bg = cv2.dilate(dilation, kernel_close3, iterations=2)

    result = cv2.bitwise_and(rgb_img, rgb_img, mask=sure_bg)

    # SPLIT MASK IF NEEDED
    # Finding certain foreground
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.65 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_cells, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed = cv2.watershed(result, markers)

    return markers, n_cells-1, watershed


def rbc(rgb, wbc_mask):
    rgb_output = rgb.copy()
    n_rbc = 0

    wbc_mask[wbc_mask < 2] = 0
    wbc_mask[wbc_mask > 0] = 255
    wbc_mask = wbc_mask.astype(np.uint8)

    rgb_img = adjust_gamma(rgb, 0.5)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    # HSV SEGMENTATION
    # Color range
    light = (0, 33, 0)
    dark = (60, 255, 255)

    light2 = (140, 30, 0)
    dark2 = (180, 255, 255)

    mask1 = cv2.inRange(hsv_img, light, dark)
    mask2 = cv2.inRange(hsv_img, light2, dark2)
    mask = mask1 + mask2 - wbc_mask
    mask[mask == 1] = 0

    # MORPHOLOGICAL OPERATIONS
    # Filter Kernels
    kernel_denoise = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(15, 15))
    kernel_erose = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10))
    kernel_opening2 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))
    kernel_close3 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10))

    closing_test2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_close3)
    opening2 = cv2.dilate(closing_test2,kernel_opening2,iterations = 4)
    opening = cv2.morphologyEx(opening2, cv2.MORPH_OPEN, kernel_denoise)
    clos = cv2.erode(opening, kernel_erose,iterations = 2)

    # REMOVES WBC
    clos2 = clos - wbc_mask
    clos2[clos2 == 1] = 0
    result = cv2.bitwise_and(rgb_img, rgb_img, mask=clos2)

    # Hough Algorithm
    circles = cv2.HoughCircles(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY), cv2.HOUGH_GRADIENT, 2, 65, param1=60, param2=30, minRadius=25, maxRadius=55)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            new_mask = np.zeros((np.shape(result)[0], np.shape(result)[1]))
            cv2.circle(new_mask, (i[0], i[1]), i[2], 1, -1)

            rectX1 = i[0].astype(np.int64) - i[2].astype(np.int64)
            if rectX1 < 0: rectX1 = 0
            rectX2 = i[0].astype(np.int64) + i[2].astype(np.int64)
            if rectX2 >= new_mask.shape[1]: rectX2 = new_mask.shape[1]-1
            rectY1 = i[1].astype(np.int64) - i[2].astype(np.int64)
            if rectY1 < 0: rectY1 = 0
            rectY2 = i[1].astype(np.int64) + i[2].astype(np.int64)
            if rectY2 >= new_mask.shape[0]: rectY2 = new_mask.shape[0] - 1

            new_mask = new_mask*clos2
            crop_img = new_mask[rectY1:rectY2, rectX1:rectX2]

            if crop_img.mean() < HOUGH_MASK_FACTOR:
                continue
            else:
                n_rbc += 1
                rgb_output = draw_boundarieRBC(rgb_output, i[0], i[1], i[2])

    return rgb_output, n_rbc
