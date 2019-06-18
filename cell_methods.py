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
    mask[mask==1] = 0
    """plt.subplot(2, 6, 3)
    plt.imshow(mask, cmap="gray")
    plt.title('HSV Segmentation')"""

    # MORPHOLOGICAL OPERATIONS
    # Filter Kernels
    kernel_close2 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(20, 20))
    kernel_open = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10))
    kernel_dilate = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))

    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    dilation = cv2.dilate(opening, kernel_dilate, iterations=5)
    sure_bg = dilation
    ##################
    """plt.subplot(2, 6, 3)
    plt.imshow(closing, cmap="gray")
    plt.title('Closing')
    plt.subplot(2, 6, 4)
    plt.imshow(opening, cmap="gray")
    plt.title('Closing')
    plt.subplot(2, 6, 5)
    plt.imshow(dilation, cmap="gray")
    plt.title('Dilation')"""
    ####################

    result = cv2.bitwise_and(rgb_img, rgb_img, mask=sure_bg)

    # SPLIT MASK IF NEEDED
    # Finding certain foreground
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.40 * dist_transform.max(), 255, 0)
    """plt.subplot(2, 6, 8)
    plt.imshow(dist_transform, cmap="gray")
    plt.title('Distance Transform')
    plt.subplot(2, 6, 9)
    plt.imshow(sure_fg, cmap="gray")
    plt.title('Threshold')"""

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_cells, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    #plt.subplot(2, 6, 10)
    #plt.imshow(markers)
    #plt.title('Unknown Region')

    markers = cv2.watershed(result, markers)
    """plt.subplot(2, 6, 11)
    plt.imshow(markers)
    plt.title('Watershed Algorithm')"""
    ########################
    #plt.show()

    return markers, n_cells-1


def wbc(rgb_img, hsv_img):
    #plt.subplot(2, 6, 1)
    #plt.imshow(rgb_img)
    #plt.title('Input Image')

    # HSV SEGMENTATION
    # Color range
    light = (150, 50, 0)
    dark = (180, 255, 255)

    light2 = (0, 50, 0)
    dark2 = (10, 255, 255)

    mask1 = cv2.inRange(hsv_img, light, dark)
    mask2 = cv2.inRange(hsv_img, light2, dark2)
    mask = mask1 + mask2
    plt.subplot(2, 6, 2)
    plt.imshow(mask, cmap="gray")
    plt.title('HSV Segmentation')

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

    ##################
    """plt.subplot(2, 6, 3)
    plt.imshow(closing_test, cmap="gray")
    plt.title('Closing')
    plt.subplot(2, 6, 4)
    plt.imshow(opening, cmap="gray")
    plt.title('Opening')
    plt.subplot(2, 6, 5)
    plt.imshow(closing, cmap="gray")
    plt.title('Closing 2')
    plt.subplot(2, 6, 6)
    plt.imshow(dilation, cmap="gray")
    plt.title('Dilation')
    plt.subplot(2, 6, 7)
    plt.imshow(sure_bg, cmap="gray")
    plt.title('Opening 2')"""
    ####################

    result = cv2.bitwise_and(rgb_img, rgb_img, mask=sure_bg)

    # SPLIT MASK IF NEEDED
    # Finding certain foreground
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.60 * dist_transform.max(), 255, 0)
    """plt.subplot(2, 6, 8)
    plt.imshow(dist_transform, cmap="gray")
    plt.title('Distance Transform')
    plt.subplot(2, 6, 9)
    plt.imshow(sure_fg, cmap="gray")
    plt.title('Threshold')"""

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_cells, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    #plt.subplot(2, 6, 10)
    #plt.imshow(markers)
    #plt.title('Unknown Region')

    watershed = cv2.watershed(result, markers)
    #plt.subplot(2, 6, 11)
    #plt.imshow(markers)
    #plt.title('Watershed Algorithm')
    ########################
    #plt.show()

    return markers, n_cells-1, watershed


def rbc(rgb, wbc_mask):
    rgb_output = rgb.copy()
    n_rbc = 0

    wbc_mask[wbc_mask < 2] = 0
    wbc_mask[wbc_mask > 0] = 255
    wbc_mask = wbc_mask.astype(np.uint8)


    rgb_img = adjust_gamma(rgb, 0.5)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    #hsv_img = cv2.normalize(hsv_img, hsv_img, 255, 0, cv2.NORM_MINMAX, 1)
    #plot3d_hsv(hsv_img, rgb_img)

    """plt.subplot(4, 4, 1)
    plt.imshow(rgb_img[:, :, 0])
    plt.title('R')
    plt.subplot(4, 4, 2)
    plt.imshow(rgb_img[:, :, 1])
    plt.title('G')
    plt.subplot(4, 4, 3)
    plt.imshow(rgb_img[:, :, 2])
    plt.title('B')
    plt.subplot(4, 4, 4)
    plt.imshow(rgb_img)
    plt.title('RGB')
    plt.subplot(4, 4, 5)
    plt.imshow(hsv_img[:, :, 0])
    plt.title('H')
    plt.subplot(4, 4, 6)
    plt.imshow(hsv_img[:, :, 1])
    plt.title('S')
    plt.subplot(4, 4, 7)
    plt.imshow(hsv_img[:, :, 2])
    plt.title('V')
    plt.subplot(4, 4, 8)
    plt.imshow(rgb_img)
    plt.title('HSV')
    plt.subplot(4, 4, 9)
    plt.imshow(lab_img[:, :, 0])
    plt.title('L')
    plt.subplot(4, 4, 10)
    plt.imshow(lab_img[:, :, 1])
    plt.title('A')
    plt.subplot(4, 4, 11)
    plt.imshow(lab_img[:, :, 2])
    plt.title('B')
    plt.subplot(4, 4, 12)
    plt.imshow(lab_img)
    plt.title('Lab')
    plt.subplot(4, 4, 13)
    plt.imshow(YCrCb_img[:, :, 0])
    plt.title('Y')
    plt.subplot(4, 4, 14)
    plt.imshow(YCrCb_img[:, :, 1])
    plt.title('Cr')
    plt.subplot(4, 4, 15)
    plt.imshow(YCrCb_img[:, :, 2])
    plt.title('Cb')
    plt.subplot(4, 4, 16)
    plt.imshow(YCrCb_img)
    plt.title('YCrCb')
    plt.show()

    return"""

    # HSV SEGMENTATION
    # Color range
    light = (0, 33, 0)
    dark = (60, 255, 255)

    light2 = (140, 30, 0)
    dark2 = (180, 255, 255)

    #plt.subplot(2, 6, 1)
    #plt.imshow(hsv_img, cmap="gray")
    #plt.title('Original')

    mask1 = cv2.inRange(hsv_img, light, dark)
    mask2 = cv2.inRange(hsv_img, light2, dark2)
    mask = mask1 + mask2 - wbc_mask
    mask[mask==1] = 0
    #plt.subplot(2, 6, 2)
    #plt.imshow(wbc_mask, cmap="gray")
    #plt.title('HSV Segmentation')

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
    clos2[clos2==1] = 0

    ##################
    """plt.subplot(2, 6, 3)
    plt.imshow(closing_test2, cmap="gray")
    plt.title('hmm')
    plt.subplot(2, 6, 4)
    plt.imshow(opening, cmap="gray")
    plt.title('Closing2')
    plt.subplot(2, 6, 5)
    plt.imshow(clos, cmap="gray")
    plt.title('Opening2')
    plt.subplot(2, 6, 6)
    plt.imshow(clos2, cmap="gray")
    plt.title('WBC Removed')"""

    result = cv2.bitwise_and(rgb_img, rgb_img, mask=clos2)
    #plt.subplot(2, 6, 7)
    #plt.imshow(result)
    #plt.title('Morph')
    ####################

    circles = cv2.HoughCircles(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY), cv2.HOUGH_GRADIENT, 2, 65, param1=60, param2=30, minRadius=25, maxRadius=55)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            new_mask = np.zeros((np.shape(result)[0], np.shape(result)[1]))
            cv2.circle(new_mask, (i[0], i[1]), i[2], 1, -1)

            #result2 = cv2.bitwise_and(result, result, mask=new_mask.astype(np.uint8))

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

            if (crop_img.mean() < HOUGH_MASK_FACTOR):
                continue
            else:
                test = cv2.bitwise_and(rgb_img.astype(np.uint8), rgb_img.astype(np.uint8), mask=new_mask.astype(np.uint8))
                #print(crop_img.mean(), rectY1,rectY2, rectX1,rectX2)
                #plt.subplot(1, 1, 1)
                #plt.imshow(test)
                #plt.show()

                n_rbc += 1
                #cv2.circle(clos_circ, (i[0], i[1]), i[2], 200, 2)
                rgb_output = draw_boundarieRBC(rgb_output, i[0], i[1], i[2])

    #plt.subplot(2, 6, 7)
    #plt.imshow(result)
    #plt.title('Hough')

    #contours, hierarchy = cv2.findContours(clos.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for cnt in contours:
        #cv2.drawContours(clos, [cnt], 0, 255, -1)



    """
    # SPLIT MASK IF NEEDED
    # Finding certain foreground
    dist_transform = cv2.distanceTransform(clos2, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.30 * dist_transform.max(), 255, 0)
    plt.subplot(2, 6, 8)
    plt.imshow(dist_transform, cmap="gray")
    plt.title('dist_transform')
    plt.subplot(2, 6, 9)
    plt.imshow(sure_fg, cmap="gray")
    plt.title('Threshold')

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(clos2, sure_fg)
    n_cells, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    plt.subplot(2, 6, 10)
    plt.imshow(markers)
    plt.title('Unknown Region')

    markers = cv2.watershed(result, markers)
    plt.subplot(2, 6, 11)
    plt.imshow(markers)
    plt.title('Watershed Algorithm')

    ########################
    rgb_img = draw_boundariesRBC(markers, n_cells, rgb_img)

    plt.subplot(2, 6, 12)
    plt.imshow(rgb_img)
    plt.title('end')"""
    #plt.show()
    return rgb_output, n_rbc
