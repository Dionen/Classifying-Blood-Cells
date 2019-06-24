# João Vitor Guino Rieswick nº9283607
# SCC0251 - Prof. Moacir Ponti
# Teaching Assistant: Aline Becher

import numpy as np
import cv2


def draw_boundariesWBC(markers, n_cells, edit_img):
    rgb_img = edit_img.copy()
    font = cv2.QT_FONT_NORMAL
    for k in range(2, n_cells + 2):
        contours, hier = cv2.findContours((markers == k).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            cv2.rectangle(rgb_img, (x - 16, y - 16), (x + w + 16, y + h + 16), (255, 0, 0), 2)
            cv2.putText(rgb_img, 'WBC', (x - 10, y + 12), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
    return rgb_img


def draw_boundariesPLATELETS(markers, n_cells, edit_img):
    rgb_img = edit_img.copy()
    font = cv2.QT_FONT_NORMAL
    for k in range(2, n_cells + 2):
        contours, hier = cv2.findContours((markers == k).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            cv2.rectangle(rgb_img, (x - 8, y - 8), (x + w + 8, y + h + 8), (0, 0, 255), 1)
            cv2.putText(rgb_img, 'Platelet', (x - 5, y + 6), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return rgb_img


def draw_boundarieRBC(edit_img, x, y, r):
    rgb_img = edit_img.copy()
    font = cv2.QT_FONT_NORMAL
    r += 5

    rectX1 = x.astype(np.int32) - r.astype(np.int32)
    if rectX1 < 0: rectX1 = 0
    rectX2 = x + r
    if rectX2 >= rgb_img.shape[1]: rectX2 = rgb_img.shape[1] - 1
    rectY1 = y.astype(np.int32) - r.astype(np.int32)
    if rectY1 < 0: rectY1 = 0
    rectY2 = y + r
    if rectY2 >= rgb_img.shape[0]: rectY2 = rgb_img.shape[0] - 1

    cv2.rectangle(rgb_img, (rectX1, rectY1), (rectX2, rectY2), (100, 0, 100), 1)
    cv2.putText(rgb_img, 'RBC', (rectX1 + 5, rectY1 + 15), font, 0.5, (100, 0, 100), 1, cv2.LINE_AA)
    return rgb_img
