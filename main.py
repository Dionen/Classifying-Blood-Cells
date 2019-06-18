import scipy.misc
from matplotlib import pyplot
from numpy import *
import numpy as np
import imageio
from cell_methods import *

filenumber = '00310'


def parsing(filenumber):
    name = ''
    wbc = 0
    platelets = 0
    rbc = 0
    tree = ET.parse('dataset-master/Annotations/BloodImage_' + filenumber + '.xml')
    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            for attr in list(elem):
                if 'name' in attr.tag:
                    name = attr.text
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                    if name[0] == "R":
                        rbc += 1
                    if name[0] == "W":
                        wbc += 1
                    if name[0] == "P":
                        platelets += 1
    return wbc, platelets, rbc


if __name__ == '__main__':
    img = cv2.imread('dataset-master/JPEGImages/BloodImage_' + filenumber + '.jpg')
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    wbc_markers, n_wbc, wbc_mask = wbc(rgb_img, hsv_img)
    platelet_markers, n_platelets = platelets(rgb_img, hsv_img, wbc_mask.copy())
    rbc_img, n_rbc = rbc(rgb_img, wbc_mask.copy())

    plat_img = draw_boundariesPLATELETS(platelet_markers, n_platelets, rbc_img)
    final_img = draw_boundariesWBC(wbc_markers, n_wbc, plat_img)
    # cv2.imwrite("out21.png", cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))

    expected_wbc, expected_platelets, expected_rbc = parsing(filenumber)
    # print(n_wbc, n_platelets, n_rbc, '\n', )

    print(expected_wbc, expected_platelets, expected_rbc)
    print(n_wbc, n_platelets, n_rbc)
    print("WBC PRECISION: %.3f" % precision(n_wbc, expected_wbc))
    print("RBC PRECISION: %.3f" % precision(n_rbc, expected_rbc))
    print("PLATELETS PRECISION: %.3f" % precision(n_platelets, expected_platelets))

    # rgb_img = draw_boundariesWBC(markers, n_cells, edit_img)