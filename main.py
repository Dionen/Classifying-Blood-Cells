from numpy import *
from cell_methods import *
import glob


def parsing(tree):
    name = ''
    wbc = 0
    platelets = 0
    rbc = 0
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


def get_total_statistics():
    jpeg_names = glob.glob("dataset-master\JPEGImages\*.jpg")
    xml_names = glob.glob("dataset-master\Annotations\*.xml")
    f = open("statistics2.txt", "w+")

    total_wbc = 0
    total_rbc = 0
    total_platelets = 0
    found_wbc = 0
    found_rbc = 0
    found_platelets = 0

    for i in range(min(len(jpeg_names), len(xml_names))):
        if jpeg_names[i][26:-4] == xml_names[i][27:-4]:

            img = cv2.imread(jpeg_names[i])
            tree = ET.parse(xml_names[i])

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            wbc_markers, n_wbc, wbc_mask = wbc(rgb_img, hsv_img)
            platelet_markers, n_platelets = platelets(rgb_img, hsv_img, wbc_mask.copy())
            rbc_img, n_rbc = rbc(rgb_img, wbc_mask.copy())

            plat_img = draw_boundariesPLATELETS(platelet_markers, n_platelets, rbc_img)
            final_img = draw_boundariesWBC(wbc_markers, n_wbc, plat_img)

            #cv2.imwrite('output/' + jpeg_names[i][26:-4] + ".png", cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))

            expected_wbc, expected_platelets, expected_rbc = parsing(tree)

            f.write("FILE: %s\t||\tfw:%d fp:%d fr:%d\t||\t" % (jpeg_names[i], n_wbc, n_platelets, n_rbc))
            f.write("ew:%d ep:%d er:%d\t||\t" % (expected_wbc, expected_platelets, expected_rbc))
            f.write("%.3f %.3f %.3f\n" % (precision(expected_wbc, n_wbc), precision(expected_platelets, n_platelets), precision(expected_rbc, n_rbc)))

            total_wbc += expected_wbc
            total_rbc += expected_rbc
            total_platelets += expected_platelets
            found_wbc += n_wbc
            found_rbc += n_rbc
            found_platelets += n_platelets

    f.write("==========================================================================\n")
    f.write("FOUND:\tew:%d\tep:%d\ter:%d\n" % (found_wbc, found_platelets, found_rbc))
    f.write("EXPECTED:\tew:%d\tep:%d\ter:%d\n" % (total_wbc, total_platelets, total_rbc))
    f.write("WBC PRECISION: %.3f\n" % precision(total_wbc, found_wbc))
    f.write("RBC PRECISION: %.3f\n" % precision(total_rbc, found_rbc))
    f.write("PLATELETS PRECISION: %.3f\n" % precision(total_platelets, found_platelets))
    f.write("TOTAL PRECISION: %.3f\n" % precision(total_rbc + total_platelets + total_wbc, found_rbc + found_wbc + found_platelets))
    f.close()


if __name__ == '__main__':
    get_total_statistics()

    """
    filenumber = '00313'

    img = cv2.imread('dataset-master/JPEGImages/BloodImage_' + filenumber + '.jpg')
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    wbc_markers, n_wbc, wbc_mask = wbc(rgb_img, hsv_img)
    platelet_markers, n_platelets = platelets(rgb_img, hsv_img, wbc_mask.copy())
    rbc_img, n_rbc = rbc(rgb_img, wbc_mask.copy())

    plat_img = draw_boundariesPLATELETS(platelet_markers, n_platelets, rbc_img)
    final_img = draw_boundariesWBC(wbc_markers, n_wbc, plat_img)
    # cv2.imwrite("out21.png", cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))

    tree = ET.parse('dataset-master/Annotations/BloodImage_' + filenumber + '.xml')
    expected_wbc, expected_platelets, expected_rbc = parsing(tree)
    # print(n_wbc, n_platelets, n_rbc, '\n', )

    print(expected_wbc, expected_platelets, expected_rbc)
    print(n_wbc, n_platelets, n_rbc)
    print("WBC PRECISION: %.3f" % precision(n_wbc, expected_wbc))
    print("RBC PRECISION: %.3f" % precision(n_rbc, expected_rbc))
    print("PLATELETS PRECISION: %.3f" % precision(n_platelets, expected_platelets))

    # rgb_img = draw_boundariesWBC(markers, n_cells, edit_img)"""