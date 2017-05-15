###################### INFORMATION ##############################
#          crass-Addin to crop and splice segements of an image
#          Optional preprocssing: deskew
#Program:  **crass**
#Info:     **Python 2.7**
#Author:   **Jan Kamlah**
#Date:     **12.05.2017**

####################### IMPORT ##################################
import os as os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imsave
import skimage.filters.thresholding as th
import skimage.filters as skfilters
import skimage.transform as sktransform
from scipy.ndimage import morphology,measurements,filters
from scipy.ndimage.morphology import *

####################### OBJECTS ###################################
class Line_Param():
    minwidth = 0.3
    maxwidth = 0.95
    minheight = 0.05
    maxheight = 0.95

class Image_Param():
    def __init__(self, image):
        self.height, self.width = image.shape

####################### FUNCTIONS ##################################
def height(s):
    return s[0].stop-s[0].start

def width(s):
    return s[1].stop-s[1].start

def linecords(s):
    return [[s[0].start,s[0].stop],[s[1].start,s[1].stop]]

def mindist(s,length):
    d1 = s[1].start
    d2 = length - s[1].stop
    if d1 < d2:
        return d1-10
    else:
        return d2-10

def deskew(image, image_param, line_param, deskew_rng,pathout):
    # Deskew
    # Calculate the angle of the points between 20% and 80% of the line
    thresh = th.threshold_sauvola(image, 31)
    binary = image > thresh
    binary = 1-binary #inverse binary
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    for i, b in enumerate(objects):
        # The line has to be bigger than minwidth, smaller than maxwidth, stay in the top (30%) of the img,
        # only one obj allowed and the line isnt allowed to start contact the topborder of the image
        if width(b) > line_param.minwidth * image_param.width and width(b) < line_param.maxwidth * image_param.width and \
                        b[0].stop < image_param.height * 0.3 and b[0].start != 0:

            obj_height, ob_jwidth = binary[b].shape

            obj_width_prc = ob_jwidth / 100
            arr = np.arange(1, (obj_width_prc * (100 - deskew_rng * 2)) + 1)
            mean_y = []
            #Calculate the mean value for every y-array
            for idx in range(obj_width_prc * (100 - deskew_rng * 2)):
                value_y = measurements.find_objects(labels[b][:, idx + (obj_width_prc * deskew_rng)] == i + 1)[0]
                mean_y.append((value_y[0].stop + value_y[0].start) / 2)
            polyfit_value = np.polyfit(arr, mean_y, 1)
            deskewangle = np.arctan(polyfit_value[0]) * (360 / (2 * np.pi))
            deskew_image = sktransform.rotate(image, deskewangle)
            deskew_path = "%s_deskew.jpg" % (pathout)
            imsave(deskew_path, deskew_image)
            break

    return deskew_path

def linecord_analyse(image, image_param, pathout, line_param):
    thresh = th.threshold_sauvola(image, 31)
    binary = image > thresh
    binary = 1-binary #inverse binary
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    count_height = 0
    count_width = 0
    list_linecords = [] # Init list of linecoordinates the format is: [0]: width.start, width.stopt,
    # [1]:height.start, height.stop, [2]: Type of line [B = blank, P = plumb]
    border = 0;
    hobj_bottom = 0;

    for i, b in enumerate(objects):
        # The line has to be bigger than minwidth, smaller than maxwidth, stay in the top (30%) of the img,
        # only one obj allowed and the line isnt allowed to start contact the topborder of the image
        if width(b) > line_param.minwidth * image_param.width and width(b) < line_param.maxwidth * image_param.width and b[
            0].stop < image_param.height * 0.3 and count_height == 0 and b[0].start != 0:

            # Distance Calculation
            border = mindist(b, image_param.width)
            hobj_bottom = b[0].stop + 5  # Lowest Point of object + 5 Pixel
            rut = image[hobj_bottom:image_param.height, border:image_param.width - border]  # Region Under Topline
            imsave("%s_crop.jpg" % (pathout), rut)

            # Get coordinats of the line
            lvl_linecords = linecords(b)
            labels[b][labels[b] == i + 1] = 0
            count_width += 1

        if height(b) > line_param.minheight * image_param.height and height(b) < line_param.maxheight * image_param.height and width(b) < 100:
            cords = linecords(b)
            if count_height == 0:
                if b[0].start - hobj_bottom > 50:
                    list_linecords.append(np.array([[hobj_bottom, b[0].start], [border, image_param.width - border], ['B']]))
                    count_height += 1
                list_linecords.append([cords[0], cords[1], ['P']])
            elif count_height != 0:
                if b[0].start - list_linecords[count_height - 1][0][1] > 50:
                    list_linecords.append(np.array(
                        [[list_linecords[count_height - 1][0][1], b[0].start], [border, image_param.width - border], ['B']]))
                    count_height += 1
                    list_linecords.append([cords[0], cords[1], ['P']])
                elif b[0].start - list_linecords[count_height - 1][0][1] < 20:
                    list_linecords[count_height - 1][0][1] = b[0].stop
                    count_height -= 1
                else:
                    list_linecords.append([cords[0], cords[1], ['P']])
            count_height += 1
            labels[b][labels[b] == i + 1] = 0
    return list_linecords, border, hobj_bottom

#def save_roi(widthstart. widthstop, heightstart, heightstop):
    #return 0


def crop(image, image_param, pathout, list_linecords, border, hobj_bottom):
    for idx, cords in enumerate(list_linecords):
        #Header
        if idx == 0:
            roi = image[0:cords[0][0] -2 , 0:image.shape[1]]  # region of interest
            imsave("%s_%d_h.jpg" % (pathout, idx), roi)
        if idx == len(list_linecords)-1:
            roi = image[cords[0][1] + 2:image.shape[0], 0:image.shape[1]]  # region of interest
            imsave("%s_%d_f.jpg" % (pathout, idx), roi)
        if cords[2][0] == 'B':
            print "Blank"
            # Add sum extra space to the cords
            roi = image[cords[0][0] + 2:cords[0][1] - 2, cords[1][0]:cords[1][1]]  # region of interest
            imsave("%s_%d_c.jpg" % (pathout, idx),roi)
        if cords[2][0] == 'P':
            if idx == 0:
                print "Plumb-First"
                roi = image[hobj_bottom + 2:cords[0][1] + 15, border:cords[1][0] - 2]  # region of interest
                imsave("%s_%d_a.jpg" % (pathout, idx), roi)
                roi = image[hobj_bottom + 2:cords[0][1] + 15, cords[1][1] + 2:image_param.width - border]
                imsave("%s_%d_b.jpg" % (pathout, idx), roi)
            else:
                print "Plumb"
                roi = image[cords[0][0] - 15:cords[0][1] + 15, border:cords[1][0] - 2]  # region of interest
                imsave("%s_%d_a.jpg" % (pathout, idx), roi)
                roi = image[cords[0][0] - 15:cords[0][1] + 15, cords[1][1] + 2:image_param.width - border]
                imsave("%s_%d_b.jpg" % (pathout, idx), roi)
    return 0

def merge():
    return 0

def plot():
    return 0
####################### MAIN ##################################
def crass():
    ####################### INIT ##################################
    fname = "hoppa-405844417-0050_0158"
    path = "U:\\Eigene Dokumente\\Literatur\\Aufgaben\\Unpaper-Ergebnisse\\"+fname
    pathout = "U:\\Eigene Dokumente\\Literatur\\Aufgaben\\Unpaper-Ergebnisse\\out\\"+fname
    # create outputdir
    if not os.path.isdir("U:\\Eigene Dokumente\\Literatur\\Aufgaben\\Unpaper-Ergebnisse\\out\\"):
        os.mkdir("U:\\Eigene Dokumente\\Literatur\\Aufgaben\\Unpaper-Ergebnisse\\out\\")

    image = imread("%s.jpg" % (path), as_grey=True)
    image_param = Image_Param(image)
    line_param = Line_Param()

    ####################### DESKEW ##################################
    # Deskew the loaded image
    if True == True:
        print "start deskew"
        deskew_rng = 20
        deskew_path = deskew(image, image_param, line_param, deskew_rng,pathout)
        image = imread("%s" % (deskew_path), as_grey=True)
        image_param = Image_Param(image)
        line_param = Line_Param()

    ####################### ANALYSE - LINECORDS ##################################
    if True == True:
        print "start linecord-analyse"
        list_linecords, border, hobj_bottom = linecord_analyse(image, image_param,pathout, line_param)

    ####################### CROP ##################################
    if True == True:
        print "start crop"
        crop(image, image_param, pathout, list_linecords, border, hobj_bottom)

    ####################### SPLICE ##################################
    if True == False:
        print "start splice"
        merge()

    ####################### PLOT ##################################
    if True == False:
        print "start plot"
        Output = np.array(labels != 0, 'B')

        fig, axes = plt.subplots(1, 3, figsize=(150, 50), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Original image')

        ax[1].imshow(binary, cmap=plt.cm.gray)
        ax[1].set_title('Sauvola')

        ax[2].imshow(Output, cmap=plt.cm.gray)
        ax[2].set_title('Finding Blackfoot')

        for a in ax:
            a.axis('off')

        plt.show()

####################### START ##################################
if __name__=="__main__":
    print "start crass"
    crass()
