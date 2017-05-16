###################### INFORMATION ##############################
#          crass-Addin to crop and splice segments of an image
#          optional preprocess: deskew
#Program:  **crass**
#Info:     **Python 2.7**
#Author:   **Jan Kamlah**
#Date:     **12.05.2017**

####################### IMPORT ##################################
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imsave
import skimage.filters.thresholding as th
import skimage.transform as sktransform
from scipy.ndimage import morphology,measurements,filters
import scipy.misc as misc

####################### OBJECTS ###################################
class Line_Param():
    minwidth = 0.3
    maxwidth = 0.95
    minheight = 0.05
    maxheight = 0.95

class Image_Param():
    def __init__(self, image, input):
        self.height, self.width = image.shape
        self.path = os.path.dirname(input)
        self.pathout = os.path.dirname(input)+"\\out\\"
        self.name = os.path.splitext(os.path.basename(input))[0]
        self.extension = os.path.splitext(os.path.basename(input))[1][1:]

class Splice_Param():
    def __init__(self, input, parts):
        self.name = os.path.splitext(input)[0]
        self.segment = parts[len(parts)-2]
        self.segmenttype = parts[len(parts)-1]
        self.extension = os.path.splitext(input)[1][1:]

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
        return d1-int(d1*0.9)
    else:
        return d2-int(d2*0.9)

def find_text_border_dist(binary, cords, border, image_param):
    print cords
    print binary
    binary_left = binary[cords[0][0]:cords[0][1],0:cords[1][0]]
    labels, numl = measurements.label(binary_left)
    objects = measurements.find_objects(labels)
    min_dist = 0
    txtborder = []
    for i, b in enumerate(objects):
        if 10 < b[1].start:
            if min_dist == 0:
                min_dist = b[1].start
            if b[1].start < min_dist:
                min_dist = b[1].start
    txtborder.append(min_dist-10)
    print txtborder[0]
    binary_right = binary[cords[0][0]:cords[0][1], cords[1][1]:image_param.width]
    labels, numl = measurements.label(binary_right)
    objects = measurements.find_objects(labels)
    min_dist = 0
    for i, b in enumerate(objects):
        print b[1].start
        if 10 < b[1].start:
            if min_dist == 0:
                min_dist = b[1].start
            if b[1].start < min_dist:
                min_dist = b[1].start
    txtborder.append(min_dist - 10)
    print txtborder[1]
    return 0

def deskew(image, image_param, line_param, deskew_rng):
    # Deskew
    # Calculate the angle of the points between 20% and 80% of the line
    thresh = th.threshold_sauvola(image, 31)
    binary = image > thresh
    binary = 1-binary #inverse binary
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    deskew_path = "0"
    for i, b in enumerate(objects):
        # The line has to be bigger than minwidth, smaller than maxwidth, stay in the top (30%) of the img,
        # only one obj allowed and the line isnt allowed to start contact the topborder of the image
        if width(b) > line_param.minwidth * image_param.width and width(b) < line_param.maxwidth * image_param.width and \
                        int(image_param.height * 0.04) < b[0].stop < int(image_param.height * 0.3) and b[0].start != 0:

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
            deskew_path = "%s_deskew.%s" % (image_param.pathout+image_param.name, image_param.extension)
            imsave(deskew_path, deskew_image)
            break

    return deskew_path

def linecord_analyse(image, image_param, line_param):
    thresh = th.threshold_sauvola(image, 31)
    binary = image > thresh
    binary = 1-binary #inverse binary
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    count_height = 0
    count_width = 0
    list_linecords = [] # Init list of linecoordinates the format is: [0]: width.start, width.stopt,
    # [1]:height.start, height.stop, [2]: Type of line [B = blank, P = plumb]
    border = 0
    hobj_bottom = 0

    for i, b in enumerate(objects):
        # The line has to be bigger than minwidth, smaller than maxwidth, stay in the top (30%) of the img,
        # only one obj allowed and the line isnt allowed to start contact the topborder of the image
        if line_param.minwidth * image_param.width <  width(b) < line_param.maxwidth * image_param.width \
                and int(image_param.height * 0.04) <  b[0].stop < int(image_param.height * 0.3) and count_width == 0 and b[0].start != 0:

            # Distance Calculation
            border = mindist(b, image_param.width)
            hobj_bottom = b[0].stop + 5  # Lowest Point of object + 5 Pixel
            roi = image[hobj_bottom:image_param.height, border:image_param.width - border]  # region of interest
            imsave("%s_crop.%s" % (image_param.pathout+image_param.name, image_param.extension), roi)

            # Get coordinats of the line
            labels[b][labels[b] == i + 1] = 0
            count_width += 1

        if  line_param.minheight * image_param.height < height(b) < line_param.maxheight * image_param.height \
                and width(b) < 50 and int(image_param.width*0.35) < (b[1].start+b[1].stop)/2 < int(image_param.width*0.75):
            cords = linecords(b)
            if count_height == 0:
                if b[0].start - hobj_bottom > 50:
                    list_linecords.append(np.array([[hobj_bottom, b[0].start], [border, image_param.width - border], ['B']]))
                    count_height += 1
                txtborder = find_text_border_dist(binary, cords, border,image_param)
                list_linecords.append([cords[0], cords[1], ['P'], [txtborder]])
            elif count_height != 0:
                if b[0].start - list_linecords[count_height - 1][0][1] > 50:
                    list_linecords.append(np.array(
                        [[list_linecords[count_height - 1][0][1], b[0].start], [border, image_param.width - border], ['B']]))
                    count_height += 1
                    txtborder = find_text_border_dist(binary, cords, border,image_param)
                    list_linecords.append([cords[0], cords[1], ['P'],[txtborder]])
                elif b[0].start - list_linecords[count_height - 1][0][1] < 20:
                    list_linecords[count_height - 1][0][1] = b[0].stop
                    count_height -= 1
                else:
                    txtborder = find_text_border_dist(binary, cords, border,image_param)
                    list_linecords.append([cords[0], cords[1], ['P'],[txtborder]])
            count_height += 1
            labels[b][labels[b] == i + 1] = 0
    return list_linecords, border, hobj_bottom

#def save_roi(widthstart. widthstop, heightstart, heightstop):
    #return 0


def crop(image, image_param, list_linecords, border, hobj_bottom):
    fpath = image_param.pathout+image_param.name
    for idx, cords in enumerate(list_linecords):
        #Header
        if idx == 0:
            "header"
            roi = image[0:cords[0][0] -2 , 0:image.shape[1]]  # region of interest
            imsave("%s_%d_h.%s" % (fpath, idx,image_param.extension), roi)
        if idx == len(list_linecords)-1:
            "footer"
            roi = image[cords[0][1] + 2:image.shape[0], 0:image.shape[1]]  # region of interest
            imsave("%s_%d_f.%s" % (fpath, idx,image_param.extension), roi)
        if cords[2][0] == 'B':
            print "blank"
            # Add sum extra space to the cords
            roi = image[cords[0][0] + 2:cords[0][1] - 2, cords[1][0]:cords[1][1]]  # region of interest
            imsave("%s_%d_c.%s" % (fpath, idx,image_param.extension),roi)
        if cords[2][0] == 'P':
            if idx == 0:
                print "plumb-first"
                roi = image[hobj_bottom + 2:cords[0][1] + 15, border:cords[1][0] - 2]  # region of interest
                imsave("%s_%d_a.%s" % (fpath, idx,image_param.extension), roi)
                roi = image[hobj_bottom + 2:cords[0][1] + 15, cords[1][1] + 2:image_param.width - border]
                imsave("%s_%d_b.%s" % (fpath, idx,image_param.extension), roi)
            else:
                print "plumb"
                roi = image[cords[0][0] - 15:cords[0][1] + 15, border:cords[1][0] - 2]  # region of interest
                imsave("%s_%d_a.%s" % (fpath, idx,image_param.extension), roi)
                roi = image[cords[0][0] - 15:cords[0][1] + 15, cords[1][1] + 2:image_param.width - border]
                imsave("%s_%d_b.%s" % (fpath, idx,image_param.extension), roi)
    return 0

def splice(input, extension):
    os.chdir(input)
    output = input+"\\spliced1\\"
    # create outputdir
    list_splice = []
    if not os.path.isdir(output):
        os.mkdir(output)
    for image in sorted(glob.glob("*.%s" % (extension))):
        Sep = ["a", "b", "c"]
        if os.path.splitext(image)[0].split("_")[len(os.path.splitext(image)[0].split("_"))-1] in Sep:
            splice_param = Splice_Param(input, os.path.splitext(image)[0].split("_"))
            if splice_param.segmenttype != 'c':
                list_splice.append(image)
            else:
                print "splice %s" % (image)
                list_splice.append(image)
                segments = [misc.imread(img,mode='RGB') for img in list_splice]
                img_height = sum(segment.shape[0] for segment in segments)
                img_width = max(segment.shape[1] for segment in segments)
                spliced_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                y = 0
                for segment in segments:
                    h, w, d = segment.shape
                    spliced_image[y:y + h, 0:w] = segment
                    y += h
                imsave("%s" % (output+"spliced"+image),spliced_image)
                list_splice = []
    return 0

def plot():
    return 0
####################### MAIN ##################################
def crass():
    ####################### INIT ##################################
    input = "U:\\Eigene Dokumente\\Literatur\\Aufgaben\\crass\\Test1957\\jpg\\hoppa-405844417-0050_0007.jpg"
    if not os.path.isfile(input):
        os.chdir(input)
        extension = "jpg"
        inputs = []
        for input in sorted(glob.glob("*.%s" % (extension))):
            inputs.append(os.getcwd()+"\\"+input)
    else:
        inputs =  []
        inputs.append(input)
    for input in inputs:
        print input
        # read image
        image = imread("%s" % (input), as_grey=True)
        image_param = Image_Param(image,input)
        line_param = Line_Param()

        # create outputdir
        if not os.path.isdir(image_param.pathout):
            os.mkdir(image_param.pathout)

        ####################### DESKEW ##################################
        # Deskew the loaded image
        if True == True:
            print "start deskew"
            #Only values between 0-49 valid
            deskew_rng = 20
            deskew_path = deskew(image, image_param, line_param, deskew_rng)
            image = imread("%s" % (deskew_path), as_grey=True)
            image_param = Image_Param(image, input)
            line_param = Line_Param()

        ####################### ANALYSE - LINECORDS ##################################
        print "start linecord-analyse"
        list_linecords, border, hobj_bottom = linecord_analyse(image, image_param, line_param)
        ####################### CROP ##################################
        if True == True:
            print "start crop"
            crop(image, image_param, list_linecords, border, hobj_bottom)
            input = image_param.pathout
            extension = image_param.extension

    ####################### SPLICE ##################################
    if True == True:
        print "start splice"
        splice(input,extension)

    ####################### PLOT ##################################
    if True == False:
        print "start plot"
        Output = 0 #np.array(labels != 0, 'B')

        fig, axes = plt.subplots(1, 3, figsize=(150, 50), sharex='all', sharey='all')
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
