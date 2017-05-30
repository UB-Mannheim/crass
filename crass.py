###################### INFORMATION ##############################
#          crass-Addin to crop and splice segments of an image
#          optional preprocess: deskew
#Program:  **crass**
#Info:     **Python 2.7**
#Author:   **Jan Kamlah**
#Date:     **22.05.2017**

####################### IMPORT ##################################
import argparse
import copy
import glob
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from scipy.ndimage import measurements
import scipy.misc as misc
import skimage.color as color
import skimage.filters.thresholding as th
from skimage.io import imread
from skimage.io import imsave
import skimage.morphology as morph
import skimage.transform as transform
import warnings

####################### CMD-PARSER-SETTINGS ########################
def get_parser():
    parser = argparse.ArgumentParser(description="Crop And Splice Segements (CRASS) of an image based on blacklines ")

    # Erease -- on input and extension
    #parser.add_argument("--input", type=str, default="C:\\Coding\\jpg\\hoppa-405844417-0050_0008.jpg",
    #                   help='Input file or folder')
    #parser.add_argument("--input", type=str, default="U:\\Eigene Dokumente\\Literatur\\Aufgaben\\crass\\1967\\jpg\\hoppa-405844417-0060_0805.jpg",
    #                    help='Input file or folder')
    parser.add_argument("--input", type=str,
                        default="U:\\Eigene Dokumente\\Literatur\\Aufgaben\\crass\\1957\\jpg\\",
                        help='Input file or folder')
    parser.add_argument("--extension", type=str, choices=["jpg"], default="jpg", help='Extension of the files, default: %(default)s')

    parser.add_argument('--crop', action="store_false", help='cropping paper into segments')
    parser.add_argument('--deskew', action="store_false", help='preprocessing: deskewing the paper')
    parser.add_argument('--minwidthpl', type=float, default=0.3, help='minwidth of the plumb lines, default: %(default)s')
    parser.add_argument('--maxwidthpl', type=float, default=0.95, help='maxwidth of the plumb lines, default: %(default)s')
    parser.add_argument('--minheightpl', type=float, default=0.00,help='minheight of the plumb lines, default: %(default)s')
    parser.add_argument('--maxheightpl', type=float, default=0.03,help='maxheight of the plumb lines, default: %(default)s')
    parser.add_argument('--minheightplmask', type=float, default=0.04, help='minheight of the plumb lines mask (search area), default: %(default)s')
    parser.add_argument('--maxheightplmask', type=float, default=0.3, help='maxheight of the plumb lines mask (search area), default: %(default)s')
    parser.add_argument('--minheightver', type=float, default=0.05, help='minheight of the vertical lines, default: %(default)s')
    parser.add_argument('--maxheightver', type=float, default=0.95, help='maxheightof the vertical lines, default: %(default)s')
    parser.add_argument('--minwidthver', type=float, default=0.00, help='minwidth of the vertical lines, default: %(default)s')
    parser.add_argument('--maxwidthver', type=float, default=0.022, help='maxwidth of the vertical lines, default: %(default)s')
    parser.add_argument('--minwidthvermask', type=float, default=0.35, help='minwidth of the vertical lines mask (search area), default: %(default)s')
    parser.add_argument('--maxwidthvermask', type=float, default=0.75, help='maxwidth of the vertical lines mask (search area), default: %(default)s')
    parser.add_argument('--parallel', type=int, default=3, help="number of CPUs to use, default: %(default)s")
    parser.add_argument('--plot',  action="store_false", help='plotting some steps in the end')
    parser.add_argument('--ramp', default=None, help='activates the function whiteout')
    parser.add_argument('--showmasks', action="store_false", help='output an image with colored masks')
    parser.add_argument('--splice', action="store_false", help='splice the cropped segments')
    parser.add_argument("--splicetypes", type=str, nargs='+', choices=['a', 'b', 'c', 'f', 'h'],
                        default=['a', 'b', 'c'],
                        help='Segmenttypes to be spliced, default: %(default)s')
    parser.add_argument("--splicemaintype", type=str, choices=['a', 'b', 'c', 'f', 'h'], default='c',
                        help='Segmenttype that indicates a new splice process, default: %(default)s')
    #Change to "store_false" for release
    parser.add_argument('--splicemaintypestart', action="store_true",
                        help='The maintype of splicetyps will be placed on the end')
    parser.add_argument('--threshwindow', type=float, default=31, help='Size of the window (binarization): %(default)s')
    parser.add_argument('--threshweight', type=float, default=0.2, help='Weight the effect of the standard deviation (binarization): %(default)s')
    parser.add_argument('-A', '--addstartheightab', type=int, default=20, help='Add some pixel for the clipping mask of segments a&b (startheight), default: %(default)s')
    parser.add_argument('-a', '--addstopheightab', type=int, default=35, help='Add some pixel for the clipping mask of segments a&b (stopheight), default: %(default)s')
    parser.add_argument('-C', '--addstartheightc', type=int, default=0,
                        help='Add some pixel for the clipping mask of segment c (startheight), default: %(default)s')
    parser.add_argument('-c', '--addstopheightc', type=int, default=0,
                        help='Add some pixel for the clipping mask of segment c (stopheight), default: %(default)s')

    parser.add_argument('-q', '--quiet', action='store_true', help='be less verbose, default: %(default)s')

    args = parser.parse_args()
    args.showmasks = True
    return args

####################### LOGGER-FILE-SETTINGS ########################
logging.basicConfig(filename=os.path.dirname(get_parser().input) + '\\Logfile_crass.log', level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

####################### CLASSES & METHODS ###########################
class Clippingmask():
    def __init__(self, image):
        self.height_start, self.width_start = 0, 0
        self.height_stop, self.width_stop = image.shape
        self.user = None

class Image_Param():
    def __init__(self, image, input):
        self.height, self.width = image.shape
        self.path = os.path.dirname(input)
        self.pathout = os.path.dirname(input)+"\\out\\"
        self.deskewpath = None
        self.name = os.path.splitext(os.path.basename(input))[0]

class Linecoords():
    def __init__(self, binary, value ,object):
        self.height_start = object[0].start
        self.height_stop = object[0].stop
        self.width_start = object[1].start
        self.width_stop = object[1].stop
        self.object = object
        self.object_value = value
        self.object_matrix = copy.deepcopy(binary[object])
        self.segmenttype = None

class Splice_Param():
    def __init__(self, input, parts):
        self.name = os.path.splitext(input)[0]
        self.segment = parts[len(parts)-2]
        self.segmenttype = parts[len(parts)-1]

####################### FUNCTIONS ##################################
def crop(args, image, image_param, list_linecoords, clippingmask):
    filepath = image_param.pathout + image_param.name
    if args.showmasks == True:
        debugimage = color.gray2rgb(copy.deepcopy(image))
    for idx, linecoords in enumerate(list_linecoords):
        # Header
        if idx == 0:
            if not args.quiet: print "header"
            roi = image[0:linecoords.height_start - 2, 0:image_param.width]  # region of interest
            imsave("%s_%d_h.%s" % (filepath, idx, args.extension), roi)
            if args.showmasks == True:
                set_colored_mask(debugimage, [[0, linecoords.height_start - 2],
                                              [0, image_param.width]], 0, 100)

        # Crop middle segments
        if linecoords.segmenttype == 'B':
            if not args.quiet: print "blank"
            # Add sum extra space to the cords
            roi = image[linecoords.height_start + 2 - args.addstartheightc:linecoords.height_stop - 2 +args.addstopheightc,
                  linecoords.width_start:linecoords.width_stop]  # region of interest
            imsave("%s_%d_c.%s" % (filepath, idx+1, args.extension), roi)
            if args.showmasks == True:
                set_colored_mask(debugimage, [[linecoords.height_start + 2- args.addstartheightc, linecoords.height_stop - 2 +args.addstopheightc],
                                              [linecoords.width_start, linecoords.width_stop]], 1, 220)
        if linecoords.segmenttype == 'P':
            if idx == 0:
                print "plumb-first"
                linecoords.height_start = clippingmask.height_start + 17
                if not args.quiet: print "plumb"
            roi = image[
                  linecoords.height_start - args.addstartheightab:linecoords.height_stop + args.addstopheightab,
                  clippingmask.width_start:linecoords.width_stop - 2]  # region of interest
            imsave("%s_%d_a.%s" % (filepath, idx+1, args.extension), roi)
            if args.showmasks == True:
                set_colored_mask(debugimage, [[linecoords.height_start - args.addstartheightab,
                                               linecoords.height_stop + args.addstopheightab],
                                              [clippingmask.width_start, linecoords.width_stop - 2]], 2, 180)
            roi = image[linecoords.height_start - args.addstartheightab:linecoords.height_stop + args.addstopheightab,
                  linecoords.width_start + 1:clippingmask.width_stop]
            imsave("%s_%d_b.%s" % (filepath, idx+1, args.extension), roi)
            if args.showmasks == True:
                set_colored_mask(debugimage, [[linecoords.height_start - args.addstartheightab,
                                               linecoords.height_stop + args.addstopheightab],
                                              [linecoords.width_start + 1, clippingmask.width_stop]], 0, 180)

        # Footer
        if idx == len(list_linecoords) - 1:
            if not args.quiet: print "footer"
            roi = image[linecoords.height_stop + 2:image_param.height,
                  0:image_param.width]  # region of interest
            imsave("%s_%d_f.%s" % (filepath, idx+2, args.extension), roi)
            if args.showmasks == True:
                set_colored_mask(debugimage,
                                 [[linecoords.height_stop + 2, image_param.height], [0, image_param.width]], 1,
                                 100)
    if args.showmasks == True:
        imsave("%s_masks.%s" % (filepath, args.extension), debugimage)
    return 0

def cropping(input):
    # read image
    print input
    args = get_parser()
    try:
        image = imread("%s" % input, as_grey=True)
    except IOError:
        print("cannot open %s" % input)
        logging.warning("cannot open %s" % input)
        return 1

    image_param = Image_Param(image, input)

    # create outputdir
    if not os.path.isdir(image_param.pathout):
        try:
            os.mkdir(image_param.pathout)
        except IOError:
            print("cannot create out directoy")

    ####################### DESKEW ####################################
    # Deskew the loaded image
    if args.deskew == True:
        if not args.quiet: print "start deskew"
        # Only values between 0-49 valid
        # deskew_linesize
        deskew_linesize = 20
        image_param.deskewpath = deskew(args, image, image_param, deskew_linesize)
        # image = misc.imread("%s" % (image_param.deskewpath),mode='RGB')
        try:
            image = imread("%s" % (image_param.deskewpath), as_grey=True)
        except IOError:
            print("cannot open %s" % input)
            logging.warning("cannot open %s" % input)
            return 1
        image_param = Image_Param(image, input)

        ####################### ANALYSE - LINECOORDS #######################
        if not args.quiet: print "start linecoord-analyse"
    clippingmask = Clippingmask(image)
    list_linecoords, border, topline_width_stop = linecoords_analyse(args, image, image_param, clippingmask)
    ####################### CROP #######################################
    if args.crop == True:
        if not args.quiet: print "start crop"
        crop(args, image, image_param, list_linecoords, clippingmask)
    return 0

def deskew(args,image, image_param, deskew_linesize):
    # Deskew
    # Calculate the angle of the points between 20% and 80% of the line
    thresh = th.threshold_sauvola(image, args.threshwindow, args.threshweight)
    binary = image > thresh
    binary = 1-binary #inverse binary
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    deskew_path = "0"
    for i, b in enumerate(objects):
        linecoords = Linecoords(image, i, b)
        # The line has to be bigger than minwidth, smaller than maxwidth, stay in the top (30%) of the img,
        # only one obj allowed and the line isnt allowed to start contact the topborder of the image
        if int(args.minwidthpl * image_param.width) < get_width(b) < int(args.maxwidthpl * image_param.width) \
                and int(image_param.height * args.minheightpl) < get_height(b) < int(image_param.height * args.maxheightpl) \
                and int(image_param.height * args.minheightplmask) < (linecoords.height_start+linecoords.height_stop)/2 < int(image_param.height * args.maxheightplmask) and linecoords.height_start != 0:

            obj_height, ob_jwidth = binary[b].shape
            obj_width_prc = ob_jwidth / 100
            arr = np.arange(1, (obj_width_prc * (100 - deskew_linesize * 2)) + 1)
            mean_y = []
            #Calculate the mean value for every y-array
            for idx in range(obj_width_prc * (100 - deskew_linesize * 2)):
                value_y = measurements.find_objects(labels[b][:, idx + (obj_width_prc * deskew_linesize)] == i + 1)[0]
                mean_y.append((value_y[0].stop + value_y[0].start) / 2)
            polyfit_value = np.polyfit(arr, mean_y, 1)
            deskewangle = np.arctan(polyfit_value[0]) * (360 / (2 * np.pi))
            args.ramp = True
            deskew_image = transform.rotate(image, deskewangle)
            deskew_path = "%s_deskew.%s" % (image_param.pathout+image_param.name, args.extension)
            with warnings.catch_warnings():
                #Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                imsave(deskew_path, deskew_image)
            break

    return deskew_path

def get_inputfiles(args):
    input = args.input
    if not os.path.isfile(input):
        os.chdir(input)
        inputfiles = []
        for input in sorted(glob.glob("*.%s" % (args.extension))):
            inputfiles.append(os.getcwd() + "\\" + input)
    else:
        inputfiles = []
        inputfiles.append(input)
    return inputfiles

def get_height(s):
    return s[0].stop-s[0].start

def get_linecoords(s):
    return [[s[0].start,s[0].stop],[s[1].start,s[1].stop]]

def get_mindist(s,length):
    d1 = s[1].start
    d2 = length - s[1].stop
    if d1 < d2:
        return d1-int(d1*0.2)
    else:
        return d2-int(d2*0.2)

def get_width(s):
    return s[1].stop-s[1].start

def linecoords_analyse(args,image, image_param, clippingmask):
    thresh = th.threshold_sauvola(image, args.threshwindow, args.threshweight)
    binary = image > thresh
    binary = 1-binary #inverse binary
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    count_height = 0
    count_width = 0
    list_linecoords = [] # Init list of linecoordinates the format is: [0]: width.start, width.stopt,
    # [1]:height.start, height.stop, [2]: Type of line [B = blank, P = plumb]

    for i, b in enumerate(objects):
        # The line has to be bigger than minwidth, smaller than maxwidth, stay in the top (30%) of the img,
        # only one obj allowed and the line isnt allowed to start contact the topborder of the image
        linecoords = Linecoords(labels, i, b)
        if args.minwidthpl * image_param.width <  get_width(b) < args.maxwidthpl * image_param.width \
                and image_param.height*args.minheightpl < get_height(b) < image_param.height*args.maxheightpl \
                and int(image_param.height * args.minheightplmask) <  linecoords.height_stop < int(image_param.height * args.maxheightplmask) and count_width == 0 and linecoords.height_start != 0:

            # Distance Calculation - defining the clippingmask
            border = get_mindist(b, image_param.width)
            topline_width_stop = b[0].stop + 5  # Lowest Point of object + 5 Pixel
            if clippingmask.user == None:
                clippingmask.width_start = linecoords.width_start - border
                if clippingmask.width_start > int(image_param.width * 0.05):
                    clippingmask.width_start = int(image_param.width * 0.05)
                clippingmask.width_stop = linecoords.width_stop + border
                if clippingmask.width_stop < int(image_param.width*0.95):
                    clippingmask.width_stop = int(image_param.width*0.95)
                clippingmask.height_start = topline_width_stop
                clippingmask.height_stop = 0

            # Test for cropping the area under the first vertical line
            #roi = image[hobj_bottom:image_param.height, border:image_param.width - border]  # region of interest
            #imsave("%s_crop.%s" % (image_param.pathout+image_param.name, args.extension), roi)

            # Get coordinats of the line
            labels[b][labels[b] == i + 1] = 0
            count_width += 1

        if args.minheightver * image_param.height < get_height(b) < args.maxheightver * image_param.height \
                and image_param.width*args.minwidthver < get_width(b) < image_param.width*args.maxwidthver \
                and int(image_param.width*args.minwidthvermask) < (linecoords.width_start+linecoords.width_stop)/2 < int(image_param.width*args.maxwidthvermask):
            linecoords.segmenttype = 'P' # Defaultvalue for segmenttype 'P' for plumb lines
            if count_height == 0:
                if b[0].start - topline_width_stop > 50:
                    blankline = Linecoords(labels,i,b)
                    blankline.segmenttype = 'B'
                    blankline.height_start = topline_width_stop
                    blankline.height_stop = linecoords.height_start
                    blankline.width_start = border
                    blankline.width_stop = image_param.width - border
                    list_linecoords.append(copy.deepcopy(blankline))
                    count_height += 1
                list_linecoords.append(copy.deepcopy(linecoords))
                if args.ramp != None:
                    whiteout_ramp(image, linecoords)

            elif count_height != 0:
                if b[0].start - list_linecoords[count_height - 1].height_stop > 50:
                    blankline = Linecoords(labels,i,b)
                    blankline.segmenttype = 'B'
                    blankline.height_start = list_linecoords[count_height - 1].height_stop
                    blankline.height_stop = linecoords.height_start
                    blankline.width_start = border
                    blankline.width_stop = image_param.width - border
                    list_linecoords.append(copy.deepcopy(blankline))
                    count_height += 1
                    list_linecoords.append(copy.deepcopy(linecoords))
                    if args.ramp != None:
                        whiteout_ramp(image, linecoords)
                elif b[0].start - list_linecoords[count_height - 1].height_stop < 35:
                    if args.ramp != None:
                        whiteout_ramp(image, linecoords)
                    list_linecoords[count_height - 1].height_stop = b[0].stop
                    count_height -= 1
                else:
                    list_linecoords.append(copy.deepcopy(linecoords))
                    if args.ramp != None:
                        whiteout_ramp(image, linecoords)
            count_height += 1
            labels[b][labels[b] == i + 1] = 0
    #imsave("%s_EDIT%d.%s" % (image_param.pathout, linecoords.object_value, args.extension), image)
    return list_linecoords, border, topline_width_stop

def plot(image, binary, Output):
    if not args.quiet: print "start plot"

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
    return 0

def set_colored_mask(image, borders, color, intensity):
    image[borders[0][0]:borders[0][0]+5,borders[1][0]:borders[1][1]] = 0
    image[borders[0][1]-6:borders[0][1]-1, borders[1][0]:borders[1][1]] = 0
    image[borders[0][0]:borders[0][1], borders[1][0]:borders[1][0]+5] = 0
    image[borders[0][0]:borders[0][1], borders[1][1]-6:borders[1][1]-1] = 0
    masked_image = np.ma.greater(image[borders[0][0]:borders[0][1], borders[1][0]:borders[1][1], color],55)
    image[borders[0][0]:borders[0][1],borders[1][0]:borders[1][1],color] += intensity
    image[borders[0][0]:borders[0][1], borders[1][0]:borders[1][1], color] = image[borders[0][0]:borders[0][1],borders[1][0]:borders[1][1],color] *masked_image
    return 0

def splice(args,input):
    os.chdir(input)
    output = input+"\\spliced\\"
    list_splice = []
    if not os.path.isdir(output):
        os.mkdir(output)
    for image in sorted(glob.glob("*.%s" % (args.extension))):
        if os.path.splitext(image)[0].split("_")[len(os.path.splitext(image)[0].split("_"))-1] in args.splicetypes:
            splice_param = Splice_Param(input, os.path.splitext(image)[0].split("_"))
            if splice_param.segmenttype != args.splicemaintype:
                list_splice.append(image)
            else:
                if not args.quiet: print "splice %s" % (image)
                if not args.splicemaintypestart:
                    list_splice.append(image)
                segments = [misc.imread(img,mode='RGB') for img in list_splice]
                img_height = sum(segment.shape[0] for segment in segments)
                img_width = max(segment.shape[1] for segment in segments)
                spliced_image = np.ones((img_height, img_width, 3), dtype=np.uint8)
                y = 0
                for segment in segments:
                    h, w, d = segment.shape
                    spliced_image[y:y + h, 0:w] = segment
                    y += h
                imsave("%s" % (output+"spliced"+image),spliced_image)
                list_splice = []
                if args.splicemaintypestart:
                    list_splice.append(image)
    return 0

def whiteout_ramp(image, linecoords):
    #for idx in range(linecoords.width_stop):
    imagesection = image[linecoords.object]
    count = 0
    # Dilation enlarge the bright segements and cut them out off the original image
    for i in morph.dilation(linecoords.object_matrix, morph.square(10)):
        whitevalue = measurements.find_objects(i == linecoords.object_value + 1)[0][0]
        imagesection[count,whitevalue.start:whitevalue.stop] = 255
        count +=1
    #imsave("%s\\whitelines\\%s.%s" %(image_param.path,linecoords.object_value,args.extension), imagesection)
    return 0

####################### MAIN-FUNCTIONS ############################################
def crass():
    args = get_parser()
    # Read inputfiles
    inputfiles = get_inputfiles(args)
    ####################### CRASS #######################################
    ####################### CROP  #######################################
    # Start crass with serialprocessing
    if args.parallel < 2:
        for input in inputfiles:
            cropping(input)
    # Start crass with multiprocessing
    else:
        pool = multiprocessing.Pool(processes=args.parallel)
        pool.map(cropping, inputfiles)
    ####################### SPLICE #######################################
    if args.splice == True:
        if not args.splicemaintype in args.splicetypes:
                print("%s is not part of the pattern %s" % (args.splicemaintype,args.splicetypes))
                logging.warning("Input error by user!")
        else:
            if not args.quiet: print "start splice"
            splice(args, os.path.dirname(args.input) + "//out//")

    ####################### TEST-PLOT ##################################
    if args.plot == None:
        plot(image, binary, Output)

####################### MAIN ############################################
if __name__=="__main__":
    crass()
