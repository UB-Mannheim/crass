###################### INFORMATION ##############################
#          crass-Addin to crop and splice segments of an image
#          optional preprocess: deskew
#Program:  **crass**
#Info:     **Python 2.7**
#Author:   **Jan Kamlah**
#Date:     **13.06.2017**

####################### IMPORT ##################################
import argparse
import copy
import glob
import logging
import multiprocessing
import numpy as np
import os
from scipy.ndimage import measurements
import skimage as skimage
import scipy.misc as misc
import skimage.color as color
import skimage.filters.thresholding as th
from skimage.io import imread, imsave, find_available_plugins
import skimage.morphology as morph
import skimage.transform as transform
import warnings

####################### CMD-PARSER-SETTINGS ########################
def get_parser():
    parser = argparse.ArgumentParser(description="Crop And Splice Segments (CRASS) of an image based on black (separator-)lines")
    #parser.add_argument("--config", action=LoadConfigAction, default=None)
    # Erease -- on input and extension
    #parser.add_argument("input", type=str,help='Input file or folder')
    #parser.add_argument("input", type=str, default="C:\\Users\\jkamlah\\Desktop\\crassWeil\\0279.jpg",
    #                    help='Input file or folder')
    parser.add_argument("--input", type=str,default="U:\\Eigene Dokumente\\Literatur\\Aufgaben\\crass\\tif\\hoppa-405844417-0050_0034.tif",
                        help='Input file or folder')
    parser.add_argument("--extension", type=str, choices=["bmp","jpg","png","tif"], default="tif", help='Extension of the files, default: %(default)s')

    parser.add_argument('-A', '--addstartheightab', type=float, default=0.01, choices=np.arange(-1.0, 1.0), help='Add some pixel for the clipping mask of segments a&b (startheight), default: %(default)s')
    parser.add_argument('-a', '--addstopheightab', type=float, default=0.011, choices=np.arange(-1.0, 1.0),help='Add some pixel for the clipping mask of segments a&b (stopheight), default: %(default)s')
    parser.add_argument('-C', '--addstartheightc', type=float, default=-0.005, choices=np.arange(-1.0, 1.0),help='Add some pixel for the clipping mask of segment c (startheight), default: %(default)s')
    parser.add_argument('-c', '--addstopheightc', type=float, default=0.0, choices=np.arange(-1.0, 1.0),help='Add some pixel for the clipping mask of segment c (stopheight), default: %(default)s')
    parser.add_argument('--bgcolor', type=int, default=0,help='Backgroundcolor of the splice image (for "uint8": 0=black,...255=white): %(default)s')
    parser.add_argument('--crop', action="store_false", help='cropping paper into segments')
    parser.add_argument("--croptypes", type=str, nargs='+', choices=['a', 'b', 'c', 'f', 'h'],
                        default=['a', 'b', 'c', 'f', 'h'],
                        help='Types to be cropped out, default: %(default)s')
    parser.add_argument('--deskew', action="store_false", help='preprocessing: deskewing the paper')
    parser.add_argument('--deskewlinesize', type=float, default=0.8, choices=np.arange(0.1, 1.0),
                        help='Percantage of the horizontal line to compute the deskewangle: %(default)s')
    parser.add_argument("--horlinepos", type=int, choices=[0, 1, 2, 3], default=0,
                        help='Position of the horizontal line(0:top, 1:right,2:bottom,3:left), default: %(default)s')
    parser.add_argument("--horlinetype", type=int, choices=[0, 1], default=0,
                        help='Type of the horizontal line (0:header, 1:footer), default: %(default)s')
    parser.add_argument("--imgmask", type=float, nargs=4, default=[0.0,1.0,0.0,1.0], help='Set a mask that only a specific part of the image will be computed, arguments =  Heightstart, Heightend, Widthstart, Widthend')
    parser.add_argument('--minwidthmask', type=float, default=0.06, choices=np.arange(0, 0.5),
                        help='min widthdistance of all masks, default: %(default)s')
    parser.add_argument('--minwidthhor', type=float, default=0.3, choices=np.arange(0, 1.0), help='minwidth of the horizontal lines, default: %(default)s')
    parser.add_argument('--maxwidthhor', type=float, default=0.95,choices=np.arange(-1.0, 1.0), help='maxwidth of the horizontal lines, default: %(default)s')
    parser.add_argument('--minheighthor', type=float, default=0.00, choices=np.arange(0, 1.0), help='minheight of the horizontal lines, default: %(default)s')
    parser.add_argument('--maxheighthor', type=float, default=0.03, choices=np.arange(0, 1.0), help='maxheight of the horizontal lines, default: %(default)s')
    parser.add_argument('--minheighthormask', type=float, default=0.04, choices=np.arange(0, 1.0), help='minheight of the horizontal lines mask (search area), default: %(default)s')
    parser.add_argument('--maxheighthormask', type=float, default=0.3, choices=np.arange(0, 1.0), help='maxheight of the horizontal lines mask (search area), default: %(default)s')
    parser.add_argument('--minheightver', type=float, default=0.0375, choices=np.arange(0, 1.0), help='minheight of the vertical lines, default: %(default)s')  # Value of 0.035 is tested (before 0.05)
    parser.add_argument('--maxheightver', type=float, default=0.95, choices=np.arange(0, 1.0), help='maxheightof the vertical lines, default: %(default)s')
    parser.add_argument('--minwidthver', type=float, default=0.00, choices=np.arange(0, 1.0), help='minwidth of the vertical lines, default: %(default)s')
    parser.add_argument('--maxwidthver', type=float, default=0.022, choices=np.arange(0, 1.0), help='maxwidth of the vertical lines, default: %(default)s')
    parser.add_argument('--minwidthvermask', type=float, default=0.35, choices=np.arange(0, 1.0), help='minwidth of the vertical lines mask (search area), default: %(default)s')
    parser.add_argument('--maxwidthvermask', type=float, default=0.75, choices=np.arange(0, 1.0), help='maxwidth of the vertical lines mask (search area), default: %(default)s')
    parser.add_argument('--maxgradientver', type=float, default=0.05, choices=np.arange(0, 1.0), help='max gradient of the vertical lines: %(default)s')
    parser.add_argument('--minsizeblank', type=float, default=0.016, choices=np.arange(0, 1.0), help='min size of the blank area between to vertical lines, default: %(default)s')
    parser.add_argument('--minsizeblankobolustop', type=float, default=0.014, choices=np.arange(0, 1.0),help='min size of the blank area between to vertical lines, default: %(default)s')
    parser.add_argument('--parallel', type=int, default=1, help="number of CPUs to use, default: %(default)s")
    parser.add_argument('--ramp', default=None, help='activates the function whiteout')
    parser.add_argument('--adaptingmasksoff', action="store_true", help='deactivates adapting maskalgorithm')
    parser.add_argument('--showmasks', action="store_false", help='output an image with colored masks')
    parser.add_argument('--specialnomoff', action="store_false", help='Disable the special nomenclature for the AKF-Project!')
    parser.add_argument('--splice', action="store_false", help='splice the cropped segments')
    parser.add_argument("--splicetypes", type=str, nargs='+', choices=['a', 'b', 'c', 'f', 'h'],
                        default=['a', 'b', 'c'],
                        help='Segmenttypes to be spliced, default: %(default)s')
    parser.add_argument("--splicemaintype", type=str, choices=['a', 'b', 'c', 'f', 'h'], default='c',
                        help='Segmenttype that indicates a new splice process, default: %(default)s')

    parser.add_argument('--splicemaintypestop', action="store_true",
                        help='The maintype of splicetyps will be placed on the end')
    parser.add_argument('--threshwindow', type=int, default=31, help='Size of the window (binarization): %(default)s')
    parser.add_argument('--threshweight', type=float, default=0.2, choices=np.arange(0, 1.0), help='Weight the effect of the standard deviation (binarization): %(default)s')
    parser.add_argument('-q', '--quiet', action='store_true', help='be less verbose, default: %(default)s')

    args = parser.parse_args()
    return args

####################### LOGGER-FILE-SETTINGS ########################
logging.basicConfig(filename=os.path.dirname(get_parser().input) + '\\Logfile_crass.log', level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

####################### CLASSES & METHODS ###########################
class Clippingmask():
    def __init__(self, image):
        self.height_start, self.width_start = 0, 0
        if len(image.shape) > 2:
            self.height_stop, self.width_stop, self.rgb = image.shape
        else:
            self.height_stop, self.width_stop = image.shape
        self.user = None

class ImageParam():
    def __init__(self, image, input):
        if len(image.shape) > 2:
            self.height, self.width, self.rgb = image.shape
        else:
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
        self.middle = None
        self.object = object
        self.object_value = value
        self.object_matrix = copy.deepcopy(binary[object])
        self.segmenttype = None

class SpliceParam():
    def __init__(self, input, parts):
        self.name = os.path.splitext(input)[0]
        self.segment = parts[len(parts)-2]
        self.segmenttype = parts[len(parts)-1]

####################### FUNCTIONS ##################################
def create_dir(newdir):
    if not os.path.isdir(newdir):
        try:
            os.mkdir(newdir)
        except IOError:
            print("cannot create %s directoy" % newdir)

def crop(args, image, image_param, list_linecoords, clippingmask):
    create_dir(image_param.pathout+"\\segments\\")
    filepath = image_param.pathout+"\\segments\\"+image_param.name
    create_dir(image_param.pathout+"\\coords\\")
    coordstxt = open(image_param.pathout+"\\coords\\"+image_param.name+"_coords.txt", "w")
    coordstxt.write("Image resolution:\t%d\t%d\n" % (image_param.height, image_param.width))
    pixelheight = set_pixelground(image_param.height)
    image = np.rot90(image, args.horlinepos)
    if args.showmasks == True:
        debugimage = color.gray2rgb(copy.deepcopy(image))
    for idx, linecoords in enumerate(list_linecoords):
        # Header
        if idx == 0:
            if not args.quiet: print "header"
            roi = image[0:linecoords.height_start - 2, 0:image_param.width]  # region of interest
            roi = np.rot90(roi, 4-args.horlinepos)
            with warnings.catch_warnings():
                # Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                if args.horlinetype == 1 and 'f' in args.croptypes:
                    imsave("%s_%d_f.%s" % (filepath, len(list_linecoords)+2, args.extension), roi)
                elif 'h' in args.croptypes:
                    imsave("%s_%d_h.%s" % (filepath, idx, args.extension), roi)
                    coordstxt.write("Header:  \t%d\t%d\t%d\t%d\n" % (0,linecoords.height_start - 2, 0,image_param.width))
            if args.showmasks == True:
                dim = 0
                if args.horlinetype == 1:
                    dim = 1
                set_colored_mask(debugimage, [[0, linecoords.height_start - 2],
                                              [0, image_param.width]], dim, 100)

        # Crop middle segments
        if linecoords.segmenttype == 'B':
            if not args.quiet: print "blank"
            if args.adaptingmasksoff != True:
                if linecoords.middle - clippingmask.width_start > clippingmask.width_stop - linecoords.middle:
                    linecoords.width_start = linecoords.middle - (clippingmask.width_stop - linecoords.middle)
                    linecoords.width_stop = linecoords.middle + (clippingmask.width_stop - linecoords.middle)
                else:
                    linecoords.width_start = linecoords.middle - (linecoords.middle - clippingmask.width_start)
                    linecoords.width_stop = linecoords.middle + (linecoords.middle - clippingmask.width_start)
            # Add sum extra space to the cords
            roi = image[linecoords.height_start + 2 - pixelheight(args.addstartheightc):linecoords.height_stop - 2 +pixelheight(args.addstopheightc),
                  linecoords.width_start:linecoords.width_stop]  # region of interest
            roi = np.rot90(roi, 4 - args.horlinepos)
            with warnings.catch_warnings():
                # Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                if args.horlinetype == 1:
                    idx = len(list_linecoords) - idx
                if 'c' in args.croptypes:
                    imsave("%s_%d_c.%s" % (filepath, idx+1, args.extension), roi)
                    coordstxt.write(
                        "Blank:  \t%d\t%d\t%d\t%d\n" % (linecoords.height_start + 2 - pixelheight(args.addstartheightc),linecoords.height_stop - 2 +pixelheight(args.addstopheightc),
                  linecoords.width_start,linecoords.width_stop))
            if args.showmasks == True:
                dim = 1
                set_colored_mask(debugimage, [[linecoords.height_start + 2- pixelheight(args.addstartheightc), linecoords.height_stop - 2 +pixelheight(args.addstopheightc)],
                                              [linecoords.width_start, linecoords.width_stop]], dim, 220)
        if linecoords.segmenttype == 'L':
            #Fixing column size
            if args.adaptingmasksoff != True:
                if linecoords.width_stop - clippingmask.width_start > clippingmask.width_stop - linecoords.width_start:
                    clippingmask.width_start = linecoords.width_stop - (clippingmask.width_stop - linecoords.width_start)
                else:
                    clippingmask.width_stop = linecoords.width_start + linecoords.width_stop - clippingmask.width_start
            if idx == 0:
                print "line-first"
                #linecoords.height_start = clippingmask.height_start + 17
            if not args.quiet: print "line"
            roi = image[
                  linecoords.height_start - pixelheight(args.addstartheightab):linecoords.height_stop + pixelheight(args.addstopheightab),
                  clippingmask.width_start:linecoords.width_stop - 2]  # region of interest
            roi = np.rot90(roi, 4 - args.horlinepos)
            with warnings.catch_warnings():
                # Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                if args.horlinetype == 1 and 'b' in args.croptypes:
                    idx = len(list_linecoords) - idx
                    imsave("%s_%d_b.%s" % (filepath, idx, args.extension), roi)
                elif 'a' in args.croptypes:
                    imsave("%s_%d_a.%s" % (filepath, idx+1, args.extension), roi)
                    coordstxt.write(
                        "A-Split:\t%d\t%d\t%d\t%d\n" % (linecoords.height_start - pixelheight(args.addstartheightab),linecoords.height_stop + pixelheight(args.addstopheightab),
                  clippingmask.width_start,linecoords.width_stop - 2))
            if args.showmasks == True:
                dim = 2
                if args.horlinetype == 1:
                    dim = 0
                set_colored_mask(debugimage, [[linecoords.height_start - pixelheight(args.addstartheightab),
                                               linecoords.height_stop + pixelheight(args.addstopheightab)],
                                              [clippingmask.width_start, linecoords.width_stop - 2]], dim, 180)
            roi = image[linecoords.height_start - pixelheight(args.addstartheightab):linecoords.height_stop + pixelheight(args.addstopheightab),
                  linecoords.width_start + 1:clippingmask.width_stop]
            roi = np.rot90(roi, 4 - args.horlinepos)
            with warnings.catch_warnings():
                # Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                if args.horlinetype == 1 and 'a' in args.croptypes:
                    imsave("%s_%d_a.%s" % (filepath, idx, args.extension), roi)
                elif 'a' in args.croptypes:
                    imsave("%s_%d_b.%s" % (filepath, idx+1, args.extension), roi)
                    coordstxt.write(
                        "B-Split:\t%d\t%d\t%d\t%d\n" % (linecoords.height_start - pixelheight(args.addstartheightab),linecoords.height_stop + pixelheight(args.addstopheightab),
                                                        linecoords.width_start + 1,clippingmask.width_stop))
            if args.showmasks == True:
                dim = 0
                if args.horlinetype == 1:
                    dim = 2
                set_colored_mask(debugimage, [[linecoords.height_start - pixelheight(args.addstartheightab),
                                               linecoords.height_stop + pixelheight(args.addstopheightab)],
                                              [linecoords.width_start + 1, clippingmask.width_stop]], dim, 180)

        # Footer
        if idx == len(list_linecoords) - 1:
            if not args.quiet: print "footer"
            roi = image[linecoords.height_stop + 2:image_param.height,
                  0:image_param.width]  # region of interest
            roi = np.rot90(roi, 4 - args.horlinepos)
            with warnings.catch_warnings():
                # Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                if args.horlinetype == 1 and 'h' in args.croptypes:
                    imsave("%s_%d_h.%s" % (filepath, 0, args.extension), roi)
                elif 'h' in args.croptypes:
                    imsave("%s_%d_f.%s" % (filepath, idx+2, args.extension), roi)
                    coordstxt.write(
                        "Footer:  \t%d\t%d\t%d\t%d\n" % (linecoords.height_stop + 2,image_param.height, 0,image_param.width))
            if args.showmasks == True:
                dim = 1
                if args.horlinetype == 1:
                    dim = 0
                set_colored_mask(debugimage,
                                 [[linecoords.height_stop + 2, image_param.height], [0, image_param.width]], dim,
                                 100)
    if args.showmasks == True:
        with warnings.catch_warnings():
            # Transform rotate convert the img to float and save convert it back
            create_dir(image_param.pathout+"\\masks\\")
            filename = (image_param.pathout+"\\masks\\"+"%s_masked.%s" % (image_param.name, args.extension))
            warnings.simplefilter("ignore")
            debugimage = np.rot90(debugimage, 4 - args.horlinepos)
            imsave(filename, debugimage)
    coordstxt.close()
    return 0

def cropping(input):
    # read image
    print input
    args = get_parser()
    try:
        image = imread("%s" % input)
        image_param = ImageParam(image, input)
        if args.imgmask != [0.0, 1.0, 0.0, 1.0]:
            image = image[int(args.imgmask[0]*image_param.height):int(args.imgmask[1]*image_param.height),
                    int(args.imgmask[2]*image_param.width):int(args.imgmask[3]*image_param.width)]
            image_param = ImageParam(image, input)
    except IOError:
        print("cannot open %s" % input)
        logging.warning("cannot open %s" % input)
        return 1
    create_dir(image_param.pathout)
    ####################### DESKEW ####################################
    # Deskew the loaded image
    if args.deskew == True:
        if not args.quiet: print "start deskew"
        deskew(args, image, image_param)
        # image = misc.imread("%s" % (image_param.deskewpath),mode='RGB')
        try:
            image = imread("%s" % (image_param.deskewpath))
            image_param = ImageParam(image, input)
        except IOError:
            print("cannot open %s" % input)
            logging.warning("cannot open %s" % input)
            return 1
    ####################### ANALYSE - LINECOORDS #######################
        if not args.quiet: print "start linecoord-analyse"
    clippingmask = Clippingmask(image)
    list_linecoords, border, topline_width_stop = linecoords_analyse(args, image, image_param, clippingmask)
    ####################### CROP #######################################
    if args.crop == True:
        if not args.quiet: print "start crop"
        crop(args, image, image_param, list_linecoords, clippingmask)
    return 0

def deskew(args,image, image_param):
    # Deskew
    # Calculate the angle of the points between 20% and 80% of the line
    uintimage = get_uintimg(image)
    thresh = th.threshold_sauvola(uintimage, args.threshwindow, args.threshweight)
    binary = uintimage > thresh
    binary = 1-binary #inverse binary
    binary = np.rot90(binary,args.horlinepos)
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    deskew_path = None
    for i, b in enumerate(objects):
        linecoords = Linecoords(image, i, b)
        # The line has to be bigger than minwidth, smaller than maxwidth, stay in the top (30%) of the img,
        # only one obj allowed and the line isn't allowed to start contact the topborder of the image
        if int(args.minwidthhor * image_param.width) < get_width(b) < int(args.maxwidthhor * image_param.width) \
                and int(image_param.height * args.minheighthor) < get_height(b) < int(image_param.height * args.maxheighthor) \
                and int(image_param.height * args.minheighthormask) < (linecoords.height_start+linecoords.height_stop)/2 < int(image_param.height * args.maxheighthormask) \
                and linecoords.height_start != 0:

            pixelwidth = set_pixelground(binary[b].shape[1])
            arr = np.arange(1, pixelwidth(args.deskewlinesize) + 1)
            mean_y = []
            #Calculate the mean value for every y-array
            for idx in range(pixelwidth(args.deskewlinesize)):
                value_y = measurements.find_objects(labels[b][:, idx + pixelwidth((1.0-args.deskewlinesize)/2)] == i + 1)[0]
                mean_y.append((value_y[0].stop + value_y[0].start) / 2)
            polyfit_value = np.polyfit(arr, mean_y, 1)
            deskewangle = np.arctan(polyfit_value[0]) * (360 / (2 * np.pi))
            args.ramp = True
            deskew_image = transform.rotate(image, deskewangle)
            create_dir(image_param.pathout+"\\deskew\\")
            deskew_path = "%s_deskew.%s" % (image_param.pathout+"\\deskew\\"+image_param.name, args.extension)
            image_param.deskewpath = deskew_path
            with warnings.catch_warnings():
                #Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                misc.imsave(deskew_path, deskew_image)
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
        return d1-int(d1*0.5)
    else:
        return d2-int(d2*0.5)

def get_uintimg(image):
    if len(image.shape) > 2:
        uintimage = color.rgb2gray(copy.deepcopy(image))
    else:
        uintimage = copy.deepcopy(image)
    if uintimage.dtype == "float64":
        with warnings.catch_warnings():
            # Transform rotate convert the img to float and save convert it back
            warnings.simplefilter("ignore")
            uintimage = skimage.img_as_uint(uintimage, force_copy=True)
    return uintimage

def get_width(s):
    return s[1].stop-s[1].start

def linecoords_analyse(args,origimg, image_param, clippingmask):
    image = get_uintimg(origimg)
    origimg = np.rot90(origimg, args.horlinepos)
    thresh = th.threshold_sauvola(image, args.threshwindow, args.threshweight)
    binary = image > thresh
    binary = 1-binary #inverse binary
    binary = np.rot90(binary, args.horlinepos)
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    count_height = 0
    count_width = 0
    pixelheight = set_pixelground(image_param.height)
    pixelwidth = set_pixelground(image_param.width)

    list_linecoords = [] # Init list of linecoordinates the format is: [0]: width.start, width.stopt,
    # [1]:height.start, height.stop, [2]: Type of line [B = blank, L = vertical line]
    for i, b in enumerate(objects):
        # The line has to be bigger than minwidth, smaller than maxwidth, stay in the top (30%) of the img,
        # only one obj allowed and the line isn't allowed to start contact the topborder of the image

        linecoords = Linecoords(labels, i, b)
        if pixelwidth(args.minwidthhor) <  get_width(b) < pixelwidth(args.maxwidthhor) \
                and pixelheight(args.minheighthor) < get_height(b) < pixelheight(args.maxheighthor) \
                and pixelheight(args.minheighthormask) <  linecoords.height_stop < pixelheight(args.maxheighthormask) \
                and count_width == 0 \
                and linecoords.height_start != 0:
            # Distance Calculation - defining the clippingmask
            border = get_mindist(b, image_param.width)
            topline_width_stop = b[0].stop + 2 # Lowest Point of object + 2 Pixel
            if clippingmask.user == None:
                clippingmask.width_start = border
                clippingmask.width_stop = image_param.width - border
                #if clippingmask.width_start > pixelwidth(args.minwidthmask):
                #    clippingmask.width_start = pixelwidth(args.minwidthmask)
                #if clippingmask.width_stop < pixelwidth(1.0-args.minwidthmask):
                #    clippingmask.width_stop = pixelwidth(1.0-args.minwidthmask)
                clippingmask.height_start = copy.deepcopy(topline_width_stop)
                clippingmask.height_stop = 0

            # Test for cropping the area under the first vertical line
            #roi = image[hobj_bottom:image_param.height, border:image_param.width - border]  # region of interest
            #imsave("%s_crop.%s" % (image_param.pathout+image_param.name, args.extension), roi)

            # Get coordinats of the line
            labels[b][labels[b] == i + 1] = 0
            count_width += 1
        if pixelheight(args.minheightver) < get_height(b) < pixelheight(args.maxheightver) \
                and pixelwidth(args.minwidthver) < get_width(b) < pixelwidth(args.maxwidthver) \
                and pixelwidth(args.minwidthvermask) < (linecoords.width_start+linecoords.width_stop)/2 < pixelwidth(args.maxwidthvermask) \
                and float(get_width(b))/float(get_height(b)) < args.maxgradientver:
            linecoords.segmenttype = 'L' # Defaultvalue for segmenttype 'P' for horizontal lines
            if count_height == 0:
                if b[0].start - topline_width_stop > pixelheight(args.minsizeblank+args.minsizeblankobolustop):
                    blankline = Linecoords(labels,i,b)
                    blankline.segmenttype = 'B'
                    blankline.height_start = topline_width_stop
                    blankline.height_stop = linecoords.height_start
                    blankline.width_start = border
                    blankline.width_stop = image_param.width - border
                    blankline.middle = int(((linecoords.width_start+linecoords.width_stop)-1)/2)
                    list_linecoords.append(copy.deepcopy(blankline))
                    count_height += 1
                    if args.ramp != None:
                        whiteout_ramp(origimg, linecoords)
                    list_linecoords.append(copy.deepcopy(linecoords))
                    count_height += 1
                else:
                    # Should fix to short vertical lines, in the height to top if they appear before any B Part in the image
                    if topline_width_stop > 0:
                        linecoords.height_start = topline_width_stop + pixelheight(args.addstartheightab)
                    list_linecoords.append(copy.deepcopy(linecoords))
                    count_height += 1
                    if args.ramp != None:
                        whiteout_ramp(origimg, linecoords)
            elif list_linecoords[count_height - 1].height_stop < b[0].stop:
                #Test argument to filter braces
                if b[0].start - list_linecoords[count_height - 1].height_stop > pixelheight(args.minsizeblank):
                    blankline = Linecoords(labels,i,b)
                    blankline.segmenttype = 'B'
                    blankline.height_start = list_linecoords[count_height - 1].height_stop
                    blankline.height_stop = linecoords.height_start
                    blankline.width_start = border
                    blankline.width_stop = image_param.width - border
                    blankline.middle = int(((linecoords.width_start+linecoords.width_stop)-1)/2)
                    list_linecoords.append(copy.deepcopy(blankline))
                    count_height += 1
                    list_linecoords.append(copy.deepcopy(linecoords))
                    if args.ramp != None:
                        whiteout_ramp(origimg, linecoords)
                    count_height += 1
                    labels[b][labels[b] == i + 1] = 0
                else:
                    if args.ramp != None:
                        whiteout_ramp(origimg, linecoords)
                    print b[0].stop
                    list_linecoords[count_height - 1].height_stop = b[0].stop
                    labels[b][labels[b] == i + 1] = 0
    #imsave("%s_EDIT%d.%s" % (image_param.pathout, linecoords.object_value, args.extension), image)
    return list_linecoords, border, topline_width_stop

def set_colored_mask(image, borders, color, intensity):
    # borders[0][.] = height, borders[1][.] = weight, borders[.][0]=start, borders[.][1]=stop
    image[borders[0][0]:borders[0][0]+5,borders[1][0]:borders[1][1]] = 0
    image[borders[0][1]-6:borders[0][1]-1, borders[1][0]:borders[1][1]] = 0
    image[borders[0][0]:borders[0][1], borders[1][0]:borders[1][0]+5] = 0
    image[borders[0][0]:borders[0][1], borders[1][1]-6:borders[1][1]-1] = 0
    # masks all values <= 55 to protect them against the color addition
    masked_image = np.ma.greater(image[borders[0][0]:borders[0][1], borders[1][0]:borders[1][1], color],55)
    image[borders[0][0]:borders[0][1],borders[1][0]:borders[1][1],color] += intensity
    image[borders[0][0]:borders[0][1], borders[1][0]:borders[1][1], color] = image[borders[0][0]:borders[0][1],borders[1][0]:borders[1][1],color] *masked_image
    return 0

def set_pixelground(image_length):
    def get_pixel(prc):
        return int(image_length*prc)
    return get_pixel

def splice(args,inputdir):
    os.chdir(inputdir+"\\segments\\")
    outputdir = inputdir + "\\splice\\"
    spliceinfo = list()
    create_dir(outputdir)
    list_splice = []
    entry_count = 1
    for image in sorted(glob.glob("*.%s" % (args.extension))):
        if os.path.splitext(image)[0].split("_")[len(os.path.splitext(image)[0].split("_"))-1] in args.splicetypes:
            splice_param = SpliceParam(inputdir, os.path.splitext(image)[0].split("_"))
            if splice_param.segmenttype != args.splicemaintype:
                list_splice.append(image)
                spliceinfo.append(image)
            else:
                if not args.quiet: print "splice %s" % (image)
                if args.splicemaintypestop:
                    list_splice.append(image)
                    spliceinfo.append(image)
                if len(list_splice) != 0:
                    segments = [misc.imread(img,mode='RGB') for img in list_splice]
                    img_height = sum(segment.shape[0] for segment in segments)
                    img_width = max(segment.shape[1] for segment in segments)
                    spliced_image = np.ones((img_height, img_width, 3), dtype=segments[0].dtype)*args.bgcolor
                    y = 0
                    for segment in segments:
                        h, w, d = segment.shape
                        spliced_image[y:y + h, 0:w] = segment
                        y += h
                    with warnings.catch_warnings():
                        # Transform rotate convert the img to float and save convert it back
                        warnings.simplefilter("ignore")
                        if args.specialnomoff == True:
                            firstitem = os.path.splitext(spliceinfo[0])[0].split("_")[0]+os.path.splitext(spliceinfo[0])[0].split("_")[1]
                            imsave("%s" % (outputdir+('{0:0>4}'.format(entry_count))+"_"+firstitem+os.path.splitext(spliceinfo[0])[1]), spliced_image)
                            spliceinfofile = open(outputdir+('{0:0>4}'.format(entry_count)) + "_" + firstitem + "_SegInfo" +".txt", "w")
                            entry_count += 1
                            spliceinfofile.writelines([x+"\n" for x in spliceinfo])
                            spliceinfofile.close()
                        else:
                            imsave("%s" % (outputdir+os.path.splitext(spliceinfo[0])[0]+"_spliced"+os.path.splitext(spliceinfo[0])[1]), spliced_image)
                            spliceinfofile = open(outputdir + os.path.splitext(spliceinfo[0])[0] + "_SegInfo" + ".txt",
                                                  "w")
                            spliceinfofile.writelines([x + "\n" for x in spliceinfo])
                            spliceinfofile.close()
                    spliceinfo = list()
                    list_splice = []
                if not args.splicemaintypestop:
                    list_splice.append(image)
                    spliceinfo.append(image)
    if len(list_splice) != 0:
        if not args.quiet: print "splice %s" % (image)
        segments = [misc.imread(img, mode='RGB') for img in list_splice]
        img_height = sum(segment.shape[0] for segment in segments)
        img_width = max(segment.shape[1] for segment in segments)
        spliced_image = np.ones((img_height, img_width, 3), dtype=segments[0].dtype) * args.bgcolor
        y = 0
        for segment in segments:
            h, w, d = segment.shape
            spliced_image[y:y + h, 0:w] = segment
            y += h
        with warnings.catch_warnings():
            # Transform rotate convert the img to float and save convert it back
            warnings.simplefilter("ignore")
            if args.specialnomoff == True:
                firstitem = os.path.splitext(spliceinfo[0])[0].split("_")[0] + \
                            os.path.splitext(spliceinfo[0])[0].split("_")[1]
                imsave("%s" % (
                outputdir + ('{0:0>4}'.format(entry_count)) + "_" + firstitem + os.path.splitext(spliceinfo[0])[1]),
                       spliced_image)
                spliceinfofile = open(outputdir + ('{0:0>4}'.format(entry_count)) + "_" + firstitem + "_SegInfo" + ".txt",
                                      "w")
                spliceinfofile.writelines([x + "\n" for x in spliceinfo])
                spliceinfofile.close()
            else:
                imsave("%s" % (outputdir +os.path.splitext(spliceinfo[0])[0]+"_spliced"+os.path.splitext(spliceinfo[0])[1]), spliced_image)
                spliceinfofile = open( outputdir + os.path.splitext(spliceinfo[0])[0] + "_SegInfo" + ".txt","w")
                spliceinfofile.writelines([x + "\n" for x in spliceinfo])
                spliceinfofile.close()
    return 0

def whiteout_ramp(image, linecoords):
    #for idx in range(linecoords.width_stop):
    imagesection = image[linecoords.object]
    count = 0
    # Dilation enlarge the bright segments and cut them out off the original image
    for i in morph.dilation(linecoords.object_matrix, morph.square(10)):
        whitevalue = measurements.find_objects(i == linecoords.object_value + 1)
        if whitevalue:
            whitevalue = whitevalue[0][0]
            imagesection[count,whitevalue.start:whitevalue.stop] = 255
            count +=1
    #imsave("%s\\whitelines\\%s.%s" %(image_param.path,linecoords.object_value,args.extension), imagesection)
    return 0

####################### MAIN-FUNCTIONS ############################################
def crass():
    args = get_parser()
    args.input = os.path.abspath(args.input)
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
        # chunksize = 1 every multiprocess gets exact the next free image (sorted order)
        pool.map(cropping, inputfiles,chunksize=1)
    ####################### SPLICE #######################################
    if args.splice == True:
        if not args.splicemaintype in args.splicetypes:
                print("%s is not part of the pattern %s" % (args.splicemaintype,args.splicetypes))
                logging.warning("Input error by user!")
        else:
            if not args.quiet: print "start splice"
            path = args.input + "//out//"
            if not os.path.isdir(args.input):
                path = os.path.dirname(args.input) + "//out//"
            splice(args, path)

####################### MAIN ############################################
if __name__=="__main__":
    crass()
