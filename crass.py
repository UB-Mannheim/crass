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
import os
import warnings

import numpy as np
import scipy.misc as misc
import skimage as ski
import skimage.color as color
import skimage.filters.thresholding as th
import skimage.morphology as morph
import skimage.transform as transform
from scipy.ndimage import measurements
from skimage.io import imread, imsave


####################### CMD-PARSER-SETTINGS ########################
def get_parser():
    parser = argparse.ArgumentParser(description="Crop And Splice Segments (CRASS) of an image based on black (separator-)lines")
    parser.add_argument("--input", type=str, default="",
                        help='Input file or folder')

    parser.add_argument("--extension", type=str, choices=["bmp","jpg","png","tif"], default="jpg", help='Extension of the files, default: %(default)s')

    parser.add_argument('-A', '--addstartheightab', type=float, default=0.01, choices=np.arange(-1.0, 1.0), help='Add some pixel for the clipping mask of segments a&b (startheight), default: %(default)s')
    parser.add_argument('-a', '--addstopheightab', type=float, default=0.011, choices=np.arange(-1.0, 1.0),help='Add some pixel for the clipping mask of segments a&b (stopheight), default: %(default)s')
    parser.add_argument('-C', '--addstartheightc', type=float, default=-0.005, choices=np.arange(-1.0, 1.0),help='Add some pixel for the clipping mask of segment c (startheight), default: %(default)s')
    parser.add_argument('-c', '--addstopheightc', type=float, default=0.0, choices=np.arange(-1.0, 1.0),help='Add some pixel for the clipping mask of segment c (stopheight), default: %(default)s')
    parser.add_argument('--bgcolor', type=int, default=1,help='Backgroundcolor of the splice image (for "uint8": 0=black,...255=white): %(default)s')
    parser.add_argument('--crop', action="store_false", help='cropping paper into segments')
    parser.add_argument("--croptypes", type=str, nargs='+', choices=['a', 'b', 'c', 'f', 'h'],
                        default=['a', 'b', 'c', 'f', 'h'],
                        help='Types to be cropped out, default: %(default)s')
    parser.add_argument('--cutwhite', action="store_true",
                        help='Cut a white area on the left side of the image')
    parser.add_argument('--deskew', action="store_false", help='preprocessing: deskewing the paper')
    parser.add_argument('--deskewlinesize', type=float, default=0.8, choices=np.arange(0.1, 1.0),
                        help='Percantage of the horizontal line to compute the deskewangle: %(default)s')
    parser.add_argument('--deskewonly', action="store_true",
                        help='Only deskew the image')
    parser.add_argument('--tablesplit', action="store_true",
                        help='Split a table with coordinates')
    parser.add_argument('--tablesplice', action="store_true",
                        help='Split a table with coordinates and merge among themselve')
    parser.add_argument("--tablewidth", type=int, default=2700,
                        help='Tablesplit/splice parameter, size of the table in pixel.')
    parser.add_argument("--tablemaxdiff", type=int, default=200,
                        help='Tablesplit/splice parameter, max range from tablestart to the right.')
    parser.add_argument("--tablecolumns", type=int, default=3,
                        help='Tablesplit/splice parameter, number of columns.')
    parser.add_argument("--tableoffset", type=int, default=10,
                        help='Tablesplit/splice parameter, offset of the spliced parts.')
    parser.add_argument("--binary_dilation", type=int, choices=[0, 1, 2, 3], default=0,
                        help='Dilate x-times the binarized areas.')
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
    parser.add_argument('--maxheighthor', type=float, default=0.95, choices=np.arange(0, 1.0), help='maxheight of the horizontal lines, default: %(default)s')
    parser.add_argument('--minheighthormask', type=float, default=0.04, choices=np.arange(0, 1.0), help='minheight of the horizontal lines mask (search area), default: %(default)s')
    parser.add_argument('--maxheighthormask', type=float, default=0.95, choices=np.arange(0, 1.0), help='maxheight of the horizontal lines mask (search area), default: %(default)s')
    parser.add_argument('--minheightver', type=float, default=0.0375, choices=np.arange(0, 1.0), help='minheight of the vertical lines, default: %(default)s')  # Value of 0.035 is tested (before 0.05)
    parser.add_argument('--maxheightver', type=float, default=0.95, choices=np.arange(0, 1.0), help='maxheightof the vertical lines, default: %(default)s')
    parser.add_argument('--minwidthver', type=float, default=0.00, choices=np.arange(0, 1.0), help='minwidth of the vertical lines, default: %(default)s')
    parser.add_argument('--maxwidthver', type=float, default=0.022, choices=np.arange(0, 1.0), help='maxwidth of the vertical lines, default: %(default)s')
    parser.add_argument('--minwidthvermask', type=float, default=0.35, choices=np.arange(0, 1.0), help='minwidth of the vertical lines mask (search area), default: %(default)s')
    parser.add_argument('--maxwidthvermask', type=float, default=0.75, choices=np.arange(0, 1.0), help='maxwidth of the vertical lines mask (search area), default: %(default)s')
    parser.add_argument('--maxgradientver', type=float, default=0.05, choices=np.arange(0, 1.0), help='max gradient of the vertical lines: %(default)s')
    # 0.016
    parser.add_argument('--minsizeblank', type=float, default=0.015, choices=np.arange(0, 1.0), help='min size of the blank area between to vertical lines, default: %(default)s')
    parser.add_argument('--minsizeblankobolustop', type=float, default=0.014, choices=np.arange(0, 1.0),help='min size of the blank area between to vertical lines, default: %(default)s')
    parser.add_argument('--nomnumber', type=int, default=4,help='Sets the quantity of numbers in the nomenclature (for "4": 000x_imagename): %(default)s')
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
    parser.add_argument('--woblankstop', action="store_true",
                        help='Deactivates the whiteout of the blank parts for the a & b parts, this will lead to less memory usage.')
    parser.add_argument('-q', '--quiet', action='store_true', help='be less verbose, default: %(default)s')

    args = parser.parse_args()
    return args

####################### LOGGER-FILE-SETTINGS ########################
logging.basicConfig(filename=os.path.dirname(get_parser().input) + os.path.normcase('//Logfile_crass.log'), level=logging.DEBUG,
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
        self.pathout = os.path.normpath(os.path.dirname(input)+"/out/")
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
            print(newdir)
        except IOError:
            print("cannot create %s directoy" % newdir)

def crop(args, image, image_param, labels,list_linecoords, clippingmask):
    # Crops the segments based on the given linecoords
    # and export the linecoords into a txt file
    create_dir(image_param.pathout+os.path.normcase("/segments/"))
    filepath = image_param.pathout+os.path.normcase("/segments/")+image_param.name
    create_dir(image_param.pathout+os.path.normcase("/coords/"))
    coordstxt = open(image_param.pathout+os.path.normcase("/coords/")+image_param.name+"_coords.txt", "w")
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
                        "C-Seg: \t%d\t%d\t%d\t%d\n" % (linecoords.height_start + 2 - pixelheight(args.addstartheightc),linecoords.height_stop - 2 +pixelheight(args.addstopheightc),
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
            if args.woblankstop == False:
                whiteout_blank(image, labels, linecoords.height_start- pixelheight(args.addstartheightab),linecoords.height_stop + pixelheight(args.addstopheightab)-linecoords.height_start- pixelheight(args.addstartheightab))
            roi = image[linecoords.height_start - pixelheight(args.addstartheightab):linecoords.height_stop + pixelheight(args.addstopheightab),
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
                        "A-Seg:\t%d\t%d\t%d\t%d\n" % (linecoords.height_start - pixelheight(args.addstartheightab),linecoords.height_stop + pixelheight(args.addstopheightab),
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
                        "B-Seg:\t%d\t%d\t%d\t%d\n" % (linecoords.height_start - pixelheight(args.addstartheightab),linecoords.height_stop + pixelheight(args.addstopheightab),
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
            create_dir(image_param.pathout+os.path.normcase("/masks/"))
            filename = (image_param.pathout+os.path.normcase("/masks/")+"%s_masked.%s" % (image_param.name, args.extension))
            warnings.simplefilter("ignore")
            debugimage = np.rot90(debugimage, 4 - args.horlinepos)
            imsave(filename, debugimage)
    coordstxt.close()
    return 0

def cropping(input):
    # Main cropping function that deskew, analyse and crops the image
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
    if args.cutwhite:
        if not args.quiet: print "start cutwhite"
        cut_white(args, image, image_param)
        return
    # Deskew the loaded image
    if args.deskew == True:
        if not args.quiet: print "start deskew"
        deskew(args, image, image_param)
        try:
            image = imread("%s" % (image_param.deskewpath))
            image_param = ImageParam(image, input)
        except IOError:
            print("cannot open %s" % input)
            logging.warning("cannot open %s" % input)
            return
    if args.deskewonly:
        print "Only Deskew mode finished!"
        return 0
    ####################### SIMPLE TABLE SPLIT AND SPLICE #######################
    if args.tablesplit or args.tablesplice:
        if not args.quiet: print "start table split and splice"
        table_split_and_splice(args, image, image_param)
        return
    ####################### ANALYSE - LINECOORDS #######################
    if not args.quiet: print "start linecoord-analyse"
    clippingmask = Clippingmask(image)
    border, labels, list_linecoords, topline_width_stop = linecoords_analyse(args, image, image_param, clippingmask)
    ####################### CROP #######################################
    if args.crop == True:
        if not args.quiet: print "start crop"
        crop(args, image, image_param, labels, list_linecoords, clippingmask)
    return 0

def cut_white(args, image, image_param):
    uintimage = get_uintimg(image)
    white_arr = np.array(uintimage).sum(axis=0) - 65535 * image_param.height
    white_arr[-1] = 0
    first_white_col = min(np.where(white_arr == 0)[0])
    create_dir(image_param.pathout + os.path.normcase("/cutwhite/"))
    deskew_path = "%s.%s" % (image_param.pathout + os.path.normcase("/cutwhite/") + image_param.name, args.extension)
    misc.imsave(deskew_path, image[:, :first_white_col])
    return

def deskew(args,image, image_param):
    # Deskew the given image based on the horizontal line
    # Calculate the angle of the points between 20% and 80% of the line
    uintimage = get_uintimg(image)
    binary = get_binary(args, uintimage)
    for x in range(0,args.binary_dilation):
        binary = ski.morphology.binary_dilation(binary,selem=np.ones((3, 3)))
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
            #arr = np.arange(1, pixelwidth(args.deskewlinesize) + 1)
            mean_y = []
            #Calculate the mean value for every y-array
            old_start = None
            for idx in range(pixelwidth(args.deskewlinesize)):
                value_y = measurements.find_objects(labels[b][:, idx + pixelwidth((1.0-args.deskewlinesize)/2)] == i + 1)[0]
                if old_start is None:
                    old_start = value_y[0].start
                #mean_y.append((value_y[0].stop + value_y[0].start) / 2)
                if abs(value_y[0].start-old_start) < 5:
                    mean_y.append(value_y[0].start)
                    old_start = value_y[0].start
            #stuff = range(1, len(mean_y) - 1)
            polyfit_value = np.polyfit(range(0,len(mean_y)), mean_y, 1)
            deskewangle = np.arctan(polyfit_value[0]) * (360 / (2 * np.pi))
            args.ramp = True
            deskew_image = transform.rotate(image, deskewangle, mode="edge")
            create_dir(image_param.pathout+os.path.normcase("/deskew/"))
            deskew_path = "%s_deskew.%s" % (image_param.pathout+os.path.normcase("/deskew/")+image_param.name, args.extension)
            deskewinfo = open(image_param.pathout+os.path.normcase("/deskew/")+image_param.name + "_deskewangle.txt", "w")
            deskewinfo.write("Deskewangle:\t%f" % deskewangle)
            deskewinfo.close()
            image_param.deskewpath = deskew_path
            with warnings.catch_warnings():
                #Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                misc.imsave(deskew_path, deskew_image)
            break
    return deskew_path

def table_split_and_splice(args,image, image_param):
    # This function splits an Image with an table by parameters
    # and merge the fragment among each other
    uintimage = get_uintimg(image)
    binary = get_binary(args, uintimage)
    for x in range(0,args.binary_dilation):
        binary = ski.morphology.binary_dilation(binary,selem=np.ones((3, 3)))
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    for i, b in enumerate(objects):
        linecoords = Linecoords(image, i, b)
        if int(args.minwidthhor * image_param.width) < get_width(b) < int(
                args.maxwidthhor * image_param.width) \
                and int(image_param.height * args.minheighthor) < get_height(b) < int(
            image_param.height * args.maxheighthor) \
                and int(image_param.height * args.minheighthormask) < (
                linecoords.height_start + linecoords.height_stop) / 2 < int(
            image_param.height * args.maxheighthormask) \
                and linecoords.height_start != 0:
            new_linecoords = objects[i]
            linecoords.width_start = new_linecoords[1].start
            linecoords.widthstop = new_linecoords[1].stop
            table_width = args.tablewidth
            max_table_diff = args.tablemaxdiff
            columns = args.tablecolumns
            spliceoffset = args.tableoffset
            col_width = (get_width(b) /columns )
            splitpoint = linecoords.width_start + (col_width)
            if abs((get_width(b) / columns) - table_width) > max_table_diff:
                col_width = (table_width / columns)
                if linecoords.width_start < image_param.width * 0.2:
                    splitpoint = linecoords.width_start + col_width
                else:
                    splitpoint = linecoords.width_stop - (col_width * (columns - 1))
            #Dynamical reszizing
            img_width = splitpoint+spliceoffset
            if columns > 1:
                last_splitpoint = splitpoint + (col_width * (columns - 2))-spliceoffset
                if splitpoint < (image_param.width - (last_splitpoint-spliceoffset)): img_width = image_param.width - (last_splitpoint-spliceoffset)
            spliced_image = np.ones((image_param.height * columns, img_width, 3)) * (255*args.bgcolor)
            startpoint = spliceoffset
            for part in range(1, columns + 1):
                if part == columns:
                    splitpoint = image_param.width - spliceoffset
                fragment = image[:, (startpoint - spliceoffset):( splitpoint + spliceoffset)]
                if args.tablesplit:
                    create_dir(image_param.pathout + os.path.normcase("/tablesplit/"))
                    misc.imsave("%s_deskew_%d.%s" % (
                        image_param.pathout + os.path.normcase("/tablesplit/") + image_param.name,part,args.extension),
                                fragment)
                if args.tablesplice:
                    spliced_image[image_param.height * (part - 1):image_param.height * part,
                    :(splitpoint + spliceoffset) - (startpoint - spliceoffset)] = fragment
                startpoint = splitpoint - spliceoffset
                splitpoint += col_width
            if args. tablesplice:
                create_dir(image_param.pathout + os.path.normcase("/tablesplice/"))
                misc.imsave("%s_deskew_merge.%s" % (
                image_param.pathout + os.path.normcase("/tablesplice/") + image_param.name, args.extension),
                        spliced_image)
            break
    return

def get_binary(args, image):
    thresh = th.threshold_sauvola(image, args.threshwindow, args.threshweight)
    binary = image > thresh
    binary = 1 - binary  # inverse binary
    binary = np.rot90(binary, args.horlinepos)
    return binary

def get_inputfiles(args):
    input = args.input
    if not os.path.isfile(input):
        os.chdir(input)
        inputfiles = []
        for input in sorted(glob.glob("*.%s" % (args.extension))):
            inputfiles.append(os.getcwd() + os.path.normcase("/") + input)
    else:
        inputfiles = []
        inputfiles.append(input)
    return inputfiles

def get_height(s):
    return s[0].stop-s[0].start

def get_linecoords(s):
    return [[s[0].start,s[0].stop],[s[1].start,s[1].stop]]

def get_mindist(s,length):
    # Computes the min. distance to the border and cuts the smallest one in half
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
            uintimage = ski.img_as_uint(uintimage, force_copy=True)
    return uintimage

def get_width(s):
    return s[1].stop-s[1].start

def linecoords_analyse(args,origimg, image_param, clippingmask):
    # Computes the clipping coords of the masks
    image = get_uintimg(origimg)
    origimg = np.rot90(origimg, args.horlinepos)
    binary = get_binary(args, image)
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
                clippingmask.height_start = copy.deepcopy(topline_width_stop)
                clippingmask.height_stop = 0
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
    return border, labels, list_linecoords, topline_width_stop

def set_colored_mask(image, borders, color, intensity):
    # Colorize the masked areas and create a black border
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
    #Computes the real pixel number out of the given percantage
    def get_pixel(prc):
        return int(image_length*prc)
    return get_pixel

def splice(args,inputdir):
    #Search the segments pattern in the given directory and splice them together
    #Spliceinfo writes a txt file with all segments in the spliced image
    #prints(os.path.normpath(inputdir+os.path.normcase("\\segments\\")))
    os.chdir(inputdir+os.path.normcase("/segments/"))
    outputdir = inputdir + os.path.normcase("/splice/")
    spliceinfo = list()
    create_dir(outputdir)
    list_splice = []
    entry_count = 1
    image = "Nothing!"
    nomnumber = '{0:0>%d}' % args.nomnumber
    for image in sorted(glob.glob("*.%s" % args.extension)):
        if os.path.splitext(image)[0].split("_")[len(os.path.splitext(image)[0].split("_"))-1] in args.splicetypes:
            splice_param = SpliceParam(inputdir, os.path.splitext(image)[0].split("_"))
            if splice_param.segmenttype != args.splicemaintype:
                list_splice.append(image)
                spliceinfo.append(image)
            else:
                if not args.quiet: print "splice %s" % image
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
                        if args.specialnomoff:
                            firstitem = os.path.splitext(spliceinfo[0])[0].split("_")[:-2]
                            firstitem = "_".join(firstitem)
                            #print(inputdir)
                            year = os.path.splitext(os.path.normpath(inputdir))[0].split(os.sep)[-3:-2][0]
                            imsave("%s" % (outputdir+(nomnumber.format(entry_count))+"_"+year+"_"+firstitem+os.path.splitext(spliceinfo[0])[1]), spliced_image)
                            spliceinfofile = open(outputdir+(nomnumber.format(entry_count)) + "_" + firstitem + "_SegInfo" +".txt", "w")
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
        if not args.quiet: print "splice %s" % image
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
            if args.specialnomoff:
                firstitem = os.path.splitext(spliceinfo[0])[0].split("_")[:-2]
                firstitem = "_".join(firstitem)
                year = os.path.splitext(os.path.normpath(inputdir))[0].split(os.sep)[-3:-2][0]
                imsave("%s" % (outputdir + (nomnumber.format(entry_count)) + "_" +year+"_"+firstitem + os.path.splitext(spliceinfo[0])[1]),
                       spliced_image)
                spliceinfofile = open(outputdir + (nomnumber.format(entry_count)) + "_" + firstitem + "_SegInfo" + ".txt",
                                      "w")
                spliceinfofile.writelines([x + "\n" for x in spliceinfo])
                spliceinfofile.close()
            else:
                imsave("%s" % (outputdir + os.path.splitext(spliceinfo[0])[0]+"_spliced"+os.path.splitext(spliceinfo[0])[1]), spliced_image)
                spliceinfofile = open(outputdir + os.path.splitext(spliceinfo[0])[0] + "_SegInfo" + ".txt","w")
                spliceinfofile.writelines([x + "\n" for x in spliceinfo])
                spliceinfofile.close()
    return 0

def whiteout_ramp(image, linecoords):
    # Dilation enlarge the bright segments and cut them out off the original image
    imagesection = image[linecoords.object]
    count = 0
    for i in morph.dilation(linecoords.object_matrix, morph.square(10)):
        whitevalue = measurements.find_objects(i == linecoords.object_value + 1)
        if whitevalue:
            whitevalue = whitevalue[0][0]
            imagesection[count,whitevalue.start:whitevalue.stop] = 255
            count +=1
    return 0

def whiteout_blank(image, labels, height, fullheight):
    # Dilation enlarge the bright segments and cut them out off the original image
    objects = measurements.find_objects(labels)
    for i, b in enumerate(objects):
        if b != None:
            #print(b[0])
            #print(height)
            if b[0].start <= height and fullheight*0.2 >= b[0].stop-b[0].start and b[0].stop != 0:
                linecoords = Linecoords(labels, i, b)
                whiteout_ramp(image, linecoords)
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
    if not args.splice == True:
        if not args.splicemaintype in args.splicetypes:
                print("%s is not part of the pattern %s" % (args.splicemaintype,args.splicetypes))
                logging.warning("Input error by user!")
        else:
            if not args.quiet: print "start splice"
            path = args.input + os.path.normcase("/out/")
            if not os.path.isdir(args.input):
                path = os.path.dirname(args.input)+os.path.normcase("/out/")
            splice(args, os.path.normpath(path))

####################### MAIN ############################################
if __name__=="__main__":
    crass()
