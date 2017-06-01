Image Processing Features
=========================

## Processing Order

Processing of the filters and auto-corrections is performed in a fixed
order according to the following sequence:

 1. **load** image file(s)
 2. perform **deskewing** (optional)
    1. **save** deskewed image file
    2. **load** deskewed image file
 3. **find** the linecoordination informations.
 4. compute **masks**
 5. cropping the areas under the **masks**
 6. **save** single segments and debuginformation (optional)
 7. **load** segments
 8. splice **segments** to a new image
 9. **save** output image file

![Processing Order](img/processing-order.png)

## Disabling Processing Steps

Each processing step can be disabled individually by a corresponding
`--xxx` option (where `xxx` stands for the feature to disable).

    ./crass (...options...) --deskew

This will disable the *deskew*.

## Multiprocessing
You can fully leverage multiple processors on a given machine with 
the feature "parallel" and the number of process. Recommended is the
number of available processors - 1 (which is used by the OS).

    ./crass (...options...) --parallel 3  # for 4 processors

## Threshold - Sauvola
[Sauvola threshold][1] is a local thresholding technique that is useful   
for images where the background is not uniform, especially for text   
recognition. Instead of calculating a single global threshold   
for the entire image, several thresholds are calculated for every pixel by   
using specific formulae that take into account the mean and standard  
deviation of the local neighborhood (defined by a window centered around   
the pixel).

In the original method a threshold T is calculated for every pixel in the   
image using the following formula:

[T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))] [2]

where m(x,y) and s(x,y) are the mean and standard deviation of pixel (x,y)   
neighborhood defined by a rectangular window with size w times w centered   
around the pixel. k is a configurable parameter that weights the effect of   
standard deviation. R is the maximum standard deviation of a greyscale image.

## Linecoord Analyse
### The different linetypes to look for

hor -> Horizontal line 
ver -> Vertical line

    xxxver
    xxxhor

(where `xxx` stands for the feature to set)

### The analyse of the Linecoordination can be influenced with the following parameters:
    
minwidth of the linetype + factor of the image width  
maxwidth of the linetype + factor of the image width  
minheight of the linetype + factor of the image height  
maxheight of the linetype + factor of the image height  

    ./crass (...options...) --minwidthver 0.5 

This will set the *minwidth of the vertical lines to find* to 50% of the
image width.
        
### Mask options (search area) for the different linetypes:

####Horizontal:   
minheighthormask + factor of the image height  
maxheighthormask + factor of the image height

0.0 -> 0% height -> top of the image  
1.0 -> 100% height -> bottom of the image  

    ./crass (...options...) --minheightplmask 0.0 --maxheightplmask 0.3 

This will set the mask (search area) between 0% (top of the image) and 
30% of the images height.
        
####Vertical:
minwidthver + factor of the image width  
maxwidthver + factor of the image width  

0.0 -> 0% width -> left side of the image  
1.0 -> 100% width -> right side of the image  
 
    ./crass (...options...) --minwidthvermask 0.3 --maxwidthvermask 0.7 

This will set the mask (search area) between 30% and 70% of the image.


## Setting Clipping Masks
Clipping mask mark the area to be crop out. This area will be compute 
automatically but the user can set some extra options.
### Add startheight
*Addstartheight* expands the mask of either a&b or c to the top of the image.

     ./crass (...options...) --addstartheightab 50
     ./crass (...options...) --addstartheightc 50

This will expand the area 50 pixels to the top.

### Add stopheight
*Addstopheight* expands the mask of either a&b or c to the bottom of the image.     
     
     ./crass (...options...) --addstopheightab 50
     ./crass (...options...) --addstopheightc 50
     
This will expand the area 50 pixels to the bottom.     

## Splice Pattern

### Splicetypes
There are 5 types of segments:
    
   - h = header
   - a = left side separated by a vertical line
   - b = right side separated by a vertical line
   - c = space between header and vertical line or vertical line and another vertical line
   - f = footer
  
    ./crass (...options...) --splicetypes a,b,f
    
Only the segments of the types a, b and f will be considered in the splicing
process. (Default. a,b and c)

### Splicemaintype
The splicemaintype starts or ends (depending on the *splicemaintypestart*-Option) the splicepattern.

#### Splicemaintypestop
The splicemaintypestop set the maintype to the end of each segment not the start (default).

    ./crass (...options...) --splicetypes a,b,h
    ./crass (...options...) --splicemaintype h

Only the segments of the types a, b and h will be considered in the splicing
process. The pattern will start with a h-segment. There can be several a and b segments
in between.

[1]: http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_niblack_sauvola.html
[2]: http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_sauvola