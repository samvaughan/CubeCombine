#!/usr/bin/env python

''' A module to modify FITS file images.
Written by Azin Khan.
Oxford University.
Summer 2012.

Modified by Ryan Houghton (RH)
'''

import numpy
try:
    import astropy.io.fits as pf
except:
    try:
        import pyfits as pf
    except:
        raise ImportError("astropy.io.fits nor pyfits found.")
import scipy
import scipy.ndimage as ndimage
import time
import numpy.ma as ma
import sys
import pdb
import pylab as pl
import os

def FOVshiftANDrot(dx,dy,dtheta):
    """
    Coordinate transform function for use with ndimage.geometric transform

    Added to allow combination of (spatially) rotated cubes

    """

def offsetting(images, offsets, order=1):
    '''
    This function takes a list of the images and arranges them separately into
    a larger array such that the correct position of the image is achieved
    using the offsets given.

    Input:
        images = List of images to be arranged. Even if only one image is to be
                 offset, the data-type of this parameter must still be a list
                 e.g. images = [some_array], not images = some_array.

        offsets = A tuple of lists documenting the correct offset for each
                  image. These must be in (x,y) form

        integer = Input to determine whether integer or non-integers offsets
                  are used. If False, then bilinear interpolation is used.
                  Non-integer offsets are set as default.
        order   = Order of the spline interpolation.
                  0=nearest ; 1=bilinear ; 2=quadratic ; 3=cubic

    Output:
        Returns a list of the images correctly positioned on a larger image,
        ready to be combined.

    Note:
        The list of images and the list of offsets must correspond; the image
        from which the offsets are calculated must have an offset entry
        of (0,0).

        The larger arrays size is calculated using the offset file.
        This ensures that the larger array is of minimum size to reduce
        filesize and inefficiencies, but will fit all of the required images.



    '''

    X_offsets = offsets[0]
    Y_offsets = offsets[1]

    if type(images) is not list:
        raise TypeError, 'The images must be given in a list'

    else:
        s = images[0].shape
        w = s[0]
        y = s[1]
        x = s[2]

    # Determine the required size of the composite image.

    



    if max(X_offsets) <0:
        X_dimension = x + iround(numpy.fabs(min(X_offsets)))
        x_pos = iround(numpy.fabs(min(X_offsets)))
    elif min(X_offsets) >0:
        X_dimension = x + iround(max(X_offsets))
        x_pos = 0
    else:
        X_dimension = x + iround(max(X_offsets)) + iround(numpy.fabs(min(X_offsets)))
        x_pos = numpy.fabs(min(X_offsets))

    if max(Y_offsets) <0:
        Y_dimension = y + iround(numpy.fabs(min(Y_offsets)))
        y_pos = iround(numpy.fabs(min(Y_offsets)))
    elif min(Y_offsets) >0:
        Y_dimension = y + iround(max(Y_offsets))
        y_pos = 0
    else:
        #Y_dimension = numpy.ceil(y + max(Y_offsets) + numpy.fabs(min(Y_offsets)))
        Y_dimension = y + iround(max(Y_offsets)) + iround(numpy.fabs(min(Y_offsets)))
        y_pos = iround(numpy.fabs(min(Y_offsets)))

    ##################################################################################
    #SPV: force the size to be a certain value- useful for my bootstrapping code!
    #force_size=[71, 122]
    #if force_size is not None:
    #    X_dimension=force_size[0]
    #    Y_dimension=force_size[1]
    #pdb.set_trace()
    ##################################################################################

    

    final_list = []

    for i, image in enumerate(images):
        # Use bi-linear interpolation to offset the images correctly.
        large = numpy.ones((w, Y_dimension, X_dimension), dtype=numpy.float) * numpy.nan
        x_offset = iround(X_offsets[i])
        y_offset = iround(Y_offsets[i])
        # Determine the 'non-integer' part of the offset by truncating it.
        # and then interpolate.
        x_shift = X_offsets[i]-x_offset # RH swapped sign here and just below
        y_shift = Y_offsets[i]-y_offset
        shift = [0,y_shift,x_shift]
        masked_image = numpy.ma.masked_array(image,numpy.isnan(image))
        interp_image = ndimage.interpolation.shift(masked_image, shift, order=order, mode='nearest')
        # Now the remaining offset is an integer and can be simply added on.
        #pdb.set_trace()
        large[:, iround(y_pos+y_offset):iround(y_pos+y_offset+y), iround(x_pos+x_offset):iround(x_pos+x_offset+x)] = 0.0
        #masked_large = ma.masked_equal(large,numpy.nan)
        masked_large = ma.masked_array(large, numpy.isnan(large)) # RH 22/1/14 numpy.nan != numpy.nan !!!!
        masked_large[0:w, iround(y_pos+y_offset):iround(y_pos+y_offset+y), iround(x_pos+x_offset):iround(x_pos+x_offset+x)] += interp_image
        masked_large.fill_value=numpy.nan # RH 23/1/14 fill with nans where bad

        #pdb.set_trace()
        final_list.append(masked_large)


        #pdb.set_trace()
    return final_list

def iround(num):
    return numpy.int(numpy.round(num))

def precise_combine(images, bpms, threshold=0, fill=0, median=False):
    '''
    This function combines a list of given images and any bad pixel masks into
    one image.

    Input:
        images = List of images to be combined.

        bpms = List of corresponding bad pixel mask arrays.

        threshold = Threshold values for bad pixels.

        fill = Fill value for masked pixels (0 by default to ensure that they
               are properly ignored).

        mask = Determines if the images are to be masked. If set to True, then
               the images will be masked according to the BPM information.

        median = Use the median rather than the mean for averaging

    Output:
        Returns the composite of the images.

    Note:
        The procedure is to take a wavelength 'slice' across all the images and
        then average using ma.average to ignore masked entries. This results in
        a final slice that is appended to a list of slices and eventually
        converted into a numpy array once all slices have been added to the
        list. This is a rather slow way to combine datacubes but fully accounts
        for the bad pixel masks, resulting in the smoothest image possible

        It is advisable to set mask = True and input an unmasked array as
        otherwise the ma.count function may not operate correctly. This appears
        to be a problem inherent in the numpy.ma module (see numpy.ma
        documentation).

    '''

#    final = []
#    bpm_info = []
#    wavelengths = images[0].shape[0]
#
#
#    for k in range(wavelengths):
#        mini = []
#        minierror = []
#        for i, image in enumerate(images):
#            # For each wavelength, find the corresponding slice in each
#            # image and create a small data cube with this information.
#            # This small data cube has a third dimension of image index.
#            mini.append(image[k])
#            if len(bpms)>0:
#                minierror.append(bpms[i][k])
#
#        # Convert to an array once all the slices have been found.
#        error = ma.array(minierror)
#        minicube = ma.array(mini)
#
#        # Mask the array then average over it, hence combining the
#        # different images at this wavelength into one final slice.
#        if len(bpms)>0:
#            masked_mini = ma.masked_where(error>threshold, minicube, copy=False)
#        else:
#            masked_mini = minicube
#
#        bpm_infoslice = ma.count(masked_mini, axis=0)
#        bpm_info.append(bpm_infoslice)
#        if median:
#            averaged_slice   = ma.median(masked_mini,axis=0)
#        else:
#            averaged_slice = ma.average(masked_mini, axis=0)
#
#        final.append(numpy.array(averaged_slice))
#
#
#    # Convert the results into numpy arrays.
#    # Pyfits cannot handle ma.array (yet).
#    # Any remaining masked pixels are set to zero in the process.
#    final_array = numpy.array(final)
#    bpm_final_info = numpy.array(bpm_info)


    # convert input to arrays
    aimages = numpy.ma.masked_invalid(images, copy=True)
    if len(bpms)>0:
        abpms = numpy.ma.masked_invalid(bpms, copy=True)
        # join masks on data
        aimages=numpy.ma.masked_where(numpy.ma.getmask(abpms),aimages,copy=False) # this augments mask
    else:
        abpms=None




    # make into masked arrays, including BPMs
    if abpms!=None:
        # set fill to -1 : always less than thresh
        aimages.set_fill_value(-1)
        aimages = numpy.ma.masked_where((abpms.filled()>threshold),aimages,copy=False) # this augments mask
        aimages.set_fill_value(numpy.nan) # set to nan

    # now make the output
    if median:
        average = numpy.ma.median(aimages,axis=0)
    else:
        average = numpy.ma.mean(aimages,axis=0)

    average.set_fill_value(numpy.nan)

    final_array = average.filled()
    bpm_final_info = numpy.ma.count(aimages,axis=0)

    return final_array, bpm_final_info


def open_refine(image_stem, error_stem, number, X_cut=0, Y_cut=0):
    '''
    This function opens a FITS file and extracts the data as a numpy array.
    The refine element is included to remove edge pixels from the outset;
    they are simply cropped out.

    Input:
        image_stem = The filname stem for the image.

        error_stem = The filename stem for the bpm file.

        number = The number of the file to be opened.

        X_cut, Y_cut = The number of pixels to be sliced from each edge.
                        This is done symmetrically

        Example input:

            open_refine('ms0%d.sci.cube.fits', 'ms0%d.bpm.cube.fits', 83, 5,10)

            This would open ms083.sci.cube.fits as the image and
            ms083.bpm.cube.fitsas the bpm array with 5 pixels cropped
            from each end of thex axis and 10 pixels cropped from each end
            of the y axis.Note that if the filenames are numbered with 0 as
            the leading digit, then this must be included in the stem, not
            the number.

    Output:
        Returns a list of two arrays: [Image, BPM].

    '''

    image = pf.open(image_stem %number)
    image_data = refine_array(image[0].data,X_cut,Y_cut)
    error = pf.open(error_stem %number)
    error_data = refine_array(error[0].data,X_cut,Y_cut)


    return [image_data, error_data]

def refine_array(data, x_cut, y_cut):
    'Simple function to crop an array symmetrically'
    size = data.shape
    w = size[0]
    y = size[1]
    x = size[2]
    refined_data = data[0:w, y_cut:y-y_cut, x_cut:x-x_cut]
    return refined_data

def background_remove(A1, A2):
    '''
    This function performs a pairwise background subtraction on two arrays

    '''
    B1 = A1 - A2
    B2 = A2 - A1

    return B1, B2


def image_combine(image_list, offsets=[], bpm_threshold=0.0, bpm_list=[], order=1, \
                  background=False, median=False, dump=False, retStack=False):
    '''
    This function combines the given images using the information given

    Input:
        image_list = List of the images to be combined, given as numpy arrays.

        offsets = Optional. Tuple of lists of the relative offset of each image.
                  The 'zeroth' image has an offset of (0,0).
                  These must be given in (x,y) form.

        bpm_list = Optional. List of the bad pixel mask arrays. These must be of
                   the same shape as the images.
                   By default, all the pixels are assumed to be good.

        bpm_threshold = Optional. The threshold value of the bpm array. If the
                        absolute value of an entry is above this threshold, the
                        corresponding pixel in the image is masked as bad. This
                        is set to 0 by default.

        offset,integer = Optional. Use to determine if the images need to be
                         offset relative to each other. If so, the offset must
                         be True. If integer is true, then a quicker offset is
                         performed. Useful for cases where the offsets are
                         actually integers in pixel units or if a rough image is
                         desired quickly. Otherwise, the images are interpolated
                         during the offset procedure.

        order       = order of spline interpolation for shift
                      0=nearest ; 1=bilinear ; 2=quadratic ; 3= cublic (spline)

        background  = Perform pairwise background subtraction (1-2, 3-4, 5-6 etc)

        Median      = Perform a median combine rather than mean

        Dump        = Save the offset images into a MEF - useful for debugging
                      problems in the final combined image (such as bad pixels
                      getting though undetected)
        retStack    = return the stack of shifted images

    Output:
        Returns one final image, which, in general, will be larger than any of
        the given images due to the offsetting. The images are averaged
        appropriately, taking into account any bad pixels. Thus, the output is
        normalized for each case. Also returns its corresponding quality array,
        the entries of which are the number of good pixels each entry in the
        image represents.

        '''


#    # check for nans
#    nan_list = numpy.array(~numpy.isfinite(numpy.array(image_list)), dtype=numpy.float)
#    if numpy.any(nan_list):
#        print "Augmenting BPMs from Nans"
#        nans_be_here = numpy.where(nan_list)
#        if len(bpm_list)==0:
#            print "No previous BPM list... creating from scratch"
#            bpm_list = numpy.zeros_like(image_list)
#            bpm_list[nans_be_here]=1.0
#        else:
#            bpm_list=numpy.array(bpm_list)
#            bpm_list[nans_be_here]=1.0
#        bpm_list=list(bpm_list)




    # Perform pairwise background subtraction if necessary.
    if background:
        bkgrnd = []
        number = len(image_list)
        # Straightforward if an even number of images given.
        if number%2 == 0:
            for k in range(0,number,2):
                bk = background_remove(image_list[k], image_list[k+1])
                bkgrnd.append(bk[0])
                bkgrnd.append(bk[1])

        else:
            # Repeat as before, but leaving the last image out.
            for k in range(0,number-1, 2):
                bk = background_remove(image_list[k], image_list[k+1])
                bkgrnd.append(bk[0])
                bkgrnd.append(bk[1])
            # For the last image, find the image that has the largest offset
            # relative to it.
            differences = []
            for i, offset in enumerate(offsets[0]):
                difference = numpy.fabs(offsets[0][-1]-offset)
                differences.append(difference)
            furthest_index = differences.index(max(differences))
            # Now do background subtraction between these two images
            bklast = background_remove(image_list[-1],
                                          image_list[furthest_index])
            bkgrnd.append(bklast[0])
    else:
        bkgrnd = image_list

    if offsets:
        # Offset the images with their appropriate offset.
        offset_images = offsetting(bkgrnd, offsets, order=order)
        # Offset the BPM arrays if given.
        if len(bpm_list)>0:
            offset_bpm = offsetting(bpm_list, offsets, order=order)
        else:
            offset_bpm = []

        if dump:
            hdu=[]
            for offim in offset_images:
                hdu.append(pf.PrimaryHDU(offim))
            hdulist = pf.HDUList(hdu)
            hdulist.writeto("offsetSCIdump.fits",clobber=True)
            hdu=[]
            for offim in offset_bpm:
                hdu.append(pf.PrimaryHDU(offim))
            hdulist = pf.HDUList(hdu)
            hdulist.writeto("offsetBPMdump.fits", clobber=True)

        combined_image = precise_combine(offset_images, offset_bpm, \
                                         bpm_threshold, median=median)
        final_image = combined_image[0]
        bpm_information = combined_image[1]
        if retStack:
            # make masked array for output stack
            offset_images = numpy.array(offset_images)
            offset_bpm    = numpy.array(offset_bpm)
            offset_images = ma.masked_invalid(offset_images, copy=False)
            offset_images = ma.masked_where(offset_bpm>bpm_threshold, offset_images, copy=False)
            return final_image, bpm_information, offset_images
        else:
            return final_image, bpm_information

    else:
        combined_image = precise_combine(bkgrnd, bpm_list, bpm_threshold)
        final_image = combined_image[0]
        bpm_information = combined_image[1]
        if retStack:
            return final_image, bpm_information, offset_images
        else:
            return final_image, bpm_information

if __name__ == '__main__':

    """
    If run from the shell, make it a quickfire script using args as
    input
    """

    # debugging: used median combine rather than mean; takes FOREVER (2h)!
    #median = True
    median = False

    # debugging: dump all shifted images and BPMs
    #dump=True
    dump=False

    start_time = time.time()

    
    if(len(sys.argv) == 4): # RH : dislike this way if reading in - limited to 25 characters. 2016 SV- not if you use dtype="str" rather than dtype="S25". Also should really use argparse not sysargv!
        files = pl.loadtxt(sys.argv[1],comments="#",dtype=str)
        offx,offy = pl.loadtxt(sys.argv[2],comments="#", unpack=True)
        offsets=(offx,offy)
        bpms=[]
        bpmthresh=0.0
        filename = sys.argv[3]
    elif (len(sys.argv)==7):
        files = pl.loadtxt(sys.argv[1],comments="#",dtype=str)
        offx,offy = pl.loadtxt(sys.argv[2],comments="#", unpack=True)
        offsets=(offx,offy)
        bpms  = pl.loadtxt(sys.argv[3],comments="#",dtype=str)
        bpmthresh = numpy.float(sys.argv[4])
        filename = sys.argv[5]
        bpmfilename = sys.argv[6]
    else: # RH
        print ""
        print "********  ERROR: incorrect input arguments ********"
        print ""
        print "USAGE: "
        print "------"
        print "[1] Without bad pixel masks (BPM)"
        print "    > python [path/]cube_combine.py SCIlist[.txt] offsetlist[.txt] SCIoutout[string]"
        print "e.g.> python $python/IFS/cube_combine.py SCIlist.txt offsets.txt combined.fits"
        print ""
        print "[2] With bad pixel masks (BPM)"
        print "    > python [path/]cube_combine.py SCI_list[.txt] offset_list[.txt] BPM_list[.txt] BPM_threshhold[float] SCIoutput[string] BPMoutput[string]"
        print "e.g.> python $python/IFS/cube_combine.py SCIlist.txt offsets.txt BPMlist.txt 0.0001 combined.fits quality.fits"
        print ""
        print "Outputs:"
        print "--------"
        print "SCIoutput: cube of the combined (mean) data."
        print "BPMoutput: cube of integers describing # of input pixels contributing to the output pixels."
        print "           A value of zero [0] implies that output pixel is still bad!"
        print ""
        sys.exit(2)

    # check if the files to create already exist; if so, exit (RH)
    if(os.path.exists(filename)):
        print ""
        print "ERROR: file "+filename+" already exists. Quiting."
        print ""
        sys.exit(2)
    if( len(sys.argv)!= 4 ):
        if(os.path.exists(bpmfilename) ):
            print ""
            print "ERROR: file "+bpmfilename+" already exists. Quiting."
            print ""
            sys.exit(2)

    # Open sci images
    imagelist=[]
    imheaderlist = []
    bpheaderlist = []

    for file in files:
        imagelist.append((pf.open(file))[0].data)
        imheaderlist.append((pf.open(file))[0].header)

    # Open bpms
    bpmlist=[]
    for file in bpms:
        bpmlist.append((pf.open(file))[0].data)
        bpheaderlist.append((pf.open(file))[0].header)

    # Create a new FITS file for the combined image.
    aim = image_combine(imagelist, offsets, bpmthresh, bpmlist, median=median)
    # Use the leftmost image's header
    min_x_offset = min(offsets[0])
    locations = numpy.where(offsets[0]==min_x_offset)
    left_location = locations[0][0]
    y_offset = offsets[1][left_location]
    header_template = imheaderlist[left_location]
    if(len(sys.argv)!=4): bpheader_template = bpheaderlist[left_location]
    # Update the reference pixel keywords. Only the y reference must be changed.
    min_y_offset = min(offsets[1])
    if y_offset == min_y_offset:
        y_ref = 0
    elif y_offset >= 0:
        y_ref = abs(min_y_offset)+y_offset
    else:
        y_ref = abs(min_y_offset)-abs(y_offset)
    y_pos = y_ref + header_template['CRPIX2']
    header_template['CRPIX2'] = y_pos
    # Write the complete FITS file with data and header.
    hdu = pf.PrimaryHDU(aim[0], header_template)
    hdulist = pf.HDUList([hdu])
    if (len(sys.argv)==7): hdulist[0].header['BPM']=bpmfilename
    hdulist.writeto(filename, clobber=True)
    if(len(sys.argv)!=4):
        # write the BPM
        hdu2 = pf.PrimaryHDU(aim[1])#, bpheader_template)
        hdulist2 = pf.HDUList([hdu2])
        # RH: fix some very odd 'features' of pyfits, so that the output BPM/quality file is readable
        #hdulist2[0].scale(type='int32')
        #hdulist2[0].header['EXTEND']=
        hdulist2[0].update_header()
        hdulist2.writeto(bpmfilename, clobber=True)
        print 'Run time', time.time()-start_time, 'seconds'
    sys.exit(0)
