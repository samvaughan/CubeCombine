import numpy as np 
from astropy.io import fits 
import continuum
import pdb
import sys
import matplotlib.pyplot as plt
import combine

def iround(num):
    return np.int(np.round(num))

def measure_aperture_counts(image, x_0, y_0, r=10):

    d2, d3=image.shape
    #Making the circular Aperture
    y,x = np.ogrid[-y_0:d2-y_0, -x_0:d3-x_0]
    mask = x*x + y*y <= r*r

    masked_image=image*mask
    return np.sum(masked_image)


def _skyfit_fullimage(data, clip_edges=True):


    """
    Fit a galaxy and sky model independently to a sky-subtracted cube. The aim is to recover the sky residuals, fit a polynomial to the continuum and subtract it off to make a sky residual cube.

    The sky model is an array of 1s, in the same shape as each slice of the data cube. The galaxy model is the mean of the object cube across the wavelength direction.

    """


    d1,d2,d3= data.shape

    median=np.median(data, axis=0)
    

    if clip_edges==True:

        median[:, :4]=0
        median[:, -4:]=0

    median=median.ravel()

    ones=np.ones(d2*d3)
    continuum=np.zeros([1, d2*d3])
    continuum[0,:]=ones

    model=np.column_stack((median,continuum.T))


    c_vals=np.zeros([1+continuum.shape[0], d1])
    sigma_vals=np.zeros_like(c_vals)



    for i in range(d1):
        sys.stdout.write('\r')
        
        sys.stdout.write("Fitting Slice {}".format(i))
        sys.stdout.flush()


        obj=data[i,:,:].ravel()

        c, resid, rank, sigma=np.linalg.lstsq(model, obj)

        #rint c

        
        c_vals[:, i]=c
        
    print "\n"
    return median.reshape(d2,d3), c_vals[0, :], c_vals[1, :]


def fix_skylines(lamdas, cube, clipping_iterations=2, low_sig=0.3, high_sig=0.5, poly_order=7):

    """Return a sky residual subtracted cube, where we've found the sky residuals using the linear algebra sky fitting fitting code"""

    d1,d2,d3=np.shape(cube)

    image, spec, sky_spec=_skyfit_fullimage(cube, clip_edges=True)

    dx=np.gradient(sky_spec)    
    fitlocs=(np.abs(dx) < 0.5)
    #Use the continuum fit function to fit polynomials, avoiding the skylines. 

    
    sky_continuum=continuum.fit_continuum(lamdas, sky_spec, np.ones_like(sky_spec), [clipping_iterations,high_sig, low_sig], poly_order, [lamdas[0], lamdas[-1]], plot=False, fitloc=fitlocs)

    sky_cube=np.repeat(sky_spec-sky_continuum, d2*d3).reshape((d1,d2,d3))


    return image, spec, cube-sky_cube


def linalg_fitting_measure_counts(cubes, xoffsets, yoffsets, subtract_residuals=True):


    #Make arrays to store the galaxy median images and spectra
    counts=np.zeros(len(cubes))
    gal_images=np.zeros([d2, d3, len(cubes)])
    gal_spectra=np.zeros([d1, len(cubes)])



    for i, (cubename, xoffset, yoffset) in enumerate(zip(cubes, xoffsets, yoffsets)):

        print "\nCube {} of {}\n".format(i+1, len(cubes))

        cube_h=fits.open(cubename)
        cube=cube_h[0].data

        image, spec, sky_residual_subtracted=fix_skylines(lamdas, cube, clipping_iterations=2, low_sig=0.3, high_sig=0.5, poly_order=poly_order)
        
        gal_spectra[:, i]=spec

        if subtract_residuals:
            datalist[:, :, :, i]=sky_residual_subtracted
        else:
            datalist[:, :, :, i]=cube



        
        x_0=iround(d3/2.0-iround(xoffset))
        y_0=iround(d2/2.0-iround(yoffset))

       

        counts[i]=measure_aperture_counts(image, x_0, y_0)

    return datalist, counts, gal_spectra

def fit_spectral_fluctations(lamdas, gal_spectra, average_gal_spec, order=2):

    continuum_fits=np.zeros_like(gal_spectra)

    for i, spec in enumerate(gal_spectra.T):
        gal_spectra[:, i]=average_gal_spec/spec

        #Use the continuum fit function to fit polynomials
        #Don't do any sigma clipping, so make the low and high sigmas very large (100)
        continuum_fits[:, i]=continuum.fit_continuum(lamdas, gal_spectra[:, i], np.ones_like(gal_spectra[:, i]), [1,100, 100], order, [lamdas[0], lamdas[-1]], plot=False)

    return continuum_fits

if __name__=="__main__":

    

    import argparse
    import time
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Fit a sky and galaxy model to an O-S cube, then fit a polynomial to the sky residuals. Subtract this continuum from each O-S cube and save')
    parser.add_argument('CubeListFile', type=str, help='A text file with one galaxy cube per line')
    parser.add_argument('OffsetFile', type=str, help='A text file with one pair of offsets per line')
    parser.add_argument("outfilename", type=str, help="Output filename for the combined cube")
    parser.add_argument("--BPM", nargs=3, help="Use the BPM mask. Order of arguments should be BPMfilelist.txt, BPM threshld (usually 0.01) and the filename of the BPM out cube")



    args=parser.parse_args()
    cubelistfile=args.CubeListFile
    offsetfile=args.OffsetFile
    filename=args.outfilename
    if args.BPM is None:
        print "Running without using any BPM cubes"
        bpmthresh=0.0
    else:
        print "Running using BPM cubes"        
        BPMlistfile=args.BPM[0] 
        bpmthresh=float(args.BPM[1])   
        bpmfilename=args.BPM[2]

    cubes=np.genfromtxt(cubelistfile, dtype=str)
    bpms=np.genfromtxt(BPMlistfile, dtype=str)

    xoffsets, yoffsets=np.genfromtxt(offsetfile, dtype=float, unpack=True)
    offsets=(xoffsets,yoffsets)





    if not cubes.shape==xoffsets.shape==yoffsets.shape:
        raise ValueError("We must have the same number of cubes and x/y offsets")

    #Load the first cube to get the shape
    data=fits.open(cubes[0])[0].data
    #Assume the first axis is the wavelength one
    d1,d2,d3=data.shape

    #Order of the polynomial fitting
    poly_order=7

    #Should make this lamda array properly...
    lamdas=np.arange(6300, 6300+d1)


    #The list to store the actual data
    datalist=np.ones([d1,d2,d3,len(cubes)])


   
    #An array to store the total counts from each cube
    



    #Do the linear algebra fitting, and subtract the skyline residuals if desired
    datalist, counts, gal_spectra=linalg_fitting_measure_counts(cubes, xoffsets, yoffsets, subtract_residuals=True)

        


    average_gal_counts=np.mean(counts)
    average_gal_spec=np.mean(gal_spectra, axis=1)



    #Fit polynomials to the continuua
    continuum_fits=fit_spectral_fluctations(lamdas, gal_spectra, average_gal_spec, order=2)

    #Scale each datacube by the correct value and polynomial
    for i, c_fit in enumerate(continuum_fits.T):
        counts_scaling_factors=average_gal_counts/counts[i]
        #spectral_scaling_factor=average_gal_spec_continuum_fit/c_fit
        spectral_scaling_cube=np.repeat(c_fit, d2*d3).reshape(d1,d2,d3)
        datalist[:, :, :, i]*=counts_scaling_factors*spectral_scaling_cube



    #Do the actual cube combine
    #Get the cubes into the correct list format that cubecombine expects
    imagelist=list(np.rollaxis(datalist, 3, 0))
    imheaderlist = []
    bpheaderlist = []

    for cube in cubes:        
        imheaderlist.append((fits.open(cube))[0].header)

    # Open bpms
    bpmlist=[]
    for file in bpms:
        bpmlist.append((fits.open(file))[0].data)
        bpheaderlist.append((fits.open(file))[0].header)

    print "Combining cubes with offsets from {}".format(offsetfile)
    # Create a new FITS file for the combined image.
    aim = combine.image_combine(imagelist, offsets, bpmthresh, bpmlist, median=False)

    # Use the leftmost image's header
    min_x_offset = min(offsets[0])
    locations = np.where(offsets[0]==min_x_offset)
    left_location = locations[0][0]
    y_offset = offsets[1][left_location]
    header_template = imheaderlist[left_location]

    if args.BPM is not None:
        bpheader_template = bpheaderlist[left_location]

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
    hdu = fits.PrimaryHDU(aim[0], header_template)
    hdulist = fits.HDUList([hdu])

    print "Writing Combined cube to {}".format(filename)
    hdulist.writeto(filename, clobber=True)

    if args.BPM is not None:
        hdulist[0].header['BPM']=bpmfilename
        # write the BPM
        hdu2 = fits.PrimaryHDU(aim[1])#, bpheader_template)
        hdulist2 = fits.HDUList([hdu2])
        # RH: fix some very odd 'features' of pyfits, so that the output BPM/quality file is readable
        #hdulist2[0].scale(type='int32')
        #hdulist2[0].header['EXTEND']=
        hdulist2[0].update_header()
        print "Writing BPM cube to {}".format(bpmfilename)
        hdulist2.writeto(bpmfilename, clobber=True)
    
    print 'Run time:', time.time()-start_time, 'seconds'
    


    sys.exit(0)




