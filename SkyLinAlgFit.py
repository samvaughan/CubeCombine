import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits

import pdb
import sys
import continuum
from spectools import spectrum

import argparse




def skyfit_rows(data, a, b, r, clip_edges=True):
    d1,d2,d3= data.shape

    median=np.median(data, axis=0)


    y,x = np.ogrid[-b:d2-b, -a:d3-a]
    mask = x*x + y*y <= r*r

    if clip_edges==True:

        median[:, :4]=0
        median[:, -4:]=0


    #median[~mask]=0



    #plt.imshow(median)
    #plt.show()

    median=median.ravel()

    ones=np.ones(d3)
    continuum=np.zeros([d2, d2*d3])
    for k in range(d2):

        continuum[k, k*d3:(k+1)*d3]=ones

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

        """
        if i >100:

            residuals=obj-(c[0]*model.T[:,0]+c[1]*model.T[:,1])
            residuals=residuals.reshape(d2, d3)
            plt.imshow(residuals)
            plt.colorbar()
            plt.show()
        """


        c_vals[:, i]=c
        
    print "\n"
    return c_vals


def skyfit_fullimage(data, clip_edges=True):


    """
    Fit a galaxy and sky model independently to a sky-subtracted cube. The aim is to recover the sky continuum, fit a polynomial to it and subtract it off from the O-S cube.

    The sky model is an array of 1s, in the same shape as each slice of the data cube. The galaxy model is the mean of the object cube across the wavelength direction

    """


    d1,d2,d3= data.shape

    median=np.median(data, axis=0)


    

    if clip_edges==True:

        median[:, :4]=0
        median[:, -4:]=0


    #median[~mask]=0



    #plt.imshow(median)
    #plt.show()

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
    return c_vals


parser = argparse.ArgumentParser(description='Fit a sky and galaxy model to an O-S cube, then fit a polynomial to the sky residuals. Subtract this continuum from each O-S cube and save')
parser.add_argument('GalName', type=str, help='Should be IC843 or NGC1277')

args = parser.parse_args()

GalName=args.GalName


if GalName=="NGC1277":
    data_list=["darc_ms047_048.fits", "darc_ms049_048.fits", "darc_ms050_051.fits", "darc_ms052_051.fits", "darc_ms056_057.fits", "darc_ms058_057.fits", "darc_ms059_060.fits"]
    cubepath="/Volumes/SPV_SWIFT/Science/SkyContinuum_LinAlg/NGC1277"
    z_gal=0.017044


elif GalName=="IC843":
    data_list=["darc_ms065_066.fits","darc_ms067_066.fits","darc_ms071_072.fits","darc_ms073_072.fits","darc_ms074_075.fits","darc_ms076_075.fits","darc_ms080_081.fits","darc_ms082_081.fits","darc_ms083_084.fits"]
    cubepath="/Volumes/SPV_SWIFT/Science/Bootstrap_CubeCombine/Cubes/IC843/"

elif GalName=="GMP4928":
    data_list=["ms059_060.fits", "ms061_060.fits", "ms064_065.fits",  "ms066_065.fits"]
    cubepath="./GMP4928"
else:
    raise NameError("Please Specify a Galaxy")



final_results_rows=np.zeros([len(data_list), 45, 4112])
final_results_fullimage=np.zeros([len(data_list), 2, 4112])


continuum_fits=np.zeros([len(data_list), 4112])
lamdas=np.arange(6300, 6300+4112)


telluric=fits.open("corrected_telluric.fits")[0].data

if GalName=="NGC277":
    fig1, axs1=plt.subplots(nrows=2, ncols=4)
    fig2, axs2=plt.subplots(nrows=2, ncols=4)
elif GalName=="IC843":
    fig1, axs1=plt.subplots(nrows=2, ncols=5)
    fig2, axs2=plt.subplots(nrows=2, ncols=5)

from matplotlib.ticker import AutoMinorLocator

for i, (name, ax1, ax2) in enumerate(zip(data_list, axs1.flatten(), axs2.flatten())):
    data=fits.open("{}/{}".format(cubepath, name))[0].data
    header=fits.open("{}/{}".format(cubepath, name))[0].header

    d1,d2,d3=data.shape


    print "\n\nFitting cube {}\n\n".format(i+1)



    #final_results_rows[i, :, :]=skyfit_rows(data, a, b, r)
    final_results_fullimage[i, :, :]=skyfit_fullimage(data)


   

    
    #Find the gradient of the sky array. Skylines correspond to sharp spikes in the gradient, so we can get rid of them by ignoring pixels with a large dx.
    dx=np.gradient(final_results_fullimage[i, 1, :])
    
    fitlocs=(np.abs(dx) < 0.5)

    #Use the continuum fit function to fit polynomials, avoiding the skylines. 
    continuum_fits[i, :]=continuum.fit_continuum(lamdas, final_results_fullimage[i,1,:], np.ones_like(final_results_fullimage[i,1,:]), [2,0.5, 0.3], 7, [6300, 10412], plot=False, fitloc=fitlocs)



    ax1.plot(lamdas, final_results_fullimage[i, 1, :], c="k")
    ax1.plot(lamdas, continuum_fits[i, :], c="r", linewidth=2.0)
    ax1.plot(lamdas, continuum_fits[i, :]-np.median(continuum_fits[i, :]), c="b", linewidth=2.0)
    ax1.set_xlim([6300,10412])
    ax1.set_xlabel("Wavelength")
    ax1.set_ylim([-20,35])
    ax1.set_ylabel("Counts")
    ax1.set_title("{}".format(name))

    
    minorLocator = AutoMinorLocator()
    ax1.yaxis.set_minor_locator(minorLocator)

    #make a cube out of the sky residuals, with the continuum subtracted,
    sky_cont_cube=np.repeat(final_results_fullimage[i,1,:]-continuum_fits[i, :], d2*d3).reshape((d1,d2,d3))
    #hdu=fits.PrimaryHDU(data-sky_cont_cube)

    
    cubename="{0}/sky_residuals_subtracted_{1}".format(GalName, name)
    print "Saving Cube {} as {}".format(i+1, cubename)
    fits.writeto(cubename, data-sky_cont_cube, header=header, clobber=True)
    
    


pdb.set_trace()

