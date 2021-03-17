# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:40:01 2017

@author: Thomas Pingel
"""

'''
TODO:
Create a routine to build 3D models.
Swiss shading
More interpolators
'''

# For development of packages, see:
# http://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html
# https://stackoverflow.com/questions/34753206/init-py-cant-find-local-modules
# https://github.com/BillMills/pythonPackageLesson
# https://biodata-club.github.io/lessons/python/packages/lesson/
# http://python-notes.curiousefficiency.org/en/latest/python_concepts/import_traps.html





#%%
import os
import inspect
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import scipy.ndimage as ndi
from scipy import stats
from scipy import sparse
from scipy import linalg
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from scipy import interpolate
from PIL import Image
from skimage.util import apply_parallel
from skimage.morphology import disk
#import cv2
import imageio

import pyproj

import geopandas
import datetime

from pyproj import Transformer

import piexif

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Global variable to help load data files (PNG-based color tables, etc.)
neilpy_dir = os.path.dirname(inspect.stack()[0][1])



#%% Coordinate transformation

'''
EPSG Hints:
Google searches that hit epsg.io are the fastest:
Google: "wgs84 zone 59N" hits
https://epsg.io/32759

wgs84 = 4326

WGS84 UTM is 326xx or 327xx (e.g., zone 17 is 32617; lookup with epsg.io)
'''
def coord_transform(x,y,from_epsg,to_epsg):
    transformer = Transformer.from_crs(from_epsg,to_epsg,always_xy=True)
    return transformer.transform(x,y)

#%% Reading data - a handy wrapper to spare some pain

def imread(fn, return_metadata=True, fix_nodata=False, force_float=False):

    with rasterio.open(fn) as src:
        metadata = src.profile
        #metadata['count'] = src.count
        #metadata['crs'] = src.crs
        #metadata['transform'] = src.transform
        metadata['bounds'] = src.bounds
        #metadata['width'] = src.width
        #metadata['height'] = src.height
        metadata['dtype'] = src.dtypes[0]  # This returns an array.  Safe to assume all the same?
        #metadata['nodata_value'] = src.nodatavals[0]
        
        
        if src.count > 1:
            X = np.stack([src.read(i+1) for i in range(src.count)],axis=2)
        else:
            X = src.read(1)
    
    # If asked to force into float, and not already in float, convert.
    if force_float and metadata['dtype'] not in ['float32','float64']:
        X = X.astype(np.float32)
        metadata['dtype'] = 'float32'
    
    # Fix nodata unless told otherwise
    if fix_nodata:
        if metadata['dtype'] in ['float32','float64']:
            X[X==metadata['nodata']] = np.nan
        else:
            print('Warning: fix_nodata requested, but ' + metadata['dtype'] + 
                  ' cannot be converted to np.nan.')
        
    # Calculate cellsize.  If directions are within a tolerance, assume the
    # mean
    cellsizes = np.abs(np.array((metadata['transform'][0],
                                 metadata['transform'][4])))
    if np.diff(cellsizes) < .00000001:
        metadata['cellsize'] = np.mean(cellsizes)
    else:
        metadata['cellsize'] = cellsizes
        
    if return_metadata:
        return X, metadata
    else:
        return X
    
    
#%%

# TODO: verify multi-layer write and colormap write

def imwrite(fn,im,metadata=None,colormap=None):
    if metadata is None:
        imageio.imwrite(fn,im)
    else:
        metadata['dtype'] = im.dtype
        with rasterio.open(fn, 'w', **metadata) as dst:
            if np.ndim(im)==2:
                dst.write(im, 1)
                if colormap is not None:
                    dst.write_colormap(1,colormap)
            else:
                bands = np.min(np.shape(im))
                metadata['count'] = bands
                for i in range(bands):
                    sm_dim = np.argsort(np.shape(im))[0]  # smallest dimension; some rasters are 3xmxn, some are mxnx3
                    if sm_dim==0:
                        dst.write(im[i,:,:],i+1)
                    else:
                        dst.write(im[:,:,i],i+1)


#%% Spatial Autocorrelation Functions

'''
Simple formula to calculate Getis-Ord Gi when given an array of values
and the pre-calculate n, global mean, and global variance. You probably don't
want to use this function.
'''
    
def gi_formula(x,n,m,v):

    k = np.sum(np.isfinite(x)).astype(np.int) # number of non-nan neighbors
    Gi =(np.nansum(x) - k*m) / np.sqrt((k * (n-1-k) * v) / (n-2))
    return Gi

def gistar_formula(x,n,m,v):
    k = np.sum(np.isfinite(x)).astype(np.int) # number of non-nan neighbors
    Gi =(np.nansum(x) - k*m) / np.sqrt((k * (n-k) * v) / (n-1))
    return Gi


'''
Calculated Getis-Ord Gi Statistic of local autocorrelation on a raster.
For vector-based operations, see the package PySAL.

The user can supply either a binary footprint (structuring element) or
can supply a scalar value to indicate a size of structuring element.  The
scalar is assumed to be a radius (i.e., if 1 if provided, the structuring
                                  element will be 3x3 (2x1+1), if 2 is 
                                  provided, then 5 (2x2+1), etc.)

The "mode" argument gets passed directly to ndimage's generic_filter, and is 
used to handle how edge cases work.  Nearest is almost always what you want, 
but "mirror" and "wrap" are interesting and could be used in very special 
circumstances.

References
----------
Ord, J.K. and A. Getis. 1995. Local Spatial Autocorrelation Statistics:
Distribution Issues and an Application. Geographical Analysis, 27(4): 286-
306. doi: 10.1111/j.1538-4632.1995.tb00912.x

https://www.researchgate.net/post/What_is_the_difference_in_interpretation_of_results_between_Local_Morans_I_and_Getis_Ord_G

https://community.esri.com/t5/arcgis-streetmap-premium/differences-between-local-spatial-statistics-results/td-p/358062#:~:text=In%20other%20words%2C%20the%20local,including%20the%20one%20in%20question.&text=Alternatively%2C%20it%20makes%20sense%20that,High%20surrounded%20by%20Low%20values.

https://www.youtube.com/watch?v=urfsjGo-XXc

https://www.youtube.com/watch?v=_0Tzo1qbN-A

'''



def rasterGi(X,footprint=1,mode='nearest',apply_correction=False,star=False):

    # Cast to a float; these operations won't all work on integers
    X = X.astype(np.float)

    # If a footprint was provided as a size, assume this is a radius (a change
    # starting in neilpy v.0.17, prior to which is was used directly as the 
    # size) and make a SQUARE structuring element.  Otherwise, assume it is 
    # a footprint/structuring element, and calculate whether this is star or 
    # not.  This will override a supplied value of star!
    # If you want a circular element, use a disk(radius)!
    if np.isscalar(footprint):
        m = footprint # This becomes the center pixel
        footprint = 2 * footprint + 1  # now a diameter
        footprint = np.ones((footprint,footprint),dtype=np.int)
        
        # Gi* includes the center value, Gi does not.
        if not star:
            footprint[m,m] = 0
    else:
        m = np.floor(np.shape(footprint)[0] / 2).astype(int)
        if footprint[m,m] == 0:
            star = False
        else:
            star = True
        
    # How many non-nans do we have in the array?
    n = np.sum(np.isfinite(X))

    # A vectorized operation to calculate the global mean and variance at each 
    # pixel.  For Gi, this needs to exclude each pixel's own value.  For Gi*
    # it is the global mean.  This could be a scalar value, but keeping it an 
    # array for consistency, and taking a small hit on memory performance.  
    # TODO: We should reexamine that choice.
    if star==False:
        global_mean = (np.nansum(X) - X) / (n-1)
        global_var = ((np.nansum(X**2) - X**2) / (n-1)) - global_mean**2
        global_mean[np.isnan(X)] = np.nan
        global_var[np.isnan(X)] = np.nan        
    else:
        global_mean = np.nanmean(X) 
        global_var = np.nanstd(X)**2

    # Within the strucutring element how many neighbors at each point?
    if np.all(np.isfinite(X)):
        w_neighbors = np.sum(footprint) * np.ones(np.shape(X),dtype=np.int)
    else:
        w_neighbors = ndi.filters.generic_filter(np.isfinite(X).astype(np.int),np.sum,footprint=footprint,mode=mode)
        w_neighbors = w_neighbors.astype(np.float)
        w_neighbors[np.isnan(X)] = np.nan

    # Calculate the numerator of Gi using a generic filter
    a = ndi.filters.generic_filter(X,np.nansum,footprint=footprint,mode=mode) - w_neighbors* global_mean
    # Different calculations for denominator, for Gi and Gi*
    if star:
        b = np.sqrt((w_neighbors / (n-1)) * (n-w_neighbors) * global_var)
    else:
        b = np.sqrt((w_neighbors / (n-2)) * (n-1-w_neighbors) * global_var)
    del global_mean, global_var
    Z = a / b
    del a,b
    
    Z[np.isnan(X)] = np.nan
    
    if apply_correction == True:
        Z = (Z-np.nanmean(Z)) / np.nanstd(Z)
        
    # P = 2 * (1 - scipy.special.ndtr(Z)) OR
    P = stats.norm.sf(abs(Z))*2    
    
    # Calculate Z-scores for CIs of 10, 5, and 1 percent (adjust for tails)
    #a = stats.norm.ppf(.95)
    #b = stats.norm.ppf(.975)
    #c = stats.norm.ppf(.995)
    
    # Create an ArcGIS-like Gi_Bin indicating CIs (90/95/99) for above-and-below
    sig_bin = np.zeros_like(X,dtype=np.float)
    np.seterr(divide='ignore', invalid='ignore')
    sig_bin[P<.1] = 1
    sig_bin[P<.05] = 2
    sig_bin[P<.01] = 3
    sig_bin[Z<0] = - sig_bin[Z<0]
    sig_bin[P>=.1] = 0
    np.seterr(divide='warn', invalid='warn')   
    sig_bin[np.isnan(X)] = np.nan
     
    
    # Return the z-score, the p-value, and the significance bin
    return Z, P, sig_bin


#%% Raster visualization functions
    
# http://edndoc.esri.com/arcobjects/9.2/net/shared/geoprocessing/spatial_analyst_tools/how_hillshade_works.htm
# esri_slope is intended to be a perfect mimic of ESRI's published slope 
# calculation routine.  This uses a generic filter to process the image, which
# is something of a slow, if intuitive approach.  It would be a lot faster
# some parallel processing added.  One could expant this to include an ESRI
# aspect calculation as well, though in practice I use the two routines
# immediately below.

def esri_slope(Z,cellsize=1,z_factor=1,return_as='degrees'):    
    def slope_filter(n):
        n = n.reshape((3,3)) # Added to accommodate filter, not strictly necessary.
        # Plus, this would be better if it factored in nans better.
        dz_dx = (np.sum(n[:,-1] * (1,2,1)) - np.sum(n[:,0] * (1,2,1))) / 8
        dz_dy = (np.sum(n[-1,:] * (1,2,1)) - np.sum(n[0,:] * (1,2,1))) / 8
        return np.sqrt(dz_dx**2 + dz_dy**2)
        
    S = ndi.filters.generic_filter(Z,slope_filter,size=3,mode='reflect')
    if cellsize != 1:
        S = S / cellsize
    if z_factor != 1:
        S = z_factor * S
    if return_as=='degrees':
        S = np.rad2deg(np.arctan(S))
    return S
        


# This is a more efficient method of calculating slope using numpy's gradient 
# routine.  Percent slope is the default, and will return a value where 1 is a
# 100 percent slope.
def slope(Z,cellsize=1,z_factor=1,return_as='degrees'):
    if return_as not in ['degrees','radians','percent']:
        print('return_as',return_as,'is not supported.')
    else:
        gy,gx = np.gradient(Z,cellsize/z_factor)
        S = np.sqrt(gx**2 + gy**2)
        if return_as=='degrees' or return_as=='radians':
            S = np.arctan(S)
            if return_as=='degrees':
                S = np.rad2deg(S)
    return S

        
# Similarly this will calculate the aspect using numpy's gradient, either
# in degrees, or radians.
def aspect(Z,return_as='degrees',flat_as='nan'):
    if return_as not in ['degrees','radians']:
        print('return_as',return_as,'is not supported.')
    else:
        gy,gx = np.gradient(Z)
        A = np.arctan2(gy,-gx) 
        A = np.pi/2 - A
        A[A<0] = A[A<0] + 2*np.pi
        if return_as=='degrees':
            A = np.rad2deg(A)
        if flat_as == 'nan':
            flat_as = np.nan
        A[(gx==0) & (gy==0)] = flat_as
        return A
    
#%%
def curvature(X,cellsize=1):
    return -100*ndi.filters.laplace(X/cellsize)

#%%     
'''
esri_curvature is intended to be a perfect copy of ESRI's planar curvature
calculations.  ESRI's calculation method is identical to Zevenbergen and
Throne (1987) except signs are reversed, and the output is mulitplied by 100x.

PROFILE curvature given here is equivalent to other authors' LONGITUDINAL 
curvature, and refers to the rate of change of potential slope down a flow 
line.

PLAN curvature here is the same as other authors' CROSS SECTIONAL curvature, 
and refers to the rate of change of aspect along a contour.

Missing values are returned with a curvature of zero.  It is recommended that
these values be filled beforehand if this behavior is not desired.  Pixels
with neighboring pixels undefined will have those missing values filled with
the center pixel value, as is ESRI's convention.  This same behavior is used
to fill necessary edge pixels as well.
    
The reference to plan and profile curvature refer to the rate of aspect and 
slope 

It returns:
    general curvature, profile curvature, and plan curvature

References
https://dx.doi.org/10.1002/esp.3290120107
https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm
https://support.esri.com/en/technical-article/000005086
'''
def esri_curvature(X,cellsize=1):

    # Match common definition (Wood, ESRI, etc.) of cell size as L
    L = cellsize
    
    # Cells are labelled from top-to-bottom, left-to-right, starting in the
    # upper left corner, and ending in the lower right corner.  Z5 is the 
    # original array, so we simply use X in place of Z5.
    Z1 = ashift(X,0)
    Z2 = ashift(X,1)
    Z3 = ashift(X,2)
    Z4 = ashift(X,7)
    Z6 = ashift(X,3)
    Z7 = ashift(X,6)
    Z8 = ashift(X,5)
    Z9 = ashift(X,4)

    # In cases where data are missing, ESRI uses the original center pixel.
    Z1[np.isnan(Z1)] = X[np.isnan(Z1)]
    Z2[np.isnan(Z2)] = X[np.isnan(Z2)]
    Z3[np.isnan(Z3)] = X[np.isnan(Z3)]
    Z4[np.isnan(Z4)] = X[np.isnan(Z4)]
    Z6[np.isnan(Z6)] = X[np.isnan(Z6)]
    Z7[np.isnan(Z7)] = X[np.isnan(Z7)]
    Z8[np.isnan(Z8)] = X[np.isnan(Z8)]
    Z9[np.isnan(Z9)] = X[np.isnan(Z9)]
    
    # Parameters given in Zevenburgen and Thorne (1987)
    # Values A to C are given here, but are not computed, 
    # as they are not used in the curvature calculations.
    # A = ((Z1 + Z3 + Z7 + Z9)/4 - (Z2 + Z4 + Z6 + Z8)/2 + X)/(L**4);
    # B = ((Z1 + Z3 - Z7 + Z9)/4 - (Z2 - Z8)/2)/(L**3);
    # C = ((-Z1 + Z3 - Z7 + Z9)/4 + (Z4 - Z6)/2) / (L**3);
    D = (((Z4 + Z6) / 2) - X) / (L**2)       # Zxx                    
    E = (((Z2 + Z8) / 2) - X) / (L**2)       # Zyy                     
    F = (-Z1 + Z3 + Z7 - Z9) / (4*(L**2))    # Zxy
    G = (-Z4 + Z6) / (2*L)                   # Zx
    H = (Z2 - Z8) / (2*L)                    # Zy

    del Z1,Z2,Z3,Z4,Z6,Z7,Z8,Z9

    K = -200 * (D + E)

    np.seterr(divide='ignore', invalid='ignore')
    
    K_plan = 200*(D*H**2 + E*G**2 - F*G*H) / (G**2 + H**2)
    K_plan[np.isnan(K_plan)] = 0;

    K_profile = -200 * (D*G**2 + E*H**2 + F*G*H) / (G**2 + H**2)
    K_profile[np.isnan(K_profile)] = 0;
    
    np.seterr(divide='warn', invalid='warn')

    
    return K, K_plan, K_profile

#%%
'''
zevenbergen_and_thorne is similar to ESRI's curvature, but with more
curvatures calculated, and scaling consistent with Z&T's original forumulation.

Returns general, profile (geometric), plan (geometric), tangential, 
longitudinal (Z&T's original and ESRI's PROFILE), and cross-sectional 
(Z&T's original and ESRI's PLAN)

K, K_profile, K_plan, K_tan, K_long, K_cross

Outputs should be the same as SAGA, except SAGA's known sign inversion for 
cross-sectional.  However, more work needs to be done to see how missing values
should be computed.  

References
https://dx.doi.org/10.1002/esp.3290120107
https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm
https://support.esri.com/en/technical-article/000005086
'''
def zevenbergen_and_thorne_curvature(X,cellsize=1):

    # Match common definition (Wood, ESRI, etc.) of cell size as L
    L = cellsize
    
    # Cells are labelled from top-to-bottom, left-to-right, starting in the
    # upper left corner, and ending in the lower right corner.  Z5 is the 
    # original array, so we simply use X in place of Z5.
    Z1 = ashift(X,0)
    Z2 = ashift(X,1)
    Z3 = ashift(X,2)
    Z4 = ashift(X,7)
    Z6 = ashift(X,3)
    Z7 = ashift(X,6)
    Z8 = ashift(X,5)
    Z9 = ashift(X,4)

    # Missing values are handled via a "finite differences" approach, given
    # by Wilson and Gallant on page 53, equation 3.8.
    idx = np.isnan(Z1); Z1[idx] = 2 * X[idx] - Z9[idx]
    idx = np.isnan(Z2); Z2[idx] = 2 * X[idx] - Z8[idx]
    idx = np.isnan(Z3); Z3[idx] = 2 * X[idx] - Z7[idx]
    idx = np.isnan(Z4); Z4[idx] = 2 * X[idx] - Z6[idx]
    idx = np.isnan(Z6); Z6[idx] = 2 * X[idx] - Z4[idx]
    idx = np.isnan(Z7); Z7[idx] = 2 * X[idx] - Z3[idx]
    idx = np.isnan(Z8); Z8[idx] = 2 * X[idx] - Z2[idx]        
    idx = np.isnan(Z9); Z9[idx] = 2 * X[idx] - Z1[idx]

    # Parameters given in Zevenburgen and Thorne (1987)
    A = ((Z1 + Z3 + Z7 + Z9)/4 - (Z2 + Z4 + Z6 + Z8)/2 + X)/(L**4);
    B = ((Z1 + Z3 - Z7 + Z9)/4 - (Z2 - Z8)/2)/(L**3);
    C = ((-Z1 + Z3 - Z7 + Z9)/4 + (Z4 - Z6)/2) / (L**3);
    D = (((Z4 + Z6) / 2) - X) / (L**2)       # Zxx                    
    E = (((Z2 + Z8) / 2) - X) / (L**2)       # Zyy                     
    F = (-Z1 + Z3 + Z7 - Z9) / (4*(L**2))    # Zxy
    G = (-Z4 + Z6) / (2*L)                   # Zx
    H = (Z2 - Z8) / (2*L)                    # Zy
    
    # These are used so often they're precomputed, as in Wilson and Gallant
    P = G**2 + H**2                          # Zx**2 + Zy**2
    Q = G**2 + H**2 + 1                      # Zx**2 + Zy**2 + 1

    del Z1,Z2,Z3,Z4,Z6,Z7,Z8,Z9

    K = 2 * (D + E)

    # Cross Sectional is PLAN in Z&T's original forumulation, but is commonly
    # known now in other domains (other than ESRI) this way (Wood, Schmidt, 
    # SAGA, etc.).  Note, SAGA incorrectly returns the sign of this as 
    # reversed.
    np.seterr(divide='ignore', invalid='ignore')
    K_cross = 2*(D*H**2 + E*G**2 - F*G*H) / P
    K_cross[np.isnan(K_cross)] = 0;

    # LONGITUDINAL is PROFILE in Z&T's original forumulation.
    K_long = -2 * (D*G**2 + E*H**2 + F*G*H) / P
    K_long[np.isnan(K_long)] = 0;
    
    # See Krcho (1991) quoted in Schmidt et al. (2003, Table 1) and 
    # Minar et al. (2020, Table 3)
    K_tan = -(D*H**2 -2*F*G*H + E*G**2) / (P*Q**.5)

    # Geometric Profile Curvature (Minar et al. 2020, Table 3)
    K_profile = (D*G**2 + 2*F*G*H + E*H**2) / (P*Q**1.5)
    
    # Geometric Plan Curvature (Minar et al. 2020, Table 3)
    K_plan = -(D*E**2 - 2*F*G*H + E*G**2) / (P**1.5)
    
    np.seterr(divide='warn', invalid='warn')
    
    
    return K, K_profile, K_plan, K_tan, K_long, K_cross

#%%

def evans_curvature(X,cellsize=1):

    # Match common definition (Wood, ESRI, etc.) of cell size as L
    L = cellsize
    
    z1 = ashift(X,0)
    z2 = ashift(X,1)
    z3 = ashift(X,2)
    z4 = ashift(X,7)
    z6 = ashift(X,3)
    z7 = ashift(X,6)
    z8 = ashift(X,5)
    z9 = ashift(X,4)
    
    # Missing values are handled via a "finite differences" approach, given
    # by Wilson and Gallant on page 53, equation 3.8.
    idx = np.isnan(z1); z1[idx] = 2 * X[idx] - z9[idx]
    idx = np.isnan(z2); z2[idx] = 2 * X[idx] - z8[idx]
    idx = np.isnan(z3); z3[idx] = 2 * X[idx] - z7[idx]
    idx = np.isnan(z4); z4[idx] = 2 * X[idx] - z6[idx]
    idx = np.isnan(z6); z6[idx] = 2 * X[idx] - z4[idx]
    idx = np.isnan(z7); z7[idx] = 2 * X[idx] - z3[idx]
    idx = np.isnan(z8); z8[idx] = 2 * X[idx] - z2[idx]        
    idx = np.isnan(z9); z9[idx] = 2 * X[idx] - z1[idx]
    

    # From Wood (1991), pages 91 and 92
    A = (z1 + z3 + z4 + z6 + z7 + z9)/(6*L**2) - (z2+X+z8)/(3*L**2)    # Fxx
    B = (z1  + z2 + z3 + z7 + z8 + z9)/(6*L**2) - (z4+X+z6)/(3*L**2)   # Fyy
    C = (z3 + z7 - z1 -z9) / (4*L**2)                                  # Fxy
    D = (z3+z6+z9-z1-z4-z7) / (6*L)                                    # Fx
    E = (z1+z2+z3-z7-z8-z9)/(6*L)                                      # Fy
    F = (2*(z2+z4+z6+z8)-(z1+z3+z7+z9)+5*X) / 9
    
    del z1,z2,z3,z4,z6,z7,z8,z9

    # From Wood, page 85-87; lon
    np.seterr(divide='ignore', invalid='ignore')
    
    # Extrapolated from the ESRI equation; note terms are different A is Fxx here
    K = -2 * (A + B)

    # These have been re-written to produce output in line with SAGA.  Minar 
    # claims (2020, page 15) that "Longitudinal, cross-sectional, maximal, and
    # minimal are miscomputed see also black boxes indicating incorrect 
    # calculation in Table 5, page 14.
    # See Schmidt (Table 1) and Wood (page 87)
    # to revisit this issue and Equations in Table 3 of Minar
    
    # K_profile = -200 * (A*D**2 + B*E**2 + C*D*E) / ((E**2+D**2)*((1+D**2+E**2)**1.5)) OLD AS SEEN IN WOOD
    K_profile = -(A*D**2 + 2*C*D*E+B*E**2) / ((D**2+E**2)*((D**2+E**2+1)**1.5))  # New as seen in Schmidt Table 1
    K_cross = -2 * (B*D**2 + A*E**2 - C*D*E) / (D**2 + E**2)
    K_long = -2 * (A*D**2 + B*E**2 + C*D*E) / (D**2 + E**2)
    K_tan = -(A*E**2 - 2*C*D*E + B*D**2) / ((D**2 + E**2)*((D**2 + E**2 + 1)**.5)) # As seen in Schmidt Table 1
    K_plan = -(A*E**2 - 2*C*D*E + B*D**2) / (D**2+E**2)**1.5 

    
    np.seterr(divide='warn', invalid='warn')
    
    # Fix nans
    K_profile[np.isnan(K_profile) & np.isfinite(X)] = 0
    K_plan[np.isnan(K_plan) & np.isfinite(X)] = 0
    K_cross[np.isnan(K_cross) & np.isfinite(X)] = 0
    K_long[np.isnan(K_long) & np.isfinite(X)] = 0
    K_tan[np.isnan(K_tan) & np.isfinite(X)] = 0
    
    return K, K_profile, K_plan, K_tan, K_long, K_cross 


#%%
'''
Returns:
    Total Curvature (K)
    Profile Curvature (Kp)
    Plan / Contour Curvature (Kc)
    Tangential Curvature (Kt)

References
----------
Wilson and Gallant. 2000. Terrain Analysis: Principles and Applications.
'''

def wilson_gallant_curvature(X,cellsize=1):
    
    # Wilson and Gallant give cellsize as h, instead of the more common L
    H = cellsize
    
    # WG use a somewhat unorthodox pixel naming scheme, going from Z1 in the
    # upper right, clockwise to the top pixel (Z8), with the center pixel Z9
    # Figure 3.1, page 52
    Z1 = ashift(X,2)
    Z2 = ashift(X,3)
    Z3 = ashift(X,4)
    Z4 = ashift(X,5)
    Z5 = ashift(X,6)
    Z6 = ashift(X,7)
    Z7 = ashift(X,8)
    Z8 = ashift(X,9)
    Z9 = X # (unshifted)
    
    # Missing values are handled via a "finite differences" approach, given
    # on page 53, equation 3.8.
    idx = np.isnan(Z1); Z1[idx] = 2 * Z9[idx] - Z5[idx]
    idx = np.isnan(Z2); Z2[idx] = 2 * Z9[idx] - Z6[idx]
    idx = np.isnan(Z3); Z3[idx] = 2 * Z9[idx] - Z7[idx]
    idx = np.isnan(Z4); Z4[idx] = 2 * Z9[idx] - Z8[idx]
    idx = np.isnan(Z5); Z5[idx] = 2 * Z9[idx] - Z1[idx]
    idx = np.isnan(Z6); Z6[idx] = 2 * Z9[idx] - Z2[idx]
    idx = np.isnan(Z7); Z7[idx] = 2 * Z9[idx] - Z3[idx]
    idx = np.isnan(Z8); Z8[idx] = 2 * Z9[idx] - Z4[idx]
    
    # Equations are given in formulas 3.1 to 3.7, page 52
    ZX  = (Z2 - Z6) / (2 * H)            # Rightmost pixel minus leftmost pixel
    ZY  = (Z8 - Z4) / (2 * H)            # Top pixel minus bottom pixel
    ZXX = (Z2 - 2*Z9 + Z6) / H**2        # Average of right and left, minus center pixel
    ZYY = (Z8 - 2*Z9 + Z4) / H**2        # Average of top and bottom, minus center pixel 
    ZXY = (-Z7 + Z1 + Z5 - Z3) / 4*H**2  # (Upper right + lower left) - (upper left + lower right) 
    P = ZX**2 + ZY**2
    Q = P + 1
    
    # "Plan or contour curvature, the rate of change of aspect along a contour"
    # Equation 3.16 / Page 57
    Kc = (ZXX*ZY**2 - 2*ZXY*ZX*ZY + ZYY*ZX**2) / (P**1.5) 
    
    # "Profile curvature, the rate of change of potential slope down a flow line"
    # Equation 3.15 / Page 57
    Kp = (ZXX*ZX**2 + 2*ZXY*ZX*ZY + ZYY*ZY**2) / (P*Q**1.5)
    
    # "Tangential Curvature, Plan curvature multipled by the sine of the slope angle"
    # Equation 3.17 / Page 57
    Kt = (ZXX*ZX**2 + 2*ZXY*ZX*ZY + ZYY*ZY**2) / (P*Q**0.5)
    
    # Curvature, Equation 3.18, Page 57
    K = ZXX**2 + 2*ZXY**2 + ZYY**2
    
    return K, Kp, Kc, Kt

    
#%%
# http://edndoc.esri.com/arcobjects/9.2/net/shared/geoprocessing/spatial_analyst_tools/how_hillshade_works.htm
# ESRI's hillshade algorithm, but using the numpy versions of slope and aspect
# given above, so results may differ slightly from ESRI's version.
# If dtype is anytho
def hillshade(Z,cellsize=1,z_factor=1,zenith=45,azimuth=315,return_uint8=True):
    zenith, azimuth = np.deg2rad((zenith,azimuth))
    S = slope(Z,cellsize=cellsize,z_factor=z_factor,return_as='radians')
    A = aspect(Z,return_as='radians',flat_as=0)
    H = (np.cos(zenith) * np.cos(S)) + (np.sin(zenith) * np.sin(S) * np.cos(azimuth - A))
    H[H<0] = 0
    if return_uint8:
        H = 255 * H
        H = np.round(H)
        H = H.astype(np.uint8)
    return H
#%%
# The user can specify a range of zeniths and azimuths to calculate a very
# rudimentary multiple illumination model, where a hillshade is a calculated
# for each combination, and the maximum illimunation retained.  This is really
# just a scratch/test function, and not intended for production use.
def multiple_illumination(Z,cellsize=1,z_factor=1,zeniths=np.array([45]),azimuths=4):
    if np.isscalar(azimuths):
        azimuths = np.arange(0,360,360/azimuths)
    if np.isscalar(zeniths):
        zeniths = 90 / (zeniths + 1)
        zeniths = np.arange(zeniths,90,zeniths)
    H = np.zeros(np.shape(Z))
    for zenith in zeniths:
        for azimuth in azimuths:
            H1 = hillshade(Z,cellsize=cellsize,z_factor=z_factor,zenith=zenith,azimuth=azimuth)
            H = np.stack((H,H1),axis=2)
            H = np.max(H,axis=2)
    return H.astype(np.uint8)

# Calculates a Perceptually Scaled Slope Map (PSSM) of the input DEM, and 
# returns a bone shaded colormapped raster.
def pssm(Z,cellsize=1,ve=2.3,reverse=False):
    P = slope(Z,cellsize=cellsize,return_as='percent')
    P = np.rad2deg(np.arctan(ve *  P))
    P = (P - P.min()) / (P.max() - P.min())
    P = np.round(255*P).astype(np.uint8)
    if reverse==False:
        P = plt.cm.bone_r(P)
    else:
        P = plt.cm.bone(P)
    return P

# A simple function to calculate a z-factor based on an input latitude to 
# calculate slopes, etc., on a degree-referenced DEM (e.g., 1 arc second)
def z_factor(latitude):
    # https://blogs.esri.com/esri/arcgis/2007/06/12/setting-the-z-factor-parameter-correctly/
    latitude = np.deg2rad(latitude)
    m=6367449;
    a=6378137;
    b=6356752.3;
    numer=(a**4)*(np.cos(latitude)**2) + (b**4)*(np.sin(latitude)**2);
    denom=(a*np.cos(latitude))**2 + (b*np.sin(latitude))**2;
    z_factor = 1 / (np.pi / 180 * np.cos(latitude) * np.sqrt(numer/denom))
    return z_factor


#%%
# A rapid, near-approximation assuming a spherical body. Earth in meters is 
# assumed but any radius can be supplied.  
# See geopy's distance calculator for more flexible and accurate options.
    
def great_circle_distance(slat,slon,elat,elon,radius=6372795):
    # Concert to radians
    slat, slon = np.deg2rad(slat), np.deg2rad(slon)
    elat, elon = np.deg2rad(elat), np.deg2rad(elon)
    
    # Calculate
    dist = np.arccos(np.cos(slat)*np.cos(slon)*np.cos(elat)*np.cos(elon) + 
                     np.cos(slat)*np.sin(slon)*np.cos(elat)*np.sin(elon) + 
                     np.sin(slat)*np.sin(elat)) * radius

    return dist


#%% Lidar routines
"""
References:
http://stackoverflow.com/questions/16573089/reading-binary-data-into-pandas
https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
LAZ http://howardbutler.com/javascript-laz-implementation.html
http://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf
"""

# Reads a file into pandas dataframe
# Originally developed as research/current/lidar/bonemap
# A pure python las reader
def read_las(filename):

    with open(filename,mode='rb') as file:
        data = file.read()
    
    # This dictionary holds the byte length of the point data (see minimum
    # PDRF Size given in LAS spec.)
    point_data_format_key = {0:20,1:28,2:26,3:34,4:57,5:63,6:30,7:36,8:38,9:59,10:67}
    
    # Read header into a dictionary
    header = {}
    header['file_signature'] = struct.unpack('<4s',data[0:4])[0].decode('utf-8')
    header['file_source_id'] = struct.unpack('<H',data[4:6])[0]
    header['global_encoding'] = struct.unpack('<H',data[6:8])[0]
    project_id = []
    project_id.append(struct.unpack('<L',data[8:12])[0])
    project_id.append(struct.unpack('<H',data[12:14])[0])
    project_id.append(struct.unpack('<H',data[14:16])[0])
    #Fix
    #project_id.append(struct.unpack('8s',data[16:24])[0].decode('utf-8').rstrip('\x00'))
    header['project_id'] = project_id
    del project_id
    header['version_major'] = struct.unpack('<B',data[24:25])[0]
    header['version_minor'] = struct.unpack('<B',data[25:26])[0]
    header['version'] = header['version_major'] + header['version_minor']/10
    header['system_id'] = struct.unpack('32s',data[26:58])[0].decode('utf-8').rstrip('\x00')
    header['generating_software'] = struct.unpack('32s',data[58:90])[0].decode('utf-8').rstrip('\x00')
    header['file_creation_day'] = struct.unpack('H',data[90:92])[0]
    header['file_creation_year'] = struct.unpack('<H',data[92:94])[0]
    header['header_size'] = struct.unpack('H',data[94:96])[0]
    header['point_data_offset'] = struct.unpack('<L',data[96:100])[0]
    header['num_variable_records'] = struct.unpack('<L',data[100:104])[0]
    header['point_data_format_id'] = struct.unpack('<B',data[104:105])[0]
    laz_format = False
    if header['point_data_format_id'] >= 128 and header['point_data_format_id'] <= 133:
        laz_format = True
        header['point_data_format_id'] = point_data_format_id - 128
    if laz_format:
        raise ValueError('LAZ not yet supported.')
    try:
        format_length = point_data_format_key[header['point_data_format_id']]
    except:
        raise ValueError('Point Data Record Format',header['point_data_format_id'],'not yet supported.')
    if header['point_data_format_id'] >= 6:
        print('Point Data Formats 6-10 have recently been added to this reader.  Please check results carefully and report any suspected errors.')
    header['point_data_record_length'] = struct.unpack('<H',data[105:107])[0]
    header['num_point_records'] = struct.unpack('<L',data[107:111])[0]
    header['num_points_by_return'] = struct.unpack('<5L',data[111:131])
    header['scale'] = struct.unpack('<3d',data[131:155])
    header['offset'] = struct.unpack('<3d',data[155:179])
    header['minmax'] = struct.unpack('<6d',data[179:227]) #xmax,xmin,ymax,ymin,zmax,zmin
    end_point_data = len(data)
    
    # For version 1.3, read in the location of the point data.  At this time
    # no wave information will be read
    header_length = 227
    if header['version']==1.3:
        header['begin_wave_form'] = struct.unpack('<q',data[227:235])[0]
        header_length = 235
        if header['begin_wave_form'] != 0:
            end_point_data = header['begin_wave_form']

    # Pare out only the point data
    data = data[header['point_data_offset']:end_point_data]

    if header['point_data_format_id']==1:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2'),('gpstime','f8')])
    
    elif header['point_data_format_id']==2:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2'),('red','u2'),
                       ('green','u2'),('blue','u2')])
    elif header['point_data_format_id']==3:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2'),('gpstime','f8'),
                       ('red','u2'),('green','u2'),('blue','u2')])
    elif header['point_data_format_id']==4:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2'),('gpstime','f8'),
                       ('wave_packet_descriptor_index','u1'),('byte_offset','u8'),
                       ('wave_packet_size','u4'),('return_point_waveform_location','f4'),
                       ('xt','f4'),('yt','f4'),('zt','f4')])
    elif header['point_data_format_id']==5:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2'),('gpstime','f8'),
                       ('red','u2'),('green','u2'),('blue','u2'),
                       ('wave_packet_descriptor_index','u1'),('byte_offset','u8'),
                       ('wave_packet_size','u4'),('return_point_waveform_location','f4'),
                       ('xt','f4'),('yt','f4'),('zt','f4')])
    elif header['point_data_format_id']==6:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('mixed_byte','u1'),('class','u1'),
                       ('user_data','u1'),('scan_angle','u2'),('point_source_id','u2'),
                       ('gpstime','f8')])
    elif header['point_data_format_id']==7:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('mixed_byte','u1'),('class','u1'),
                       ('user_data','u1'),('scan_angle','u2'),('point_source_id','u2'),
                       ('gpstime','f8'),('red','u2'),('green','u2'),('blue','u2')])        
    elif header['point_data_format_id']==8:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('mixed_byte','u1'),('class','u1'),
                       ('user_data','u1'),('scan_angle','u2'),('point_source_id','u2'),
                       ('gpstime','f8'),('red','u2'),('green','u2'),('blue','u2'),
                       ('near_infrared','u2')])    
    elif header['point_data_format_id']==9:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('mixed_byte','u1'),('class','u1'),
                       ('user_data','u1'),('scan_angle','u2'),('point_source_id','u2'),
                       ('gpstime','f8'),('wave_packet_descriptor_index','u1'),
                       ('byte_offset','u8'),('wave_packet_size','u4'),
                       ('return_point_waveform_location','f4'),
                       ('xt','f4'),('yt','f4'),('zt','f4')])
    elif header['point_data_format_id']==10:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('mixed_byte','u1'),('class','u1'),
                       ('user_data','u1'),('scan_angle','u2'),('point_source_id','u2'),
                       ('gpstime','f8'),('red','u2'),('green','u2'),('blue','u2'),
                       ('near_infrared','u2'),('wave_packet_descriptor_index','u1'),
                       ('byte_offset','u8'),('wave_packet_size','u4'),
                       ('return_point_waveform_location','f4'),
                       ('xt','f4'),('yt','f4'),('zt','f4')])         
        
        
    # Transform to Pandas dataframe, via a numpy array
    data = pd.DataFrame(np.frombuffer(data,dt))
    data['x'] = data['x']*header['scale'][0] + header['offset'][0]
    data['y'] = data['y']*header['scale'][1] + header['offset'][1]
    data['z'] = data['z']*header['scale'][2] + header['offset'][2]

    def get_bit(byteval,idx):
        return ((byteval&(1<<idx))!=0);

    # Recast the mixed bytes as specified in the LAS specification
    if header['point_data_format_id'] < 6:
        data['return_number'] = 4 * get_bit(data['return_byte'],2).astype(np.uint8) + 2 * get_bit(data['return_byte'],1).astype(np.uint8) + get_bit(data['return_byte'],0).astype(np.uint8)
        data['return_max'] = 4 * get_bit(data['return_byte'],5).astype(np.uint8) + 2 * get_bit(data['return_byte'],4).astype(np.uint8) + get_bit(data['return_byte'],3).astype(np.uint8)
        data['scan_direction'] = get_bit(data['return_byte'],6)
        data['edge_of_flight_line'] = get_bit(data['return_byte'],7)
        del data['return_byte']
    else:
        data['return_number'] = 8 * get_bit(data['return_byte'],3).astype(np.uint8) + 4 * get_bit(data['return_byte'],2).astype(np.uint8) + 2 * get_bit(data['return_byte'],1).astype(np.uint8) + get_bit(data['return_byte'],0).astype(np.uint8)
        data['return_max'] = 8 * get_bit(data['return_byte'],7).astype(np.uint8) + 4 * get_bit(data['return_byte'],6).astype(np.uint8) + 2 * get_bit(data['return_byte'],5).astype(np.uint8) + get_bit(data['return_byte'],4).astype(np.uint8)
        # data['scan_direction'] = get_bit(data['return_byte'],6)
        # data['edge_of_flight_line'] = get_bit(data['return_byte'],7)
        del data['return_byte']        
    if header['point_data_format_id'] >= 6:
        data['classification_bit_synthetic'] = get_bit(data['mixed_byte'],0)
        data['classification_bit_keypoint'] = get_bit(data['mixed_byte'],1)
        data['classification_bit_withheld'] = get_bit(data['mixed_byte'],2)
        data['classification_bit_overlap'] = get_bit(data['mixed_byte'],3)
        data['scanner_channel'] = 2 * get_bit(data['mixed_byte'],5).astype(np.uint8) + 1 * get_bit(data['mixed_byte'],4).astype(np.uint8)
        data['scan_direction'] = get_bit(data['mixed_byte'],6)
        data['edge_of_flight_line'] = get_bit(data['mixed_byte'],7)
        del data['mixed_byte']
    

    
    return header,data

#%%

# Using scipy's binned statistic would be preferable here, but it doesn't do
# min/max natively, and is too slow when not cython.
# It would look like: 
# Z,xi,yi,binnum = stats.binned_statistic_2d(x,y,z,statistic='min',bins=(x_edge,y_edge))
def create_dem(x,y,z,cellsize=1,bin_type='max',use_binned_statistic=False,inpaint=False):
    
    #x = df.x.values
    #y = df.y.values
    #z = df.z.values
    #resolution = 1
    #bin_type = 'max' 
    floor2 = lambda x,v: v*np.floor(x/v)
    ceil2 = lambda x,v: v*np.ceil(x/v)
    
    
    xedges = np.arange(floor2(np.min(x),cellsize)-.5*cellsize,
                       ceil2(np.max(x),cellsize) + 1.5*cellsize,cellsize)
    yedges = np.arange(ceil2(np.max(y),cellsize)+.5*cellsize,
                       floor2(np.min(y),cellsize) - 1.5*cellsize,-cellsize)
    nx, ny = len(xedges)-1,len(yedges)-1
    
    I = np.empty(nx*ny)
    I[:] = np.nan
    
    # Define an affine matrix, and convert realspace coordinates to integer pixel
    # coordinates
    t = rasterio.transform.from_origin(xedges[0], yedges[0], cellsize, cellsize)
    c,r = ~t * (x,y)
    c,r = np.floor(c).astype(np.int64), np.floor(r).astype(np.int64)
    
    # Old way:
    # Use pixel coordiantes to create a flat index; use that index to aggegrate, 
    # using pandas.
    if use_binned_statistic:
        I = stats.binned_statistic_2d(x,y,z,statistic='min',bins=(xedges,yedges))
    else:        
        mx = pd.DataFrame({'i':np.ravel_multi_index((r,c),(ny,nx)),'z':z}).groupby('i')
        del c,r
        if bin_type=='max':
            mx = mx.max()
        elif bin_type=='min':
            mx = mx.min()
        else:
            raise ValueError('This type not supported.')
        
        I.flat[mx.index.values] = mx.values
        I = I.reshape((ny,nx))
        
    if inpaint==True:
        I = inpaint_nans_by_springs(I)
    
    return I,t


#%% Inpainting.  See research/current/inpaint/inpaint_nans.py for full details
# Finite difference approximation
def inpaint_nans_by_fda(A,fast=True,inplace=False):
    m,n = np.shape(A)
    nanmat = np.isnan(A)

    nan_list = np.flatnonzero(nanmat)
    known_list = np.flatnonzero(~nanmat)
    
    index = np.arange(m*n,dtype=np.int64).reshape((m,n))
    
    i = np.hstack( (np.tile(index[1:-1,:].ravel(),3),
                    np.tile(index[:,1:-1].ravel(),3)
                    ))
    j = np.hstack( (index[0:-2,:].ravel(),
                    index[2:,:].ravel(),
                    index[1:-1,:].ravel(),
                    index[:,0:-2].ravel(),
                    index[:,2:].ravel(),
                    index[:,1:-1].ravel()
                    ))
    data = np.hstack( (np.ones(2*n*(m-2),dtype=np.int64),
                       -2*np.ones(n*(m-2),dtype=np.int64),
                       np.ones(2*m*(n-2),dtype=np.int64),
                       -2*np.ones(m*(n-2),dtype=np.int64)
                       ))
    if fast==True:
        goodrows = np.in1d(i,index[ndi.binary_dilation(nanmat)])
        i = i[goodrows]
        j = j[goodrows]
        data = data[goodrows]
        del goodrows

    fda = sparse.coo_matrix((data,(i,j)),(m*n,m*n),dtype=np.int8).tocsr()
    del i,j,data,index
    
    rhs = -fda[:,known_list] * A[np.unravel_index(known_list,(m,n))]
    k = fda[:,np.unique(nan_list)]
    k = k.nonzero()[0]
    a = fda[k][:,nan_list]
    results = sparse.linalg.lsqr(a,rhs[k])[0]

    if inplace:
        A[np.unravel_index(nan_list,(m,n))] = results
    else:
        B = A.copy()
        B[np.unravel_index(nan_list,(m,n))] = results
        return B
        
    
#%%    
    
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
            
# At the moment, only 4 neighbors are supported.
def inpaint_nans_by_springs(A,inplace=False,neighbors=4):

    m,n = np.shape(A)
    nanmat = np.isnan(A)

    nan_list = np.flatnonzero(nanmat)
    known_list = np.flatnonzero(~nanmat)
    
    r,c = np.unravel_index(nan_list,(m,n))
    
    num_neighbors = neighbors
    neighbors = np.array([[0,1],[0,-1],[-1,0],[1,0]]) #r,l,u,d

    neighbors = np.vstack([np.vstack((r+i[0], c+i[1])).T for i in neighbors])
    del r,c
    
    springs = np.tile(nan_list,num_neighbors)
    good_rows = (np.all(neighbors>=0,1)) & (neighbors[:,0]<m) & (neighbors[:,1]<n)
    
    neighbors = np.ravel_multi_index((neighbors[good_rows,0],neighbors[good_rows,1]),(m,n))
    springs = springs[good_rows]
    
    springs = np.vstack((springs,neighbors)).T
    del neighbors,good_rows
    
    springs = np.sort(springs,axis=1)
    springs = unique_rows(springs)
    
    n_springs = np.shape(springs)[0]
    
    i = np.tile(np.arange(n_springs),2)
    springs = springs.T.ravel()
    data = np.hstack((np.ones(n_springs,dtype=np.int8),-1*np.ones(n_springs,dtype=np.int8)))
    springs = sparse.coo_matrix((data,(i,springs)),(n_springs,m*n),dtype=np.int8).tocsr()
    del i,data
    
    rhs = -springs[:,known_list] * A[np.unravel_index(known_list,(m,n))]
    results = sparse.linalg.lsqr(springs[:,nan_list],rhs)[0]       

    if inplace:
        A[np.unravel_index(nan_list,(m,n))] = results
    else:
        B = A.copy()
        B[np.unravel_index(nan_list,(m,n))] = results
        return B
    
    
    
#%%
        
def inpaint_nearest(X):
    idx = np.isfinite(X)
    RI,CI = np.meshgrid(np.arange(X.shape[0]),np.arange(X.shape[1]))
    f_near = interpolate.NearestNDInterpolator((RI[idx],CI[idx]),X[idx])
    idx = ~idx
    X[idx] = f_near(RI[idx],CI[idx])
    return X

#%%
    
# ashift pulls a copy of the raster shifted.  0 shifts upper-left to lower right
# 1 shifts top-to-bottom, etc.  Clockwise from upper left. Use 0 to "grab" the
# upper left pixel, 1 to "grab" up top pixel, etc.
def ashift(surface,direction,n=1):
    surface = surface.copy()
    if direction==0:
        surface[n:,n:] = surface[0:-n,0:-n]
    elif direction==1:
        surface[n:,:] = surface[0:-n,:]
    elif direction==2:
        surface[n:,0:-n] = surface[0:-n,n:]
    elif direction==3:
        surface[:,0:-n] = surface[:,n:]
    elif direction==4:
        surface[0:-n,0:-n] = surface[n:,n:]
    elif direction==5:
        surface[0:-n,:] = surface[n:,:]
    elif direction==6:
        surface[0:-n,n:] = surface[n:,0:-n]
    elif direction==7:
        surface[:,n:] = surface[:,0:-n]
    return surface


#%%


def progressive_window(min_value,max_value,percent):
    this_list = np.array([min_value],dtype=np.int32)
    last_value = min_value
    while last_value < max_value:
        last_value = np.ceil(last_value*(100+percent)/100).astype(np.int32)
        if last_value <= max_value:
            this_list = np.append(this_list,last_value)
    return this_list

#%%

def openness(Z,cellsize=1,lookup_pixels=1,neighbors=np.arange(8),skyview=False,fast=False,how_fast=20):

    nrows, ncols = np.shape(Z)
        
    # neighbor directions are clockwise from top left,starting at zero
    # neighbors = np.arange(8)   
    
    # Define a (fairly large) 3D matrix to hold the minimum angle for each pixel
    # for each of the requested directions (usually 8)
    opn = np.Inf * np.ones((len(neighbors),nrows,ncols))
    
    # Define an array to calculate distances to neighboring pixels
    dlist = np.array([np.sqrt(2),1])

    # Calculate minimum angles        
    test_range = np.arange(1,lookup_pixels+1)   
    if fast==True:
        test_range = progressive_window(1, lookup_pixels, how_fast)
    for L in test_range:
        for i,direction in enumerate(neighbors):
            # Map distance to this pixel:
            dist = dlist[direction % 2]
            dist = cellsize * L * dist
            # Angle is the arctan of the difference in elevations, divided by distance
            these_angles = (np.pi/2) - np.arctan((ashift(Z,direction,L)-Z)/dist)
            this_layer = opn[i,:,:]
            this_layer[these_angles < this_layer] = these_angles[these_angles < this_layer]
            opn[i,:,:] = this_layer

    # Openness is definted as the mean of the minimum angles of all 8 neighbors  
    # Return in degrees, though:
    return np.rad2deg(np.mean(opn,0))

#%% 
    
def skyview_factor(Z,cellsize=1,lookup_pixels=1):

    nrows, ncols = np.shape(Z)

    # This will sum the max angles    
    sum_matrix = np.zeros_like(Z,dtype=np.float)
    
    # Define an array to calculate distances to neighboring pixels
    dlist = np.array([np.sqrt(2),1])

    for direction in np.arange(8):
        max_angles = np.zeros_like(Z,dtype=np.float)
        z_shift = Z.copy()
        for L in range(1,lookup_pixels+1):
            # Map distance to this pixel:
            dist = dlist[direction % 2]
            dist = cellsize * L * dist
            # Angle is the arctan of the difference in elevations, divided by distance
            z_shift = ashift(z_shift,direction,1)
            these_angles = np.clip(np.arctan((z_shift-Z)/dist),0,np.inf)
            max_angles = np.nanmax(np.stack((max_angles,these_angles),axis=0),axis=0)
        sum_matrix += np.sin(max_angles)
    sum_matrix = 1 - sum_matrix / 8

    return sum_matrix



#%%
    
# This routine uses openness to generate a ternary pattern based on the 
# difference of the positive and negative openness values.  If the difference
# is above a supplied threshold, the value is "high" or 2.  If the difference
# is below the threshold, it is 1 or "equal".  If the difference is less than 
# the negative threshold, it is 0 or "low".
    
# The algorithm proceeds through each 8 directions, one at a time, building
# a list of 8 ternary values (e.g., 21120210).  Previously, these would have 
# been recorded, and then converted to decimal; here they are converted
# to decimal as it progresses.  Upper left pixel is the least significant
# digit, left pixel is the most significant pixel.

# For a binary equivalent, see https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
    
def ternary_pattern_from_openness(Z,cellsize=1,lookup_pixels=1,threshold_angle=0,use_negative_openness=True,lowest=False):
    pows = 3**np.arange(8)
    #bc = np.zeros(np.shape(Z),dtype=np.uint32)
    tc = np.zeros(np.shape(Z),dtype=np.uint16)
    f = 1
    for i in range(8):
        O = openness(Z,cellsize,lookup_pixels,neighbors=np.array([i]))
        if use_negative_openness:
            O = O - openness(-Z,cellsize,lookup_pixels,neighbors=np.array([i]))
        else:
            O = O - 90.0
        tempMat = np.ones(np.shape(tc),dtype=np.uint32)
        tempMat[O > threshold_angle] = 2;
        tempMat[O < -threshold_angle] = 0;
    
        # Record the result.
        #bc = bc + f*tempMat;
        tc = tc + tempMat*pows[i] 
    
        # Increment f
        f = f * 10;
        
    if lowest:
        lookup_table = np.array([get_lowest_equivalent(x) for x in np.arange(3**8)])
        tc = lookup_table[tc]
    
    return tc


#%%
# This is a general purpose function that coverts a base 10 integer to a string
# representation of a base b number.
# e.g. 5,2 -> '00000101'
# Adapted from https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-in-any-base-to-a-string
def int2base(x,b,alphabet='0123456789abcdefghijklmnopqrstuvwxyz',min_digits=8):
    rets=''
    while x>0:
        x,idx = divmod(x,b)
        rets = alphabet[idx] + rets
    if len(rets) < min_digits:
        pad = ''
        for i in range(min_digits - len(rets)):
            pad = pad + '0'
        rets = pad + rets
    return rets



#%%
# A helper function that generates a lookup table mapping each 8-digit string
# code to its rotational and reflectional equivalents, and return back the 
# base 10 integer that corresponds to the minimum of these.
# For instance, going from top left to left (clockwise):
# 0 0 0        2 1 0           0 1 2
# 1   2   ->   2   0      ->   0   2
# 2 2 2        2 2 0           0 2 2
# Original    90 degree        reflection
# '00022221'  '21000222'      '01222200'    : String codes (Base 3 representation)
# 241        5129             1449          : Terrain code (Base 10 Representation)
#
# Lowest Equivalent: 161, '00012222'

def get_lowest_equivalent(terrain_code):
    s = int2base(terrain_code,3)
    min_val = int(s,3)
    for j in range(1,16):
        s = s[-1] + s[:7]
        min_val = min(min_val,int(s,3))
        if j==7:
            s = s[::-1]
    return min_val
    
#%%
    
# Applies either a strict or loose lookup table to a terrain code to transform
# into a standardized list of terrain types.  Zero is "undefined".
# 
# Strict uses only the eight neighbors, and must match precisely.  For instance,
# to be "flat", ALL of the eight neighbors must have been classified as equal
# within the tolerance value (i.e., all are "1").
#
# In contrast, the loose method counts the number of cells higher, and the 
# number of cells lower, so a "flat" area is any where the  number of cells
# with a higher value is three or less, and the number of cells with a lower
# value is also three of less, AND the total number of higher or lower 
# is not more than 3 (i.e., at least 5 are strictly "flat")
def terrain_code_to_geomorphon(terrain_code,method='loose'):
    geomorphon = None
    method_options = ['strict','loose']
    if method not in method_options:    
        print('method should be one of',method_options)
    else:
        lookup_table = np.zeros(3**8,np.uint8)
        if method=='strict':
            lookup_table[3280] = 1  # Flat
            lookup_table[0] = 2     # Peak
            lookup_table[82] = 3    # Ridge
            lookup_table[121] = 4   # Shoulder
            lookup_table[26] = 5    # Spur
            lookup_table[160] = 6   # Slope
            lookup_table[242] = 7   # Hollow
            lookup_table[3293] = 8  # Footslope
            lookup_table[4346] = 9  # Valley
            lookup_table[6560] = 10 # Pit
        elif method=='loose':
            lookup_table = np.zeros(3**8,np.uint8)
            strict_table = np.zeros((9,9),dtype=np.uint8)
            #                      (Fig 4., Jasiewicz and Stepinksi, 2013)
            #                      Number of cells higher
            strict_table[0,:]   = [1,1,1,8,8,9,9,9,10] # 
            strict_table[1,:8]  = [1,1,8,8,8,9,9,9]    # Num
            strict_table[2,:7]  = [1,4,6,6,7,7,9]      # Cells
            strict_table[3,:6]  = [4,4,6,6,6,7]        # Lower
            strict_table[4,:5]  = [4,4,5,6,6]
            strict_table[5,:4]  = [3,3,5,5]
            strict_table[6,:3]  = [3,3,3]
            strict_table[7,:2]  = [3,3]
            strict_table[8,:1]  = [2]
            for i in range(3**8):
                base = int2base(i,3)
                r,c = base.count('2'), base.count('0')
                lookup_table[i] = strict_table[r,c]
    geomorphon = lookup_table[terrain_code]
    return geomorphon
                
#%%
def geomorphon_cmap_old():
    lut = [255,255,255, \
    220,220,220, \
    56,0,0, \
    200,0,0, \
    255,80,20, \
    250,210,60, \
    255,255,60, \
    180,230,20, \
    60,250,150, \
    0,0,255, \
    0,0,56]
    return lut

def geomorphon_cmap():
    d = {1: (220,220,220),
         2: (56,0,0), 
         3: (200,0,0),
         4: (255,80,20),
         5: (250,210,60),
         6: (255,255,60),
         7: (180,230,20),
         8: (60,250,150),
         9: (0,0,255),
         10: (0,0,56)}
    return d
    
#%%
'''
This is a helper function to easily write out a text-based worldfile to 
accompany an image used as raster data.
Source: http://www.perrygeo.com/python-affine-transforms.html
'''

def write_worldfile(affine_matrix,output_file):
    outfile = 'test.pgw'
    x_ul_center,y_ul_center = affine_matrix * (.5,.5)
    pixel_width, row_rotation = affine_matrix[0],affine_matrix[1]
    pixel_height, col_rotation = affine_matrix[4],affine_matrix[3]
    world_data = [pixel_width,col_rotation,row_rotation,pixel_height,x_ul_center,y_ul_center]
    np.savetxt(output_file,np.array([world_data]).reshape((6,1)),fmt='%0.10f')
    

#%%

# https://stackoverflow.com/questions/14448763/is-there-a-convenient-way-to-apply-a-lookup-table-to-a-large-array-in-numpy
# This was the first function I wrote to do the calculation, but is actually 
# fairly unnecessary... get_geomorphone_from_openness has fewer steps (but
# actually doesn't take that much less time to calculate).
def get_geomorphons(Z,cellsize=1,lookup_pixels=5,threshold_angle=1,use_negative_openness=True,method='loose',outfile=None,out_transform=None):
    terrain_code = ternary_pattern_from_openness(Z,cellsize=cellsize, \
                                                 lookup_pixels=lookup_pixels, \
                                                 threshold_angle=threshold_angle, \
                                                 use_negative_openness=use_negative_openness)
    lookup_table = np.array([get_lowest_equivalent(x) for x in np.arange(3**8)])
    terrain_code = lookup_table[terrain_code]
    geomorphon = terrain_code_to_geomorphon(terrain_code,method='loose')
    
    if not outfile==None:
        im = Image.fromarray(geomorphon,mode='L')
        im.putpalette(geomorphon_cmap())
        im.save(outfile)
        if not out_transform==None:
            write_worldfile(out_transform,outfile[:-3] + 'pgw')
        del im

    return geomorphon


#%%  Edit to try to include the "correction of forms" section in J&S
def count_openness(Z,cellsize,lookup_pixels,threshold_angle,fast=False,how_fast=20):
    
    num_pos = np.zeros(np.shape(Z),dtype=np.uint8)
    num_neg = np.zeros(np.shape(Z),dtype=np.uint8)
        
    for i in range(8):        
        O = openness(Z,cellsize,lookup_pixels,neighbors=np.array([i]),fast=fast,how_fast=how_fast)
        O = O - openness(-Z,cellsize,lookup_pixels,neighbors=np.array([i]),fast=fast,how_fast=how_fast)
        num_pos[O > threshold_angle] = num_pos[O > threshold_angle] + 1
        num_neg[O < -threshold_angle] = num_neg[O < -threshold_angle] + 1
    return num_pos, num_neg


    
#%%
# This is the best go-to function for calcluating a geomorhon from an openness
# calculation.    
def get_geomorphon_from_openness(Z,cellsize=1,lookup_pixels=1,threshold_angle=1,enhance=False,fast=False,how_fast=20):

    
    num_pos, num_neg = count_openness(Z,cellsize,lookup_pixels,threshold_angle,fast,how_fast)
          
    
    lookup_table = np.zeros((9,9),dtype=np.uint8)

    # 1 – flat, 2 – peak, 3 - ridge, 4 – shoulder, 5 – spur, 6 – slope, 7 – hollow, 8 – footslope, 9 – valley, and 10 – pit
    #                      Number of cells higher
    lookup_table[0,:]   = [1,1,1,8,8,9,9,9,10] # 
    lookup_table[1,:8]  = [1,1,8,8,8,9,9,9]    # Num
    lookup_table[2,:7]  = [1,4,6,6,7,7,9]      # Cells
    lookup_table[3,:6]  = [4,4,6,6,6,7]        # Lower
    lookup_table[4,:5]  = [4,4,5,6,6]
    lookup_table[5,:4]  = [3,3,5,5]
    lookup_table[6,:3]  = [3,3,3]
    lookup_table[7,:2]  = [3,3]
    lookup_table[8,:1]  = [2]    
    
    geomorphons = lookup_table[num_pos.ravel(),num_neg.ravel()].reshape(np.shape(Z))
    
    # Edit to try to include the "correction of forms" section in J&S
    if enhance==True and lookup_pixels > 16:
        lookup_pixels_sm = np.floor(lookup_pixels / 4).astype(np.int)
        if lookup_pixels_sm < 4:
            lookup_pixels_sm = 4
        num_pos_sm, num_neg_sm = count_openness(Z,cellsize,lookup_pixels_sm,threshold_angle)
        
        geomorphons_sm = lookup_table[num_pos_sm.ravel(),num_neg_sm.ravel()].reshape(np.shape(Z))
        geomorphons[(geomorphons==4) & (geomorphons_sm==1)] = 1
        geomorphons[(geomorphons==8) & (geomorphons_sm==1)] = 1
        geomorphons[(geomorphons==2) | (geomorphons==3)] = geomorphons_sm[(geomorphons==2) | (geomorphons==3)]
        
        
    
    
    return geomorphons


#%% The Simple Morphological Filter

def progressive_filter(Z,windows,cellsize=1,slope_threshold=.15):
    last_surface = Z.copy()
    elevation_thresholds = slope_threshold * (windows * cellsize)  
    is_object_cell = np.zeros(np.shape(Z),dtype=np.bool)
    for i,window in enumerate(windows):
        elevation_threshold = elevation_thresholds[i]
        this_disk = disk(window)
        if window==1:
            this_disk = np.ones((3,3),dtype=np.uint8)
        this_surface = ndi.morphology.grey_opening(last_surface,footprint=disk(window)) 
        is_object_cell = (is_object_cell) | (last_surface - this_surface > elevation_threshold)
        if i < len(windows) and len(windows)>1:
            last_surface = this_surface.copy()
    return is_object_cell


#%%

'''
x,y,z are points in space (e.g., lidar points)

windows is a scalar value specifying the maximum radius in pixels.  One can also 
supply an array of specific radii to test.  Very often, increasing the radius by 
one each time (as is the default) is unnecessary, especially for EDA.

Final classification of points is done using elevation_threshold and elevation_scaler.
points are compared against the provisional surface with a threshold modulated by the 
scaler value.  However, often the provisional surface (itself interpolated) works
quite well.  

Two additional parameters are being test to assist in low outlier removal.
low_filter_slope provides a slope value for an inverted surface.  Its default
value is 5 (meaning 500% slope).  However, we have noticed that in very rugged 
and forested terrain, that an even larger value may be necessary.  Alternatively,
low noise can be scrubbed using other means, and then this value can be set to a very high 
value to avoid its use entirely.  A second parameter (low_outlier_fill) will 
remove the points from the provisional DTM, and then fill them in before the main
body of the SMRF algorithm proceeds.  This should aid in preventing the "damage"
to the DTM that can happen when low outliers are present.
'''

def smrf(x,y,z,cellsize=1,windows=18,slope_threshold=.15,elevation_threshold=.5,
         elevation_scaler=1.25,low_filter_slope=5,low_outlier_fill=False):

    if np.isscalar(windows):
        windows = np.arange(windows) + 1
    
    Zmin,t = create_dem(x,y,z,cellsize=cellsize,bin_type='min');
    is_empty_cell = np.isnan(Zmin)
    Zmin = inpaint_nans_by_springs(Zmin)
    low_outliers = progressive_filter(-Zmin,np.array([1]),cellsize,slope_threshold=low_filter_slope); 
    
    # perhaps best to remove and interpolate those low values before proceeding?
    if low_outlier_fill:
        Zmin[low_outliers] = np.nan
        Zmin = inpaint_nans_by_springs(Zmin)
    
    # This is the main crux of the algorithm
    object_cells = progressive_filter(Zmin,windows,cellsize,slope_threshold);
    
    # Create a provisional surface
    Zpro = Zmin
    del Zmin
    # For the purposes of returning values to the user, an "object_cell" is
    # any of these: empty cell, low outlier, object cell
    object_cells = is_empty_cell | low_outliers | object_cells
    Zpro[object_cells] = np.nan
    Zpro = inpaint_nans_by_springs(Zpro)
    
    # Use provisional surface to interpolate a height at each x,y point in the
    # point cloud.  This uses a linear interpolator, where the original SMRF
    # used a splined cubic interpolator.  Perhaps use RectBivariateSpline instead.
    col_centers = np.arange(0.5,Zpro.shape[1]+.5)
    row_centers = np.arange(0.5,Zpro.shape[0]+.5)
    
    # Regular Grid Interpolator, keep here for a reference for a short time
    # Example syntax: x,y = t * (col,row)
#    xi, _ = t * (col_centers, np.zeros(np.shape(col_centers)))
#    _, yi = t * (np.zeros(np.shape(row_centers)), row_centers)
#    f1 = interpolate.RegularGridInterpolator((yi[::-1],xi),np.flipud(Zpro))
#    elevation_values = f1((y,x))
    
    # RectBivariateSpline Interpolator
    c,r = ~t * (x,y)
    f1 = interpolate.RectBivariateSpline(row_centers,col_centers,Zpro)
    elevation_values = f1.ev(r,c)
    
    # Calculate a slope value for each point.  This is used to apply a some "slop"
    # to the ground/object ID, since there is more uncertainty on slopes than on
    # flat areas.
    gy,gx = np.gradient(Zpro,cellsize)
    S = np.sqrt(gy**2 + gx**2)
    del gy,gx
    f2 = interpolate.RectBivariateSpline(row_centers,col_centers,S)
    del S
    slope_values = f2.ev(r,c)
    
    # Use elevation and slope values and thresholds interpolated from the 
    # provisional surface to classify as object/ground
    required_value = elevation_threshold + (elevation_scaler * slope_values)
    is_object_point = np.abs(elevation_values - z) > required_value
    
    # Return the provisional surface, affine matrix, raster object cells
    # and boolean vector identifying object points from point cloud
    return Zpro,t,object_cells,is_object_point


#%%
'''
h0 is the height of the neighbor pixel in one direction, relative to the center
h1 is the height of the pixel on the other size of the center pixel (same dir)
xdist is the real distance between them (as some neighbors are diagnonal)
'''
    
def triangle_height(h0,h1,x_dist=1):
    n = np.shape(h0)

    # The area of the triangle is half of the cross product    
    h0 = np.column_stack((-x_dist*np.ones(n),h0))
    h1 = np.column_stack(( x_dist*np.ones(n),h1))
    cp = np.abs(np.cross(h0,h1))
    
    # Find the base from the original coords
    base = np.sqrt( (2*x_dist)**2 + (h1[:,1]-h0[:,1])**2 )
    
    # Triangle height is the cross product divided by the base
    return cp/base

def vip_score(Z,cellsize=1):
    heights = np.zeros(np.size(Z))
    dlist = np.array([np.sqrt(2),1])
    for direction in range(4):
        dist = dlist[direction % 2]
        h0 = ashift(Z,direction) - Z
        h1 = ashift(Z,direction+4) - Z
        heights += triangle_height(h0.ravel(),h1.ravel(),dist*cellsize)
        
    # The original VIP spec simply used the sum; here an average is calculated
    # to make for a more direct comparison to other average-based methods
    heights = heights / 4
    heights = heights.reshape(np.shape(Z))
    return heights

#%%
def swiss_shading(Z,cellsize=1):
    lut = plt.imread(neilpy_dir + '/swiss_shading_lookup.png')[:,:,:3]
    lut = np.round(255 * lut)
    lut = lut.astype(np.uint8)
    
    z_min_prc, z_max_prc = 0,100
    Z_norm = np.round(255 * (Z - Z.min()) / (Z.max() - Z.min())).astype(np.uint8)
    Z_norm = Z_norm.astype(np.uint8)
    H= hillshade(Z,cellsize)
    
    RGB = np.zeros((np.shape(Z)[0],np.shape(Z)[1],3),dtype=np.uint8)
    RGB[:,:,0] = lut[:,:,0][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))
    RGB[:,:,1] = lut[:,:,1][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))
    RGB[:,:,2] = lut[:,:,2][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))
    
    return RGB




#%%
    
def colortable_shade(Z,name='swiss',cellsize=1):
    if type(name) == str:
        if name=='gray_high_contrast':
            lut = plt.imread(neilpy_dir + '/' + 'gray_high_contrast_lookup.png')
            lut = np.stack((lut,lut,lut),axis=2)
            lut = np.round(255 * lut)
            lut = lut.astype(np.uint8)
        elif name.endswith('.png'):
            lut = plt.imread(neilpy_dir + '/' + name)
            lut = np.stack((lut,lut,lut),axis=2)
            lut = np.round(255 * lut)
            lut = lut.astype(np.uint8)
        else:
            if name=='bare_earth_dark':
                spec = np.array([[90,74,84],[95,77,85],[40,38,74],[116,102,109]])
            if name=='bare_earth_medium':
                spec = np.array([[189,169,107],[203,179,114],[0,0,10],[116,102,109]])
            if name=='bare_earth_light':
                spec = np.array([[189,169,107],[203,179,114],[0,0,10],[255,255,255]])
            if name=='swiss_dark':
                spec = np.array([[110,79,107],[190,192,173],[40,38,74],[244,244,190]])
            elif name=='swiss':
                spec = np.array([[129,137,131],[190,192,173],[117,124,121],[244,244,190]])
            elif name=='swiss_green':
                spec = np.array([[118,162,120],[177,232,158],[111,123,115],[242,254,186]])
            elif name=='gray':
                spec = np.array([[0,0,0],[119,119,119],[1,1,1],[255,255,255]])
                lut = np.zeros((256,256,3),dtype=np.uint8)
            lut[:,:,0] = ndi.zoom([[spec[0,0],spec[1,0]],[spec[2,0],spec[3,0]]],128)
            lut[:,:,1] = ndi.zoom([[spec[0,1],spec[1,1]],[spec[2,1],spec[3,1]]],128)
            lut[:,:,2] = ndi.zoom([[spec[0,2],spec[1,2]],[spec[2,2],spec[3,2]]],128)
    else:
        lut = name
        if np.ndim(lut)!=3:
            lut = np.stack((lut,lut,lut),axis=2)

    H= hillshade(Z,cellsize,return_uint8=True)
    Z = np.round(255 * (Z - Z.min()) / (Z.max() - Z.min())).astype(np.uint8)
    
    RGB = np.zeros((np.shape(Z)[0],np.shape(Z)[1],3),dtype=np.uint8)
    RGB[:,:,0] = lut[:,:,0][Z.ravel(),H.ravel()].reshape(np.shape(Z))
    RGB[:,:,1] = lut[:,:,1][Z.ravel(),H.ravel()].reshape(np.shape(Z))
    RGB[:,:,2] = lut[:,:,2][Z.ravel(),H.ravel()].reshape(np.shape(Z))
    
    return RGB


#%%
def rmse(X):
    return np.sqrt(np.nansum(X**2)/np.size(X))

#%%
    
'''
Convenience function to split a raster into r and c pieces. 
Returns a list of lists, in row-column form. 
Example:
    X = tifffile.imread('bigraster.tif')
    X = cutter(X,3,6)
    upper_right_piece = X[0][5]
    
See also: "Split Raster" tool in ArcGIS.
'''
    
def cutter(x,r,c):
    return [np.hsplit(i,c) for i in np.vsplit(x,r)]


#%%
'''
Convenience function to change an array from min/max to 0/1 or a variety
of other mappings.  Simply specify calculate values to xrange parameter, or 
use simple keywords like min,max,mean,median.  You can specify more than just
two endpoints as well, letting you simply specify a piecewise curve re-mapping

Examples:
    Z, metadata = neilpy.imread('dem.tif')
    Zn = neilpy.normalize(N)
    
    or
    Zn = neilpy.normalize(Z,yrange=[-1,1])
    
    or
    Zn = neilpy.normalize(Z,xrange=['min','mean','max'],yrange=[-1,0,1])
    
    or
    Zmax = np.nanmax(Z)
    Zmin = np.nanmin(Z)
    Zmean = np.nanmean(Z)
    Zn = neilpy.normalize(Z,xrange=[Zmin,Zmean,Zmax],yrange=[-1,0,1])
'''

def normalize(X,xrange=['min','max'],yrange=[0,1]):
    xrange_fixed = []
    for item in xrange:
        if item=='max':
            item = np.nanmax(X)
        elif item=='min':
            item = np.nanmin(X)
        elif item=='mean':
            item = np.nanmean(X)
        elif item=='median':
            item = np.nanmedian(X)
        xrange_fixed.append(item)
    return np.interp(X,xrange_fixed,yrange)

#%%
'''
An implementation of Brassel's 1974 atmospheric correction routine for shaded
relief images.

Parameters:
    k, which controls the amount of correction. Should be >= 1.  This is the 
        only required parameter.
    flat, which is a value (integer or 0-1) that specifies the shading of 
        flat areas.  Automatically set to 180, for a standard hillshade.
    Zmid, which lets you specify a midpoint for the effects.  This is usually
        the average of the maximum and minimum values, but can be set as 
        described by Jenny (2000).  Setting to the mean or median is nice.
    reverse, a boolean value to de-emphasize higher areas
    C2, a tonal adjustment value, to be set between -1 and 1. 
'''

def brassel_atmospheric_perspective(H,Z,k,flat=180,Zmid=None,reverse=False,C2=0):
    
    if k<1:
        raise('k must be equal to or greater than one.')
    
    was_int = False
    if np.any(H>1):
        H = H / 255
        was_int = True
    
    if flat>1:
        flat = flat / 255
    
    Zmin = np.nanmin(Z)
    Zmax = np.nanmax(Z)
    
    if Zmid is None:
        Zstar = (Z - ((Zmax+Zmin) / 2)) / ((Zmax-Zmin)/2)
    else:
        Zstar = normalize(Z,xrange=[Zmin,Zmid,Zmax],yrange=[-1,0,1])
        
    if reverse:
        Zstar = -Zstar
    
    exponent = np.e**(Zstar*np.log(k))
    
    H_new = ((H - flat) * exponent) + flat
    
    H_new[H_new<0] = 0
    H_new[H_new>1] = 1
    
    if C2 != 0:
        H_new = H_new + (C2 * (Zstar-1))/2
    
    if was_int:
        H_new = np.round(255*H_new).astype(np.uint8)
        
    
    return H_new

#%%

'''
References:
    http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf
    http://www.jennessent.com/downloads/TPI_Documentation_online.pdf
'''

def topographic_position_index(X,radius=1,standardize=True):

    # If radius is one, use a 3x3 structuring element, otherwise, use this
    # as the radius of a disk       
    if radius==1:
        strel = np.ones((3,3),dtype=np.uint8)
    else:
        strel = disk(radius)
    # But remove the center, so we don't include the center pixel in the 
    # mean calculation
    strel[radius,radius] = 0
    # Define the weights (divide by total number of ones)
    strel = strel / np.sum(strel)
   
    mean = ndi.convolve(X,strel,mode='nearest')
    result = X - mean
    
    # If standardization is requested, construct a standard deviation filter
    # and divide the result by that
    if standardize:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        sd = np.sqrt(np.mean(ndi.convolve(X**2,strel,mode='nearest')) - np.mean(result)**2)
        result = result / sd
    
    return result

#%%

# Reader for LLH data returned by Emlid Reach and RTKlib
# Takes a filename, returns a geodataframe
# https://community.emlid.com/t/reach-llh-protocol-format/1354/4

def read_llh(fn,return_datetimes=True,skiprows=0):
    
    df = pd.read_csv(fn,header=None,delim_whitespace=True,skiprows=skiprows)
    
    df = df.rename({0:'date_gps',1:'time_gps',2:'lat',3:'lon',4:'alt',5:'Q',
                    6:'num_sat',7:'sdn',8:'sde',9:'sdu',10:'sdne',11:'sdeu',
                    12:'sdun',13:'age',14:'ratio'},axis=1)
    
    # Q=1:fix,2:float,3:sbas,4:dgps,5:single,6:ppp
    
    if return_datetimes:
        tm = df.iloc[:,0] + " " + df.iloc[:,1]
        df['datetime_gps'] = pd.to_datetime(tm)
        df['datetime_utc'] = df['datetime_gps'] - datetime.timedelta(seconds=18)
    
    df = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
    df = df.set_crs(epsg=4326)
    
    return df



#%%
def read_pos(fn,return_datetimes=True):
    df = read_llh(fn,return_datetimes,skiprows=11)
    return df

#%%


def exif_dict_to_dd(exif_dict):
    lat = exif_dict['GPS'][2][0][0] + exif_dict['GPS'][2][1][0]/60 + exif_dict['GPS'][2][2][0]/(exif_dict['GPS'][2][2][1]*3600)
    if exif_dict['GPS'][1] == b'S':
        lat = -lat
    lon = exif_dict['GPS'][4][0][0] + exif_dict['GPS'][4][1][0]/60 + exif_dict['GPS'][4][2][0]/(exif_dict['GPS'][4][2][1]*3600)
    if exif_dict['GPS'][3] == b'W':
        lon = -lon
    alt, gpstime, gpsdate, clockdatetime = np.nan, np.nan, np.nan, np.nan
    try:
        alt = exif_dict['GPS'][6][0] / exif_dict['GPS'][6][1]
        if exif_dict['GPS'][5]==1:
            alt = -alt
    except:
        pass
    try:
        gpstime = str(exif_dict['GPS'][7][0][0]) + ':' + str(exif_dict['GPS'][7][1][0]).zfill(2) + ':' + str(exif_dict['GPS'][7][2][0]).zfill(2)
    except:
        pass
    try:
        gpsdate = exif_dict['GPS'][29].decode("utf-8") 
    except:
        pass
    try:
        clockdatetime = exif_dict['Exif'][36867].decode('utf-8')
    except:
        pass
        
    return lon,lat,alt,gpstime,gpsdate,clockdatetime


# Decimal Degree to EXIF tuple
# You still need to add manually add correction for NS / EW in EXIF
def dd_to_exif_tuple(dd):
    dd = np.abs(dd)
    d = int(np.floor(dd))
    m = int(np.floor(60 * (dd - d)))
    s = (dd - d - m/60) * 3600
    ss = int(np.floor(10000 * s))

    tup = ((d,1),(m,1),(ss,10000))
    return tup


def read_geotags_into_df(fns,return_datetimes=True):
    df = pd.DataFrame()
    for fn in fns:
        # Open the image, and read its exif information
        with Image.open(fn) as im:
            exif_dict = piexif.load(im.info["exif"])
            lon,lat,alt,gpstime,gpsdate,clockdatetime = exif_dict_to_dd(exif_dict)
            if isinstance(gpsdate,str):
                gpsdate = str(gpsdate).replace(':','-')
                gpsdatetime = gpsdate + ' ' + gpstime
            else:
                gpsdatetime = np.nan
            
            #print(gpsdatetime)
            df = df.append([[fn,lat,lon,alt,gpsdatetime,clockdatetime]],ignore_index=True)
    df = df.rename({0:'fn',1:'lat',2:'lon',3:'alt',4:'datetime_gps',5:'datetime_clock'},axis=1)   
    if return_datetimes:
        df['datetime_gps'] = pd.to_datetime(df['datetime_gps'])

        # TODO
        # Need to convert datetime clock to datetime here.        
        
    return df


#%%
def stringify_time(series,how='time'):
    if how=='datetime':
        return series.dt.strftime('%Y:%m:%d %H:%M:%S.%f').str[:-5]
    else:
        return series.dt.strftime('%H:%M:%S.%f').str[:-5]  

#%%

def fix_gopro_bad_time_resolution(series):
    df = pd.DataFrame(series)
    df = df.rename({df.columns.values[0]:'key'},axis=1)
    df['count'] = 0
    group = df.groupby('key') 
    datetime_counts = pd.DataFrame(group.count()).iloc[:,0].reset_index() 
    df.drop('count',axis=1,inplace=True)
    #datetime_counts['count'] = datetime_counts.iloc[:,-1]
    #datetime_counts.drop(1,axis=1,inplace=True)    
    #datetime_counts = df.rename(columns={ df.columns[0]: "count" })
    # datetime_counts = datetime_counts.rename({'datetime_gps':'count'},axis=1)
    df = df.merge(datetime_counts,how='left',on='key')
    
    df['increment'] = 1  
    df['add_to'] = 0
    
    last_time = -1
    increment = 1
    for i in range(len(df)):
        this_time = df.loc[i,'key']
        if this_time != last_time:
            increment = 1
        else:
            increment = increment + 1
        df.loc[i,'increment'] = increment      
        last_time = this_time
    
    idx = (df['count']>=2) & (df['increment']==2)
    df.loc[idx,'add_to'] = .5
    idx = (df['count']==1) & (df['increment']==1)
    df.loc[idx,'add_to'] = .5
    idx = (df['count']==3) & (df['increment']==3)
    df.loc[idx,'add_to'] = 1
        
    value = df['key'] + pd.to_timedelta(df['add_to'], unit='seconds')
    
    return value


#%%
# https://pythonhealthcare.org/2018/04/15/64-numpy-setting-width-and-number-of-decimal-places-in-numpy-print-output/

def set_print_options(places=2,width=0):
    set_np = '{0:' + str(width) + '.' + str(places) + 'f}'
    np.set_printoptions(formatter={'float': lambda x: set_np.format(x)})
    pd.options.display.float_format = set_np.format
    
#%%

# For development, see \data\Projects\rpy_to_opk
# Yaw is the same as Heading

def ypr2opk(yaw,pitch,roll=0):
    
    if roll is not 0:
        print('Roll values other than zero not yet supported.')
    
    kappa = -yaw
    
    # Phi and Omega are x and y cartesian coordinates on the unit circle, 
    # multiplied by the pitch angle off nadir.  The function assumes the pitch
    # is specied as off the horizon angle, the same as DJI drones.
    phi = -(90+pitch)*np.cos((2.5*np.pi - np.deg2rad(yaw)) % (2*np.pi))     #x
    omega = (90+pitch)*np.sin((2.5*np.pi - np.deg2rad(yaw)) % (2*np.pi))    #y
    
    return omega, phi, kappa


#%%

def track2azimuth(lat,lon):

    lat1 = lat[:-1]
    lat2 = lat[1:]
    lon1 = lon[:-1]
    lon2 = lon[1:]

    geodesic = pyproj.Geod(ellps='WGS84')
    fwd_azimuth,back_azimuth,distance = geodesic.inv(lon1, lat1, lon2, lat2)
    
    # Add last value on to the end, 
    fwd_azimuth = np.append(fwd_azimuth,fwd_azimuth[-1])
    
    fwd_azimuth = np.mod(fwd_azimuth + 360,360)
    
    return fwd_azimuth



#%%

# This has not been thoroughly tested
# TODO, add other kernels at https://www.nearearthimaginglab.org/lab_reports/2020/2020-08-13/pingel.20200813.pdf
# For simple binary kernels, just use disk(radius)

def distance_kernel(radius,cellsize=1,method='binary',idw_power=2):

    radius_in_pixels = radius / cellsize
    window = (np.round(2 * radius_in_pixels)).astype(int)
    if window%2 == 0:
        window = window + 1
    xi,yi = np.meshgrid(np.arange(window)-np.floor(window/2),np.arange(window)-np.floor(window/2))
    D = (xi**2 + yi**2)**.5
    
    if method=='idw':
        return 1 / D**idw_power
    elif method=='binary':
        return D < radius/cellsize
    elif method=='distance':
        return D
    else:
        return D
    
#%%
# From Tobler. 1993. THREE PRESENTATIONS ON GEOGRAPHICAL ANALYSIS AND MODELING
# Returns velocity in km/hr
def lcp_cost_tobler_hiking_function(S,symmetric=True):

    # Convert to dz/dx
    S = np.tan(np.deg2rad(S))
    
    V = 6 * np.exp(-3.5 * np.abs(S + .05))
    
    if symmetric:
        V2 = 6 * np.exp(-3.5 * np.abs(-S + .05))
        V = (V + V2) / 2
        
    return 1 / V

#%%%
# From Rademaker et al. (2012)

# weight of traveler is given in kg
# weight of pack is given in kg
# terrain coefficients greater than 1 introduce "friction"
# velocity is Walking speed in meters per second

def lcp_cost_rademaker(S,weight=50,pack_weight=0,terrain_coefficient=1.1,velocity=1.2):
   
    # Rademaker assumes a grade in percent (0 to 100, rather than 0 to 1):
    G = 100 * np.arctan(np.deg2rad(S))
    
    W = weight
    L = pack_weight
    tc = terrain_coefficient
    V = velocity
    
    # Cost, in MWatts
    MW = 1.5*W + 2.0 * (W + L) * ((L/W)**2) + tc * (W+L) * (1.5 * V**2 + .35 * V * G)
    
    return MW


#%%

def lcp_cost_pingel_exponential(S,scale_factor=9.25):

    EXP = stats.expon.pdf(0,0,scale_factor) / stats.expon.pdf(S,0,scale_factor) 
    
    return EXP
    
#%%    
    
def ve(S,ve=2.3):
    S = np.tan(np.deg2rad(S))
    S = np.rad2deg(np.arctan(ve *  S))
    return S

#%%

def scaled_morphometry(X,cellsize=1,lookup_pixels=1):
    
    L = cellsize * lookup_pixels

    nrows, ncols = np.shape(X)
    
    z1 = ashift(X,0,lookup_pixels)
    z2 = ashift(X,1,lookup_pixels)
    z3 = ashift(X,2,lookup_pixels)
    z4 = ashift(X,7,lookup_pixels)
    z6 = ashift(X,3,lookup_pixels)
    z7 = ashift(X,6,lookup_pixels)
    z8 = ashift(X,5,lookup_pixels)
    z9 = ashift(X,4,lookup_pixels)

    # From Wood (1991), pages 91 and 92
    A = (z1 + z3 + z4 + z6 + z7 + z9)/(6*L**2) - (z2+X+z8)/(3*L**2)    # Fxx
    B = (z1  + z2 + z3 + z7 + z8 + z9)/(6*L**2) - (z4+X+z6)/(3*L**2)   # Fyy
    C = (z3 + z7 - z1 -z9) / (4*L**2)                                  # Fxy
    D = (z3+z6+z9-z1-z4-z7) / (6*L)                                    # Fx
    E = (z1+z2+z3-z7-z8-z9)/(6*L)                                      # Fy
    F = (2*(z2+z4+z6+z8)-(z1+z3+z7+z9)+5*X) / 9
    
    SM = {}


    np.seterr(divide='ignore', invalid='ignore')    
    SM['A'] = np.mod(270-np.rad2deg(np.arctan2(E,D)),360)
    SM['S'] = np.rad2deg(np.arctan((D**2 + E**2)**.5))
    SM['K'] = -2 * (A + B)
    SM['K_profile'] = -(A*D**2 + 2*C*D*E+B*E**2) / ((D**2+E**2)*((D**2+E**2+1)**1.5))  # New as seen in Schmidt Table 1
    SM['K_cross'] = -2 * (B*D**2 + A*E**2 - C*D*E) / (D**2 + E**2)
    SM['K_long'] = -2 * (A*D**2 + B*E**2 + C*D*E) / (D**2 + E**2)
    SM['K_tan'] = -(A*E**2 - 2*C*D*E + B*D**2) / ((D**2 + E**2)*((D**2 + E**2 + 1)**.5)) # As seen in Schmidt Table 1
    SM['K_plan'] = -(A*E**2 - 2*C*D*E + B*D**2) / (D**2+E**2)**1.5
    np.seterr(divide='warn', invalid='warn')
             
        
    return SM


#%%

def score(A,B,k=100000,mask=None):
    if mask is None:
        A = A.flatten()
        B = B.flatten()
    else:
        A = A[mask].flatten()
        B = B[mask].flatten()
        
    if k > len(A):
        k = len(A)
        
    s = np.random.choice(len(A),k,replace=True)
    kappa = cohen_kappa_score(A[s],B[s])
    cmatrix = confusion_matrix(A[s],B[s])
    f1 = f1_score(A[s],B[s])
    ac = accuracy_score(A[s],B[s])
    
    result = {'cohen_kappa_score':kappa,
              'confusion_matrix':cmatrix,
              'f1_score':f1,
              'accuracy_score':ac}
    
    return result
    