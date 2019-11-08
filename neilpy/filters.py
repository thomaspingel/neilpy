# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:09:52 2018

@author: Thomas Pingel
"""

import numpy as np


#%%
    '''
    Simple formula to calculate Getis-Ord Gi when given an array of values
    and the pre-calculate n, global mean, and global variance.    
    '''
def gi_formula(x,n,m,v):
    k = np.sum(np.isfinite(x)).astype(np.int) # number of non-nan neighbors
    Gi =(np.nansum(x) - k*m) / np.sqrt((k * (n-1-k) * v) / (n-2))
    return Gi

#%%
     '''
    Calculated Getis-Ord Gi Statistic of local autocorrelation on a raster.
    For vector-based operations, see the package PySAL.
    
    The user can supply either a bineary footprint (structuring element) or
    can supply a scaler value to indicate a size of structuring element.  In
    this case, a square structuring element with zero at its center is used.
    Users should supply odd-dimension neighborhoods (3x3, 5x5, etc).
    
    References
    ----------
    Ord, J.K. and A. Getis. 1995. Local Spatial Autocorrelation Statistics:
    Distribution Issues and an Application. Geographical Analysis, 27(4): 286-
    306. doi: 10.1111/j.1538-4632.1995.tb00912.x
    
    '''   
def rasterGi(X,footprint,mode='nearest'):
    
    # Cast to a float; these operations won't all work on integers
    X = X.astype(np.float)

    # If a footprint was provided as a size, make a square structuring element 
    # with a zero at the center.
    if np.isscalar(footprint):
        m = np.floor(footprint/2).astype(np.int)
        w = np.ones((footprint,footprint))
        w[m,m] = 0
    else:
        w = footprint
        
    # How many non-nans do we have in the array?
    n = np.sum(np.isfinite(X))

    # A vectorized operation to calculate the global mean and variance at each
    # pixel, excluding that pixel.
    mean_not_me = (np.nansum(X) - X) / (n-1)
    var_not_me = ((np.nansum(X**2) - X**2) / (n-1)) - mean_not_me**2

    # Within the strucutring element how many neighbors at each point?
    if np.all(np.isfinite(X)):
        w_neighbors = np.sum(w) * np.ones(np.shape(X),dtype=np.int)
    else:
        w_neighbors = ndi.filters.generic_filter(np.isfinite(X).astype(np.int),np.sum,footprint=w,mode=mode)

    # Calculate Gi
    a = (ndi.filters.generic_filter(X,np.nansum,footprint=w,mode=mode)) - (w_neighbors* mean_not_me)
    b = np.sqrt((w_neighbors * (n-1-w_neighbors) * var_not_me) / (n-2))
    del mean_not_me, var_not_me
    Z = a / b
    del a,b
    
    # Calculate Z-scores for CIs of 10, 5, and 1 percent (adjust for tails)
    a = st.norm.ppf(.95)
    b = st.norm.ppf(.975)
    c = st.norm.ppf(.995)
    
    # Create an ArcGIS-like Gi_Bin indicating CIs (90/95/99) for above-and-below
    Gi_Bin = np.zeros(np.shape(X)).astype(np.float)
    Gi_Bin[Z>a] = 1
    Gi_Bin[Z>b] = 2
    Gi_Bin[Z>c] = 3
    Gi_Bin[Z<-a] = -1
    Gi_Bin[Z<-b] = -2
    Gi_Bin[Z<-c] = -3
    
    # Return the binned value and the Z-score.  P is not returned since this
    # is directly calculable from Z
    return Gi_Bin, Z

#%%

def terrain_ruggedness(X):
    '''
    The square root of the average of the squared differences between a pixel 
    and its neighbors.
    
    The Terrain Ruggedness Index (TRI) is a measure of the heterogeneity
    of a local neighboorhood in a digital elevation model.  According to the 
    original specification, the neighboorhood is a 3x3 window.  However, any
    odd-sized (3x3, 5x5, etc.) neighbhorhood can be calculated.
    
    References
    ----------
    Riley, S.J., S.D. DeGloria, and R. Elliot. 1999. A Terrain Ruggedness Index That 
    Quantifies Topographic Heterogeneity. Intermountain Journal of Sciences, 
    5(1-4): 23-27.
    
    '''
    
    if X.ndim > 1:
        X = X.ravel()
    n = np.size(X)
    center = int(n / 2)
    X = (X - X[center]) ** 2
    X = np.sum(X).astype(np.float) / (n-1)
    X = np.sqrt(X)
    return X

#%%    
def esri_planar_slope(X):
    '''
    The maximum areal slope of a 3x3 neighborhood.
    
    This function uses the planar formula to calculate a percent slope of the 
    given surface, assuming that horizonal and vertical units are the same.
    
    If cellsize is not 1, you must divide the result by the cellsize for the 
    correct slope value.
    
    If degrees are required, transform the resulting surface using 
    np.rad2deg(np.arctan(S))
    

    References
    ----------
    ESRI. ArcMap version 10.5. How Slope Works.
    
    http://desktop.arcgis.com/en/arcmap/10.5/tools/spatial-analyst-toolbox/how-slope-works.htm

    
    Examples
    ----------
    
    Example 1
    
    >>> import rasterio
    >>> import scipy.ndimage as ndi
    
    >>> with rasterio.open('../sample_data/sample_dem.tif') as src:
        Z = src.read(1)
        Zt = src.affine
        cellsize = Zt[0]
    
    >>> S = ndi.filters.generic_filter(Z,neilpy.filters.esri_planar_slope,size=3)
    >>> S = S / cellsize
    
    >>> S_deg = np.rad2deg(np.arctan(S))
    
    Example 2
    
    >>> Z = np.array([[50,45,50],[30,30,30],[8,10,10]])
    >>> cellsize = 5
    >>> S = neilpy.filters.esri_planar_slope(Z) / cellsize
    >>> print(S)
    3.800328933131973
    >>> S_deg = np.rad2deg(np.arctan(S))
    >>> print(S_deg)
    75.25765769167738
    
 
    
    '''
    
    X = X.reshape((3,3)) 
    dz_dx = (np.sum(X[:,-1] * (1,2,1)) - np.sum(X[:,0] * (1,2,1))) / 8
    dz_dy = (np.sum(X[-1,:] * (1,2,1)) - np.sum(X[0,:] * (1,2,1))) / 8
    return np.sqrt(dz_dx**2 + dz_dy**2)


#%%

 # Found on https://stackoverflow.com/questions/40820955/numpy-average-distance-from-array-center
def grid_distance(shp):
    grid_x, grid_y = np.mgrid[0:shp[0], 0:shp[1]]
    center = int(shp[0] / 2)
    grid_x = grid_x - center
    grid_y = grid_y - center
    grid_distance = np.hypot(grid_x, grid_y)
    return grid_distance


#%%
    
def skyview_filter(X,cellsize=1):
    z=np.size(X)
    w=int(np.sqrt(z))
    c =(int(w/2))
    X=X.reshape(w,w)

    #Calculate height difference
    height = X-X[c,c]
    height = np.clip(height,0,np.inf)
    height[c,c] = np.nan

    g_dist = cellsize * grid_distance(np.shape(X))     

    #Calculate Horizon angle
    horizon_angle=np.arctan(height/g_dist)

    max_angles = [np.max(fetch_values(horizon_angle,i)) for i in range(8)]
    sv = 1 - np.mean(np.sin(max_angles))
    
    return sv

#%%


def openness_filter(X,cellsize=1,skyview=False):
    n = np.size(X)
    n_rows = np.int(np.sqrt(n))
    center = np.int(np.floor(n_rows / 2))
    if np.ndim(X)==1:
        X = np.reshape(X,(n_rows,n_rows))
    X = X - X[center,center]
    
    # Distance matrix
    D = np.meshgrid(np.arange(n_rows) - center, np.arange(n_rows) - center)
    D = cellsize * np.sqrt(D[0]**2 + D[1]**2)
    D[center,center] = np.inf
    
    # Slope to each pixel
    O = 90 - np.rad2deg(np.arctan(X/D))
    
    # Calculate maximum angle for each of the 8 primary directions
    angles = np.array([np.min(fetch_values(O,direction)) for direction in range(8)])
    
    # Skyview is limited to a maximum angle of 90 degrees.  The sin of the 
    # angle is used to normalize to 0/1
    if skyview:
        angles[angles>90] = 90
        angles = np.sin(angles)
    
    # The result is the mean of these angles
    openness = np.mean(angles)
    
    return openness

#%%

def fetch_values(X,direction):
    n_rows, n_cols = np.shape(X)
    center = np.int(np.floor(n_rows / 2))
    if direction==0:
        return X[np.arange(center-1,-1,-1),np.arange(center-1,-1,-1)]
    elif direction==1:
        return X[np.arange(center-1,-1,-1),center]
    elif direction==2:
        return X[np.arange(center-1,-1,-1),np.arange(center+1,n_rows,1)]
    elif direction==3:
        return X[center,np.arange(center+1,2*center+1)]
    elif direction==4:
        return X[np.arange(center+1,2*center+1),np.arange(center+1,2*center+1)]
    elif direction==5:
        return X[np.arange(center+1,2*center+1),center]
    elif direction==6:
        return X[np.arange(center+1,2*center+1),np.arange(center-1,-1,-1)]    
    elif direction==7:
        return X[center,np.arange(center-1,-1,-1)]    
        
        
        
    
    