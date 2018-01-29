# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:09:52 2018

@author: Thomas Pingel
"""

import numpy as np

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


def openness(X,cellsize=1,skyview=False):
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
    if skyview:
        angles[angles>90] = 90
    
    # Openness is the mean of these angles
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
        
        
        
    
    