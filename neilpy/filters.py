# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:09:52 2018

@author: Thomas Pingel
"""


def topographic_position_index_filter(X):

    if X.ndim > 1:
        X = X.ravel()
    n = np.size(X)

    center = np.floor(n / 2).astype(np.int)
    center_value = X[center]
    rest = np.delete(X,center)
    
    value = (center_value - np.nanmean(rest)) / np.nanstd(rest)
    
    if np.isnan(value) and np.isfinite(center_value):
        value = 0
    
    return value


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
def esri_planar_slope(X,cellsize=1,degrees=True):
    '''
    The maximum areal slope of a 3x3 neighborhood.  No sizes other than 3x3
    should be supplied; unpredictable behavior will result.
    
    This function uses the planar formula to calculate a percent slope of the 
    given surface, assuming that horizonal and vertical units are the same.

    

    References
    ----------
    ESRI. ArcMap version 10.5. How Slope Works.
    
    http://desktop.arcgis.com/en/arcmap/10.5/tools/spatial-analyst-toolbox/how-slope-works.htm

    
    Examples
    ----------
    
    Example 1
    
    >>> import rasterio
    >>> import scipy.ndimage as ndi
    >>> from neilpy.filters import esri_planar_slope
    
    >>> with rasterio.open('../sample_data/sample_dem.tif') as src:
        Z = src.read(1)
        Zt = src.affine
        cellsize = Zt[0]
    
    >>> S = ndi.filters.generic_filter(Z,esri_planar_slope,size=3,mode='nearest',extra_keywords={'cellsize':cellsize,'degrees':True})
    
   
    '''
    
    X = X.reshape((3,3)) 
    dz_dx = (np.sum(X[:,-1] * (1,2,1)) - np.sum(X[:,0] * (1,2,1))) / 8
    dz_dy = (np.sum(X[-1,:] * (1,2,1)) - np.sum(X[0,:] * (1,2,1))) / 8
    S = np.sqrt(dz_dx**2 + dz_dy**2) / cellsize
    if degrees:
        S = np.rad2deg(np.arctan(S))
    return S


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
        
        
        
#%%

def life_filter(neighborhood):
    '''
    A Conway's Game of Life Filter
    '''
    neighborhood = neighborhood.reshape((3,3)) # Added to accommodate filter
    result = 0
    this_cell  = neighborhood[1,1]
    live_neighbors = np.sum(neighborhood) - this_cell
    if this_cell == 1:
        if live_neighbors==2 or live_neighbors==3:
            result = 1
    else:
        if live_neighbors==3:
            result=1
    return result        

#%%

def esri_curvature(X,cellsize=1,kind='curvature'):
    '''
    A simple, pedagogical implementation of ESRI's curvature.  Call with, 
    for example:

        profile_curvature = ndi.filters.generic_filter(Z,esri_curvature,size=3,extra_keywords={'kind':'profile','cellsize':5})    

    References:
        https://www.usna.edu/Users/oceano/pguth/md_help/html/geomorph_curvature.htm
        http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm
    '''
    X = X.reshape((3,3)) 
    
    L = cellsize
    
    # An inefficient, but pedagogically useful expression:
    Z1 = X[0,0]
    Z2 = X[0,1]
    Z3 = X[0,2]
    Z4 = X[1,0]
    Z5 = X[1,1]
    Z6 = X[1,2]
    Z7 = X[2,0]
    Z8 = X[2,1]
    Z9 = X[2,2]
    
    # making some sub-calculations
    A = ((Z1 + Z3 + Z7 + Z9)/4 - (Z2 + Z4 + Z6 + Z8)/2 + Z5)/(L**4);
    B = ((Z1 + Z3 - Z7 + Z9)/4 - (Z2 - Z8)/2)/(L**3);
    C = ((-Z1 + Z3 - Z7 + Z9)/4 + (Z4 - Z6)/2) / (L**3);
    D = (((Z4 + Z6) / 2) - Z5) / (L**2);
    E = (((Z2 + Z8) / 2) - Z5) / (L**2);
    F = (-Z1 + Z3 + Z7 - Z9) / (4*(L**2));
    G = (-Z4 + Z6) / (2*L);
    H = (Z2 - Z8) / (2*L);

    if kind=='curvature':
        curvature = -200 * (D + E)
        if np.isnan(curvature):
            curvature = 0
        return curvature
        
    elif kind=='plan':
        # Plan curvature
        P1 = D*(H**2);
        P2 = E*(G**2);
        P3 = F*G*H;
        P4 = (G**2) + (H**2);
        planc = -200 * ((P1 + P2 - P3) / P4);
        planc[np.isnan(planc)] = 0;
        return planc
    
    elif kind=='profile':
        P1 = D*(G**2);
        P2 = E*(H**2);
        P3 = F*G*H;
        P4 = (G**2) + (H**2);
        profc = 200 * ((P1 + P2 + P3) / P4);
        profc[np.isnan(profc)] = 0;
        return profc
  