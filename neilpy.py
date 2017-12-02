# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:40:01 2017

@author: Thomas Pingel
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import rasterio

#%%

with rasterio.open('sample_dem.tif') as src:
    Z = src.read(1)
    #Z = Z.astype(np.float)
    # Z[Z==src.nodata] = np.nan
    
#%%
# http://edndoc.esri.com/arcobjects/9.2/net/shared/geoprocessing/spatial_analyst_tools/how_hillshade_works.htm
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

def hillshade(Z,cellsize=1,z_factor=1,zenith=45,azimuth=315):
    zenith, azimuth = np.deg2rad((zenith,azimuth))
    S = slope(Z,cellsize=cellsize,z_factor=z_factor,return_as='radians')
    A = aspect(Z,return_as='radians',flat_as=0)
    H = (np.cos(zenith) * np.cos(S)) + (np.sin(zenith) * np.sin(S) * np.cos(azimuth - A))
    H[H<0] = 0
    return H

def multiple_illumination(Z,cellsize=1,z_factor=1,zeniths=np.array([45]),azimuths=4):
    if np.isscalar(azimuths):
        azimuths = np.arange(0,360,360/azimuths)
    if np.isscalar(zeniths):
        zeniths = 90 / (zeniths + 1)
        zeniths = np.arange(zeniths,90,zeniths)
    H = np.zeros(np.shape(Z))
    for zenith in zeniths:
        for azimuth in azimuths:
            H1 = hillshade(Z,src.transform[0],z_factor=z_factor,zenith=zenith,azimuth=azimuth)
            H = np.stack((H,H1),axis=2)
            H = np.max(H,axis=2)
    return H

def pssm(Z,cellsize=1,ve=2.3,reverse=False):
    P = slope(Z,cellsize=cellsize,return_as='percent')
    P = np.rad2deg(np.arctan(2.3 *  P))
    P = (P - P.min()) / (P.max() - P.min())
    P = np.round(255*P).astype(np.uint8)
    if reverse==False:
        P = plt.cm.bone_r(P)
    else:
        P = plt.cm.bone(P)
    return P

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
S = slope(Z,src.transform[0])   
A = aspect(Z)    
H = hillshade(Z,src.transform[0],z_factor=20)
plt.imshow(H,cmap='gray',vmin=0,vmax=1)

#%%
H = multiple_illumination(Z,cellsize=src.transform[0],z_factor=1,zeniths=2,azimuths=3);
plt.imshow(H,cmap='gray_r',aspect='equal')

#%%
P = pssm(Z,cellsize=src.transform[0],reverse=True)
plt.imshow(P,aspect='equal')

#%%
# zvalue = 1/((np.pi / 180) * cosd(lat)*(sqrt(numer/denom)));
