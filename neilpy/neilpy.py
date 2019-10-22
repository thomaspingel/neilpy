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
from scipy import interpolate
from PIL import Image
from skimage.util import apply_parallel
from skimage.morphology import disk

# Global variable to help load data files (PNG-based color tables, etc.)
neilpy_dir = os.path.dirname(inspect.stack()[0][1])



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
"""

# Reads a file into pandas dataframe
# Originally developed as research/current/lidar/bonemap
# A pure python las reader
def read_las(filename):

    with open(filename,mode='rb') as file:
        data = file.read()
    
    point_data_format_key = {0:20,1:28,2:26,3:34,4:57,5:63}
    
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
    format_length = point_data_format_key[header['point_data_format_id']]
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

    # Transform to Pandas dataframe, via a numpy array
    data = pd.DataFrame(np.frombuffer(data,dt))
    data['x'] = data['x']*header['scale'][0] + header['offset'][0]
    data['y'] = data['y']*header['scale'][1] + header['offset'][1]
    data['z'] = data['z']*header['scale'][2] + header['offset'][2]

    def get_bit(byteval,idx):
        return ((byteval&(1<<idx))!=0);

    # Recast the return_byte to get return number (3 bits), the maximum return (3
    # bits), and the scan direction and edge of flight line flags (1 bit each)
    data['return_number'] = 4 * get_bit(data['return_byte'],2).astype(np.uint8) + 2 * get_bit(data['return_byte'],1).astype(np.uint8) + get_bit(data['return_byte'],0).astype(np.uint8)
    data['return_max'] = 4 * get_bit(data['return_byte'],5).astype(np.uint8) + 2 * get_bit(data['return_byte'],4).astype(np.uint8) + get_bit(data['return_byte'],3).astype(np.uint8)
    data['scan_direction'] = get_bit(data['return_byte'],6)
    data['edge_of_flight_line'] = get_bit(data['return_byte'],7)
    del data['return_byte']
    
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
# 1 shifts top-to-bottom, etc.  Clockwise from upper left.
def ashift(surface,direction,n=1,fillnan=False):
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

def openness(Z,cellsize=1,lookup_pixels=1,neighbors=np.arange(8),skyview=False):

    nrows, ncols = np.shape(Z)
        
    # neighbor directions are clockwise from top left,starting at zero
    # neighbors = np.arange(8)   
    
    # Define a (fairly large) 3D matrix to hold the minimum angle for each pixel
    # for each of the requested directions (usually 8)
    opn = np.Inf * np.ones((len(neighbors),nrows,ncols))
    
    # Define an array to calculate distances to neighboring pixels
    dlist = np.array([np.sqrt(2),1])

    # Calculate minimum angles        
    for L in np.arange(1,lookup_pixels+1):
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
    return np.mean(opn,0)

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
def geomorphon_cmap():
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
    
def count_openness(Z,cellsize,lookup_pixels,threshold_angle):
    
    num_pos = np.zeros(np.shape(Z),dtype=np.uint8)
    num_neg = np.zeros(np.shape(Z),dtype=np.uint8)
        
    for i in range(8):        
        O = openness(Z,cellsize,lookup_pixels,neighbors=np.array([i]))
        O = O - openness(-Z,cellsize,lookup_pixels,neighbors=np.array([i]))
        num_pos[O > threshold_angle] = num_pos[O > threshold_angle] + 1
        num_neg[O < -threshold_angle] = num_neg[O < -threshold_angle] + 1
    return num_pos, num_neg
    
#%%
# This is the best go-to function for calcluating a geomorhon from an openness
# calculation.    
def get_geomorphon_from_openness(Z,cellsize=1,lookup_pixels=1,threshold_angle=1,enhance=False):

    
    num_pos, num_neg = count_openness(Z,cellsize,lookup_pixels,threshold_angle)
          
    
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
        this_surface = ndi.morphology.grey_opening(last_surface,footprint=disk(window)) 
        is_object_cell = (is_object_cell) | (last_surface - this_surface > elevation_threshold)
        if i < len(windows) and len(windows)>1:
            last_surface = this_surface.copy()
    return is_object_cell


#%%
def smrf(x,y,z,cellsize=1,windows=18,slope_threshold=.15,elevation_threshold=.5,elevation_scaler=1.25):

    if np.isscalar(windows):
        windows = np.arange(windows) + 1
    
    Zmin,t = create_dem(x,y,z,cellsize=cellsize,bin_type='min');
    is_empty_cell = np.isnan(Zmin)
    Zmin = inpaint_nans_by_springs(Zmin)
    low_outliers = progressive_filter(-Zmin,np.array([1]),cellsize,slope_threshold=5); 
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


