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
    

"""
References:
http://stackoverflow.com/questions/16573089/reading-binary-data-into-pandas
https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
LAZ http://howardbutler.com/javascript-laz-implementation.html
"""

# Reads a file into pandas dataframe
# Originally developed as research/current/bonemap
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
