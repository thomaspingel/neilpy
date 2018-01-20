# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:38:51 2017

@author: Thomas Pingel
"""
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rasterio
#%%

with rasterio.open('../sample_data/sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.affine
    
    
#%%

with rasterio.open('../neilpy_data/poland_30m.tif') as src:
    Z = src.read(1)
    Zt = src.transform
    
#%% 42 mins
then = time.time()
G = get_geomorphon_from_openness(Z,Zt[0],10,1)
now = time.time()
print(now-then)

#%% at 1000, 26 minutes

# Calculate the geomorphons (a numeric code, 1-10)
then = time.time()
cellsize = Zt[0]
lookup_pixels = 50
threshold_angle = 1
def gm_wrap(I):
    this_G = get_geomorphon_from_openness(I,cellsize,lookup_pixels,threshold_angle)
    return this_G
G = apply_parallel(gm_wrap,Z.copy(),1000,lookup_pixels)
now = time.time()
print(now-then)

#%%

im = Image.fromarray(G,mode='L')
im.putpalette(geomorphon_cmap())
plt.imshow(im)
plt.show()


im.save('../neilpy_data/poland_30m_geomorphons.png')
#%%
write_worldfile(Zt,'../neilpy_data/poland_30m_geomorphons.pgw')

#%%  SMRF TESTING
fns = glob.glob(r'C:\Temp\Reference\*.txt')
total_error = np.zeros(len(fns))
for i,fn in enumerate(fns):
    df = pd.read_csv(fn,header=None,names=['x','y','z','g'],delimiter='\t')
    x,y,z = df.x.values,df.y.values,df.z.values
    windows = np.arange(18) + 1
    cellsize= 1
    slope_threshold = .15
    elevation_threshold = .5
    elevation_scaler = 1.25
    
    result = neilpy.smrf(x,y,z,windows,cellsize,slope_threshold,elevation_threshold,elevation_scaler)
    
    total_error[i] = 1 - np.sum(result[3] == df.g) / len(df)
    
    print(fn,':',total_error[i])

print('Mean total error',np.mean(total_error))
print('Median total error',np.median(total_error))

#%%
plt.imshow(Zpro)


#%%
values = [f(p[0],p[1])[0][0] for p in zip(row,col)]

#%% progressive_filter
header, df = read_las('DK22_partial.las')
Z,t = create_dem(df.x,df.y,df.z,resolution=5,bin_type='min');
Z = apply_parallel(inpaint_nans_by_springs,Z.copy(),100,10)


#%%
slope_threshold = .15
windows = np.arange(1,10,2)
cellsize = 5
OC = progressive_filter(Z,np.arange(1,10),cellsize=5,slope_threshold=.2)
plt.imshow(is_object_cell)   
    
#%%
a = np.arange(3) # cols, x
b = np.arange(3) + 1 # rows, y
c = np.arange(9).reshape((3,3))
print(c)
g = interpolate.RegularGridInterpolator((a,b),c)
print(g((2,3)))  #col/x "2" and row/y "3"

#%%
c,r = ~t * (df.x.values,df.y.values)
f = interpolate.RectBivariateSpline(row_centers,col_centers,Zpro)
values = [f(p[0],p[1])[0][0] for p in zip(r,c)]

#%%

#Z = np.arange(9).reshape((3,3))
def vipmask(Z,cellsize=1):
    heights = np.zeros(np.size(Z))
    dlist = np.array([np.sqrt(2),1])
    for direction in range(4):
        dist = dlist[direction % 2]
        h0 = ashift(Z,direction) - Z
        h1 = ashift(Z,direction+4) - Z
        heights += triangle_height(h0.ravel(),h1.ravel(),dist*cellsize)
    return heights.reshape(np.shape(Z))
#print(heights)

with rasterio.open('../sample_data/sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.affine
    
V = vipmask(Z)
    

#%% third go
#h0 = np.array([-1,0,0])
#h1 = np.array([1,1,1])
#x_dist = 1

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
    


#%% second go
    
y = np.array([[0,1,2],[2,2,3],[4,4,5]])
z_diff = np.diff(y)
z_diff[:,0] = -z_diff[:,0]

n = np.shape(z_diff)[0]

xdist = 1

a = np.ones(np.shape(z_diff))
b = np.ones(np.shape(z_diff))
a[:,0] = -xdist
b[:,0] = xdist
a[:,1] = z_diff[:,0]
b[:,1] = z_diff[:,1]

cp = np.sqrt(np.abs(np.cross(a,b)))

# Calculate base
base = np.sqrt((2**xdist*np.ones(n))**2 + (z_diff[:,1])**2)
# Calculate height
h = cp/base
print(h)

#%%  First go
y = np.array([[0,1,2],[2,2,3],[4,4,5]])
# y = np.random.rand(100,3)
xdist = 1


n = np.shape(y)[0]

# Calculate cross-product
a = np.hstack((-xdist*np.ones((n,1)),np.reshape(y[:,0]-y[:,1],(n,1)),np.zeros((n,1))))
b = np.hstack((xdist*np.ones((n,1)),np.reshape(y[:,2]-y[:,1],(n,1)),np.zeros((n,1))))
cp = np.abs(np.cross(a,b))
print(cp)
#del a,b
cp = np.sqrt(np.sum(cp**2,axis=1))
# Calculate base
base = np.sqrt((2**xdist*np.ones(n))**2 + (y[:,2] - y[:,1])**2)
# Calculate height
h = cp/base
print(h)


#%%
with rasterio.open('../sample_data/sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.affine
plt.imshow(Z,cmap='terrain',vmin=-500,vmax=2000)
plt.show()
#%%

G = get_geomorphon_from_openness(Z,cellsize=Zt[0],lookup_pixels=25,threshold_angle=1,enhance=False)

#%%
G2 = get_geomorphon_from_openness(Z,cellsize=Zt[0],lookup_pixels=6,threshold_angle=1,enhance=True)
# repair peaks
G[G==2] = G2[G==2]
# repair ridges
G[G==3] = G2[G==3]
#%%
# Apply a "standard" colormap and display the image
im = Image.fromarray(G,mode='L')
im.putpalette(geomorphon_cmap())
plt.imshow(im)
plt.show()

#%%

with rasterio.open('../sample_data/sample_dem_geomorphons.tif') as src:
    G3 = src.read(1)
np.sum(G==G3) / np.size(G3)  
    
#%% Develop for Swiss Shading

# Uh, awesome one!

with rasterio.open('../sample_data/sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.affine
cellsize = Zt[0]
    
"""
color_table = np.zeros((2,2,3),dtype=np.uint8)
color_table[0,0,:] = [110,120,117] # Top Left
color_table[0,1,:] = [242,245,173] # Top Right
color_table[1,0,:] = [128,148,138] # Bottom Left
color_table[1,1,:] = [196,201,168] # Bottom Right

# Top Left, Top Right, Bottom Left, Bottom Right
R = ndi.zoom(np.array([[110,242],[128,196]]).astype(np.uint8),8)
G = ndi.zoom(np.array([[120,245],[148,138]]).astype(np.uint8),8)
B = ndi.zoom(np.array([[117,173],[138,168]]).astype(np.uint8),8)
"""
lut = plt.imread('swiss_shading_lookup_flipped.png')[:,:,:3]
lut = (255*lut).astype(np.uint8)

# Subtract Z_norm from 255 here to invert the colormap
Z_norm = np.round(255 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))).astype(np.uint8)
H = hillshade(Z,cellsize,return_uint8=True)

RGB = np.zeros((np.shape(Z)[0],np.shape(Z)[1],3))
RGB[:,:,0] = lut[:,:,0][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))
RGB[:,:,1] = lut[:,:,1][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))
RGB[:,:,2] = lut[:,:,2][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))

plt.imshow(RGB)

#%% Develop for Swiss Shading; got it!

# Uh, awesome one!

with rasterio.open('../sample_data/sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.affine
cellsize = Zt[0]
    
"""
color_table = np.zeros((2,2,3),dtype=np.uint8)
color_table[0,0,:] = [110,120,117] # Top Left
color_table[0,1,:] = [242,245,173] # Top Right
color_table[1,0,:] = [128,148,138] # Bottom Left
color_table[1,1,:] = [196,201,168] # Bottom Right

# Top Left, Top Right, Bottom Left, Bottom Right
R = ndi.zoom(np.array([[110,242],[128,196]]).astype(np.uint8),8)
G = ndi.zoom(np.array([[120,245],[148,138]]).astype(np.uint8),8)
B = ndi.zoom(np.array([[117,173],[138,168]]).astype(np.uint8),8)
"""
lut = plt.imread('swiss_shading_lookup.png')[:,:,:3]
lut = (255*lut).astype(np.uint8)

# Subtract 255 here to invert the colormap!
# 
Z_norm = np.round(255 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))).astype(np.uint8)
H = hillshade(Z,cellsize,return_uint8=True)

RGB = np.zeros((np.shape(Z)[0],np.shape(Z)[1],3))
RGB[:,:,0] = lut[:,:,0][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))
RGB[:,:,1] = lut[:,:,1][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))
RGB[:,:,2] = lut[:,:,2][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))

plt.imshow(RGB)


#%%

fig = plt.imshow(RGB)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.margins(0,0)
plt.savefig('just_the_image.png',bbox_inches='tight',pad_inches=0.0)



#%% play around
with rasterio.open('../sample_data/sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.affine
cellsize = Zt[0]
    
"""
color_table = np.zeros((2,2,3),dtype=np.uint8)
color_table[0,0,:] = [110,120,117] # Top Left
color_table[0,1,:] = [242,245,173] # Top Right
color_table[1,0,:] = [128,148,138] # Bottom Left
color_table[1,1,:] = [196,201,168] # Bottom Right
"""
# Top Left, Top Right, Bottom Left, Bottom Right
#R = ndi.zoom(np.array([[40,116],[90,95]]).astype(np.uint8),128)
#G = ndi.zoom(np.array([[38,102],[74,77]]).astype(np.uint8),128)
#B = ndi.zoom(np.array([[74,109],[84,85]]).astype(np.uint8),128)
#lut = np.stack((R,G,B),axis=2)
spec = np.array([[90,74,84],[95,77,85],[40,38,74],[116,102,109]]) # dark
spec = np.array([[129,137,131],[190,192,173],[117,124,121],[244,244,190]]) # swiss
lut = np.zeros((256,256,3),dtype=np.uint8)
lut[:,:,0] = ndi.zoom([[spec[0,0],spec[1,0]],[spec[2,0],spec[3,0]]],128)
lut[:,:,1] = ndi.zoom([[spec[0,1],spec[1,1]],[spec[2,1],spec[3,1]]],128)
lut[:,:,2] = ndi.zoom([[spec[0,2],spec[1,2]],[spec[2,2],spec[3,2]]],128)


#lut = plt.imread('swiss_shading_test.png')[:,:,:3]
#lut = (255*lut).astype(np.uint8)

Z_norm = np.round(255 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))).astype(np.uint8)
H = hillshade(Z,cellsize,return_uint8=True)

#                                                      this is the key
# without the uint8, it's processed as a float, andthe resulting image is
# very different (and cool!                                \/
RGB = np.zeros((np.shape(Z)[0],np.shape(Z)[1],3),dtype=np.uint8)
RGB[:,:,0] = lut[:,:,0][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))
RGB[:,:,1] = lut[:,:,1][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))
RGB[:,:,2] = lut[:,:,2][Z_norm.ravel(),H.ravel()].reshape(np.shape(Z))

plt.imshow(RGB)

#%%
from scipy.misc import imsave
imsave('lut.png',lut)
imsave('orig5.png',RGB)


#%%
with rasterio.open('../sample_data/sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.affine
cellsize = Zt[0]

name = 'swiss'    
RGB = colortable_shade(Z,name,Zt[0])
imsave(name + '.png',RGB)




