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
import scipy.ndimage as ndi
from skimage.util import apply_parallel
#%%
import neilpy
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
#spec = np.array([[129,137,131],[190,192,173],[117,124,121],[244,244,190]]) # swiss
spec = np.array([])
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

name = 'gray_high_contrast'
RGB = colortable_shade(Z,name,Zt[0])
imsave(name + '.png',RGB)

#%%
with rasterio.open('../sample_data/sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.affine
cellsize = Zt[0]
I = plt.imread('gray_mash.png')
I = (255 * I[:,:,0]).astype(np.uint8)
H = hillshade(Z,cellsize,return_uint8=True)
Zn = np.round(255 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))).astype(np.uint8)

#%%

O = np.zeros((256,256),dtype=np.float)
O[:] = np.nan
for z in range(256):
    print(z)
    for h in range(256):
        pixels = I[(Zn==z) & (H==h)]
        if len(pixels) > 0:
            m = np.median(pixels)
        else:
            m = np.nan
        O[z,h] = m

#%%

from scipy import interpolate

k = np.random.choice(np.size(Z),1000,replace=False)

# Alternatively, interp2 also works, but is designed for unstructured data and can run more slowly
f = interpolate.interp2d(H.flatten()[k],Z.flatten()[k],I.flatten()[k],kind='cubic')

#%%

def inpaint_nearest(X):
    idx = np.isfinite(X)
    RI,CI = np.meshgrid(np.arange(X.shape[0]),np.arange(X.shape[1]))
    f_near = interpolate.NearestNDInterpolator((RI[idx],CI[idx]),X[idx])
    idx = ~idx
    X[idx] = f_near(RI[idx],CI[idx])
    return X


#%%
X = O.copy()
idx = np.isfinite(X)
RI,CI = np.meshgrid(np.arange(X.shape[0]),np.arange(X.shape[1]))
f = interpolate.interp2d(RI[idx],CI[idx],X[idx],kind='cubic')

#%%
def logfit(x,k=12.5,low=0,high=0):
    low_idx  = x < low
    high_idx = x > 1 - high

    idx = ~((low_idx) | (high_idx))
    
    a = x[idx]
    a = (a - a.min()) / (a.max() - a.min())
    x[idx] = a
    
    
    y = 1 / (1 + np.e **(-k * (x - .5)))
    y[low_idx] = 0
    y[high_idx] = 1
    
    y = (y - y.min()) / (y.max() - y.min())

    
    return y

def roundfit(x):
    y = (x) ** 2
    return y

#%%
    
#%%

def ifit(x,lows=(.1,0),highs=(.9,1),kind='quadratic'):
    a = np.array([0,lows[0],highs[0],1])
    b = np.array([lows[1],lows[1],highs[1],highs[1]])
    if lows[0]==0:
        a,b=a[1:],b[1:]
    if highs[0]==1:
        a,b=a[:-1],b[:-1]
    f = interpolate.interp1d(a,b,kind)
    result = f(x)
    result[result < lows[1]] = lows[1]
    result[result > highs[1]] = highs[1]
    return result

x = np.linspace(0,1)
y = ifit(x,lows = (.1,.5),highs=(.9,.8))
plt.plot(x,y)




#%%
#def loglut():
spec = np.array([[119,119,119],[255,255,255]])
lut = np.zeros((256,256),dtype=np.float)
lut[:] = np.nan
lut[-1,:] = ifit(np.linspace(0,1,256),(.3,0),(.8,spec[1,0]))
lut[:,-1] = ifit(np.linspace(0,1,256),(.05,spec[0,0]),(.7,spec[1,0]))
lut[0,:] = ifit(np.linspace(0,1,256),(.35,0),(.975,spec[0,0]))
lut[np.arange(256),np.arange(256)] = ifit(np.linspace(0,1,256),(.35,0),(.85,spec[1,0]))
lut[:,0] = 0
lut = inpaint_nans_by_springs(lut)

RGB = colortable_shade(Z,lut,Zt[0])
imsave('gray.png',RGB)
imsave('gray_lut_exp.png',lut)

#%%
with rasterio.open('../sample_data/sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.affine
cellsize = Zt[0]

name = 'gray_nice.png'
RGB = colortable_shade(Z,name,Zt[0])
imsave(name + '.png',RGB)

#%%
with rasterio.open('../sample_data/sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.affine
cellsize = Zt[0]

#%%
lookup_pixels = 20
O = ndi.filters.generic_filter(Z,openness_filter,size=2*lookup_pixels+1,extra_keywords={'cellsize':cellsize})

#%%
lookup_pixels = 20
SV = ndi.filters.generic_filter(Z,skyview_filter,size=2*lookup_pixels+1,extra_keywords={'cellsize':cellsize})

#%%
Osk = ndi.filters.generic_filter(Z,openness_filter,size=2*lookup_pixels+1,extra_keywords={'cellsize':cellsize,'skyview':True})

#%%
O2 = neilpy.openness(Z,cellsize,lookup_pixels=20)

#%%


import numpy as np
from skimage import graph
import matplotlib.pyplot as plt

#%%
cs = Z
cellSize = Zt[0]
lg = graph.MCP_Geometric(cs, sampling=(cellSize, cellSize))
startCell = (5, 5)
lcd = lg.find_costs(starts=[startCell])[0]

#%%

I =  np.array([[1,1,1],[1,100,1],[1,1,1]])
lg = graph.MCP_Geometric(I)
startCell = (0,0)
cumCost, tb = lg.find_costs(starts=[startCell])
print(cumCost)

#%%

import numpy as np
import skimage.graph.mcp as mcp

a = np.ones((8, 8), dtype=np.float32)
a[1::2] *= 2.0


class FlexibleMCP(mcp.MCP_Flexible):
    """ Simple MCP subclass that allows the front to travel 
    a certain distance from the seed point, and uses a constant
    cost factor that is independant of the cost array.
    """
    
    def _reset(self):
        mcp.MCP_Flexible._reset(self)
        self._distance = np.zeros((8, 8), dtype=np.float32).ravel()
    
    def goal_reached(self, index, cumcost):
        if self._distance[index] > 4:
            return 2
        else:
            return 0
    
    def travel_cost(self, index, new_index, offset_length):
        return 1.0  # fixed cost
    
    def examine_neighbor(self, index, new_index, offset_length):
        pass  # We do not test this
        
    def update_node(self, index, new_index, offset_length):
        self._distance[new_index] = self._distance[index] + 1

#%%

def test_flexible():
    # Create MCP and do a traceback
    mcp = FlexibleMCP(a)
    costs, traceback = mcp.find_costs([(0, 0)])
    
    # Check that inner part is correct. This basically
    # tests whether travel_cost works.
    assert_array_equal(costs[:4, :4], [[1, 2, 3, 4],
                                       [2, 2, 3, 4],
                                       [3, 3, 3, 4],
                                       [4, 4, 4, 4]])
    
    # Test that the algorithm stopped at the right distance.
    # Note that some of the costs are filled in but not yet frozen,
    # so we take a bit of margin
    assert np.all(costs[-2:, :] == np.inf)
    assert np.all(costs[:, -2:] == np.inf)
    
#%%
    
from skimage import graph

with rasterio.open('../../neilpy_data/peru.tif') as src:
    Z = src.read(1)
    Zt = src.affine
cellsize = Zt[0]
zf = z_factor(Zt[5])

#%%

G = 100 * slope(Z,cellsize,z_factor=zf,return_as='percent')
G[Z==src.nodata] = np.inf
#%%
W = 50
L = 0
tc = 1.1
V = 1.2

MW = 1.5*W + 2.0 * (W + L) * ((L/W)**2) + tc * (W+L) * (1.5 * V**2 + .35 * V * G)

#%%

start = np.round(~Zt * (-72.872,-16.509)).astype(int)[::-1]
end = np.round(~Zt * (-72.708,-15.123)).astype(int)[::-1]
route, cost = graph.route_through_array(MW,start,end)

route = np.array(route)
#%%
plt.imshow(Z,cmap='terrain',vmin=-500,vmax=5000)
plt.plot(route[:,1],route[:,0],'r-')
plt.show()


#%%


def svf2(Z,cellsize=1,lookup_pixels=1):

    nrows, ncols = np.shape(Z)
    neighbors=np.arange(8)        
    # neighbor directions are clockwise from top left,starting at zero

    # This will sum the max angles    
    sum_matrix = np.zeros_like(Z,dtype=np.float)
    
    # Define an array to calculate distances to neighboring pixels
    dlist = np.array([np.sqrt(2),1])

    # Calculate minimum angles        
    for direction in neighbors:
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
tic = time.time()
lookup_pixels = 20
SV = ndi.filters.generic_filter(Z,skyview_filter,size=2*lookup_pixels+1,extra_keywords={'cellsize':cellsize})
toc = time.time()
print(toc-tic)
