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

with rasterio.open('sample_dem.tif') as src:
    Z = src.read(1)
    Zt = src.transform
    
    
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

