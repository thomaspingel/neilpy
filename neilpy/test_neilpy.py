# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:38:51 2017

@author: Thomas Pingel
"""
import time
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