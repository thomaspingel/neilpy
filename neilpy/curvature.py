# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:51:37 2020

@author: Thomas Pingel
"""
import numpy as np
import scipy.ndimage as ndi
import neilpy

#%%
'''
Laplacian curvature.  Multiply by -100x to get an equivalent result to
ESRI's general curvature.
'''
def curvature(X,cellsize=1):
    return ndi.filters.laplace(X/cellsize)


#%%
'''
ashift pulls a copy of the raster shifted.  0 shifts upper-left to lower right
1 shifts top-to-bottom, etc.  Clockwise from upper left. Use 0 to "grab" the
upper left pixel, 1 to "grab" up top pixel, etc.

'''
def ashift(surface,direction,n=1):
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
'''
esri_curvature is intended to be a perfect copy of ESRI's planar curvature
calculations.  ESRI's calculation method is identical to Zevenbergen and
Throne (1987) except signs are reversed, and the output is mulitplied by 100x.

PROFILE curvature given here is equivalent to other authors' LONGITUDINAL 
curvature, and refers to the rate of change of potential slope down a flow 
line.

PLAN curvature here is the same as other authors' CROSS SECTIONAL curvature, 
and refers to the rate of change of aspect along a contour.

Missing values are returned with a curvature of zero.  It is recommended that
these values be filled beforehand if this behavior is not desired.  Pixels
with neighboring pixels undefined will have those missing values filled with
the center pixel value, as is ESRI's convention.  This same behavior is used
to fill necessary edge pixels as well.
    
The reference to plan and profile curvature refer to the rate of aspect and 
slope 

It returns:
    general curvature, profile curvature, and plan curvature

References
https://dx.doi.org/10.1002/esp.3290120107
https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm
https://support.esri.com/en/technical-article/000005086
'''
def esri_curvature(X,cellsize=1):

    # Match common definition (Wood, ESRI, etc.) of cell size as L
    L = cellsize
    
    # Cells are labelled from top-to-bottom, left-to-right, starting in the
    # upper left corner, and ending in the lower right corner.  Z5 is the 
    # original array, so we simply use X in place of Z5.
    Z1 = ashift(X,0)
    Z2 = ashift(X,1)
    Z3 = ashift(X,2)
    Z4 = ashift(X,7)
    Z6 = ashift(X,3)
    Z7 = ashift(X,6)
    Z8 = ashift(X,5)
    Z9 = ashift(X,4)

    # In cases where data are missing, ESRI uses the original center pixel.
    Z1[np.isnan(Z1)] = X[np.isnan(Z1)]
    Z2[np.isnan(Z2)] = X[np.isnan(Z2)]
    Z3[np.isnan(Z3)] = X[np.isnan(Z3)]
    Z4[np.isnan(Z4)] = X[np.isnan(Z4)]
    Z6[np.isnan(Z6)] = X[np.isnan(Z6)]
    Z7[np.isnan(Z7)] = X[np.isnan(Z7)]
    Z8[np.isnan(Z8)] = X[np.isnan(Z8)]
    Z9[np.isnan(Z9)] = X[np.isnan(Z9)]
    
    # Parameters given in Zevenburgen and Thorne (1987)
    # Values A to C are given here, but are not computed, 
    # as they are not used in the curvature calculations.
    # A = ((Z1 + Z3 + Z7 + Z9)/4 - (Z2 + Z4 + Z6 + Z8)/2 + X)/(L**4);
    # B = ((Z1 + Z3 - Z7 + Z9)/4 - (Z2 - Z8)/2)/(L**3);
    # C = ((-Z1 + Z3 - Z7 + Z9)/4 + (Z4 - Z6)/2) / (L**3);
    D = (((Z4 + Z6) / 2) - X) / (L**2)       # Zxx                    
    E = (((Z2 + Z8) / 2) - X) / (L**2)       # Zyy                     
    F = (-Z1 + Z3 + Z7 - Z9) / (4*(L**2))    # Zxy
    G = (-Z4 + Z6) / (2*L)                   # Zx
    H = (Z2 - Z8) / (2*L)                    # Zy

    del Z1,Z2,Z3,Z4,Z6,Z7,Z8,Z9

    K = -200 * (D + E)

    np.seterr(divide='ignore', invalid='ignore')
    
    K_plan = 200*(D*H**2 + E*G**2 - F*G*H) / (G**2 + H**2)
    K_plan[np.isnan(K_plan)] = 0;

    K_profile = -200 * (D*G**2 + E*H**2 + F*G*H) / (G**2 + H**2)
    K_profile[np.isnan(K_profile)] = 0;
    
    np.seterr(divide='warn', invalid='warn')

    
    return K, K_plan, K_profile

#%%
'''
zevenbergen_and_thorne is similar to ESRI's curvature, but with more
curvatures calculated, and scaling consistent with Z&T's original forumulation.

References
https://dx.doi.org/10.1002/esp.3290120107
https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm
https://support.esri.com/en/technical-article/000005086
'''


def zevenburgen_and_thorne_curvature(X,cellsize=1):

    # Match common definition (Wood, ESRI, etc.) of cell size as L
    L = cellsize
    
    # Cells are labelled from top-to-bottom, left-to-right, starting in the
    # upper left corner, and ending in the lower right corner.  Z5 is the 
    # original array, so we simply use X in place of Z5.
    Z1 = ashift(X,0)
    Z2 = ashift(X,1)
    Z3 = ashift(X,2)
    Z4 = ashift(X,7)
    Z6 = ashift(X,3)
    Z7 = ashift(X,6)
    Z8 = ashift(X,5)
    Z9 = ashift(X,4)

    # Missing values are handled via a "finite differences" approach, given
    # by Wilson and Gallant on page 53, equation 3.8.
    idx = np.isnan(Z1); Z1[idx] = 2 * X[idx] - Z9[idx]
    idx = np.isnan(Z2); Z2[idx] = 2 * X[idx] - Z8[idx]
    idx = np.isnan(Z3); Z3[idx] = 2 * X[idx] - Z7[idx]
    idx = np.isnan(Z4); Z4[idx] = 2 * X[idx] - Z6[idx]
    idx = np.isnan(Z6); Z6[idx] = 2 * X[idx] - Z4[idx]
    idx = np.isnan(Z7); Z7[idx] = 2 * X[idx] - Z3[idx]
    idx = np.isnan(Z8); Z8[idx] = 2 * X[idx] - Z2[idx]        
    idx = np.isnan(Z9); Z9[idx] = 2 * X[idx] - Z1[idx]

    # Parameters given in Zevenburgen and Thorne (1987)
    A = ((Z1 + Z3 + Z7 + Z9)/4 - (Z2 + Z4 + Z6 + Z8)/2 + X)/(L**4);
    B = ((Z1 + Z3 - Z7 + Z9)/4 - (Z2 - Z8)/2)/(L**3);
    C = ((-Z1 + Z3 - Z7 + Z9)/4 + (Z4 - Z6)/2) / (L**3);
    D = (((Z4 + Z6) / 2) - X) / (L**2)       # Zxx                    
    E = (((Z2 + Z8) / 2) - X) / (L**2)       # Zyy                     
    F = (-Z1 + Z3 + Z7 - Z9) / (4*(L**2))    # Zxy
    G = (-Z4 + Z6) / (2*L)                   # Zx
    H = (Z2 - Z8) / (2*L)                    # Zy
    
    # These are used so often they're precomputed, as in Wilson and Gallant
    P = G**2 + H**2                          # Zx**2 + Zy**2
    Q = G**2 + H**2 + 1                      # Zx**2 + Zy**2 + 1

    del Z1,Z2,Z3,Z4,Z6,Z7,Z8,Z9

    K = 2 * (D + E)

    # Cross Sectional is PLAN in Z&T's original forumulation, but is commonly
    # known now in other domains (other than ESRI) this way (Wood, Schmidt, 
    # SAGA, etc.).  Note, SAGA incorrectly returns the sign of this as 
    # reversed.
    np.seterr(divide='ignore', invalid='ignore')
    K_cross = 2*(D*H**2 + E*G**2 - F*G*H) / P
    K_cross[np.isnan(K_cross)] = 0;

    # LONGITUDINAL is PROFILE in Z&T's original forumulation.
    K_long = -2 * (D*G**2 + E*H**2 + F*G*H) / P
    K_long[np.isnan(K_long)] = 0;
    
    # See Krcho (1991) quoted in Schmidt et al. (2003, Table 1) and 
    # Minar et al. (2020, Table 3)
    K_tan = -(D*H**2 -2*F*G*H + E*G**2) / (P*Q**.5)

    # Geometric Profile Curvature (Minar et al. 2020, Table 3)
    K_profile = (D*G**2 + 2*F*G*H + E*H**2) / (P*Q**1.5)
    
    # Geometric Plan Curvature (Minar et al. 2020, Table 3)
    K_plan = -(D*E**2 - 2*F*G*H + E*G**2) / (P**1.5)
    
    np.seterr(divide='warn', invalid='warn')
    
    
    return K, K_cross, K_long, K_tan, K_profile, K_plan

#%%

X = np.array([[2,4,6],[3,6,9],[1,2,4]])
#%%
# Should get .86
print(zevenburgen_and_thorne_curvature(X))