# general
import numpy as np
from numpy.random import PCG64, SeedSequence
import pandas as pd
import verde as vd
import xarray as xr
import skgstat as skg
from skgstat import models
import gstatsim as gsm
from scipy.interpolate import RBFInterpolator, griddata, RegularGridInterpolator
from scipy.stats import qmc
from sklearn.preprocessing import QuantileTransformer
from tqdm.auto import tqdm
from numba import njit, prange
from numba_progress import ProgressBar
import harmonica as hm
from pyproj import Transformer, CRS

# plotting
import matplotlib.pyplot as plt
import cmocean
from cmcrameri import cm

# io
from tqdm.auto import tqdm
import os
from pathlib import Path
import time
import numbers
import warnings
import multiprocessing as mp

import sys
sys.path.append('../src')

import gstatsim_custom as gsim

from gstatsim_custom import interpolate
import prisms

def load_antgg():
    das = []
    for item in os.scandir(Path('D:/AntGG2021_allfiles')):
        if item.name.endswith('.nc'):
            print(item.name)
            da_tmp = xr.open_dataset(item.path)
            das.append(da_tmp)

    antgg = xr.merge(das, compat='override')

    antgg = xr.Dataset(
        data_vars = dict(
            Boug_anom = (['y', 'x'], antgg.Boug_anom.values[::-1,:]),
            grav_dist = (['y', 'x'], antgg.grav_dist.values[::-1,:]),
            gravity_disturbance_5000m = (['y', 'x'], antgg.gravity_disturbance_5000m.values[::-1,:]),
        ),
        coords=dict(
            y=("y", antgg.y.values[::-1]),
            x=("x", antgg.x.values),
        )
    )

    return antgg

def local_gravity(i, j, hw, ds, density_dict):
    if ds.bed_topography.values[i,j]==np.nan:
        return np.nan
    else:
        ni, nj = ds.bed_topography.shape
        xx, yy = np.meshgrid(ds.x, ds.y)
        mask = np.full((ni,nj), False)
    
        ilow = max(0, i-hw)
        ihigh = min(ni, i+hw+1)
        jlow = max(0, j-hw)
        jhigh = min(nj, j+hw+1)
    
        mask[ilow:ihigh,jlow:jhigh] = True
        
        p, dens = prisms.make_prisms(ds, ds.bed_topography.values, density_dict, msk=mask)
        coords_i = (xx[i,j], yy[i,j], 5000)
        g_z = hm.prism_gravity(coords_i, p, dens, field='g_z', progressbar=False)
    return g_z

if __name__ == '__main__':

    tic = time.time()
    
    ds = xr.open_dataset(Path('G:/bedmap_interpolation/processed_data/bedmap3_mod_1000.nc'))
    ds = ds.coarsen(x=10, y=10, boundary='trim').median()
    
    antgg = load_antgg()
    antgg = antgg.coarsen(x=2, y=2, boundary='trim').median()
    
    xmin = antgg.x.min().values
    xmax = antgg.x.max().values
    ymin = antgg.y.min().values
    ymax = antgg.y.max().values
    
    ds = ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    
    # interpolate antgg onto ds coords
    
    xx, yy = np.meshgrid(ds.x, ds.y)
    xx_ant, yy_ant = np.meshgrid(antgg.x, antgg.y)
    
    ant_coords = np.array([xx_ant.flatten(), yy_ant.flatten()]).T
    pred_coords = np.array([xx.flatten(), yy.flatten()]).T
    
    grav_dist = antgg.gravity_disturbance_5000m.values
    grav_dist = griddata(ant_coords, grav_dist.flatten(), pred_coords)
    grav_dist = grav_dist.reshape(xx.shape)
    
    # get bed data
    thick_cond = np.where(ds.mask == 4, 0, ds.thick_cond.values)
    
    bed_cond = ds.surface_topography.values - thick_cond
    ice_rock_msk = (ds.mask == 1) | (ds.mask == 4) | (ds.mask == 2)
    bed_cond = np.where(ice_rock_msk, bed_cond, np.nan)
    xx, yy = np.meshgrid(ds.x, ds.y)
    
    cond_msk = ~np.isnan(bed_cond)
    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    data_cond = bed_cond[cond_msk]
    trend = ds.trend.values
    
    res_cond = bed_cond - trend
    
    ### IBCSO
    bath = xr.open_dataset(Path('D:/IBCSO_v2_allfiles/IBCSO_v2_bed.nc'))
    bath = bath.coarsen(x=20, y=20, boundary='trim').median()
    
    tid = xr.open_dataset(Path('D:/IBCSO_v2_allfiles/IBCSO_v2_tid.nc'))
    tid = tid.coarsen(x=20, y=20, boundary='trim').median()
    
    xx_ibcso, yy_ibcso = np.meshgrid(bath.x, bath.y)
    
    transformer = Transformer.from_crs(9354, 3031)
    xx_ps, yy_ps = transformer.transform(xx_ibcso, yy_ibcso)
    
    ibcso_coords = np.array([xx_ps.flatten(), yy_ps.flatten()]).T
    
    preds_bath = griddata(ibcso_coords, bath.z.values.flatten(), pred_coords)
    preds_tid = griddata(ibcso_coords, tid.tid.values.flatten(), pred_coords)
    
    preds_bath = preds_bath.reshape(xx.shape)
    preds_tid = preds_tid.reshape(xx.shape)
    
    preds_tid = np.rint(preds_tid).astype(int)
    ocean_msk = (preds_tid==10) | (preds_tid==11) | (preds_tid==12) | (preds_tid==13) | (preds_tid==46)
    
    ocean_cond = np.where(ocean_msk, preds_bath, np.nan)
    combined_cond = np.where(ocean_msk, preds_bath, np.nan)
    combined_cond = np.where(cond_msk, bed_cond, combined_cond)
    combined_mask = ~np.isnan(combined_cond)
    
    ds = ds.assign_attrs({'res' : 10_000})
    
    bedmachine_mask = np.where(ds.mask.values==np.nan, 0, ds.mask.values)
    bedmachine_mask = np.where(ds.mask.values==1, 2, bedmachine_mask)
    ds['mask'] = (('y', 'x'), bedmachine_mask)
    ds = ds.rename_vars({'surface_topography' : 'surface'})
    ds = ds.rename_vars({'ice_thickness' : 'thickness'}) 
    
    ds['surface'] = (('y', 'x'), np.where(np.isnan(ds.surface.values), 0, ds.surface.values))
    ds['thickness'] = (('y', 'x'), np.where(np.isnan(ds.thickness.values), 0, ds.thickness.values))
    
    density_dict = {
        'ice' : 917,
        'water' : 1027,
        'rock' : 2670
    }
    
    hw = 10
    ni, nj = ds.bed_topography.values.shape

    params = []

    for i in range(ni):
        for j in range(nj):
            params.append([i, j, hw, ds, density_dict])

    with mp.Pool(7) as p:
        result = p.starmap(local_gravity, params)

    result = np.array(result).reshape(xx.shape)
    np.save('tmp.npy', result)

    toc = time.time()

    print(f'time elapsed: {toc-tic} seconds')