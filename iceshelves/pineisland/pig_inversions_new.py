# general
import numpy as np
from numpy.random import PCG64, SeedSequence
import pandas as pd
import verde as vd
import harmonica as hm
from scipy import interpolate
import xarray as xr
import cmocean
from cmcrameri import cm
import geopandas as gpd
from skgstat import models
import gstatsim as gsm

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import animation

# io
from tqdm.auto import tqdm
import os
from pathlib import Path
import time
import argparse

import sys
sys.path.append('../../src')
sys.path.append('src')

from preprocessing import *
from block_update import *
from bouguer import *
from diagnostics import *
from rfgen import *
from utilities import *
from postprocessing import *
# from gstatsim_custom import *

from gstatsim_custom import interpolate, utilities

# get argumetns from command line
parser = argparse.ArgumentParser(description='Run bathymetry inversions with SGS interpolation')
parser.add_argument('-n', '--ninvs', default=100, type=int, help='number of inversions')
parser.add_argument('-f', '--filt', action='store_true', default=False, help='filter SGS')

args = parser.parse_args()

dir_path = Path('G:/antarctic_iceshelves/iceshelves/pineisland')

ds = xr.load_dataset(dir_path/'pig_new.nc')
grav = pd.read_csv(dir_path/'pig_grav_new.csv')

xx, yy = np.meshgrid(ds.x.values, ds.y.values)
grav_mask = ds.surface.values < 1500

g_z = bm_terrain_effect(ds, grav)
g_z_grid = xy_into_grid(ds, (xx[grav_mask], yy[grav_mask]), g_z)
gdist_masked = np.where(grav_mask, ds.grav_1500.values, np.nan)
boug_grid = gdist_masked-g_z_grid
boug_dist = boug_grid[grav_mask]

# make trend with RBF and get residual field
trend = rbf_trend(ds, grav, boug_dist, smoothing=1e10, full_grid=True)
residual_grid = boug_grid-trend

# exclude residual data more than 25 mGal from conditioning
cond_msk = (np.abs(residual_grid)<25) & (ds.inv_msk==False)
res_grid_mod = np.where(cond_msk, residual_grid, np.nan)

# experimental variogram and model for interpolation
vgrams, experimental, bins = utilities.variograms(xx, yy, res_grid_mod, maxlag=30e3, n_lags=30)
parameters = vgrams['matern']
vario = {
    'azimuth' : 0,
    'nugget' : parameters[-1],
    'major_range' : parameters[0],
    'minor_range' : parameters[0],
    'sill' : parameters[1],
    'vtype' : 'matern',
    's' : parameters[2]
}

bed_max = np.where(ds.mask==3, (ds.surface-ds.thickness).values, ds.bed.values)

density_dict = {
    'ice' : 917,
    'water' : 1027,
    'rock' : 2670
}
pred_coords = (grav.x, grav.y, grav.height)

prisms, densities = make_prisms(ds, bed_max, density_dict)
g_z_max = hm.prism_gravity(pred_coords, prisms, densities, field='g_z')
boug_max = grav.faa - g_z_max
boug_max_grid = xy_into_grid(ds, (pred_coords[0], pred_coords[1]), boug_max)
min_bound = boug_max_grid - trend

bounds = (min_bound, 100)

# number of neighbors and max radius
k = 50
rad = 500_000

# random number generator
rng = np.random.default_rng(seed=0)

# make arrays for random field generation
range_max = [50e3, 50e3]
range_min = [30e3, 30e3]
high_step = 300
nug_max = 0.0
eps = 3e-4

density_dict = {
    'ice' : 917,
    'water' : 1027,
    'rock' : 2670
}

# gravity calculation coordinates
grav_mskd = grav[grav.inv_pad==True]
pred_coords = (grav_mskd.x.values, grav_mskd.y.values, grav_mskd.height.values)

# block size, range, amplitude, iterations
sequence = [
    # [21, 10, 60, 1000],
    # [15, 8, 40, 1000],
    [9, 6, 40, 5000],
    [5, 5, 40, 10000]
]

# gravity uncertainty
sigma = 1.6

# RMSE stopping condition
stop = 0.8

# make base PRNG
root_seed = 328613813390984468677358742156199349641
base_seq = SeedSequence()
rng = np.random.default_rng(base_seq)

n_invs = args.ninvs

target_cache_nodens = np.zeros((n_invs, grav.shape[0]))

print(f'running {n_invs} inversions of Abbot')

for i in tqdm(range(n_invs)):
    rng_i = np.random.default_rng([i, root_seed])

    # bouguer SGS interpolation
    sim = interpolate.sgs(xx, yy, res_grid_mod, vario, rad, k, sim_mask=ds.inv_msk.values, quiet=True, seed=rng_i, bounds=bounds)

    target = grav.faa - (sim + trend)[grav_mask]

    if args.filt == True:
        boug_filt = filter_boug(ds, grav, target, cutoff=12e3, pad=0)
        target = grav.faa.values - boug_filt

    # save target
    target_cache_nodens[i,:] = target
    
    # trim to mask
    target = target[grav.inv_pad==True]

    # initial pertubation away from BedMachine
    rfgen = RFGen(ds, range_max, range_min, high_step, nug_max, eps, 'Gaussian', rng=rng_i)
    x0 = ds.bed.data + rfgen.generate_field(condition=True, seed=rng_i.integers(10_000, 20_000, 1))
    x0 = np.where(x0>ds.surface-ds.thickness, ds.surface-ds.thickness, x0)
    
    path = dir_path/f'results_new/result_{i}.npy'
    
    result = chain_sequence(sequence, ds, x0, pred_coords, target, sigma, density_dict, rng_i, 
                            weights=None, stop=stop, save=path, full_cache=False, quiet=True, num_mp=i+1)

np.save(dir_path/'results_new/bouguer_cache_new.npy', target_cache_nodens)

### Upscale beds to 500 m resolution
grid = xr.open_dataset(Path('G:/stochastic_bathymetry/raw_data/bedmachine/BedMachineAntarctica-v3.nc'))

xx, yy = np.meshgrid(ds.x, ds.y)

# trim original BedMachine, get coordinates
x_trim = (grid.x >= np.min(xx)) & (grid.x <= np.max(xx))
y_trim = (grid.y >= np.min(yy)) & (grid.y <= np.max(yy))
grid = grid.sel(x=x_trim, y=y_trim)
xx_bm, yy_bm = np.meshgrid(grid.x.values, grid.y.values)

# interpolate inversion mask to original resolution
kn = vd.KNeighbors(1)
kn.fit((xx.flatten(), yy.flatten()), ds.inv_msk.values.flatten())
preds_msk = kn.predict((xx_bm, yy_bm))
preds_msk = preds_msk.reshape(xx_bm.shape) > 0.5

# save ensemble with conditioning and density
print('upscaling beds')
save_upscale(ds, grid, preds_msk,
             dir_path/'results_new',
             dir_path/'results_new/ensemble_geoid_2000.nc',
             dir_path/'results_new/ensemble_geoid_500.nc')