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
from gstatsim_custom import *


# get argumetns from command line
parser = argparse.ArgumentParser(description='Run bathymetry inversions with SGS interpolation')
parser.add_argument('-n', '--ninvs', default=100, type=int, help='number of inversions')
parser.add_argument('-f', '--filt', action='store_true', default=False, help='filter SGS')

args = parser.parse_args()

dir_path = Path('G:/antarctic_iceshelves/iceshelves/totten')

ds = xr.load_dataset(dir_path/'totten.nc')
grav = pd.read_csv(dir_path/'totten_grav.csv')

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
vgrams, experimental, bins = do_variograms(xx, yy, res_grid_mod, maxlag=30e3, n_lags=20)
parameters = vgrams['matern']
vario = {
    'azimuth' : 0,
    'nugget' : parameters[-1],
    'major_range' : parameters[0],
    'minor_range' : parameters[0],
    'sill' : parameters[1],
    'vtype' : 'matern',
    'smoothness' : parameters[2]
}

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
    [21, 10, 60, 1000],
    [15, 8, 40, 1000],
    [9, 6, 40, 5000],
    [5, 5, 40, 10000]
]

# gravity uncertainty
sigma = 1.6

# RMSE stopping condition
stop = 2

# make base PRNG
root_seed = 328613813390984468677358742156199349641
base_seq = SeedSequence()
rng = np.random.default_rng(base_seq)

n_invs = args.ninvs

target_cache_nodens = np.zeros((n_invs, grav.shape[0]))

print(f'running {n_invs} inversions of Totten')

for i in tqdm(range(n_invs)):
    rng_i = np.random.default_rng([i, root_seed])

    # bouguer SGS interpolation
    sim = sgs(xx, yy, res_grid_mod, vario, rad, k, sim_mask=ds.inv_msk.values, quiet=True, seed=rng_i)

    # resample portions that want ice bed above ice shelf bottom
    final_boug = boug_resample(ds, grav, sim, trend, cond_msk, k, vario, rad, rng, density_dict, max_iter_no_change=100, verbose=False)
    final_boug = np.where(ds.inv_msk==False, residual_grid, final_boug)
    target = grav.faa - (final_boug + trend)[grav_mask]

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
    
    path = dir_path/f'results/result_{i}.npy'
    
    result = chain_sequence(sequence, ds, x0, pred_coords, target, sigma, density_dict, rng_i, 
                            weights=None, stop=stop, save=path, full_cache=False, quiet=True, num_mp=i+1)

np.save(dir_path/'results/bouguer_cache.npy', target_cache_nodens)

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
             dir_path/'results',
             dir_path/'results/ensemble_geoid_2000.nc',
             dir_path/'results/ensemble_geoid_500.nc')