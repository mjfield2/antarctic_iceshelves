# general
import numpy as np
import pandas as pd
import xarray as xr
from skgstat import models
import gstatsim as gsm

# io
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
from gaussnewton import *

tic = time.time()

parser = argparse.ArgumentParser(description="Run bathymetry inversions")
# parser.add_argument("--ninvs", type=int, default=200, help="number of inversions to run")
# parser.add_argument("--lamb", type=float, default=0, help="lambda regularization parameter")
parser.add_argument("--filt", type=bool, default=False, help="True for filter")
parser.add_argument("--name", type=str, default='abbot', help="Name of ice shelf")
parser.add_argument("--i", type=int, default=0, help="index of seed")
args = parser.parse_args()

lopts = {
    'abbot' : 5.99e-8,
    'george' : 1.67e-7,
    'getz' : 1.67e-7,
    'larsen' : 4.64e-7,
    'maudeast' : 5.99e-8,
    'maudwest' : 4.64e-7,
    'pineisland' : 4.64e-7,
    'pineisland_new' : 1e-4,
    'salzberger' : 1.67e-7,
    'shackleton' : 1.67e-7,
    'totten' : 1.29e-6
}

lopt = lopts[args.name]

print(f'Running inversion for {args.name} index {args.i}')
print(f'lambda: {lopt}')
print(f'filtering: {args.filt}')

with open(Path('../200_seeds.txt'), 'r') as f:
    lines = f.readlines()

seeds = []
for line in lines:
    seeds.append(int(line.strip()))


n_invs = int(args.high-args.low)

density_dict = {
    'ice' : 917,
    'water' : 1027,
    'rock' : 2670
}

# number of neighbors and max radius
k = 48
rad = 500_000

# make arrays for random field generation
range_max = [50e3, 50e3]
range_min = [30e3, 30e3]
high_step = 50
nug_max = 0.0
eps = 3e-4

print(f'Starting inversions for {args.name}')
dir_path = Path(f'../iceshelves/{args.name}')
os.makedirs(dir_path/'results_gn', exist_ok=True)

with os.scandir(dir_path) as iceshelf_files:
    for iceshelf_file in iceshelf_files:
        if (iceshelf_file.name.endswith('.nc')==True) & (iceshelf_file.name.startswith('hubs')==False):
            dataset_path = iceshelf_file.path
        elif iceshelf_file.name.endswith('csv'):
            grav_path = iceshelf_file.path
            
print(dataset_path)

ds = xr.load_dataset(dataset_path)
grav = pd.read_csv(grav_path)
grav_mskd = grav[grav.inv_pad==True]

xx, yy = np.meshgrid(ds.x.values, ds.y.values)
grav_mask = ds.surface.values < 1500

g_z = bm_terrain_effect(ds, grav)
g_z_grid = xy_into_grid(ds.x.values, ds.y.values, (xx[grav_mask], yy[grav_mask]), g_z, quiet=True)
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
vgrams, experimental, bins = utilities.variograms(xx, yy, res_grid_mod, maxlag=30e3, n_lags=20)
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

min_bound, boug_max_grid = bouguer_lower_bound(ds, grav, trend, density_dict)
bounds = (min_bound, 100)

# gravity calculation coordinates
grav_mskd = grav[grav.inv_pad==True]
pred_coords = (grav_mskd.x.values, grav_mskd.y.values, grav_mskd.height.values)

# make operators
lapop = make_laplace(ds.bed.values)
lapop_trim = trim_laplace(lapop, ds.inv_pad.values)

Wm = lapop_trim
Cm = Wm.T@Wm

stdev = ds.stdev.values[grav_mask & ds.inv_pad.values]
weights = 1/stdev**2

Wd = np.diag(weights)

# run inversion
rng = np.random.default_rng(seeds[args.i])

sim = interpolate.sgs(xx, yy, res_grid_mod, vario, rad, k, sim_mask=ds.inv_msk.values, bounds=bounds, quiet=True, seed=rng)
sim = np.where(ds.inv_msk==False, residual_grid, sim)

target = grav.faa - (sim + trend)[grav_mask]

if args.filt==True:
    boug_filt = filter_boug(ds, grav, target, cutoff=20e3, pad=0)
    target = grav.faa.values - boug_filt

target = target[grav.inv_pad==True]

rfgen = RFGen(ds, range_max, range_min, high_step, nug_max, eps, 'Gaussian', const_var=True, rng=rng)
field = rfgen.generate_field(condition=True)

x0 = ds.bed.values + field
x0 = np.where(x0>ds.surface-ds.thickness, ds.surface-ds.thickness, x0)
prior = x0.astype(np.float64)

bed_cache, rmse_cache = gauss_newton_regularized(ds, prior, target, pred_coords, density_dict, Cm, Wd, max_iter=30, alpha='search', perturb_scale=1, lamb=lopt, stop=0.001, quiet=True)

result = {
    'bed_cache' : bed_cache,
    'loss_cache' : rmse_cache,
    'target' : target,
    'density' : 2670
}

np.save(dir_path/'results_gn'/f'result_{str(seeds[i])[:6]}_{i}', result)

print(f'Inversion {i} done. Seed {seeds[i]}')
    
toc = time.time()
print(f'Time elapsed: {toc-tic:.2f} s')