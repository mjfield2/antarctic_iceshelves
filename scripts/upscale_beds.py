import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
import multiprocessing as mp
from pathlib import Path
import psutil
import time
import os

import sys
sys.path.append('../src')

from postprocessing import get_beds, upscale_beds, hub_elevations
from bouguer import bm_terrain_effect
from utilities import xy_into_grid

if __name__ == '__main__':

    tic_begin = time.time()

    with os.scandir(Path('../iceshelves')) as entries:
        for entry in entries:
            if entry.name.startswith('.'):
                continue
            print(f'Upscaling and finding hubs for for {entry.name}')
            base_path = Path(entry.path)
            os.makedirs(base_path/'results_gn', exist_ok=True)
    
            with os.scandir(entry.path) as iceshelf_files:
                for iceshelf_file in iceshelf_files:
                    if (iceshelf_file.name.endswith('.nc')==True) & (iceshelf_file.name.startswith('hubs')==False):
                        dataset_path = iceshelf_file.path
                    if iceshelf_file.name.endswith('.csv'):
                        grav_path = iceshelf_file.path
    
            
            ds = xr.load_dataset(dataset_path)
            grav = pd.read_csv(grav_path)
            
            # load results
            beds, losses, targets, seed_ids = get_beds(base_path/'results_gn')
            beds = beds[seed_ids]
            targets = targets[seed_ids]
            
            # forward model BM terrain effect
            g_z = bm_terrain_effect(ds, grav)
            g_z_grid = xy_into_grid(ds.x.values, ds.y.values, (grav.x.values, grav.y.values), g_z)
            grav_mskd = grav[grav.inv_pad==True]
            pred_coords = (grav_mskd.x.values, grav_mskd.y.values, grav_mskd.height.values)
            
            target_grids = np.full(beds.shape, np.nan)
            
            # put targets into grids for saving
            for i in range(targets.shape[0]):
                target_grid = xy_into_grid(ds.x.values, ds.y.values, (grav_mskd.x.values, grav_mskd.y.values), targets[i])
                target_grids[i,...] = np.where(ds.inv_pad.values, target_grid, g_z_grid)

            # reference beds to geoid
            beds -= ds.geoid.values
            
            grid = xr.open_dataset(Path('../raw_data/BedMachineAntarctica-v4.nc'))
            
            xx, yy = np.meshgrid(ds.x, ds.y)
            
            # trim original BedMachine, get coordinates
            x_trim = (grid.x >= np.min(xx)) & (grid.x <= np.max(xx))
            y_trim = (grid.y >= np.min(yy)) & (grid.y <= np.max(yy))
            grid = grid.sel(x=x_trim, y=y_trim)
            
            print('upscaling beds')
            xx_bm, yy_bm = np.meshgrid(grid.x.values, grid.y.values)
            kn = vd.KNeighbors(1)
            kn.fit((xx.flatten(), yy.flatten()), ds.inv_msk.values.flatten())
            inv_dens = kn.predict((xx_bm, yy_bm))
            inv_dens = inv_dens.reshape(xx_bm.shape) > 0.5
            
            beds_up = upscale_beds(beds, ds, grid, inv_dens, 5e3, 'exponential')

            ice_bottom = grid.surface.values - grid.thickness.values

            for i in range(beds_up.shape[0]):
                beds_up[i,...] = np.where(beds_up[i,...] > ice_bottom, ice_bottom, beds_up[i,...])

            ii = np.arange(beds.shape[0])
            ds_beds = xr.DataArray(beds, coords = {'seed_id' : ii, 'y' : ds.y.values, 'x' : ds.x.values})
            ds_beds_up = xr.DataArray(beds_up, coords = {'seed_id' : ii, 'y' : grid.y.values, 'x' : grid.x.values})
            target_da = xr.DataArray(target_grids, coords = {'seed_id' : ii, 'y' : ds.y.values, 'x' : ds.x.values})
            
            ds_beds.to_netcdf(base_path/'results_gn/ensemble_geoid_5000.nc')
            ds_beds_up.to_netcdf(base_path/'results_gn/ensemble_geoid_500.nc')
            target_da.to_netcdf(base_path/'results_gn/target_cache.nc')

    toc_end = time.time()
    print(f'total time: {toc_end-tic_begin} seconds')