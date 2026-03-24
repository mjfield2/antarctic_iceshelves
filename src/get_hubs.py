import numpy as np
import xarray as xr
import multiprocessing as mp
from pathlib import Path
import psutil
import time
import os

from postprocessing import get_beds, upscale_beds, hub_elevations

if __name__ == '__main__':

    tic_begin = time.time()

    hubs_continent = np.load(Path('../continent_hubs.npy'))

    with os.scandir(Path('../iceshelves')) as entries:
        for entry in entries:
            if entry.name.startswith('.'):
                continue
            print(f'Upscaling and finding hubs for for {entry.name}')
            base_path = Path(entry.path)
            os.makedirs(base_path/'results_gn', exist_ok=True)
    
            with os.scandir(entry.path) as iceshelf_files:
                for iceshelf_file in iceshelf_files:
                    if iceshelf_file.name.endswith('.nc'):
                        dataset_path = iceshelf_file.path
    
            
            ds = xr.load_dataset(dataset_path)
            
            beds, losses = get_beds(base_path/'results_gn')
            
            grid = xr.open_dataset(Path('D:/bedmachine/BedMachineAntarctica-v3.nc'))
            
            xx, yy = np.meshgrid(ds.x, ds.y)
            
            # trim original BedMachine, get coordinates
            x_trim = (grid.x >= np.min(xx)) & (grid.x <= np.max(xx))
            y_trim = (grid.y >= np.min(yy)) & (grid.y <= np.max(yy))
            grid = grid.sel(x=x_trim, y=y_trim)
            xx_bm, yy_bm = np.meshgrid(grid.x.values, grid.y.values)

            trim_msk = np.outer(y_trim, x_trim)
            hub_trim = hubs_continent[trim_msk].reshape((np.count_nonzero(y_trim), np.count_nonzero(x_trim)))
            
            print('upscaling beds')
            
            beds_up = upscale_beds(beds, ds, grid, 10e3, 'spherical')
            
            # HUB calculation
            water_msk = (grid.mask.values==0) | (grid.mask.values==3)
            
            bed = grid.bed.values
            min_bed = bed[water_msk].min()
            min_bed = -2000
            max_bed = bed[water_msk].max()
            vert_res = 1
            
            params = []
            for i in range(beds.shape[0]):
                params.append([grid, beds_up[i,...], water_msk, min_bed, max_bed, vert_res, hub_trim, False, True])
            
            print('starting parallel hubs')
            tic = time.time()
            
            # run in parallel
            n_cores = psutil.cpu_count(logical=False)-1
            with mp.Pool(n_cores) as p:
                result = p.starmap(hub_elevations, params)
            
            toc = time.time()
            print(f'{toc-tic} seconds')
            
            hubs_ensemble = np.array(result)
            np.save(base_path/'results_gn/upscaled_ensemble_gn.npy', beds_up)
            np.save(base_path/'results_gn/upscaled_hubs.npy', hubs_ensemble)

    toc_end = time.time()
    print(f'total time: {toc_end-tic_begin} seconds')