import numpy as np
import xarray as xr
from pathlib import Path
import os
import glob

for entry in os.scandir(Path('../iceshelves')):
    if entry.name.startswith('.')==False:
        print(f'Creating hub dataset for {entry.name}')
        
        base_path = Path(entry.path)

        ds = xr.load_dataarray(base_path/'results_gn/ensemble_geoid_500.nc')
        
        fnames = glob.glob(str(base_path/'hubs/*.npy'))

        hubs = np.empty((1, *ds.values.shape)).squeeze()

        for fname in fnames:
            # get the seed id in the file name located at the end
            seed_id = int(fname.split('/')[-1][11:-4])
            hubs[seed_id,...] = np.load(fname)
        
        ii = np.arange(hubs.shape[0])
        hubs_da = xr.DataArray(hubs, coords = {'seed_id' : ii, 'y' : ds.y.values, 'x' : ds.x.values})
        hubs_da.to_netcdf(base_path/'hubs/ensemble_hubs.nc')
        
        
                    
            