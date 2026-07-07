import numpy as np
import xarray as xr
from pathlib import Path
import sys
sys.path.append('../src')

from postprocessing import hub_elevations

grid = xr.open_dataset(Path('../raw_data/BedMachineAntarctica-v4.nc'))

water_msk = (grid.mask.values==0) | (grid.mask.values==3)

bed = grid.bed.values
min_bed = -2000
max_bed = bed[water_msk].max()
vert_res = 1

hubs = hub_elevations(grid, bed, water_msk, min_bed, max_bed, vert_res, save_connects=False)

np.save('../continent_hubs.npy', hubs)