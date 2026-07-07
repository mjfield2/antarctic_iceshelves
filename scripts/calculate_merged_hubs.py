import numpy as np
import xarray as xr
from pathlib import Path
from tqdm.auto import tqdm
import sys
import os
import verde as vd
import multiprocessing as mp
import argparse
sys.path.append('../src')

from utilities import emplace_data
from postprocessing import hub_elevations

parser = argparse.ArgumentParser(description="calculate HUB")
parser.add_argument("--low", type=int, default=0, help="low index of seed")
parser.add_argument("--high", type=int, default=100, help="high index of seed")
args = parser.parse_args()

print(f'running {args.low} to {args.high}')

grid = xr.open_dataset(Path('../raw_data/BedMachineAntarctica-v4.nc'))
xx_bm, yy_bm = np.meshgrid(grid.x, grid.y)

msks = np.full(xx_bm.shape, False)

water_msk = (grid.mask.values==0) | (grid.mask.values==3)

bed = grid.bed.values
min_bed = -2000
max_bed = bed[water_msk].max()
vert_res = 1

ensembles = []
ensemble_dict = {}

for entry in os.scandir(Path('../iceshelves')):
    if entry.name.startswith('.')==False:
        with os.scandir(entry.path) as iceshelf_files:
            for iceshelf_file in iceshelf_files:
                if iceshelf_file.name.endswith('.nc'):
                    dataset_path = iceshelf_file.path
        ds_i = xr.open_dataarray(Path(entry.path)/'results_gn/ensemble_geoid_500.nc')
        ensembles.append(ds_i)
        ensemble_dict[entry.name] = ds_i

mask_dict = {}

merged_i = np.full(grid.bed.shape, np.nan)
    
for seed_id in tqdm(range(args.low, args.high)):
    print(seed_id)
    for name in ensemble_dict.keys():
        ds_j = ensemble_dict[name]

        xx_j, yy_j = np.meshgrid(ds_j.x, ds_j.y)
        xmin = np.min(xx_j)
        xmax = np.max(xx_j)
        ymin = np.min(yy_j)
        ymax = np.max(yy_j)
        msk_j = (xx_bm >= xmin) & (xx_bm <= xmax) & (yy_bm >= ymin) & (yy_bm <= ymax)
        msks += msk_j

        mask_dict[name] = msk_j

        bed_j = ds_j.sel(i=seed_id).values
        kn = vd.KNeighbors(k=1)
        kn.fit((xx_j, yy_j), bed_j)
        preds_j = kn.predict((xx_bm[msk_j], yy_bm[msk_j]))

        merged_i = np.where(msk_j, emplace_data(msk_j, preds_j), merged_i)
    merged_i = np.where((msks==True) & (grid.mask.values==3), merged_i, grid.bed.values)
    merged_i = merged_i.astype(np.float32)

    hubs_i = hub_elevations(grid, merged_i, water_msk, min_bed, max_bed, vert_res, save_connects=False, quiet=True)

    for name in mask_dict.keys():
        msk = mask_dict[name]
        hubs_msk = hubs_i[msk]

        ni = (np.count_nonzero(msk, axis=1) > 0).sum()
        nj = (np.count_nonzero(msk, axis=0) > 0).sum()

        hubs_msk = hubs_msk.reshape((ni,nj))

        save_path = Path(f'../iceshelves/{name}/hubs')
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path/f'hub_{seed_id}.npy', hubs_msk)
