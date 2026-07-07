import numpy as np
import xarray as xr
from scipy.interpolate import griddata, RegularGridInterpolator
from tqdm.auto import tqdm
from pathlib import Path
import verde as vd
import os
from utilities import lowpass_filter_invpad
from skgstat import models
from scipy.spatial import KDTree
from skimage.measure import label
import multiprocessing as mp

from utilities import min_dist_from_mask, emplace_data

"""
Run this script to load inversions, upscale them, and save upscaled beds.
"""

def get_beds(path):
    beds = []
    losses = []
    targets = []
    seed_ids = []
    
    for entry in os.scandir(path):
        if 'result' in entry.name and '._' not in entry.name:
            result = np.load(entry.path, allow_pickle=True).item()
            
            bed = result['bed_cache']
            if len(bed.shape)==3:
                beds.append(bed[-1,...])
            else:
                beds.append(bed)
            losses.append(result['loss_cache'])
            targets.append(result['target'])
            seed_ids.append(int(entry.name[:-4].split('_')[-1]))

    beds = np.array(beds)
    targets = np.array(targets)
    return beds, losses, targets, seed_ids

def upscale_beds(beds, ds, grid, inv_msk, weight_range=10e3, weight_fun='spherical'):
    xx_bm, yy_bm = np.meshgrid(grid.x, grid.y)
    distance = min_dist_from_mask(xx_bm, yy_bm, ~inv_msk)
    var_model = {
        'spherical' : models.spherical,
        'gaussian' : models.gaussian,
        'exponential' : models.exponential
    }
    weights = var_model[weight_fun](distance.ravel(), weight_range, 1, 0).reshape(xx_bm.shape)

    pts_eval = np.array([yy_bm.ravel(), xx_bm.ravel()]).T

    interp_beds = np.full((beds.shape[0], *xx_bm.shape), np.nan)

    for i in tqdm(range(beds.shape[0])):
        grid_interp = RegularGridInterpolator((ds.y.values, ds.x.values), beds[i,...], bounds_error=True, method='cubic')
        interp = grid_interp(pts_eval).reshape(xx_bm.shape)
        interp_beds[i,...] = (1-weights)*grid.bed.values + weights*interp

    return interp_beds

def hub_elevations(ds, bed, mask, min_bed, max_bed, vert_res=1, prior=None, save_connects=False, quiet=False):
    ii, jj = np.meshgrid(np.arange(bed.shape[0]), np.arange(bed.shape[1]), indexing='ij')
    amin = np.argmin(bed)
    imin = ii.ravel()[amin]
    jmin = jj.ravel()[amin]

    hubs = np.full(ii.shape, np.nan)
    last_connect = np.full(ii.shape, False)
    elevations = np.arange(min_bed+vert_res, max_bed, vert_res)
    if save_connects==True:
        connects = np.full((elevations.size, *ii.shape), False)

    for i, bed_i in enumerate(tqdm(elevations, disable=quiet)):
        thresh = np.where(mask, bed < bed_i, False)
        groups = label(thresh, connectivity=1)
        connect = groups==groups[imin,jmin]
        if prior is not None:
            connect = connect | (prior < bed_i)
        hubs[connect ^ last_connect] = bed_i
        last_connect = connect

        if save_connects==True:
            connects[i,...] = connect

    if save_connects==True:
        return hubs, connects
    else:
        return hubs

def merged_hubs(i):
    grid = xr.open_dataset(Path('D:/bedmachine/BedMachineAntarctica-v3.nc'))
    xx_bm, yy_bm = np.meshgrid(grid.x, grid.y)
    
    msks = np.full(xx_bm.shape, False)
    
    water_msk = (grid.mask.values==0) | (grid.mask.values==3)
    
    bed = grid.bed.values
    min_bed = -2000
    max_bed = bed[water_msk].max()
    vert_res = 1
    
    ensembles = []
    ensemble_dict = {}
    
    for i, entry in enumerate(os.scandir(Path('iceshelves'))):
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
    
    for name in ensemble_dict.keys():
        ds_j = ensemble_dict[name]
        
        xx_j, yy_j = np.meshgrid(ds_j.x, ds_j.y)
        xmin = np.min(xx_j)
        xmax = np.max(xx_j)
        ymin = np.min(yy_j)
        ymax = np.max(yy_j)
        msk_j = (xx_bm > xmin) & (xx_bm < xmax) & (yy_bm > ymin) & (yy_bm < ymax)
        msks += msk_j
    
        mask_dict[name] = msk_j
    
        bed_j = ds_j.sel(i=i).values
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
    
        save_path = Path(f'iceshelves/{name}/hubs')
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path/f'hub_{i}.npy', hubs_msk)

def get_ensembles(ds, beds, bm_path, filt=False, quiet=True):
    grid = xr.open_dataset(bm_path)
        
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

    beds_up = upscale_beds(beds, ds, grid, 10e3, 'spherical')

    
    beds_up = np.zeros((beds.shape[0], *grid.bed.shape))
    if filt==True:
        beds_filt = np.zeros((beds.shape[0], *ds.bed.values.shape))
        beds_filt_up = np.zeros((beds.shape[0], *grid.bed.shape))

    for i in tqdm(range(beds.shape[0]), disable=quiet):
        beds_up[i,...] = np.where(beds_up_i > grid.surface-grid.thickness, grid.surface-grid.thickness, beds_up_i)

        if filt==True:
            beds_filt[i,...] = lowpass_filter_invpad(ds, beds[i,...], cutoff=10e3)
    
            beds_filt_up_i = upscale_data(ds, grid, beds_filt[i], grid.bed.values, preds_msk, ds.inv_msk.values)
            beds_filt_up[i,...] = np.where(beds_filt_up_i > grid.surface-grid.thickness, grid.surface-grid.thickness, beds_filt_up_i)

    ii = np.arange(beds.shape[0])
    ds_beds = xr.DataArray(beds, coords = {'i' : ii, 'y' : ds.y.values, 'x' : ds.x.values})
    ds_beds_up = xr.DataArray(beds_up, coords = {'i' : ii, 'y' : grid.y.values, 'x' : grid.x.values})
    
    if filt==True:
        ds_beds_filt = xr.DataArray(beds_filt, coords = {'i' : ii, 'y' : ds.y.values, 'x' : ds.x.values})
        ds_beds_filt_up = xr.DataArray(beds_filt_up, coords = {'i' : ii, 'y' : grid.y.values, 'x' : grid.x.values})

        return ds_beds, ds_beds_filt, ds_beds_up, ds_beds_filt_up
    else:
        return ds_beds, ds_beds_up

def load_data(ds, res_path, geoid=False, filter=False):
    """
    Load inversions results from directory

    Args:
        ds : Xarray.Dataset preprocessed BedMachine data
        res_path : path to directory with inversion results
        geoid : reference beds to geoid from ellipsoid
        filter : apply Gaussian lowpass filter to remove edges
    Outputs:
        Beds, mean of beds, stdev of beds, densities, losses
    """
    count = 0
    for entry in os.scandir(res_path):
        if 'result' in entry.name and '._' not in entry.name:
            count += 1
    print(f' {count} inversions')
    
    densities = np.zeros(count)
    last_iter = np.zeros((count, *ds.bed.shape))
    losses = np.zeros((count, 47_000))
    
    i = 0
    for entry in os.scandir(res_path):
        if 'result' in entry.name and '._' not in entry.name:
            result = np.load(entry.path, allow_pickle=True).item()
            # densities[i] = result['density'][0]
            bed = result['bed_cache']
            if len(bed.shape)==3:
                bed = bed[-1,...]
            if filter==True:
                bed_filt = lowpass_filter_invpad(ds, bed, cutoff=5e3)
                last_iter[i] = bed_filt.reshape(bed.shape)
            else:
                last_iter[i] = bed
            losses[i,:result['loss_cache'].size] = result['loss_cache']
            i += 1
            
            
    mean = np.mean(last_iter, axis=0)
    std = np.std(last_iter, axis=0)

    if geoid==True:
        last_iter -= ds.geoid.values
        mean -= ds.geoid.values
        std -= ds.geoid.values

    return last_iter, mean, std, densities, losses

def upscale_data(ds, grid, data, grid_vals, preds_msk, inv_msk, outside=True):
    """
    Upscale data to BedMachine v3 500 m resolution. Data and grid_vals are
    not bathymetry specific so that other fields like standard deviation can
    be upscaled as well.
    Args:
        ds : trimmed and coarsened BedMachine xarray.Dataset used for the inversions
        grid : trimmed BedMachine xarray.Dataset with original resolution
        data : array to upscale
        grid_vals : conditioning data at higher resolution
        preds_msk : inversion domain at higher resolution
        outside : if True, interpolate between the grid_vals outside inversion domain.
            Use True if interpolating coarse bathymetry to higher resolution bathymetry.
    Outputs:
        Data at 500m BedMachine resolution.
    """
    xx_i, yy_i = np.meshgrid(grid.x.values, grid.y.values)
    pred_coords = np.stack([xx_i.flatten(), yy_i.flatten()]).T
    xx_int = xx_i[~preds_msk]
    yy_int = yy_i[~preds_msk]
    interp_coords = np.stack([xx_int, yy_int]).T
    interp_vals = grid_vals[~preds_msk]
    
    xx_g, yy_g = np.meshgrid(ds.x.values, ds.y.values)
    xx_g = xx_g[inv_msk]
    yy_g = yy_g[inv_msk]
    interp_coords_grav = np.stack([xx_g.flatten(), yy_g.flatten()]).T
    interp_vals_grav = data[inv_msk]
    if outside==True:
        interp_vals_i = np.concatenate([interp_vals, interp_vals_grav])
        interp_coords_i = np.concatenate([interp_coords, interp_coords_grav], axis=0)
        
        upscale = griddata(interp_coords_i, interp_vals_i, pred_coords, method='cubic').reshape(grid.bed.shape)
    else:
        upscale = griddata(interp_coords_grav, interp_vals_grav, pred_coords, method='cubic').reshape(grid.bed.shape)
    upscale = np.where(preds_msk, upscale, grid.bed.values)
    return upscale

def save_upscale(ds, grid, preds_msk, data_path, out_path, out_path_up):
    """
    Load inversions, upscale beds, save upscaled beds

    Args:
        ds : trimmed and coarsened BedMachine xarray.Dataset used for the inversions
        grid : trimmed BedMachine xarray.Dataset at original resolution
        preds_msk : inversion domain mask at higher resolution
        data_path : path to directory with inversions
        out_path : path to save upscaled beds
    Outputs:
        None
    """
    beds, _, _, _, _ = load_data(ds, data_path, geoid=True, filter=True)

    beds_up = np.zeros((beds.shape[0], *grid.bed.shape))
    
    for i in tqdm(range(beds.shape[0])):
        beds_up_i = upscale_data(ds, grid, beds[i], grid.bed.values, preds_msk, ds.inv_msk.values)
        beds_up[i,...] = np.where(beds_up_i > grid.surface-grid.thickness, grid.surface-grid.thickness, beds_up_i)

    ii = np.arange(beds.shape[0])
    ds_beds = xr.DataArray(beds, coords = {'i' : ii, 'y' : ds.y.values, 'x' : ds.x.values})
    ds_beds_up = xr.DataArray(beds_up, coords = {'i' : ii, 'y' : grid.y.values, 'x' : grid.x.values})

    # save as netcdf
    ds_beds.to_netcdf(out_path)
    ds_beds_up.to_netcdf(out_path_up)

if __name__ == '__main__':
    # load preprocessed and original BedMachine
    ds = xr.open_dataset(Path('processed_data/xr_2000.nc'))
    grid = xr.open_dataset(Path('raw_data/bedmachine/BedMachineAntarctica-v3.nc'))

    xx, yy = np.meshgrid(ds.x, ds.y)

    # trim original BedMachine, get coordinates
    x_trim = (grid.x >= np.min(xx)) & (grid.x <= np.max(xx))
    y_trim = (grid.y >= np.min(yy)) & (grid.y <= np.max(yy))
    grid = grid.sel(x=x_trim, y=y_trim)
    xx_bm, yy_bm = np.meshgrid(grid.x.values, grid.y.values)

    # interpolate inversion mask to original resolution
    kn = vd.KNeighbors(1)
    kn.fit((xx.flatten(), yy.flatten()), ds.inv_no_muto.values.flatten())
    preds_msk = kn.predict((xx_bm, yy_bm))
    preds_msk = preds_msk.reshape(xx_bm.shape) > 0.5

    # path to where inversion directories are
    base_path = Path('results')

    # save ensemble with conditioning and density
    print('upscaling beds cd')
    save_upscale(ds, grid, preds_msk,
                 base_path/'dens',
                 base_path/'dens_geoid_2000.nc',
                 base_path/'dens_geoid_500.nc')

    # save ensemble with conditioning and no density
    print('upscaling beds cnd')
    save_upscale(ds, grid, preds_msk,
                 base_path/'nodens',
                 base_path/'nodens_geoid_2000.nc',
                 base_path/'nodens_geoid_500.nc')

    # save ensemble with conditioning and no deteministic bouger
    print('upscaling beds c determ')
    save_upscale(ds, grid, preds_msk,
                 base_path/'krige',
                 base_path/'krige_geoid_2000.nc',
                 base_path/'krige_geoid_500.nc')

    # # save ensemble with no conditioning and density
    # print('upscaling beds ucd')
    # save_upscale(grid, preds_msk,
    #              base_path/'uncond_dens',
    #              base_path/'uncond_dens_geoid_2000.nc',
    #              base_path/'uncond_dens_geoid_500.nc')

    # # save ensemble with no conditioning and no density
    # print('upscaling beds ucd')
    # save_upscale(grid, preds_msk,
    #              base_path/'uncond_nodens',
    #              base_path/'uncond_nodens_geoid_2000.nc',
    #              base_path/'uncond_nodens_geoid_500.nc')

    # # save ensemble with no conditioning and no deteministic bouger
    # print('upscaling beds uc determ')
    # save_upscale(grid, preds_msk,
    #              base_path/'uncond_deterministic',
    #              base_path/'uncond_determ_geoid_2000.nc',
    #              base_path/'uncond_determ_geoid_500.nc')