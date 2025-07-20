# general
import numpy as np
import pandas as pd
import verde as vd
import harmonica as hm
from scipy import interpolate
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
from cmcrameri import cm

# io
from tqdm.auto import tqdm
import os
from pathlib import Path
import itertools

def bedmachine_grid(path, region, res=2000):
    grid = xr.open_dataset(path)

    region_pad = vd.pad_region(region, 10_000)

    x_trim = (grid.x > region_pad[0]) & (grid.x < region_pad[1])
    y_trim = (grid.y > region_pad[2]) & (grid.y < region_pad[3])
    
    grid = grid.sel(x=x_trim, y=y_trim)
    
    xx, yy = np.meshgrid(grid.x, grid.y)

    # block reduce with median to desired resolution
    reducer = vd.BlockReduce(np.median, spacing=res, region=region, adjust='region', 
                                 center_coordinates=True)
    
    coordinates = (xx, yy)
    
    data = (
        grid['bed'].values,
        grid['surface'].values,
        grid['thickness'].values,
    )
    
    coords, data = reducer.filter(coordinates, data)
    
    x_uniq = np.unique(coords[0])
    y_uniq = np.unique(coords[1])
    xx_red, yy_red = np.meshgrid(x_uniq, y_uniq)
    
    # use nearest point to predict categorical variables
    kn = vd.KNeighbors(k=1)
    
    cat_data = (
        grid['mask'].values,
        grid['geoid'].values,
        grid['source'].values,
        grid['dataid'].values,
        grid['errbed'].values
    )
    kn_preds = []
    for i, cdata in enumerate(cat_data):
        kn.fit(coordinates, cdata)
        kn_preds.append(kn.predict(coords))
    
    # make new xarray
    ds = vd.make_xarray_grid(
        coordinates=(xx_red, yy_red), 
        data=(
            data[0].reshape(xx_red.shape), 
            data[1].reshape(xx_red.shape), 
            data[2].reshape(xx_red.shape), 
            kn_preds[0].reshape(xx_red.shape), 
            kn_preds[1].reshape(xx_red.shape), 
            kn_preds[2].reshape(xx_red.shape), 
            kn_preds[3].reshape(xx_red.shape), 
            kn_preds[4].reshape(xx_red.shape)),
        data_names=('bed', 'surface', 'thickness', 'mask', 'geoid', 'source', 'dataid', 'errbed'),
        dims=('y', 'x')
    )
    
    # make sure surface is 0 in open water
    ds['surface'] = (('y', 'x'), np.where(ds.mask==0, 0, ds.surface))
    
    # make sure thickness is 0 in open water and at exposed rock
    ocean_rock_msk = (ds.mask==0)^(ds.mask==1)
    ds['thickness'] = (('y', 'x'), np.where(ocean_rock_msk, 0, ds.thickness))
    
    # make sure bed is equal to surface minus thickness under grounded ice
    ds['bed'] = (('y', 'x'), np.where(ds.mask==2, ds.surface-ds.thickness, ds.bed))
    
    # make sure surface is equal to bed at exposed rock
    ds['bed'] = (('y', 'x'), np.where(ds.mask==1, ds.surface, ds.bed))
    
    # make sure bed not above ice in ice shelf
    bed_above_ice_bottom = np.where(ds.bed > (ds.surface-ds.thickness), True, False)
    ds['bed'] = (('y', 'x'), np.where(bed_above_ice_bottom, ds.surface-ds.thickness, ds.bed))
    
    # make mask grounded ice where bed was above ice bottom
    ds['mask'] = (('y', 'x'), np.where(bed_above_ice_bottom, 2, ds.mask))
    
    # reference elevations to WGS84
    ds['bed'] += ds['geoid']
    ds['surface'] += ds['geoid']
    
    return ds

def bedmap3_grid(path, region, res=2000):
    grid = xr.open_dataset(path)

    region_pad = vd.pad_region(region, 10_000)

    x_trim = (grid.x > region_pad[0]) & (grid.x < region_pad[1])
    y_trim = (grid.y > region_pad[2]) & (grid.y < region_pad[3])
    
    grid = grid.sel(x=x_trim, y=y_trim)
    
    xx, yy = np.meshgrid(grid.x, grid.y)

    # block reduce with median to desired resolution
    reducer = vd.BlockReduce(np.median, spacing=res, region=region, adjust='region', 
                                 center_coordinates=True)
    
    coordinates = (xx, yy)
    
    data = (
        grid['bed_topography'].values,
        grid['surface_topography'].values,
        grid['ice_thickness'].values,
        grid['geoid'],
        grid['thickness_uncertainty'].values,
        grid['thick_cond']
    )
    
    coords, data = reducer.filter(coordinates, data)
    
    x_uniq = np.unique(coords[0])
    y_uniq = np.unique(coords[1])
    xx_red, yy_red = np.meshgrid(x_uniq, y_uniq)
    
    # use nearest point to predict categorical variables
    kn = vd.KNeighbors(k=1)
    
    cat_data = (
        grid['mask'].values,
    )
    kn_preds = []
    for i, cdata in enumerate(cat_data):
        kn.fit(coordinates, cdata)
        kn_preds.append(kn.predict(coords))
    
    # make new xarray
    ds = vd.make_xarray_grid(
        coordinates=(xx_red, yy_red), 
        data=(
            data[0].reshape(xx_red.shape), 
            data[1].reshape(xx_red.shape), 
            data[2].reshape(xx_red.shape),
            data[3].reshape(xx_red.shape), 
            data[4].reshape(xx_red.shape), 
            data[5].reshape(xx_red.shape),
            kn_preds[0].reshape(xx_red.shape)
        ),
        data_names=('bed', 'surface', 'thickness', 'geoid', 'uncertainty', 'thick_cond', 'mask'),
        dims=('y', 'x')
    )

    # change mask to BedMachine convention
    new_mask = np.zeros(ds.mask.shape)
    new_mask[ds.mask==3] = 3
    new_mask[ds.mask==np.nan] = 0
    new_mask[ds.mask==1] = 2
    new_mask[ds.mask==4] = 1
    new_mask[ds.mask==2] = 3

    ds['mask'] = (('y', 'x'), new_mask)
    
    # make sure surface is 0 in open water
    ds['surface'] = (('y', 'x'), np.where(ds.mask==0, 0, ds.surface))
    
    # make sure thickness is 0 in open water and at exposed rock
    ocean_rock_msk = (ds.mask==0)^(ds.mask==1)
    ds['thickness'] = (('y', 'x'), np.where(ocean_rock_msk, 0, ds.thickness))
    
    # make sure bed is equal to surface minus thickness under grounded ice
    ds['bed'] = (('y', 'x'), np.where(ds.mask==2, ds.surface-ds.thickness, ds.bed))
    
    # make sure surface is equal to bed at exposed rock
    ds['bed'] = (('y', 'x'), np.where(ds.mask==1, ds.surface, ds.bed))
    
    # make sure bed not above ice in ice shelf
    bed_above_ice_bottom = np.where(ds.bed > (ds.surface-ds.thickness), True, False)
    ds['bed'] = (('y', 'x'), np.where(bed_above_ice_bottom, ds.surface-ds.thickness, ds.bed))
    
    # make mask grounded ice where bed was above ice bottom
    ds['mask'] = (('y', 'x'), np.where(bed_above_ice_bottom, 2, ds.mask))
    
    # reference elevations to WGS84
    ds['bed'] += ds['geoid']
    ds['surface'] += ds['geoid']
    
    return ds

def antgg_grid_from_bm(path, grid, max_height=1500):
    ds = xr.open_dataset(path)
    x = ds.x.values
    y = ds.y.values[::-1]
    h_ell = ds.h_ell.values[::-1,:]
    grav_dist = ds.grav_dist.values[::-1,:]

    xmin = np.min(grid.x.values)
    xmax = np.max(grid.x.values)
    ymin = np.min(grid.y.values)
    ymax = np.max(grid.y.values)

    xx, yy = np.meshgrid(grid.x.values, grid.y.values)

    region = [xmin-10e3, xmax+10e3, ymin-10e3, ymax+10e3]

    x_mask = (x > region[0]) & (x < region[1])
    y_mask = (y > region[2]) & (y < region[3])
    x_size = np.count_nonzero(x_mask)
    y_size = np.count_nonzero(y_mask)

    grav_dist = grav_dist[np.ix_(y_mask, x_mask)]
    h_ell = h_ell[np.ix_(y_mask, x_mask)]
    y = y[y_mask]
    x = x[x_mask]

    interp = interpolate.RegularGridInterpolator((y, x), grav_dist)
    surface_preds = interp((yy, xx))

    dampings = [1, 10, 100, 1000]
    depths = [1e3, 3e3, 4e3, 5e3]
    parameter_sets = [
        dict(damping=combo[0], depth=combo[1])
        for combo in itertools.product(dampings, depths)
    ]
    # Gradient Boosted Equivalent sources
    equivalent_sources = hm.EquivalentSourcesGB(window_size=20e3)
    
    # Use downsampled data since so dense
   #  coordinates = (grav.x[::10], grav.y[::10], grav.height[::10])

    xx_grav, yy_grav = np.meshgrid(x, y)
    pred_coords = (xx, yy, np.full(xx.shape, max_height))
    coordinates = (xx_grav, yy_grav, h_ell)
    
    scores = []
    for params in tqdm(parameter_sets):
        equivalent_sources.set_params(**params)
        score = np.mean(
            vd.cross_val_score(
                equivalent_sources,
                coordinates,
                grav_dist,
            )
        )
        scores.append(score)
    best = np.argmax(scores)
    print("Best score:", scores[best])
    print("Best parameters:", parameter_sets[best])

    

    equivalent_sources = hm.EquivalentSourcesGB(**parameter_sets[best], window_size=20e3)
    equivalent_sources.fit(coordinates, grav_dist)
    leveled = equivalent_sources.predict(pred_coords).reshape(xx.shape)

    return surface_preds, leveled

def bedmachine_plots(bm, figsize=(12,4), vmax=2000):
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
    ax = axs[0]
    im = ax.pcolormesh(bm.x/1000, bm.y/1000, bm.bed, cmap=cm.bukavu, vmin=-vmax, vmax=vmax)
    ax.axis('scaled')
    plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
    ax.set_title('Bed')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    
    ax = axs[1]
    im = ax.pcolormesh(bm.x/1000, bm.y/1000, bm.surface, cmap=cm.batlow)
    ax.axis('scaled')
    plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
    ax.set_title('Surface')
    ax.set_xlabel('X [km]')
    
    ax = axs[2]
    im = ax.pcolormesh(bm.x/1000, bm.y/1000, bm.mask, cmap=cm.glasgow)
    ax.axis('scaled')
    plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
    ax.set_title('Mask')
    ax.set_xlabel('X [km]')
    
    plt.show()

def plot_gravity(bm, surface, upcon, max_height, figsize, vmax):
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
    ax = axs[0]
    im = ax.pcolormesh(bm.x/1000, bm.y/1000, surface, cmap=cm.vik, vmin=-vmax, vmax=vmax)
    ax.axis('scaled')
    ax.set_title('Surface')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    plt.colorbar(im, ax=ax, pad=0.03, aspect=40)

    upcon_masked = np.where(bm.surface > max_height, np.nan, upcon)

    ax = axs[1]
    im = ax.pcolormesh(bm.x/1000, bm.y/1000, upcon_masked, cmap=cm.vik, vmin=-vmax, vmax=vmax)
    ax.axis('scaled')
    ax.set_title('Upward continued')
    ax.set_xlabel('X [km]')
    plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
    plt.show()