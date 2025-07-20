import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import harmonica as hm
from sklearn.preprocessing import QuantileTransformer
import gstatsim as gsm
import skgstat as skg
import xarray as xr
import xrft
import verde as vd
from scipy.interpolate import RBFInterpolator
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

from prisms import make_prisms
from utilities import xy_into_grid, lowpass_filter_invpad
from gstatsim_custom import *

def bm_terrain_effect(ds, grav, rock_density=2670):
    """
    Forward model gravity response of terrain

    Args:
        ds : preprocessed BedMachine xarray.Dataset
        grav : pandas.DataFrame of gravity data
        rock_density : float of rock or background density
    Outputs:
        Terrain effect for use as target of inversion.
    """
    density_dict = {
        'ice' : 917,
        'water' : 1027,
        'rock' : rock_density
    }
    
    prisms, densities = make_prisms(ds, ds.bed.values, density_dict)
    pred_coords = (grav.x, grav.y, grav.height)
    g_z = hm.prism_gravity(pred_coords, prisms, densities, field='g_z')

    return g_z

def variograms(grav, data, bin_func='even', maxlag=100e3, n_lags=70, covmodels=['gaussian', 'spherical', 'exponential']):
    """
    Make experimental variogram and fit covariance models.

    Args:
        grav : pandas.DataFrame of gravity data
        data : the data to make the variogram of
        bin_func : binning function or array of bin edges
        maxlag : maximum lag for experimental variogram
        n_lags : number of lag bins for variogram
        covmodels : covariance models to fit to variogram
        azimuth : orientation in degrees of primary range
    Outputs:
        Dictionary of variograms, pd.DataFrame of dataset, experimental variogram values, bins, and nscore transformer
    """
    x_cond = grav.loc[grav.inv_msk==False, 'x'].values
    y_cond = grav.loc[grav.inv_msk==False, 'y'].values
    data_cond = data[grav.inv_msk==False].reshape(-1,1)
    pred_grid = np.stack([x_cond, y_cond]).T
    
    # normal score transformation
    nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data_cond)
    norm_data = nst_trans.transform(data_cond).squeeze()

    vgrams = {}

    # compute experimental (isotropic) variogram
    V = skg.Variogram(pred_grid, norm_data, bin_func=bin_func, n_lags=n_lags, 
                   maxlag=maxlag, normalize=False)
    
    V.model = covmodels[0]
    vgrams[covmodels[0]] = V.parameters

    if len(covmodels) > 1:
        for i, cov in enumerate(covmodels[1:]):
            V_i = deepcopy(V)
            V_i.model = cov
            vgrams[cov] = V_i.parameters

    df_grid = pd.DataFrame({'X' : x_cond, 'Y' : y_cond, 'residual' : data_cond.squeeze(), 'NormZ' : norm_data})

    return vgrams, df_grid, V.experimental, V.bins, nst_trans

def boug_interpolation_sgs(ds, grav, density, maxlag=100e3, n_lags=70, covmodel='spherical', azimuth=0, minor_range_scale=1, k=64, rad=100e3, trend=False, smoothing=None, quiet=True, rng=None):
    """
    Stochastically interpolate gridded Bouguer disturbance using SGS

    Args:
        ds : preprocessed BedMachine xarray.Dataset
        grav : pandas.DataFrame of gravity data
        density : float of rock or background density
        maxlag : maximum range distance for variogram
        n_lags : number of lag bins for variogram
        covmodel : covariance model for interpolation
        azimuth : orientation in degrees of primary range
        minor_range_scale : scale the major range by this to make the minor range
        k : number of neighboring data points to estimate a point in SGS
        rad : maximum search distance for SGS 
    Outputs:
        Terrain effect for use as target of inversion.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    g_z = bm_terrain_effect(ds, grav, density)

    residual = grav.faa.values-g_z
    
    if trend==True:
        boug_trend = rbf_trend(ds, grav, residual, smoothing=smoothing, full_grid=False)
        residual -= boug_trend

    vgrams, df_grid, experimental, bins, nst_trans = variograms(grav, residual, bin_func='even', maxlag=maxlag, n_lags=n_lags, covmodels=[covmodel])
    parameters = vgrams[covmodel]
    
    # set variogram parameters
    nugget = parameters[-1]

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = parameters[0]
    minor_range = parameters[0] * minor_range_scale
    sill = parameters[1]

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, covmodel]

    if covmodel.lower() == 'matern':
        smoothness = parameters[2]
        vario.append(smoothness)

    pred_grid = np.stack([grav.x, grav.y]).T
    sim = gsm.Interpolation.okrige_sgs(pred_grid, df_grid, 'X', 'Y', 'NormZ', k, vario, rad, quiet=quiet, seed=rng)
    sim_trans = nst_trans.inverse_transform(sim.reshape(-1,1)).squeeze()

    if trend==True:
        sim_trans += boug_trend
    
    terrain_effect = grav.faa.values - sim_trans

    return terrain_effect

def filter_boug(ds, grav, target, cutoff=10e3, pad=0):
    """
    Filter Bouguer disturbance with lowpass Gaussian filter
    given a terrain effect simulation.

    Args:
        ds : preprocessed BedMachine xarray.Dataset
        grav : pandas.DataFrame gravity data
        target : terrain effect resulting from Bouguer SGS interpolation
        cutoff : pass frequencies below this
        pad : amount to pad inversion domain for filtering
    Outputs:
        Filtered Bouguer disturbance
    """
    xx, yy = np.meshgrid(ds.x, ds.y)

    target_grid = xy_into_grid(ds, (grav.x.values, grav.y.values), target)
    faa_grid = xy_into_grid(ds, (grav.x, grav.y), grav.faa)
    boug_grid = faa_grid - target_grid
    
    grav_msk = ~np.isnan(boug_grid)

    nearest = vd.KNeighbors(k=10)
    nearest.fit(
        coordinates=(grav.x.values, grav.y.values),
        data = grav.faa-target
    )
    boug_fill = nearest.predict((xx.flatten(), yy.flatten()))
    boug_fill = np.where(grav_msk==True, boug_grid, boug_fill.reshape(xx.shape))
    
    boug_filt = lowpass_filter_invpad(ds, boug_fill, cutoff, pad)
    boug_filt = boug_filt[grav_msk]
    
    return boug_filt

def sgs_filt(ds, grav, density, maxlag=100e3, n_lags=70, covmodel='spherical', azimuth=0, minor_range_scale=1, k=64, rad=100e3, trend=False, smoothing=None, quiet=True, cutoff=10e3, pad=0, rng=None):
    """
    Performs SGS Bouguer interpolation, filters Bouguer,
    returns new target terrain effect

    Args:
        ds : preprocessed BedMachine xarray.Dataset
        grav : pandas.DataFrame gravity data
        target : terrain effect resulting from Bouguer SGS interpolation
        cutoff : pass frequencies below this
        pad : amount to pad inversion domain for filtering
    Outputs:
        Target terrain effect from filtered Bouguer SGS interpolation
    """
    target = boug_interpolation_sgs(ds, grav, density, maxlag, n_lags, covmodel, azimuth, minor_range_scale, k, rad, trend, smoothing, quiet, rng=rng)
    boug_filt = filter_boug(ds, grav, target, cutoff, pad)
    new_target = grav.faa.values - boug_filt
    
    return new_target

def rbf_trend(ds, grav, boug_dist, smoothing=1e11, full_grid=False):
    """
    Calculate a trend using Radial Basis Functions

    Args:
        ds : preprocessed BedMachine xarray.Dataset
        grav : pandas.DataFrame gravity data
        boug_dist : bouguer disturbance at the gravity coordinates
        smoothing : how smooth to make the trend
        full_trend : if True, put the trend on the full grid, otherwise
            return the trend only at the gravity coordinates
    Outputs:
        Trend on either the gravity coordinates or on the full grid
    """
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    
    x_cond = grav.loc[grav.inv_msk==False, 'x'].values
    y_cond = grav.loc[grav.inv_msk==False, 'y'].values
    boug_cond = boug_dist[grav.inv_msk==False]
    cond_coords = np.array([x_cond, y_cond]).T
    
    rbf = RBFInterpolator(cond_coords, boug_cond, smoothing=smoothing)

    # if True solve for trend on whole grid
    if full_grid == True:
        pred_grid = np.stack([xx.flatten(), yy.flatten()]).T
        trend_rbf = rbf(pred_grid).reshape(xx.shape)

    # else solve for trend only at gravity coordinates
    else:
        grav_coords = grav[['x', 'y']].values
        trend_rbf = rbf(grav_coords)
    
    return trend_rbf

# def block_resample(xx, yy, df_grid, field, bad_msk, grav_cond_msk, k, vario, rad, bsize, rng, verbose=True):
    
#     # choose block
#     ni, nj = field.shape

#     # find an index inside the inversion domain
#     goodInd = False
#     while goodInd==False:
#         ci = rng.integers(0, ni, size=1)[0]
#         cj = rng.integers(0, nj, size=1)[0]
#         if bad_msk[ci,cj]==True:
#             goodInd = True

#     # half width of the block
#     hw = bsize//2

#     # make sure block extent inside domain
#     ilow = max(0, ci-hw)
#     ihigh = min(ni-1, ci+hw+1)
#     jlow = max(0, cj-hw)
#     jhigh = min(nj-1, cj+hw+1)

#     # extract everything else as conditioning data
#     block_msk = np.zeros(xx.shape).astype(bool)
#     block_msk[ilow:ihigh, jlow:jhigh] = ~grav_cond_msk[ilow:ihigh, jlow:jhigh]
#     #block_msk[grav_cond_msk] = True
#     x_cond = xx[block_msk==False]
#     y_cond = yy[block_msk==False]
#     data_cond = field[block_msk==False].reshape(-1,1)
    
#     # normalize the data
#     nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data_cond)
#     norm_data = nst_trans.transform(data_cond).squeeze()

#     df_block = pd.DataFrame({'X' : x_cond, 'Y' : y_cond, 'residual' : data_cond.squeeze(), 'NormZ' : norm_data})
#     df_new = pd.concat([df_grid, df_block])
    
#     # resimulate
#     pred_grid = np.array([xx[block_msk==True], yy[block_msk==True]]).T
#     # bug: pred grid can have 0 entries. couldn't easily fix so just return original
#     if pred_grid.shape[0] == 0:
#         if verbose==True:
#             print('warning: 0 grid cells in pred grid')
#         return field
#     else:
#         new_field = field.copy()
#         sim_tmp = gsm.Interpolation.okrige_sgs(pred_grid, df_new, 'X', 'Y', 'NormZ', k, vario, rad, quiet=True, seed=rng)
#         np.place(new_field, block_msk, nst_trans.inverse_transform(sim_tmp.reshape(-1,1)))
#     return new_field

def block_resample(xx, yy, field, sim_mask, og_cond_mask, k, vario, rad, bsize, rng, verbose=True):
    
    # choose block
    ni, nj = field.shape

    # find an index inside the inversion domain
    goodInd = False
    while goodInd==False:
        ci = rng.integers(0, ni, size=1)[0]
        cj = rng.integers(0, nj, size=1)[0]
        if sim_mask[ci,cj]==True:
            goodInd = True

    # half width of the block
    hw = bsize//2

    # make sure block extent inside domain
    ilow = max(0, ci-hw)
    ihigh = min(ni-1, ci+hw+1)
    jlow = max(0, cj-hw)
    jhigh = min(nj-1, cj+hw+1)

    # extract everything else as conditioning data
    block_msk = np.full(xx.shape, False)
    # block_msk[ilow:ihigh, jlow:jhigh] = ~og_cond_mask[ilow:ihigh, jlow:jhigh]
    block_msk[ilow:ihigh, jlow:jhigh] = (~og_cond_mask & sim_mask)[ilow:ihigh, jlow:jhigh]
    new_field = np.where(block_msk==True, np.nan, field)
    
    # bug: pred grid can have 0 entries. couldn't easily fix so just return original
    if np.count_nonzero(np.isnan(new_field)) == 0:
        if verbose==True:
            print('warning: 0 grid cells in pred grid')
        return field
    else:
        new_sim = sgs(xx, yy, new_field, vario, rad, k, sim_mask=block_msk, quiet=True, seed=rng)
        
    return new_sim

# def boug_resample(ds, grav, df_grid, boug, trend, k, vario, rad, rng, density_dict, max_iter_no_change=100, verbose=True):

#     grav_cond = grav.loc[grav.inv_msk==False]
#     grav_mskd = grav.loc[grav.inv_pad==True]

#     pred_coords = (grav_mskd.x.values, grav_mskd.y.values, grav_mskd.height.values)
#     # x_cond = grav.loc[grav.inv_msk==False, 'x'].values
#     # y_cond = grav.loc[grav.inv_msk==False, 'y'].values
#     # data_cond = data[grav.inv_msk==False].reshape(-1,1)
#     # pred_grid = np.stack([x_cond, y_cond]).T

#     xx, yy = np.meshgrid(ds.x, ds.y)

#     bed_max = np.where(ds.mask==3, (ds.surface-ds.thickness).values, ds.bed.values)

#     prisms, densities = make_prisms(ds, bed_max, density_dict)
#     g_z_max = hm.prism_gravity(pred_coords, prisms, densities, field='g_z')
#     boug_max = grav_mskd.faa - g_z_max
#     boug_max_grid = xy_into_grid(ds, (pred_coords[0], pred_coords[1]), boug_max)

#     grav_cond_msk = ~np.isnan(xy_into_grid(ds, (grav_cond.x, grav_cond.y), grav_cond.faa))
    
#     prev_boug = deepcopy(boug)
#     bad_msk_prev = prev_boug < (boug_max_grid-trend)
#     n_bad_prev = np.count_nonzero(bad_msk_prev)

#     i = 0
#     iter_no_change = 0
    
#     while n_bad_prev > 0:
#         bsize = rng.choice([1, 3, 5, 7])
#         next_boug = block_resample(xx, yy, df_grid, prev_boug, bad_msk_prev, grav_cond_msk, k, vario, rad, bsize, rng, verbose)
#         bad_msk_next = next_boug < (boug_max_grid-trend)
#         n_bad_next = np.count_nonzero(bad_msk_next)
#         if n_bad_next < n_bad_prev:
#             prev_boug = next_boug
#             n_bad_prev = n_bad_next
#             bad_msk_prev = bad_msk_next
#             iter_no_change = 0
#         else:
#             iter_no_change += 1
#         i += 1
#         if verbose==True:
#             print(f'{i}: {n_bad_prev} bad grid cells. {n_bad_next} tried.')
        
#         if iter_no_change >= max_iter_no_change:
#             print(f'max iterations with no change reached after {i} iterations')
#             break

#     print(f'{n_bad_prev} bad grid cells remaining were hard-corrected')
#     final_boug = np.where(prev_boug<(boug_max_grid-trend), boug_max_grid - trend, prev_boug)

#     return final_boug

def boug_resample(ds, grav, boug, trend, cond_msk, k, vario, rad, rng, density_dict, max_iter_no_change=100, verbose=True):

    # grav_cond = grav.loc[grav.inv_msk==False]
    grav_mskd = grav.loc[grav.inv_pad==True]

    pred_coords = (grav_mskd.x.values, grav_mskd.y.values, grav_mskd.height.values)

    xx, yy = np.meshgrid(ds.x, ds.y)

    bed_max = np.where(ds.mask==3, (ds.surface-ds.thickness).values, ds.bed.values)

    prisms, densities = make_prisms(ds, bed_max, density_dict)
    g_z_max = hm.prism_gravity(pred_coords, prisms, densities, field='g_z')
    boug_max = grav_mskd.faa - g_z_max
    boug_max_grid = xy_into_grid(ds, (pred_coords[0], pred_coords[1]), boug_max)

    #grav_cond_msk = ~np.isnan(xy_into_grid(ds, (grav_cond.x, grav_cond.y), grav_cond.faa))
    
    prev_boug = deepcopy(boug)
    bad_msk_prev = prev_boug < (boug_max_grid-trend)
    n_bad_prev = np.count_nonzero(bad_msk_prev)

    i = 0
    iter_no_change = 0
    
    while n_bad_prev > 0:
        bsize = rng.choice([3, 5, 7])
        next_boug = block_resample(xx, yy, prev_boug, bad_msk_prev, cond_msk, k, vario, rad, bsize, rng, verbose)
        bad_msk_next = next_boug < (boug_max_grid-trend)
        n_bad_next = np.count_nonzero(bad_msk_next)
        if n_bad_next < n_bad_prev:
            prev_boug = next_boug
            n_bad_prev = n_bad_next
            bad_msk_prev = bad_msk_next
            iter_no_change = 0
        else:
            iter_no_change += 1
        i += 1
        if verbose==True:
            print(f'{i}: {n_bad_prev} bad grid cells. {n_bad_next} tried.')
        
        if iter_no_change >= max_iter_no_change:
            if verbose==True:
                print(f'max iterations with no change reached after {i} iterations')
            break

    if verbose==True:
        print(f'{n_bad_prev} bad grid cells remaining were hard-corrected')
    final_boug = np.where(prev_boug<(boug_max_grid-trend), boug_max_grid - trend, prev_boug)

    return final_boug