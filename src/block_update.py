import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm
import harmonica as hm
import xarray as xr
import time
import verde as vd

from prisms import *

def sum_sq_err(data, pred):
    """
    Sum of squared error
    """
    return np.sum(np.square(data-pred))

def mse(data, pred):
    """
    Mean squared error
    """
    return np.mean(np.square(data-pred))

def block_update_sample(field, inv_msk, bsize, sample, rng, weights=None):
    """
    Update bed with a block random field

    Args:
        field : 2D array of the bed
        inv_msk : 2D boolean array of the inversion mask
        bsize : int size of the square block
        sample : 2D array of size (bsize, bsize) zero-mean random field
        rng : Numpy random number generator
        weights : conditioning on the edge of the inversion mask to lower amplitude
    Outputs:
        updated bed as 2D array, tuple of indices of extent of random update in
        order of (bottom, top, left, right)
    """
    ni, nj = field.shape

    # find an index inside the inversion domain
    goodInd = False
    while goodInd==False:
        ci = rng.integers(0, ni, size=1)[0]
        cj = rng.integers(0, nj, size=1)[0]
        if inv_msk[ci,cj]==True:
            goodInd = True

    # half width of the block
    hw = bsize//2

    # make sure block extent inside domain
    ilow = max(0, ci-hw)
    ihigh = min(ni-1, ci+hw+1)
    jlow = max(0, cj-hw)
    jhigh = min(nj-1, cj+hw+1)
    field_next = field.copy()

    # apply weights if there are weights
    if weights is None:
        field_next[ilow:ihigh, jlow:jhigh] = field[ilow:ihigh, jlow:jhigh]+sample[0:ihigh-ilow, 0:jhigh-jlow]
    else:
        field_next[ilow:ihigh, jlow:jhigh] = field[ilow:ihigh, jlow:jhigh]+sample[0:ihigh-ilow, 0:jhigh-jlow]*weights[ilow:ihigh, jlow:jhigh]
        
    return np.where(inv_msk==True, field_next, field), (ilow, ihigh, jlow, jhigh)

def make_cov(bsize, corr_dist):
    """
    Other covariance matrix generator
    """
    bi, bj = np.meshgrid(np.arange(bsize), np.arange(bsize))
    X_b = np.stack([bi.flatten(), bj.flatten()]).T
    D = pairwise_distances(X_b)**2
    return np.exp(-D/corr_dist) + np.diag(np.full(D.shape[0], 1e-10))

def gauss_cov(bsize, corr_dist):
    """
    Gaussian-like covariance matrix for block that produces variogram with
    range approximately corr_dist
    Args:
        bsize : (int) size of square block
        corr_dist : correlation distance or range in pixels
    Outputs:
        2D covariance matrix
    """
    bi, bj = np.meshgrid(np.arange(bsize), np.arange(bsize))
    X_b = np.stack([bi.flatten(), bj.flatten()]).T
    D = pairwise_distances(X_b)**2
    return np.exp(-D/(0.5*corr_dist)**2) + np.diag(np.full(D.shape[0], 1e-10))

def chain_sequence(sequence, ds, x0, pred_coords, target, sigma, density_dict, rng, weights=None, stop=None, save=None, full_cache=False, quiet=False, num_mp=0):
    """
    Run an inversion with a sequence of block update parameters.

    Args:
        sequence : list or tuple of block update parameter lists. Each elemet should be [block size, range, amplitude, iterations]
        ds : Xarray.Dataset of BedMachine preprocessed for problem
        x0 : 2D numpy array of initial bed model
        pred_coords : tuple of gravity observation coordinates in order of (x, y, height)
        target : 1D numpy array of gravity terrain effect. This is the target of the inversion.
        sigma : uncertainty of gravity data in mGals, or tunable parameter for acceptance rate
        density_dict : dictionary of densities with keys rock, ice, and water
        rng : numpy random number generator
        weights : 2D array of conditioning weights for limiting change near edge of inversion domain. Maximum amplitude 1 and minimum 0.
        stop : RMSE stopping condition, if None then all iterations will be completed
        save : path to save at
        full_cache : if True, return all bed iterations, else return only the last bed
        quiet : if True, no progress bars are displayed, else display progress bars
        num_mp : int number to print out when running multiple inversions
    Outputs:
        Dictionary of inversions results. Keys are:
        bed_cache : cache of beds, or last bed iteration
        loss_cache : cache of losses
        step_cache : cache of wether iterations were accepted or not
        grav_cache : last forward modeled gravity data
        density : float of rock or background density
        target : target terrain effect, same as input
    """
    results = []
    for i, seq in enumerate(sequence):
        # block update parameters
        bsize, corr_dist, var, iter_num = seq
        # make Gaussian covariance matrix
        Sigma = gauss_cov(bsize, corr_dist)

        # run chain with current block parameters
        tic = time.time()
        result = mcmc_block(ds, x0, pred_coords, target, sigma, density_dict, Sigma, bsize, var, rng, weights, iter_num=iter_num, stop=stop, quiet=quiet)
        toc = time.time()

        # calculate RMSE/sec efficiency
        rmse_start = np.sqrt(result['loss_cache'][0]/len(target))
        rmse_finish = np.sqrt(result['loss_cache'][-1]/len(target))
        efficiency = (rmse_finish-rmse_start)/(toc-tic)
        
        if quiet==False:
            print(f'chain {i} efficiency: {efficiency:.3f} RMSE/sec')

        # next initial condition is last bed iteration
        x0 = result['bed_cache'][-1]
        results.append(result)

    # concatenate chains together
    bed_cache = np.concatenate([r['bed_cache'] for r in results])
    loss_cache = np.concatenate([r['loss_cache'] for r in results])
    step_cache = np.concatenate([r['step_cache'] for r in results])
    grav_cache = np.concatenate([r['grav_cache'] for r in results])

    # return results as dictionary
    result = {
        'bed_cache' : bed_cache if full_cache==True else bed_cache[-1,...],
        'loss_cache' : loss_cache,
        'step_cache' : step_cache,
        'grav_cache' : grav_cache[-1,:],
        'density' : np.array([density_dict['rock']]),
        'target' : target
    }

    # save results if there is a path
    if save is not None:
        np.save(save, result)
        
    if quiet==False:
        print(f'{num_mp} finished')
    
    return result

def chain_sequence_mp(args):
    return chain_sequence(*args)

def mcmc_block(ds, x0, pred_coords, target, sigma, density_dict, Sigma, bsize, var, rng, weights=None, save_path=None, iter_num=500, stop=None, quiet=False, parallel=True, num_mp=1):
    """
    Random Walk Metropolis MCMC with block updates for inverting for bathymetry from gravity data

    Args:
        ds : Xarray.Dataset of BedMachine preprocessed for problem
        x0 : 2D numpy array of initial bed model
        pred_coords : tuple of gravity observation coordinates in order of (x, y, height)
        target : 1D numpy array of gravity terrain effect. This is the target of the inversion.
        sigma : uncertainty of gravity data in mGals, or tunable parameter for acceptance rate
        density_dict : dictionary of densities with keys rock, ice, and water
        Sigma : covariance matrix for blocks of bsize
        bsize : size of square blocks
        var : ampltitude of block random fields
        rng : numpy random number generator
        weights : 2D array of conditioning weights for limiting change near edge of inversion domain. Maximum amplitude 1 and minimum 0.
        stop : RMSE stopping condition, if None then all iterations will be completed
        save : path to save at
        full_cache : if True, return all bed iterations, else return only the last bed
        quiet : if True, no progress bars are displayed, else display progress bars
        num_mp : int number to print out when running multiple inversions
    Outputs:
        Dictionary of inversions results. Keys are:
        bed_cache : cache of beds, or last bed iteration
        loss_cache : cache of losses
        step_cache : cache of wether iterations were accepted or not
        grav_cache : last forward modeled gravity data
    """
    
    bed = x0
    y = np.unique(ds.y.data)
    x = np.unique(ds.x.data)
    te_dist = target
    
    # initialize caches
    bed_cache = np.zeros((iter_num, bed.shape[0], bed.shape[1]))
    loss_cache = np.zeros(iter_num)
    step_cache = np.zeros(iter_num)
    grav_cache = np.zeros((iter_num, len(target)))

    # generate block updates
    samples = rng.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma, size=iter_num, method='cholesky')*var/3
    samples = samples.reshape((iter_num, bsize, bsize))
    
    # initialize loss
    prisms, densities = make_prisms(ds, bed, density_dict)
    g_z = hm.prism_gravity(pred_coords, prisms, densities, field='g_z', parallel=parallel)
    loss_prev = sum_sq_err(te_dist, g_z)
    
    pbar = tqdm(range(iter_num), position=0, leave=True, disable=quiet)
    for i in pbar:
        # random gaussian field perturbation
        if weights is None:
            bed_next, idxs_next = block_update_sample(bed, ds.inv_msk.data, bsize, samples[i], rng)
        else:
            bed_next, idxs_next = block_update_sample(bed, ds.inv_msk.data, bsize, samples[i], rng, weights)

        # make sure below below ice bottom
        bed_next = np.where(bed_next>ds.surface-ds.thickness, ds.surface-ds.thickness, bed_next)
        
        # trim dataset to area
        i_msk = np.full(ds.y.size, False)
        i_msk[idxs_next[0]:idxs_next[1]] = True
        j_msk = np.full(ds.x.size, False)
        j_msk[idxs_next[2]:idxs_next[3]] = True
        ds_trim = ds.isel(x=j_msk, y=i_msk)

        bed_trim = bed[idxs_next[0]:idxs_next[1],idxs_next[2]:idxs_next[3]]
        bed_next_trim = bed_next[idxs_next[0]:idxs_next[1],idxs_next[2]:idxs_next[3]]

        # find gravity within 10 km of block
        xx_trim, yy_trim = np.meshgrid(ds_trim.x, ds_trim.y)
        coords_ind = vd.distance_mask((xx_trim, yy_trim), 10e3, pred_coords)
        pred_coords_trim = (pred_coords[0][coords_ind], pred_coords[1][coords_ind], pred_coords[2][coords_ind])

        # calculate prev
        prisms, densities = make_prisms(ds_trim, bed_trim, density_dict, ds_trim.inv_msk.values, ice=False)
        g_z_prev_block = hm.prism_gravity(pred_coords_trim, prisms, densities, field='g_z', parallel=parallel)
        
        # calculate new
        prisms, densities = make_prisms(ds_trim, bed_next_trim, density_dict, ds_trim.inv_msk.values, ice=False)
        g_z_next_block = hm.prism_gravity(pred_coords_trim, prisms, densities, field='g_z', parallel=parallel)

        # place gravity subset back in
        g_z_change = np.zeros(pred_coords[0].size)
        np.place(g_z_change, coords_ind, -g_z_prev_block+g_z_next_block)

        # keep track of total gravity
        g_z_next = g_z + g_z_change
        
        # compute loss
        loss_next = sum_sq_err(te_dist, g_z_next)
        
        # metrpolis acceptance
        alpha = min(1,np.exp((loss_prev-loss_next)/(2*sigma**2)))
        
        # accept or not, save cachestep
        u = rng.uniform(size = 1)
        if (u <= alpha):
            bed = bed_next
            g_z = g_z_next
            loss_cache[i] = loss_next
            step_cache[i] = True
            loss_prev = loss_next
        else:
            loss_cache[i] = loss_prev
            step_cache[i] = False
        
        bed_cache[i,:,:] = bed
        grav_cache[i,:] = g_z

        if stop is not None and np.sqrt(loss_prev/target.size)<stop:
            bed_cache = bed_cache[:i+1,...]
            loss_cache = loss_cache[:i+1]
            step_cache = step_cache[:i+1]
            grav_cache = grav_cache[:i+1,:]
            break
                
        pbar.set_description(f'#{num_mp} RMSE: {np.sqrt(loss_cache[i]/target.size):.3f}')
        
    result = {
        'bed_cache' : bed_cache,
        'loss_cache' : loss_cache,
        'step_cache' : step_cache,
        'grav_cache' : grav_cache
    }
    
    return result

def simulated_annealing(ds, x0, pred_coords, target, sigma, density_dict, Sigma, bsize, var, temp, rng, weights=None, iter_num=500, stop=None, full_cache=False, parallel=True, quiet=False):
    """
    Random Walk Metropolis simulated annealing with block updates for inverting for bathymetry from gravity data

    Args:
        ds : Xarray.Dataset of BedMachine preprocessed for problem
        x0 : 2D numpy array of initial bed model
        pred_coords : tuple of gravity observation coordinates in order of (x, y, height)
        target : 1D numpy array of gravity terrain effect. This is the target of the inversion.
        sigma : uncertainty of gravity data in mGals, or tunable parameter for acceptance rate
        density_dict : dictionary of densities with keys rock, ice, and water
        Sigma : covariance matrix for blocks of bsize
        bsize : size of square blocks
        var : ampltitude of block random fields
        temp : temperature; array controlling how much the MCMC "explores". As temp approaches 0 the algorithm becomes greedy.
        rng : numpy random number generator
        weights : 2D array of conditioning weights for limiting change near edge of inversion domain. Maximum amplitude 1 and minimum 0.
        stop : RMSE stopping condition, if None then all iterations will be completed
        save : path to save at
        full_cache : if True, return all bed iterations, else return only the last bed
        quiet : if True, no progress bars are displayed, else display progress bars
        num_mp : int number to print out when running multiple inversions
    Outputs:
        Dictionary of inversions results. Keys are:
        bed_cache : cache of beds, or last bed iteration
        loss_cache : cache of losses
        step_cache : cache of wether iterations were accepted or not
        grav_cache : last forward modeled gravity data
        metropolis : cache of metropolis values, how likely an update is to be accepted
        diff_cache : cache of differences between iterations
    """
    bed = x0
    y = np.unique(ds.y.data)
    x = np.unique(ds.x.data)
    te_dist = target
    
    # initialize caches
    bed_cache = np.zeros((iter_num, bed.shape[0], bed.shape[1]))
    loss_cache = np.zeros(iter_num)
    step_cache = np.zeros(iter_num)
    grav_cache = np.zeros((iter_num, len(target)))

    samples = rng.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma, size=iter_num, method='cholesky')
    samples = samples.reshape((iter_num, bsize, bsize))

    met_cache = np.zeros(iter_num)
    diff_cache = np.zeros(iter_num)

    # initialize loss
    prisms, densities = make_prisms(ds, bed, density_dict)
    g_z = hm.prism_gravity(pred_coords, prisms, densities, field='g_z', parallel=parallel)
    loss_prev = sum_sq_err(te_dist, g_z)

    pbar = tqdm(range(iter_num), position=0, leave=True, disable=quiet)
    for i in pbar:
        # random gaussian field perturbation
        if weights is not None:
            bed_next, idxs_next = block_update_sample(bed, ds.inv_msk.data, bsize, samples[i]*var[i]/3, rng, weights)
        else:
            bed_next, idxs_next = block_update_sample(bed, ds.inv_msk.data, bsize, samples[i]*var[i]/3, rng)

        # make sure bed below ice bottom
        bed_next = np.where(bed_next>ds.surface-ds.thick, ds.surface-ds.thick, bed_next)
        
        # trim dataset to area
        i_msk = np.full(ds.y.size, False)
        i_msk[idxs_next[0]:idxs_next[1]] = True
        j_msk = np.full(ds.x.size, False)
        j_msk[idxs_next[2]:idxs_next[3]] = True
        ds_trim = ds.isel(x=j_msk, y=i_msk)

        bed_trim = bed[idxs_next[0]:idxs_next[1],idxs_next[2]:idxs_next[3]]
        bed_next_trim = bed_next[idxs_next[0]:idxs_next[1],idxs_next[2]:idxs_next[3]]

        # find gravity within 10 km of block
        xx_trim, yy_trim = np.meshgrid(ds_trim.x, ds_trim.y)
        coords_ind = vd.distance_mask((xx_trim, yy_trim), 10e3, pred_coords)
        pred_coords_trim = (pred_coords[0][coords_ind], pred_coords[1][coords_ind], pred_coords[2][coords_ind])

        # calculate prev
        prisms, densities = make_prisms(ds_trim, bed_trim, density_dict, msk=ds.inv_msk.values, ice=False)
        g_z_prev_block = hm.prism_gravity(pred_coords_trim, prisms, densities, field='g_z', parallel=parallel)
        
        # calculate new
        prisms, densities = make_prisms(ds_trim, bed_next_trim, density_dict, msk=ds.inv_msk.values, ice=False)
        g_z_next_block = hm.prism_gravity(pred_coords_trim, prisms, densities, field='g_z', parallel=parallel)

        # place gravity subset back in
        g_z_change = np.zeros(pred_coords[0].size)
        np.place(g_z_change, coords_ind, -g_z_prev_block+g_z_next_block)

        # keep track of total gravity
        g_z_next = g_z + g_z_change
        
        # compute loss
        loss_next = sum_sq_err(te_dist, g_z_next)

        diff = loss_next - loss_prev
        
        metropolis = np.exp(-diff / temp[i])
        if diff < 0 or rng.random() < metropolis:
            bed = bed_next
            g_z = g_z_next.copy()
            loss_prev = loss_next
            step_cache[i] = True
        else:
            step_cache[i] = False
        
        loss_cache[i] = loss_prev
        bed_cache[i,:,:] = bed
        met_cache[i] = metropolis
        diff_cache[i] = diff

        if stop is not None and np.sqrt(loss_prev/target.size)<stop:
            loss_cache = loss_cache[:i+1]
            step_cache = step_cache[:i+1]
            grav_cache = grav_cache[:i+1,:]
            met_cache = met_cache[:i+1]
            diff_cache = diff_cache[:i+1]
            break

        pbar.set_description(f'#{num_mp} RMSE: {np.sqrt(loss_cache[i]/target.size):.3f}')

    result = {
        'bed_cache' : bed_cache if full_cache==True else bed_cache[-1,...],
        'loss_cache' : loss_cache,
        'step_cache' : step_cache,
        'grav_cache' : grav_cache,
        'metropolis' : met_cache,
        'diff_cache' : diff_cache
    }
    
    return result

def min_dist(ds, metric='l2'):
    """
    Compute minimum distance from conditioning data.
    Note: Uses np.outer, can have memory error if domain is too large.
    Use min_dist_simple if domain is too large.

    Args:
        ds : Xarray.Dataset of preprocessed BedMachine
        metric : metrics accepted by sklearn.metrics.pairwise_distances
    Outputs:
        2D array of minimum distance from conditioning data
    """
    x = ds.x.values
    y = ds.y.values
    xx, yy = np.meshgrid(x, y)
    XX = np.array([xx.flatten(), yy.flatten()]).T
    inv_flat = ds.inv_msk.data.flatten()

    dist = pairwise_distances(XX, metric=metric)

    dist_to_cond = dist*np.outer(inv_flat, ~inv_flat)
    min_dist = np.nanmin(np.where(dist_to_cond==0, np.nan, dist_to_cond), axis=1)
    return min_dist.reshape(ds.bed.shape)

def min_dist_simple(hard_mat, xx, yy):
    """
    Compute minimum distance from conditioning data.

    Args:
        hard_mat : 2D boolean array where True is conditioning data
        xx, yy : 2D x and y-coordinates
    Outputs:
        2D array of minimum distance from conditioning data
    """
    dist = np.zeros(xx.shape)
    xx_hard = np.where(hard_mat==False, np.nan, xx)
    yy_hard = np.where(hard_mat==False, np.nan, yy)
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            dist[i,j] = np.nanmin(np.sqrt(np.square(yy[i,j]-yy_hard)+np.square(xx[i,j]-xx_hard)))
    return dist

def rescale(x, a=0, b=1):
    """
    Rescale x between a and b
    """
    return a+((x-np.nanmin(x))*(b-a))/(np.nanmax(x)-np.nanmin(x))

def logistic(x, L, x0, k):
    """
    logistic function

    Args:
        x : array
        L : curve's maximum value
        k : steepness
        x0 : x-value of sigmoid midpoint
    Outputs:
        logistic function of x
    """
    return L/(1+np.exp(-k*(x-x0)))