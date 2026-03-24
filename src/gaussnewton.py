import numpy as np
import xarray as xr
from tqdm.auto import tqdm
import harmonica as hm
import time
from copy import deepcopy
from numba import njit, prange
from choclo.prism import gravity_u
from scipy.optimize import minimize_scalar

from gstatsim_custom import *
from prisms import *
from utilities import *

def jacobian(xx, yy, ds, bed, density_dict, pred_coords, perturb_scale=1, quiet=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    res = np.abs(ds.x.values[1]-ds.x.values[0])

    x_vary = xx[ds.inv_msk.values==True]
    y_vary = yy[ds.inv_msk.values==True]
    bed_vary = ds.bed.values[ds.inv_msk.values==True]
    surface_vary = ds.surface.values[ds.inv_msk.values==True]
    thickness_vary = ds.thickness.values[ds.inv_msk.values==True]

    water_dens = density_dict['water']
    rock_dens = density_dict['rock']
    
    jac = np.zeros((pred_coords[0].size, bed_vary.size))

    pbar = tqdm(range(bed_vary.size), position=0, leave=True, disable=quiet)
    for i in pbar:
        pos = rng.random()*perturb_scale
        neg = -1*rng.random()*perturb_scale
        bed_pos = bed_vary[i]+pos
        bed_neg = bed_vary[i]+neg
        water_top = surface_vary[i]-thickness_vary[i]
        j = 0
        while bed_pos > water_top:
            pos = rng.random()
            bed_pos = bed_vary[i]+pos
            if j > 20:
                bed_pos = water_top - 1e-8
                break
            j += 1
        
        prism_pos = [x_vary[i]-res/2, x_vary[i]+res/2, y_vary[i]-res/2, y_vary[i]+res/2, bed_pos, water_top]
        prism_neg = [x_vary[i]-res/2, x_vary[i]+res/2, y_vary[i]-res/2, y_vary[i]+res/2, bed_neg, water_top]
    
        prisms_water, idx_water_pos = split_prisms(np.array([prism_pos]))
        d_water = np.where(idx_water_pos, water_dens, water_dens-rock_dens)
        g_z_pos = hm.prism_gravity(pred_coords, prisms_water, d_water, field='g_z')
    
        prisms_water, idx_water_pos = split_prisms(np.array([prism_neg]))
        d_water = np.where(idx_water_pos, water_dens, water_dens-rock_dens)
        g_z_neg = hm.prism_gravity(pred_coords, prisms_water, d_water, field='g_z')
    
        grad = (-g_z_pos+g_z_neg)/(bed_pos-bed_neg)
        jac[:,i] = grad

    return jac

def gauss_newton(ds, target, pred_coords, density_dict, max_iter=5, alpha=1, perturb_scale=1, lamb=0, stop=None, quiet=False):

    tic = time.time()

    xx, yy = np.meshgrid(ds.x, ds.y)

    ice_bottom = ds.surface.values-ds.thickness.values
    bed = ds.bed.values.astype(np.float64)
    inv_msk = ds.inv_msk.values
    
    x_vary = xx[inv_msk==True]
    y_vary = yy[inv_msk==True]
    prev_bed = ds.bed.values[inv_msk==True]

    bed_cache = np.zeros((max_iter+1, *ds.bed.shape))
    rmse_cache = np.full(max_iter+1, np.nan)

    g_z_prev = forward_model(ds, bed, pred_coords, density_dict)
    g_z_inv_prev = forward_model(ds, bed, pred_coords, density_dict, msk=inv_msk, ice=False)
    residual = target-g_z_prev
    rmse_next = np.sqrt(np.mean(np.square(residual)))
    rmse_prev = rmse_next + 9999

    bed_cache[0,...] = bed
    rmse_cache[0] = rmse_next

    alpha_i = alpha

    if quiet==False:
        print(f'# start \t RMSE: {rmse_next:.3f}')
    
    for i in range(max_iter):
        next_bed, next_bed_grid = gn_step(xx, yy, ds, bed, prev_bed, target, residual, density_dict, pred_coords, perturb_scale, lamb, alpha_i, inv_msk, ice_bottom, g_z_prev, g_z_inv_prev, quiet)
        g_z_inv_new = forward_model(ds, next_bed_grid, pred_coords, density_dict, msk=inv_msk, ice=False)
        g_z_new = g_z_prev - g_z_inv_prev + g_z_inv_new
        residual = target-g_z_new
        rmse_next = np.sqrt(np.mean(np.square(residual)))

        j = 0
        divergence = False
        while rmse_next > rmse_prev:
            if j > 4:
                divergence = True
                break
            if alpha_i=='search':
                divergence = True
                break
            alpha_i = alpha_i / 2
            
            next_bed, next_bed_grid = gn_step(xx, yy, ds, bed, prev_bed, target, residual, density_dict, pred_coords, perturb_scale, lamb, alpha_i, inv_msk, ice_bottom, g_z_prev, g_z_inv_prev, quiet)
            g_z_inv_new = forward_model(ds, next_bed_grid, pred_coords, density_dict, msk=inv_msk, ice=False)
            g_z_new = g_z_prev - g_z_inv_prev + g_z_inv_new
            residual = target-g_z_new
            rmse_next = np.sqrt(np.mean(np.square(residual)))
            j += 1

        bed_cache[i+1,...] = next_bed_grid
        rmse_cache[i+1,...] = rmse_next

        rmse_prev = rmse_next
        prev_bed = next_bed
        g_z_prev = g_z_new
        g_z_inv_prev = g_z_inv_new

        toc = time.time()

        if quiet==False:
            print(f'# {i+1} \t RMSE: {rmse_next:.3f} \t time elapsed: {toc-tic:.2f} s')

        is_stop, bed_cache, rmse_cache = stopping_conditions(i, rmse_cache, bed_cache, stop, divergence, quiet)
        if is_stop==True:
            break

    return bed_cache, rmse_cache

def gn_step(xx, yy, ds, bed, prev_bed, target, residual, density_dict, pred_coords, perturb_scale, lamb, alpha, inv_msk, ice_bottom, g_z_prev, g_z_inv_prev, quiet):
    
    J = jacobian(xx, yy, ds, prev_bed, density_dict, pred_coords, perturb_scale=perturb_scale, quiet=True)
    delta = -1*np.linalg.inv(J.T@J + lamb*np.eye(J.shape[1]))@J.T@residual

    if alpha=='search':
        args = (delta, prev_bed, bed, ds, pred_coords, density_dict, target, inv_msk, ice_bottom, g_z_prev, g_z_inv_prev)

        search_result = minimize_scalar(next_step_rmse, bounds=(0, 1), args=args, tol=0.01, options=dict(maxiter=15))
        alpha = search_result['x']
    
    next_bed = prev_bed + alpha*delta
    next_bed_grid = emplace_bed(next_bed, bed, inv_msk, ice_bottom)
    
    return next_bed, next_bed_grid

def next_step_rmse(alpha, delta, prev_bed, bed, ds, pred_coords, density_dict, target, inv_msk, ice_bottom, g_z_prev, g_z_inv_prev):
    next_bed = prev_bed + alpha*delta
    next_bed_grid = emplace_bed(next_bed, bed, inv_msk, ice_bottom)

    g_z_inv_new = forward_model(ds, next_bed_grid, pred_coords, density_dict, msk=inv_msk, ice=False)
    g_z_new = g_z_prev - g_z_inv_prev + g_z_inv_new
    residual = target-g_z_new
    rmse_next = np.sqrt(np.mean(np.square(residual)))
    return rmse_next

def gauss_newton_regularized(ds, prior, target, pred_coords, density_dict, Cm, Wd, max_iter=5, alpha=1, perturb_scale=1, lamb=0, stop=None, quiet=False):

    tic = time.time()

    xx, yy = np.meshgrid(ds.x, ds.y)

    ice_bottom = ds.surface.values-ds.thickness.values
    bed = deepcopy(prior)
    inv_msk = ds.inv_msk.values

    g_z_prev = forward_model(ds, bed, pred_coords, density_dict)
    g_z_inv_prev = forward_model(ds, bed, pred_coords, density_dict, msk=inv_msk, ice=False)
    residual = target-g_z_prev
    rw = np.diag(Wd)*residual
    rmse_next = np.sqrt(np.mean(np.square(residual)))
    rmse_prev = rmse_next + 9999
    
    x_vary = xx[inv_msk==True]
    y_vary = yy[inv_msk==True]
    prev_bed = ds.bed.values[inv_msk==True]
    prior = prior[inv_msk==True].astype(np.float64)

    bed_cache = np.zeros((max_iter+1, *ds.bed.shape))
    rmse_cache = np.full(max_iter+1, np.nan)

    bed_cache[0,...] = bed
    rmse_cache[0] = rmse_next

    alpha_i = alpha

    if quiet==False:
        print(f'# start \t RMSE: {rmse_next:.3f}')
    
    for i in range(max_iter):
        next_bed, next_bed_grid = rgn_step(xx, yy, ds, bed, prev_bed, target, density_dict, pred_coords, perturb_scale, rw, Wd, Cm, lamb, prior, alpha_i, inv_msk, ice_bottom, g_z_prev, g_z_inv_prev, quiet)
        g_z_inv_new = forward_model(ds, next_bed_grid, pred_coords, density_dict, msk=inv_msk, ice=False)
        g_z_new = g_z_prev - g_z_inv_prev + g_z_inv_new
        residual = target-g_z_new
        rw = np.diag(Wd)*residual
        rmse_next = np.sqrt(np.mean(np.square(residual)))

        j = 0
        divergence = False
        while rmse_next > rmse_prev:
            if j > 4:
                divergence = True
                break
            if alpha_i=='search':
                divergence = True
                break
            alpha_i = alpha_i / 2
            
            next_bed, next_bed_grid = rgn_step(xx, yy, ds, bed, prev_bed, target, density_dict, pred_coords, perturb_scale, rw, Wd, Cm, lamb, prior, alpha_i, inv_msk, ice_bottom, g_z_prev, g_z_inv_prev, quiet)
            g_z_inv_new = forward_model(ds, next_bed_grid, pred_coords, density_dict, msk=inv_msk, ice=False)
            g_z_new = g_z_prev - g_z_inv_prev + g_z_inv_new
            residual = target-g_z_new
            rw = np.diag(Wd)*residual
            rmse_next = np.sqrt(np.mean(np.square(residual)))
            j += 1

        bed_cache[i+1,...] = next_bed_grid
        rmse_cache[i+1] = rmse_next
        
        rmse_prev = rmse_next
        prev_bed = next_bed
        g_z_prev = g_z_new
        g_z_inv_prev = g_z_inv_new

        toc = time.time()

        if quiet==False:
            print(f'# {i+1} \t RMSE: {rmse_next:.3f} \t time elapsed: {toc-tic:.2f} s')

        is_stop, bed_cache, rmse_cache = stopping_conditions(i, rmse_cache, bed_cache, stop, divergence, quiet)
        if is_stop==True:
            break

    return bed_cache, rmse_cache

def rgn_step(xx, yy, ds, bed, prev_bed, target, density_dict, pred_coords, perturb_scale, rw, Wd, Cm, lamb, prior, alpha, inv_msk, ice_bottom, g_z_prev, g_z_inv_prev, quiet):
    
    J = jacobian(xx, yy, ds, prev_bed, density_dict, pred_coords, perturb_scale=perturb_scale, quiet=True)
    Jw = Wd@J
    H = Jw.T@Jw + lamb*Cm
    I = Jw.T@rw + lamb*Cm@(prev_bed-prior)
    
    delta = -1*np.linalg.inv(H)@I

    if alpha=='search':
        args = (delta, prev_bed, bed, ds, pred_coords, density_dict, target, inv_msk, ice_bottom, g_z_prev, g_z_inv_prev, Wd, Cm, prior, lamb)

        search_result = minimize_scalar(next_step_rgn, bounds=(0, 1), args=args, tol=0.01, options=dict(maxiter=15))
        alpha = search_result['x']
        
    next_bed = prev_bed + alpha*delta
    
    next_bed_grid = emplace_bed(next_bed, bed, inv_msk, ice_bottom)

    return next_bed, next_bed_grid

def next_step_rgn(alpha, delta, prev_bed, bed, ds, pred_coords, density_dict, target, inv_msk, ice_bottom, g_z_prev, g_z_inv_prev, Wd, Cm, prior, lamb):
    next_bed = prev_bed + alpha*delta
    next_bed_grid = emplace_bed(next_bed, bed, inv_msk, ice_bottom)

    g_z_inv_new = forward_model(ds, next_bed_grid, pred_coords, density_dict, msk=inv_msk, ice=False)
    g_z_new = g_z_prev - g_z_inv_prev + g_z_inv_new
    residual = target-g_z_new
    
    data_l2_norm = np.sum(np.square(np.diag(Wd)*residual))
    model_l2_norm = np.sum(np.square(Cm@(prev_bed-prior)))
    
    return data_l2_norm + lamb*model_l2_norm


def emplace_bed(next_bed, bed, inv_msk, ice_bottom):
    next_bed_grid = bed
    np.place(next_bed_grid, inv_msk==True, next_bed)
    next_bed_grid = np.where(next_bed_grid > ice_bottom, ice_bottom, next_bed_grid)
    return next_bed_grid

def forward_model(ds, bed, pred_coords, density_dict, msk=None, ice=True):
    prisms, densities = make_prisms(ds, bed, density_dict, msk=msk, ice=ice)
    g_z = hm.prism_gravity(pred_coords, prisms, densities, field='g_z')
    return g_z

def stopping_conditions(i, rmse_cache, bed_cache, stop, divergence, quiet):
    if stop is not None:
        if i >= 2:
            if np.mean(np.abs(np.diff(rmse_cache[i-1:i+2]))) < stop:
                bed_cache = bed_cache[:i+2,...]
                rmse_cache = rmse_cache[:i+2]
                if quiet==False:
                    print(f'reached stopping criterion after {i+1} iterations')
                return True, bed_cache, rmse_cache

    if divergence == True:
        bed_cache = bed_cache[:i+1,...]
        rmse_cache = rmse_cache[:i+1]
        if quiet==False:
            print(f'stopping due to diverging solution after {i+1} iterations')
        return True, bed_cache, rmse_cache
        
    return False, bed_cache, rmse_cache

def make_laplace(a):
    ny, nx = a.shape
    lapop_small = np.diag(np.ones(nx)*-4, 0) + np.diag(np.ones(nx-1), -1) + np.diag(np.ones(nx-1), 1)
    lapop = np.kron(np.eye(ny), lapop_small)

    lapop += np.diag(np.ones(nx*ny-nx), -nx)
    lapop += np.diag(np.ones(nx*ny-nx), nx)
    return lapop

def make_derivative(a):
    ny, nx = a.shape
    dop = np.diag(np.ones(nx*ny), 0) + np.diag(np.ones(nx*ny-1), 1)
    return dop

def trim_laplace(lapop, inv_msk):
    inv_msk = inv_msk
    inv_msk_flat = inv_msk.ravel()
    
    lapop_trim = lapop[inv_msk_flat,:]
    lapop_trim = lapop_trim[:,inv_msk_flat]
    
    bound_cells = np.count_nonzero(lapop_trim, axis=1) < 5
    lapop_trim[bound_cells,:] = 0

    return lapop_trim

def lcurve(lambdas, ds, prior, target, pred_coords, density_dict, Cm, Wd, max_iter=30, alpha=1, perturb_scale=1, stop=0.001, quiet=True):
    data_fits = []
    model_fits = []
    
    for l in tqdm(lambdas):
        bed_cache, rmse_cache = gauss_newton_regularized(ds, prior, target, pred_coords, density_dict, Cm, Wd, max_iter=30, alpha=0.5, perturb_scale=1, lamb=l, stop=0.001, quiet=True)
        
        prisms, densities = make_prisms(ds, bed_cache[-1,...], density_dict)
        g_z_new = hm.prism_gravity(pred_coords, prisms, densities, field='g_z')
    
        inv_msk = ds.inv_msk.values
    
        residual = target - g_z_new
        data_fits.append(np.sqrt(np.sum(np.square(np.diag(Wd)*residual))))
        model_fits.append(np.sqrt(np.sum(np.square(Cm@(bed_cache[-1,...][inv_msk]-prior[inv_msk])))))

    min_model_fit = np.min(np.log(model_fits))
    min_data_fit = np.min(np.log(data_fits))
    dists = np.sqrt((np.log(data_fits)-min_data_fit)**2+(np.log(model_fits)-min_model_fit)**2)

    lopt = lambdas[np.argmin(dists)]

    return lopt, data_fits, model_fits, dists
    