import numpy as np
from sklearn.preprocessing import QuantileTransformer
from scipy.spatial import distance_matrix
from scipy.special import kv, gamma
from sklearn.metrics import pairwise_distances
from copy import deepcopy
import numbers
from tqdm.auto import tqdm
import skgstat as skg

def sgs(xx, yy, grid, variogram, radius=100e3, num_points=20, sim_mask=None, quiet=False, seed=None):

    # make random number generator if not provided
    rng = get_random_generator(seed)

    # get masks and gaussian transform data
    cond_msk = ~np.isnan(grid)
    grid_norm, nst_trans = gaussian_transformation(grid, cond_msk)
    out_grid = deepcopy(grid_norm)

    if sim_mask is None:
        sim_mask = np.full(xx.shape, True)

    # get index coordinates and filter with sim_mask
    ii, jj = np.meshgrid(np.arange(xx.shape[0]), np.arange(xx.shape[1]), indexing='ij')
    inds = np.array([ii[sim_mask].flatten(), jj[sim_mask].flatten()]).T
    
    # randomize inds for path
    rng.shuffle(inds)

    vario = deepcopy(variogram)

    # turn scalar variogram parameters into grid
    for key in vario:
        if isinstance(vario[key], numbers.Number):
            vario[key] = np.full(grid.shape, vario[key])

    for k in tqdm(range(inds.shape[0]), disable=quiet):
        i, j = inds[k]
        if cond_msk[i, j] == False:

            azimuth = vario['azimuth'][i,j]
            nugget = vario['nugget'][i,j]
            major_range = vario['major_range'][i,j]
            minor_range = vario['minor_range'][i,j]
            sill = vario['sill'][i,j]

            local_vario = {
                'azimuth' : azimuth,
                'nugget' : nugget,
                'major_range' : major_range,
                'minor_range' : minor_range,
                'sill' : sill,
                'vtype' : vario['vtype'],
                'smoothness' : vario['smoothness'][i,j]
            }
            
            rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range)
            
            nearest = neighbors(i, j, xx, yy, out_grid, cond_msk, radius, num_points)
            
            norm_data_val = nearest[:,-1]
            xy_val = nearest[:,:-1]
            local_mean = np.mean(norm_data_val)
            new_num_pts = len(nearest)
            
            # covariance between data
            covariance_matrix = np.zeros(shape=((new_num_pts+1, new_num_pts+1))) 
            covariance_matrix[0:new_num_pts,0:new_num_pts] = Covariance.make_covariance_matrix(xy_val, 
                                                                                    local_vario, rotation_matrix)
            covariance_matrix[new_num_pts,0:new_num_pts] = 1
            covariance_matrix[0:new_num_pts,new_num_pts] = 1

            # Set up Right Hand Side (covariance between data and unknown)
            covariance_array = np.zeros(shape=(new_num_pts+1))
            k_weights = np.zeros(shape=(new_num_pts+1))
            covariance_array[0:new_num_pts] = Covariance.make_covariance_array(xy_val, 
                                                                    np.tile([xx[i,j], yy[i,j]], new_num_pts), 
                                                                    local_vario, rotation_matrix)
            covariance_array[new_num_pts] = 1 
            covariance_matrix.reshape(((new_num_pts+1)), ((new_num_pts+1)))

            # any of these work: lstsq, pinv, inv
            # k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array)
            # k_weights = np.linalg.pinv(covariance_matrix)@covariance_array
            k_weights = np.linalg.inv(covariance_matrix)@covariance_array
            est = local_mean + np.sum(k_weights[0:new_num_pts]*(norm_data_val - local_mean)) 
            var = sill - np.sum(k_weights[0:new_num_pts]*covariance_array[0:new_num_pts]) 
            var = np.absolute(var)

            # put value in grid
            out_grid[i,j] = rng.normal(est, np.sqrt(var), 1)
            cond_msk[i,j] = True

    # back transform data
    sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1,1)).squeeze().reshape(xx.shape)

    return sim_trans

def neighbors(i, j, xx, yy, grid, cond_msk, radius, num_points):
    # get neighbors
    distances = np.sqrt((xx[i,j] - xx)**2 + (yy[i,j] - yy)**2)
    angles = np.arctan2(yy[i,j] - yy, xx[i,j] - xx)

    points = []
    for b in np.arange(-4, 4):
        msk = (distances < radius) & (angles >= b/4*np.pi) & (angles < (b+1)/4*np.pi) & cond_msk
        sort_inds = np.argsort(distances[msk])
        p = np.array([xx[msk], yy[msk], grid[msk]]).T
        p = p[sort_inds,:]
        p = p[:num_points//8,:]
        points.append(p)
    points = np.concatenate(points)
    return points

def gaussian_transformation(grid, cond_msk, n_quantiles=500):
    data_cond = grid[cond_msk].reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal").fit(data_cond)
    norm_data = nst_trans.transform(data_cond).squeeze()
    grid_norm = np.full(grid.shape, np.nan)
    np.place(grid_norm, cond_msk, norm_data)

    return grid_norm, nst_trans

def make_rotation_matrix(azimuth, major_range, minor_range):
    """
    Make rotation matrix for accommodating anisotropy
    
    Parameters
    ----------
        azimuth : int, float
            angle (in degrees from horizontal) of axis of orientation
        major_range : int, float
            range parameter of variogram in major direction, or azimuth
        minor_range : int, float
            range parameter of variogram in minor direction, or orthogonal to azimuth
    
    Returns
    -------
        rotation_matrix : numpy.ndarray
            2x2 rotation matrix used to perform coordinate transformations
    """
    
    theta = (azimuth / 180.0) * np.pi 
    
    rotation_matrix = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],]),
        np.array([[1 / major_range, 0], [0, 1 / minor_range]]))
    
    return rotation_matrix

class Covariance:
    
    def covar(effective_lag, sill, nug, vtype, s=None):
        """
        Compute covariance
        
        Parameters
        ----------
            effective_lag : int, float
                lag distance that is normalized to a range of 1
            sill : int, float
                sill of variogram
            nug : int, float
                nugget of variogram
            vtype : string
                type of variogram model (Exponential, Gaussian, Spherical, or Matern)
            s : float
                smoothness for Matern covariance
        Raises
        ------
        AtrributeError : if vtype is not 'Exponential', 'Gaussian', or 'Spherical'

        Returns
        -------
            c : numpy.ndarray
                covariance
        """
        
        if vtype.lower() == 'exponential':
            c = (sill - nug)*np.exp(-3 * effective_lag)
        elif vtype.lower() == 'gaussian':
            c = (sill - nug)*np.exp(-3 * np.square(effective_lag))
        elif vtype.lower() == 'spherical':
            c = sill - nug - 1.5 * effective_lag + 0.5 * np.power(effective_lag, 3)
            c[effective_lag > 1] = sill - 1
        elif vtype.lower() == 'matern':
            scale = 0.45246434*np.exp(-0.70449189*s)+1.7863836
            effective_lag[effective_lag==0.0] = 1e-8
            c = (sill-nug)*2/gamma(s)*np.power(scale*effective_lag*np.sqrt(s), s)*kv(s, 2*scale*effective_lag*np.sqrt(s))
            c[np.isnan(c)] = sill-nug
        else: 
            raise AttributeError(f"vtype must be 'Exponential', 'Gaussian', 'Spherical', or Matern")
        return c

    def make_covariance_matrix(coord, vario, rotation_matrix):
        """
        Make covariance matrix showing covariances between each pair of input coordinates
        
        Parameters
        ----------
            coord : numpy.ndarray
                coordinates of data points
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            rotation_matrix : numpy.ndarray
                rotation matrix used to perform coordinate transformations
        
        Returns
        -------
            covariance_matrix : numpy.ndarray 
                nxn matrix of covariance between n points
        """
        
        nug = vario['nugget']
        sill = vario['sill']  
        vtype = vario['vtype']
        if vtype.lower() == 'matern':
            if 'smoothness' in vario:
                s = vario['smoothness']
            else:
                raise ValueError("smoothness s must be specified for Matern covariance")
        else:
            s = None
        mat = np.matmul(coord, rotation_matrix)
        effective_lag = pairwise_distances(mat,mat)
        covariance_matrix = Covariance.covar(effective_lag, sill, nug, vtype, s=s)

        return covariance_matrix

    def make_covariance_array(coord1, coord2, vario, rotation_matrix):
        """
        Make covariance array showing covariances between each data points and grid cell of interest
        
        Parameters
        ----------
            coord1 : numpy.ndarray
                coordinates of n data points
            coord2 : numpy.ndarray
                coordinates of grid cell of interest (i.e. grid cell being simulated) that is repeated n times
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            rotation_matrix - rotation matrix used to perform coordinate transformations
        
        Returns
        -------
            covariance_array : numpy.ndarray
                nx1 array of covariance between n points and grid cell of interest
        """
        
        nug = vario['nugget']
        sill = vario['sill']  
        vtype = vario['vtype']
        if vtype.lower() == 'matern':
            if 'smoothness' in vario:
                s = vario['smoothness']
            else:
                raise ValueError("smoothness s must be specified for Matern covariance")
        else:
            s = None
        mat1 = np.matmul(coord1, rotation_matrix) 
        mat2 = np.matmul(coord2.reshape(-1,2), rotation_matrix) 
        effective_lag = np.sqrt(np.square(mat1 - mat2).sum(axis=1))
        covariance_array = Covariance.covar(effective_lag, sill, nug, vtype, s=s)

        return covariance_array

def get_random_generator(seed):
    """
    Conveniance function to get numpy random number generator for SGS. If seed is None, a random
    seed is used. If seed is an integer, that integer is used to seed the RNG. If seed is
    already an instance of a numpy RNG that is returned.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, int):
        rng = np.random.default_rng(seed=seed)
    elif isinstance(seed, np.random._generator.Generator):
        rng = seed
    else:
        raise ValueError('Seed should be an integer, a NumPy random Generator, or None')
    return rng

def do_variograms(xx, yy, grid, bin_func='even', maxlag=100e3, n_lags=70, covmodels=['gaussian', 'spherical', 'exponential', 'matern'], downsample=None):
    """
    Make experimental variogram and fit covariance models.

    Args:
        grid : gridded data, nan where there is not conditioning data
        bin_func : binning function or array of bin edges
        maxlag : maximum lag for experimental variogram
        n_lags : number of lag bins for variogram
        covmodels : covariance models to fit to variogram
        azimuth : orientation in degrees of primary range
    Outputs:
        Dictionary of variograms, pd.DataFrame of dataset, experimental variogram values, bins, and nscore transformer
    """
    cond_msk = ~np.isnan(grid)
    grid_norm, nst_trans = gaussian_transformation(grid, cond_msk)
    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    data_norm = grid_norm[cond_msk]
    coords_cond = np.array([x_cond, y_cond]).T

    if isinstance(downsample, int):
        data_norm = data_norm[::10]
        coords_cond = coords_cond[::10]

    vgrams = {}

    # compute experimental (isotropic) variogram
    V = skg.Variogram(coords_cond, data_norm, bin_func=bin_func, n_lags=n_lags, 
                   maxlag=maxlag, normalize=False)
    
    V.model = covmodels[0]
    vgrams[covmodels[0]] = V.parameters

    if len(covmodels) > 1:
        for i, cov in enumerate(covmodels[1:]):
            V_i = deepcopy(V)
            V_i.model = cov
            vgrams[cov] = V_i.parameters

    return vgrams, V.experimental, V.bins