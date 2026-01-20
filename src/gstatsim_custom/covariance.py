import numpy as np
from scipy.special import kv, gamma
from skgstat import models

def exponential_cov_norm(h, sill, nugget, **kwargs):
    return sill*np.exp(-3*h) - nugget

def gaussian_cov_norm(h, sill, nugget, **kwargs):
    return sill*np.exp(-4*h**2) - nugget

def spherical_cov_norm(h, sill, nugget, **kwargs):
    c = sill - sill*(1.5*h - 0.5*np.power(h, 3)) - nugget 
    c[h > 1] = -1*nugget
    return c

def matern_cov_norm(h, sill, nugget, s, **kwargs):
    scale = 0.45246434*np.exp(-0.70449189*s)+1.7863836
    h[h==0.0] = 1e-8
    c = sill*2/gamma(s)*np.power(2*h*np.sqrt(s), s)*kv(s, 4*h*np.sqrt(s)) - nugget
    c[np.isnan(c)] = sill-nugget
    return c

# def exponential_cov_norm(h, c0, b, **kwargs):
#     c = c0 - models.exponential(h, 1, c0, b)
#     return c

# def gaussian_cov_norm(h, c0, b, **kwargs):
#     c = c0 - models.gaussian(h, 1, c0, b)
#     return c

# def spherical_cov_norm(h, c0, b, **kwargs):
#     c = c0 - models.spherical(h, 1, c0, b)
#     return c

# def matern_cov_norm(h, c0, b, s, **kwargs):
#     c = c0 - models.matern(h, 1, c0, s, b)
#     return c

covmodels = {
    'matern' : matern_cov_norm,
    'exponential' : exponential_cov_norm,
    'gaussian' : gaussian_cov_norm,
    'spherical' : spherical_cov_norm
}