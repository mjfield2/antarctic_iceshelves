import os
import xarray as xr
from pathlib import Path


def load_antgg():
    """
    Load different AntGG2021 files and put them into correct format in a single xarray.Dataset
    """
    
    das = []
    for item in os.scandir(Path('D:/AntGG2021_allfiles')):
        if item.name.endswith('.nc'):
            print(item.name)
            da_tmp = xr.open_dataset(item.path)
            das.append(da_tmp)

    antgg = xr.merge(das, compat='override')

    antgg = xr.Dataset(
        data_vars = dict(
            Boug_anom = (['y', 'x'], antgg.Boug_anom.values[::-1,:]),
            grav_anom = (['y', 'x'], antgg.grav_anom.values[::-1,:]),
            grav_dist = (['y', 'x'], antgg.grav_dist.values[::-1,:]),
            gravity_dist_5000m = (['y', 'x'], antgg.gravity_disturbance_5000m.values[::-1,:]),
            stdev = (('y', 'x'), antgg.std_grav_anom.values[::-1,:]),
            ell_surf_height = (('y', 'x'), antgg.h_ell.values[::-1,:]),
            geoid = (('y', 'x'), antgg.h_anomaly_ell.values[::-1,:]),
            d2T_dr2 = (('y', 'x'), antgg.d2T_dr2.values[::-1,:])
        ),
        coords=dict(
            y=("y", antgg.y.values[::-1]),
            x=("x", antgg.x.values),
        )
    )

    return antgg


if __name__ == '__main__':
    antgg = load_antgg()
    antgg.to_netcdf(Path('../raw_data/antgg.nc'))