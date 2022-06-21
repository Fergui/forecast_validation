from scipy.interpolate import griddata, interp1d
from wrf import getvar
import numpy as np
import netCDF4 as nc
import glob
import re
from datetime import datetime
import pandas as pd
import os.path as osp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging

groups = {
    0: ((0, 12), '12h'),
    1: ((12, 24), '24h'),
    2: ((24, 36), '36h'),
    3: ((36, 48), '48h')
}

obs2wrf = {
    'wind_speed': lambda d,t,h: getvar(d, 'wspd_wdir10', timeidx=t)[0], 
    'wind_direction': lambda d,t,h: getvar(d, 'wspd_wdir10', timeidx=t)[1], 
    'air_temp': lambda d,t,h: getvar(d, 'T2', timeidx=t) - 273.15,
    'relative_humidity': lambda d,t,h: getvar(d, 'rh2', timeidx=t),
    'PM_25_concentration': lambda d,t,h: d['tr17_1'][t][0]
}

def wrf_time(ds, tindx=0):
    str_time = ''.join([c.decode() for c in ds['Times'][tindx]])
    dt_time = datetime.strptime(str_time,'%Y-%m-%d_%H:%M:%S')
    return dt_time

def wrf_bbox(ds):
    xlat = ds.variables['XLAT'][0]
    xlon = ds.variables['XLONG'][0]
    return xlon.min(), xlon.max(), xlat.min(), xlat.max()

def meso_opts(fc_paths):
    fc_paths = sorted(fc_paths)
    with nc.Dataset(fc_paths[0]) as ds:
        tm_start = wrf_time(ds, 0)
    with nc.Dataset(fc_paths[-1]) as ds:
        ds = nc.Dataset(fc_paths[-1])
        tm_end = wrf_time(ds, -1)
        bb = wrf_bbox(ds)
        bbox = (bb[0],bb[2],bb[1],bb[3])
    vars = obs2wrf.keys()
    return tm_start, tm_end, bbox, vars

def wrf_spatial_interp(wrf_files, obs):
    logging.info('wrf_spatial_interp - starting spatial interpolation')
    obs['date_time'] = pd.to_datetime(obs['date_time'])
    stats = obs.groupby('STID').first()[['LONGITUDE','LATITUDE']] 
    fields = ['STID', 'LATITUDE', 'LONGITUDE', 'fc_time', 'wrf_time', 'offset']
    fields += ['wrf_'+k for k in obs2wrf.keys()]
    result = {f:[] for f in fields}
    for file in wrf_files:
        logging.debug('wrf_spatial_interp - spatial interpolation of WRF data from {}'.format(osp.basename(file)))
        m = re.match(r'.*-([0-9]{4})-([0-9]{2})-([0-9]{2})_([0-9]{2}):([0-9]{2}):([0-9]{2})-*',file)
        if m is not None:
            fcstarttime = '{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}'.format(*[int(_) for _ in m.groups()])#converting strings into integers 
            with nc.Dataset(file, 'r', format='NETCDF4') as ds:
                ft_dt = datetime.strptime(fcstarttime,'%Y-%m-%d_%H:%M:%S')
                xlat = ds.variables['XLAT'][0]
                xlon = ds.variables['XLONG'][0]
                minlon, maxlon, minlat, maxlat = xlon.min(), xlon.max(), xlat.min(), xlat.max()
                mask = np.logical_and(stats.LONGITUDE  >= minlon, 
                        np.logical_and(stats.LONGITUDE <= maxlon, 
                            np.logical_and(stats.LATITUDE >= minlat, stats.LATITUDE <= maxlat)))
                coords = stats[mask]
                points = np.c_[xlon.ravel(), xlat.ravel()]
                xi = np.c_[coords.LONGITUDE, coords.LATITUDE]
                stid = coords.index
                for t in range(len(ds['Times'])):
                    wt_dt = wrf_time(ds, t)
                    offset = abs(wt_dt-ft_dt).total_seconds()/3600.
                    result['STID'].append(stid)
                    result['LONGITUDE'].append(xi[:, 0])
                    result['LATITUDE'].append(xi[:, 1])
                    result['fc_time'].append([ft_dt]*len(xi))
                    result['wrf_time'].append([wt_dt]*len(xi))
                    result['offset'].append([offset]*len(xi))
                    for var in obs.columns:
                        if var in obs2wrf.keys():
                            # get data
                            values = np.ravel(obs2wrf[var](ds, t, h=None))
                            # interpolate data
                            yi = griddata(points, values, xi, method='linear')
                            # concatenate data
                            result['wrf_'+var].append(yi)
    for k,v in result.items():
        result[k] = np.concatenate(v)
    df = pd.DataFrame(result)
    return df

def wrf_temporal_interp(wrf, obs):
    logging.info('wrf_temporal_interp - starting temporal interpolation')
    wrf['wrf_time'] = pd.to_datetime(wrf['wrf_time'])
    obs['date_time'] = pd.to_datetime(obs['date_time'])
    obs.sort_values(['STID', 'date_time'], inplace=True)
    wrf.sort_values(['STID', 'wrf_time'], inplace=False)
    df = [[] for _ in range(len(groups))]
    for g,((ioff,foff),label) in groups.items():
        logging.info('wrf_temporal_interp - starting temporal interpolation of group {}'.format(label))
        for stid,data in wrf[wrf.offset.between(ioff,foff)].groupby('STID'):
            logging.debug('wrf_temporal_interp - temporal interpolation of station {}'.format(stid))
            df_st = obs.set_index('STID').loc[[stid]].reset_index()
            ti = data.wrf_time.to_numpy().astype('datetime64[m]').astype('float64')
            t = df_st.date_time.to_numpy().astype('datetime64[m]').astype('float64')
            for var in obs2wrf.keys():
                wrf_var = 'wrf_'+var
                fi = data[wrf_var].to_numpy().astype('float64')
                ft = interp1d(ti, fi, bounds_error=False)
                df_st[wrf_var] = ft(t)
            df_st['offset'] = label
            df[g].append(df_st)
    df_g = []
    for g in groups.keys():
        df_g.append(pd.concat(df[g]))
    df = pd.concat(df_g)
    return df

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    fc_path = '/home/akochanski/scratch/Caldor/wfc-from-web-*-16-10-*-48/wrf/'
    obs_path = '/home/afarguell/scratch/forecasts/caldor/raws_data/meso_data_caldor.csv'
    fc_files = sorted(glob.glob(fc_path + "wrfout_d03*"))
    obs = pd.read_csv(obs_path)
    fcst = wrf_spatial_interp(fc_files, obs)
    fcst.to_csv('wrf_data.csv', index=False)
    df = wrf_temporal_interp(fcst, obs)
    df.to_csv('interp_data.csv', index=False)