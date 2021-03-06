from perimeters import Perimeter
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
    'wind_speed': lambda d,t,h: wind_at_h(d,t,h)[0],
    'wind_direction': lambda d,t,h: wind_at_h(d,t,h)[1], 
    'air_temp': lambda d,t,h: getvar(d, 'T2', timeidx=t) - 273.15,
    'relative_humidity': lambda d,t,h: getvar(d, 'rh2', timeidx=t),
    'PM_25_concentration': lambda d,t,h: d['tr17_1'][t][0],
    'plume_height': lambda d,t,h: plume_height(d,t),
    'fire_area': lambda d,t,h: None,
    'fire_perim': lambda d,t,h: None
}

external_vars = ['plume_height', 'fire_area', 'fire_perim']

def height8w(d,t):
    """
    Compute height at mesh bottom a.k.a. w-points 
    :param d: open NetCDF4 dataset
    :param t: number of timestep
    """
    ph = d.variables['PH'][t,:,:,:]  
    phb = d.variables['PHB'][t,:,:,:]
    return (phb + ph)/9.81 # geopotential height at W points

def height8p(d,t):
    """
    Compute height of mesh centers (p-points)
    :param d: open NetCDF4 dataset
    :param t: number of timestep
    """
    z8w = height8w(d,t)
    return 0.5*(z8w[0:z8w.shape[0]-1,:,:]+z8w[1:,:,:])

def plume_height(d,t):
    """
    Compute plume height
    :param d: open NetCDF4 dataset
    :param t: number of timestep
    """
    smoke_threshold = 1
    z =  height8p(d,t)
    tr = d.variables['tr17_1'][t,:,:,:]
    h = np.zeros(tr.shape[1:])
    for i in range(0, tr.shape[2]):
      for j in range(0, tr.shape[1]):
          for k in range(tr.shape[0]-1, -1, -1):
               if tr[k,j,i] > smoke_threshold:
                    h[j,i] = z[k,j,i]
                    break
    return h

def wind_at_h(ds, t, h):
    """
    Compute wind at a certain height
    :param d: open NetCDF4 dataset
    :param t: number of timestep
    :param h: height in meters
    """
    #wind_speed,wind_direction = getvar(ds,'wspd_wdir10',timeidx=t) # old version of wind
    k = 0.41 # Karman's constant
    u10,v10 = getvar(ds,'uvmet10',timeidx=t)
    ust = getvar(ds,'UST',timeidx=t)
    uh = u10 + ust/k*np.log(10/h)
    vh = v10 + ust/k*np.log(10/h)
    wind_speed = np.sqrt(uh**2+vh**2)
    wind_direction = 270 - np.arctan2(vh,uh)*180/np.pi
    return wind_speed,wind_direction

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
    vars = [v for v in obs2wrf.keys() if v not in external_vars]
    return tm_start, tm_end, bbox, vars

def wrf_spatial_interp(wrf_files, obs):
    logging.info('wrf_spatial_interp - starting spatial interpolation')
    obs['date_time'] = pd.to_datetime(obs['date_time'])
    # set default wind height
    default_wind_height = 10
    if 'default_wind_speed_height' not in obs.keys():
        # if variable default_wind_speed_height doesn't exists, created with values default_wind_height
        obs['default_wind_speed_height'] = default_wind_height
    else:
        # else, missing values in default_wind_speed_height, give a value of default_wind_height
        obs.loc[obs['default_wind_speed_height'].isna(), 'default_wind_speed_height'] = default_wind_height
    if 'wind_speed_height' not in obs.keys():
        # if variable wind_speed_height doesn't exists, created with values from default_wind_speed_height
        obs['wind_speed_height'] = obs['default_wind_speed_height']
    else:
        # else, missing values in wind_speed_height, give a value from default_wind_speed_height
        if obs['wind_speed_height'].isna().sum():
            obs.loc[obs['wind_speed_height'].isna(), 'wind_speed_height'] = obs['default_wind_speed_height'][obs['wind_speed_height'].isna()]
    stats = obs.groupby(['STID','wind_speed_height']).first()[['LONGITUDE','LATITUDE']].reset_index('wind_speed_height')
    fields = ['STID', 'LATITUDE', 'LONGITUDE', 'fc_time', 'wrf_time', 'offset']
    fields += ['wrf_'+k for k in obs2wrf.keys()]
    result = {f:[] for f in fields}
    for file in wrf_files:
        logging.debug('wrf_spatial_interp - spatial interpolation of WRF data from {}'.format(osp.basename(file)))
        m = re.match(r'.*-([0-9]{4})-([0-9]{2})-([0-9]{2})_([0-9]{2}):([0-9]{2}):([0-9]{2})-*',file)
        if m is not None:
            fcstarttime = '{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}'.format(*[int(_) for _ in m.groups()]) #converting strings into integers 
            try:
                with nc.Dataset(file, 'r', format='NETCDF4') as ds:
                    ft_dt = datetime.strptime(fcstarttime,'%Y-%m-%d_%H:%M:%S')
                    xlat = ds.variables['XLAT'][0]
                    xlon = ds.variables['XLONG'][0]
                    minlon, maxlon, minlat, maxlat = xlon.min(), xlon.max(), xlat.min(), xlat.max()
                    mask = np.logical_and(stats.LONGITUDE  >= minlon, 
                            np.logical_and(stats.LONGITUDE <= maxlon, 
                                np.logical_and(stats.LATITUDE >= minlat, stats.LATITUDE <= maxlat)))
                    coords = stats[mask]
                    pidx = np.where(coords.index == 'IR_DATA')[0]
                    points = np.c_[xlon.ravel(), xlat.ravel()]
                    xi = np.c_[coords.LONGITUDE, coords.LATITUDE]
                    stid = coords.index
                    for t in range(len(ds['Times'])):
                        wt_dt = wrf_time(ds, t)
                        if len(pidx) == 1:
                            logging.debug('wrf_spatial_interp - processing fire perimeters')
                            fxlat = ds.variables['FXLAT'][...]
                            fxlong = ds.variables['FXLONG'][...]
                            fa = ds.variables['FIRE_AREA'][...]
                            aff = 1.-fa
                            array = np.concatenate((fxlong,fxlat,aff))
                            p = Perimeter({'array': array, 'time': wt_dt})
                            perim = [np.nan]*len(xi)
                            perim[pidx[0]] = p
                            area = [np.nan]*len(xi)
                            area[pidx[0]] = p.area
                        logging.debug('wrf_spatial_interp - processing weather data')
                        offset = abs(wt_dt-ft_dt).total_seconds()/3600.
                        result['STID'].append(stid)
                        result['LONGITUDE'].append(xi[:, 0])
                        result['LATITUDE'].append(xi[:, 1])
                        result['fc_time'].append([ft_dt]*len(xi))
                        result['wrf_time'].append([wt_dt]*len(xi))
                        result['offset'].append([offset]*len(xi))
                        for var in obs.columns:
                            if var in obs2wrf.keys():
                                if 'wind' in var:
                                    data = pd.DataFrame(np.zeros(len(coords)), columns=[var])
                                    # interpolate at each unique height
                                    for h in coords.wind_speed_height.unique():
                                        # get index where to interpolate to
                                        idx = coords['wind_speed_height'] == h
                                        # get data
                                        values = np.ravel(obs2wrf[var](ds, t, h=h))
                                        # interpolate data
                                        yi = griddata(points, values, xi[idx], method='linear')
                                        # set value for the right heights
                                        data[var][idx.values] = yi
                                    result['wrf_'+var].append(data[var].values)
                                elif 'fire' not in var:
                                    # get data
                                    values = np.ravel(obs2wrf[var](ds, t, h=None))
                                    # interpolate data
                                    yi = griddata(points, values, xi, method='linear')
                                    # concatenate data
                                    result['wrf_'+var].append(yi)
                                elif var == 'fire_area':
                                    result['wrf_'+var].append(area)
                                elif var == 'fire_perim':
                                    result['wrf_'+var].append(perim)
                                else:
                                    logging.warning('wrf_spatial_interp - var {} not recognized'.format(var))
            except Exception as e:
                logging.warning('wrf_spatial_interp - some issue processing file {}, more details below:\n{}'.format(file,e))
    for k,v in result.items():
        result[k] = np.concatenate(v)
    df = pd.DataFrame(result)
    return df

def wrf_temporal_interp(wrf, obs):
    logging.info('wrf_temporal_interp - starting temporal interpolation')
    wrf['wrf_time'] = pd.to_datetime(wrf['wrf_time'])
    obs['date_time'] = pd.to_datetime(obs['date_time'])
    obs.sort_values(['STID', 'date_time'], inplace=True)
    wrf.sort_values(['STID', 'wrf_time'], inplace=True)
    df = [[] for _ in range(len(groups))]
    for g,((ioff,foff),label) in groups.items():
        logging.info('wrf_temporal_interp - starting temporal interpolation of group {}'.format(label))
        for stid,data in wrf[wrf.offset.between(ioff,foff,inclusive='right')].groupby('STID'):
            logging.debug('wrf_temporal_interp - temporal interpolation of station {}'.format(stid))
            df_st = obs.set_index('STID').loc[[stid]].reset_index()
            twrf = data.wrf_time.to_numpy().astype('datetime64[m]').astype('float64')
            tobs = df_st.date_time.to_numpy().astype('datetime64[m]').astype('float64')
            for var in obs2wrf.keys():
                wrf_var = 'wrf_'+var
                if var == 'fire_perim':
                    fwrf = data[wrf_var].to_numpy()
                    tindx = [np.argmin([abs(tw-to)/3600. for tw in twrf]) for to in tobs]
                    fobs = fwrf[tindx]
                    tmask = np.logical_or(tobs < twrf[0], tobs > twrf[1])
                    fobs[tmask] = np.nan
                else:
                    fwrf = data[wrf_var].to_numpy().astype('float64')
                    ft = interp1d(twrf, fwrf, bounds_error=False)
                    fobs = ft(tobs)
                df_st[wrf_var] = fobs
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
