from MesoPy import Meso
from datetime import datetime,timezone,timedelta
import pandas as pd
import logging

_meso_token = 'a597022227f14b7f80c6cfa3f896624b'

def meso2df(mesowestData):
    if mesowestData is None:
        return pd.DataFrame([]),pd.DataFrame([])
    site_keys = ['STID', 'MNET_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'STATE']
    site_dic = {key: [] for key in site_keys}
    vars = ['date_time'] + [key for key in mesowestData['UNITS'].keys() if key not in ['position', 'elevation']]
    data_keys = ['STID'] + vars + ['wind_speed_height']
    data_dic = {key: [] for key in data_keys}
    for stData in mesowestData['STATION']:
        for site_key in site_keys:
            site_dic[site_key].append(stData[site_key]) 
        len_obs = len(stData['OBSERVATIONS']['date_time'])
        data_dic['STID'] += [stData['STID']]*len_obs
        wind_found = False
        for var in vars:
            var_found = False
            if var in stData['SENSOR_VARIABLES'].keys():
                for key in sorted(stData['SENSOR_VARIABLES'][var].keys(), reverse=True):
                    if key in stData['OBSERVATIONS'].keys():
                        data_dic[var] += stData['OBSERVATIONS'][key]
                        if var == 'wind_speed':
                            data_dic['wind_speed_height'] += [stData['SENSOR_VARIABLES'][var][key].get('position', None)]*len_obs 
                            wind_found = True 
                        var_found = True
                        break
            if not var_found:
                data_dic[var] += [None]*len_obs
        if not wind_found:
            data_dic['wind_speed_height'] += [None]*len_obs
    data = pd.DataFrame.from_dict(data_dic)
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['wind_speed_height'] = data['wind_speed_height'].astype(float)
    sites = pd.DataFrame.from_dict(site_dic).set_index('STID')
    sites['LONGITUDE'] = sites['LONGITUDE'].astype(float)
    sites['LATITUDE'] = sites['LATITUDE'].astype(float)
    sites['ELEVATION'] = sites['ELEVATION'].astype(int)
    return data, sites

def meso_time(dt):
    # example: 201603311600
    return '%04d%02d%02d%02d%02d' % (dt.year, dt.month, dt.day, dt.hour, dt.minute)

def get_mesowest(tm_start, tm_end, bbox, vars):
    assert tm_end > tm_start, 'ERROR: ending time must be larger than start time'
    results = []
    from_date = tm_start
    while from_date < tm_end:
        to_date = from_date + timedelta(days=1)
        logging.info('getting data from {} to {}...'.format(from_date,to_date))
        meso_tstart = meso_time(from_date)
        meso_tend = meso_time(to_date)
        meso_bbox = '{},{},{},{}'.format(*bbox)
        meso_vars = ','.join(vars)
        meso = Meso(_meso_token)
        meso_obss = meso.timeseries(
                        meso_tstart, meso_tend,
                        showemptystations='0', 
                        bbox=meso_bbox,
                        vars=meso_vars)
        data, sites = meso2df(meso_obss)
        results.append(data.join(sites,'STID'))
        from_date = to_date
    return pd.concat(results).reset_index(drop=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    tm_start = datetime(2021,8,18,tzinfo=timezone.utc)
    tm_end = datetime(2021,9,18,tzinfo=timezone.utc)
    bbox = (-120.92843627929688,38.372676849365234,-119.6753158569336,39.34367370605469)
    vars = ['air_temp', 'relative_humidity', 'wind_speed', 'wind_direction', 'PM_25_concentration']
    meso_data = get_mesowest(tm_start, tm_end, bbox, vars)
    meso_data.to_csv('meso_data.csv', index=False)