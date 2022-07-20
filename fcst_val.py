from meso import get_mesowest
from wrf_interp import meso_opts, wrf_spatial_interp, wrf_temporal_interp
import pandas as pd
import os.path as osp
import logging

_obs_path = 'obs_data.csv'
_wrf_path = 'wrf_data.csv'
_interp_path = 'interp_data.csv'

def fcst_val(wrf_paths, misr_path=None):
    if osp.exists(_interp_path):
        logging.info('fcst_val - found interp data {}'.format(_interp_path))
        df = pd.read_csv(_interp_path)
        df.date_time = pd.to_datetime(df.date_time)
    else:
        if osp.exists(_obs_path):
            logging.info('fcst_val - found meso data {}'.format(_obs_path))
            obs_data = pd.read_csv(_obs_path)
        else:
            tm_start, tm_end, bbox, vars = meso_opts(wrf_paths)
            logging.info('fcst_val - meso options: \n   tm_start={}\n   tm_end={}\n   bbox={}\n   vars={}'.format(tm_start, tm_end, bbox, vars))
            meso_data = get_mesowest(tm_start, tm_end, bbox, vars)
            if osp.exists(misr_path):
                misr_data = pd.read_csv(misr_path)
                misr_data['date_time'] = pd.to_datetime(misr_data['date_time'])
            else:
                misr_data = pd.DataFrame([])
            obs_data = pd.concat((meso_data, misr_data))
            obs_data.to_csv(_obs_path, index=False)
            logging.info('fcst_val - meso data saved {}'.format(_obs_path))
        if osp.exists(_wrf_path):
            logging.info('fcst_val - found wrf data {}'.format(_wrf_path))
            wrf_data = pd.read_csv(_wrf_path)
        else:
            wrf_data = wrf_spatial_interp(wrf_paths, obs_data)
            wrf_data.to_csv(_wrf_path, index=False)
            logging.info('fcst_val - wrf data saved {}'.format(_wrf_path))
        df = wrf_temporal_interp(wrf_data, obs_data)
        df.to_csv(_interp_path, index=False)
        logging.info('fcst_val - interp data saved {}'.format(_interp_path))

if __name__ == '__main__':
    import glob
    import sys
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    if len(sys.argv) < 2:
        logging.error('usage - python {} forecasts [misr_path]'.format(sys.argv[0]))
        sys.exit(1)
    wrf_paths = sorted(glob.glob(sys.argv[1]))
    misr_path = sys.argv[2]
    fcst_val(wrf_paths, misr_path)