from perimeters import Perimeters
from meso import get_mesowest
from wrf_interp import meso_opts,wrf_spatial_interp,wrf_temporal_interp,obs2wrf
from datetime import datetime
import pandas as pd
import os.path as osp
import numpy as np
import logging
import json
import glob
import sys

def load_cfg():
    base_path = osp.dirname(sys.argv[0])
    js = json.load(open(osp.join(base_path,'conf.json')))
    js['sims_path'] = js.get('sims_path', None)
    if js['sims_path'] is None or not osp.exists(js['sims_path']):
        logging.critical('load_cfg - sims_path not found {}'.format(js['sims_path']))
    js['domain'] = js.get('domain', 3)
    js['meso_vars'] = js.get('meso_vars', None)
    js['start_utc'] = js.get('start_utc', None)
    js['end_utc'] = js.get('end_utc', None)
    js['ir_path'] = js.get('ir_path', None)
    js['misr_path'] = js.get('misr_path', None)
    js['obs_path'] = js.get('obs_path', osp.join(base_path, 'obs_data.pkl'))
    if js['obs_path'] is None:
        js['obs_path'] = osp.join(base_path, 'obs_data.pkl')
    js['wrf_path'] = js.get('wrf_path', osp.join(base_path, 'wrf_data.pkl'))
    if js['wrf_path'] is None:
        js['wrf_path'] = osp.join(base_path, 'wrf_data.pkl')
    js['res_path'] = js.get('res_path', osp.join(base_path, 'res_data.pkl'))
    if js['res_path'] is None:
        js['res_path'] = osp.join(base_path, 'res_data.pkl')
    js['network_height_path'] = js.get('network_height_path', osp.join(base_path, 'network_height.csv'))
    if js['network_height_path'] is None:
        js['network_height_path'] = osp.join(base_path, 'network_height_data.pkl')
    js['stats_path'] = js.get('stats_path', osp.join(base_path, 'result_stats.csv'))
    if js['stats_path'] is None:
        js['stats_path'] = osp.join(base_path, 'result_stats.csv')
    js['plots_path'] = js.get('plots_path', osp.join(base_path, 'plots'))
    if js['plots_path'] is None:
        js['plots_path'] = osp.join(base_path, 'plots')
    return js

def fcst_val():
    js = load_cfg()
    if osp.exists(js['res_path']):
        logging.info('fcst_val - found interp data {}'.format(js['res_path']))
        df = pd.read_pickle(js['res_path'])
        df.date_time = pd.to_datetime(df.date_time)
    else:
        sims_path = osp.join(js['sims_path'],'wrf/wrfout_d{:02d}*'.format(js['domain']))
        wrf_paths = sorted(glob.glob(sims_path))
        if osp.exists(js['obs_path']):
            logging.info('fcst_val - found obs data {}'.format(js['obs_path']))
            obs_data = pd.read_pickle(js['obs_path'])
        else:    
            tm_start, tm_end, bbox, vars = meso_opts(wrf_paths)
            if js['start_utc'] is not None:
                start_utc = datetime.strptime(js['start_utc'],'%Y%m%d%H%M')
                tm_start = max(start_utc, tm_start)
            if js['end_utc'] is not None:
                end_utc = datetime.strptime(js['end_utc'],'%Y%m%d%H%M')
                tm_end = min(end_utc, tm_end)
            if js['meso_vars'] is not None and isinstance(js['meso_vars'],list) and len(js['meso_vars']):
                user_vars = [v for v in js['meso_vars'] if v in obs2wrf.keys()]
                if len(user_vars):
                    vars = user_vars
            logging.info('fcst_val - meso options: \n   tm_start={}\n   tm_end={}\n   bbox={}\n   vars={}'.format(tm_start, tm_end, bbox, vars))
            obs_data = get_mesowest(tm_start, tm_end, bbox, vars)
            tm_start = pd.to_datetime(tm_start, utc=True)
            tm_end = pd.to_datetime(tm_end, utc=True)
            if osp.exists(js['misr_path']):
                logging.info('fcst_val - found MISR data {}'.format(js['misr_path']))
                misr_data = pd.read_csv(js['misr_path'])
                misr_data['date_time'] = pd.to_datetime(misr_data['date_time'])
                misr_data = misr_data[misr_data['date_time'].between(tm_start, tm_end, inclusive='both')]
                misr_data = misr_data[
                    np.logical_and(misr_data.LONGITUDE >= bbox[0], 
                        np.logical_and(misr_data.LONGITUDE <= bbox[2],
                            np.logical_and(misr_data.LATITUDE >= bbox[1], 
                                            misr_data.LATITUDE <= bbox[3])))]
                obs_data = pd.concat((obs_data, misr_data))
            if osp.exists(js['network_height_path']):
                logging.info('fcst_val - found network height data {}'.format(js['network_height_path']))
                net = pd.read_csv(js['network_height_path'])
                obs_data.loc[obs_data['MNET_ID'].isna(), 'MNET_ID'] = 0
                obs_data['MNET_ID'] = obs_data.MNET_ID.astype(int)
                obs_data = obs_data.merge(net, how='left', on='MNET_ID')
            if osp.exists(js['ir_path']):
                logging.info('fcst_val - found IR perimeters data {}'.format(js['ir_path']))
                perims = Perimeters(js['ir_path'])
                if len(perims):
                    ir_perim_info = {
                        'STID': 'IR_DATA', 'date_time': [], 'LONGITUDE': (bbox[0]+bbox[2])/2,
                        'LATITUDE': (bbox[1]+bbox[3])/2, 'fire_area': [], 'fire_perim': []
                    }
                    n = 1
                    for perim in perims:
                        if perim.area > 0 and perim.time >= tm_start and perim.time <= tm_end:
                            ir_perim_info['date_time'].append(perim.time)
                            ir_perim_info['fire_area'].append(perim.area)
                            ir_perim_info['fire_perim'].append(perim)
                            n += 1
                    ir_data = pd.DataFrame(ir_perim_info)
                    obs_data = pd.concat((obs_data, ir_data))
            obs_data.reset_index(drop=True).to_pickle(js['obs_path'])
            logging.info('fcst_val - obs data saved {}'.format(js['obs_path']))
        if osp.exists(js['wrf_path']):
            logging.info('fcst_val - found wrf data {}'.format(js['wrf_path']))
            wrf_data = pd.read_pickle(js['wrf_path'])
        else:
            wrf_data = wrf_spatial_interp(wrf_paths, obs_data)
            wrf_data.to_pickle(js['wrf_path'])
            logging.info('fcst_val - wrf data saved {}'.format(js['wrf_path']))
        df = wrf_temporal_interp(wrf_data, obs_data)
        df.to_pickle(js['res_path'])
        logging.info('fcst_val - interp data saved {}'.format(js['res_path']))
        from stat_plot_results import stat_plot_results
        stat_plot_results(df,js)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    fcst_val()