import warnings
from fcst_val import load_cfg
warnings.filterwarnings('ignore')
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from geometry import lonlat_to_merc,merc_to_lonlat,validate
from wrf_interp import obs2wrf, groups
from scipy.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as osp
import logging
import sys

# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])
    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

# global params
fontsize = 30
simplify_tol = 1e-6
sorenson_vars = ['intersect', 'underpred', 'overpred', 'sorenson']
# statistical measures of error
pearson = lambda x,y: pearsonr(x,y)[0]
spearman = lambda x,y: spearmanr(x,y)[0]
rmse = lambda x,y: np.sqrt(abs((x-y)**2).mean())
mae = lambda x,y: abs(x-y).mean()
smape = lambda x,y: int(np.round((abs(x-y)/(abs(x)+abs(y))).mean()*100.))
T = lambda x: np.sin(np.pi*x/180)
Tinv = lambda x: np.arcsin(x)*180/np.pi
cpearson = lambda x,y: pearson(T(x),T(y))
cspearman = lambda x,y: spearman(T(x),T(y))
crmse = lambda x,y: np.sqrt(abs(T(x-y)**2).mean())
cmae = lambda x,y: abs(T(x-y)).mean()
csmape = lambda x,y: int(np.round((abs(T(x-y))/(abs(T(x))+abs(T(y)))).mean()*100.))
def sorenson(wrf_poly, ir_poly):
    a = validate(ir_poly.intersection(wrf_poly))
    b = ir_poly.difference(a)
    c = wrf_poly.difference(a)
    return a,b,c

def stat_plot_results(js):
    res_path = js['res_path']
    if osp.exists(res_path):
        df = pd.read_pickle(res_path)
    else:
        logging.critical(f'res_path files {res_path} not found, can not compute statistics and plots')
        sys.exit(1)
    vars = [v for v in df.keys() if v in obs2wrf.keys()]
    data = {}
    for g in groups.values():
        for v in vars:
            if v != 'fire_perim':
                data.update({g[1]+'_'+v: []})
            else:
                data.update({g[1]+'_'+s: [] for s in sorenson_vars})
    df.date_time = pd.to_datetime(df.date_time)
    for var in vars:
        wrf_var = 'wrf_'+var
        df_var_ = df.dropna(subset=[var, wrf_var])
        for _,label in groups.values():
            df_var = df_var_.loc[df_var_.offset == label,['date_time', var, wrf_var]]
            df_var = df_var.set_index('date_time')
            if len(df_var):
                if var == 'fire_perim':
                    stats = []
                    for date_time,wrf_perim,ir_perim in zip(df_var.index,df_var[wrf_var],df_var[var]):
                        fig,ax = plt.subplots(1,1,figsize=(20,10))
                        wrf_poly = lonlat_to_merc(wrf_perim.poly.simplify(simplify_tol))
                        ir_poly = lonlat_to_merc(ir_perim.poly.simplify(simplify_tol))
                        a,b,c = sorenson(wrf_poly, ir_poly)
                        A = a.area/4047.
                        B = b.area/4047.
                        C = c.area/4047. 
                        S = 2*A+B+C
                        stats.append(list([A,B,C,S]))
                        a = merc_to_lonlat(a)
                        b = merc_to_lonlat(b)
                        c = merc_to_lonlat(c)
                        plot_polygon(ax, a, facecolor='green')
                        plot_polygon(ax, b, facecolor='blue')
                        plot_polygon(ax, c, facecolor='red')
                        plt.legend(['Intersect','Underpred','Overpred'])
                        plt.title(f'S={S}')
                        plt.savefig('sorenson_{}_{}.png'.format(label,date_time))
                        plt.close()
                    stats = np.array(stats)
                    stats = stats.mean(axis=0)
                    for i in range(len(stats)):
                        data[label+'_'+sorenson_vars[i]].append(stats[i])
                else:
                    if var == 'wind_direction':
                        stats = {'Pearson': cpearson, 'Spearman': cspearman, 'RMSE': crmse, 'MAE': cmae, 'SMAPE': csmape}
                        xx = Tinv(T(df_var[wrf_var]))
                        yy = Tinv(T(df_var[var]))
                    else:
                        stats = {'Pearson': pearson, 'Spearman': spearman, 'RMSE': rmse, 'MAE': mae, 'SMAPE': smape}
                        xx = df_var[wrf_var]
                        yy = df_var[var]
                    for f in stats.values():
                        data[label+'_'+var].append(f(df_var[var], df_var[wrf_var]))
                    if var == 'plume_height':
                        fig,ax = plt.subplots(1,1,figsize=(20,10))
                        corr, _ = pearsonr(xx, yy)
                        ax.plot(xx, yy, 'b.')
                        ax_min = max(xx.min(), yy.min())
                        ax_max = min(xx.max(), yy.max())
                        ax.plot([ax_min, ax_max], [ax_min, ax_max], 'k-')
                        ax.set_title('{} {} corr={}'.format(var, label, corr), fontsize=fontsize)
                        ax.set_xlabel(wrf_var, fontsize=fontsize)
                        ax.set_ylabel(var, fontsize=fontsize)
                        ax.tick_params(axis='both', labelsize=20)
                    else:
                        diff = xx - yy
                        fig,ax = plt.subplots(1,2,figsize=(40,20))
                        diff.reset_index().groupby('date_time').mean().plot(style='k-', ax=ax[0], legend=False)
                        ax[0].set_title('Error {} {}'.format(var, label), fontsize=fontsize)
                        ax[0].set_xlabel('time', fontsize=fontsize)
                        ax[0].set_ylabel('{} - {}'.format(wrf_var,var), fontsize=fontsize)
                        ax[0].tick_params(axis='both', labelsize=20)
                        if var == 'PM_25_concentration':
                            corr, _ = pearsonr(df_var[wrf_var], df_var[var])
                            df_var.loc[df_var[var] < 1.,var] = 0.
                            df_var.loc[df_var[wrf_var] < 1.,wrf_var] = 0.
                            ax[1].loglog(df_var[wrf_var], df_var[var], 'b.')
                            ax_min = max(df_var[wrf_var].min(), df_var[var].min())
                            ax_max = min(df_var[wrf_var].max(), df_var[var].max())
                            ax[1].plot([ax_min, ax_max], [ax_min, ax_max], 'k-')
                            ax[1].set_title('{} {} corr={}'.format(var, label, corr), fontsize=fontsize)
                            ax[1].set_xlabel(wrf_var, fontsize=fontsize)
                            ax[1].set_ylabel(var, fontsize=fontsize)
                            ax[1].tick_params(axis='both', labelsize=20)
                        else:
                            corr, _ = pearsonr(df_var[wrf_var], df_var[var])
                            ax[1].plot(xx, yy, 'b.')
                            ax_min = max(xx.min(), yy.min())
                            ax_max = min(xx.max(), yy.max())
                            ax[1].plot([ax_min, ax_max], [ax_min, ax_max], 'k-')
                            ax[1].set_title('{} {} corr={}'.format(var, label, corr), fontsize=fontsize)
                            ax[1].set_xlabel(wrf_var, fontsize=fontsize)
                            ax[1].set_ylabel(var, fontsize=fontsize)
                            ax[1].tick_params(axis='both', labelsize=20)
                    plt.savefig('{}_{}.png'.format(var, label))
                    plt.close()
            else:
                for f in stats.values():
                    data[label+'_'+var].append(np.nan)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    js = load_cfg()
    stat_plot_results(js)