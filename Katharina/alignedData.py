import numpy as np
import matplotlib.pyplot as plt
from optimalEmbedding import run_optEmbedding
from scipy.stats import pearsonr
from GCCM_gao_corrected import run_GCCM_corrected
import pandas as pd
from optimalEmbedding_sampling import run_optEmbedding_sampling
from GCCM_sampling import run_GCCM_sampling
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MaxNLocator
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import random

from plot import make_Eplot

import rasterio

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

lib_sizes = np.arange(10,120,20)
random.seed( 100 )


enviNames={'dTRI':'Industry','nlights':'Nightlight'}
#%%
for hm in('Cu','Mg'):
    y = rasterio.open('data/'+hm+'_aligned.tif')  # effect
    yMatrix_al = y.read(1)
    yMatrix_al = yMatrix_al.T
    yMatrix_al = yMatrix_al.astype(float)
    for env in('dTRI','nlights'):
        x = rasterio.open('data/'+env+'_aligned.tif')  # cause
        xMatrix_al = x.read(1)
        xMatrix_al = xMatrix_al.T
        xMatrix_al = xMatrix_al.astype(float)
        x_flat = xMatrix_al.flatten().reshape(-1, 1)
        y_flat = yMatrix_al.flatten().reshape(-1, 1)
        # Fit linear model y = M*x + c
        model = LinearRegression()
        model.fit(x_flat, y_flat)
        y_pred = model.predict(x_flat)
        y_star_flat = y_flat - y_pred
        yMatrix_al = y_star_flat.reshape(y.shape)
        results_sampling = run_GCCM_sampling(xMatrix_al, yMatrix_al, lib_sizes, E=4, cores=6)

        results = pd.DataFrame(columns=['x_xmap_y_mean', 'x_xmap_y_sig', 'x_xmap_y_conf_1', 'x_xmap_y_conf_2', \
                                        'y_xmap_x_mean', 'y_xmap_x_sig', 'y_xmap_x_conf_1', 'y_xmap_x_conf_2'])
        for i in lib_sizes:
            curlineX = results_sampling['x_xmap_y'][i]
            curlineY = results_sampling['y_xmap_x'][i]
            results.loc[len(results.index)] = [curlineX['mean'], curlineX['sig'], curlineX['conf'][0], curlineX['conf'][1], \
                                               curlineY['mean'], curlineY['sig'], curlineY['conf'][0], curlineY['conf'][1]]
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)

        p1, = ax.plot(lib_sizes, results.loc[:, 'x_xmap_y_mean'], c='#006EBC', lw=2,
                      label=hm + r' $\rightarrow$ ' + enviNames[env])
        ax.fill_between(lib_sizes, results.loc[:, 'x_xmap_y_conf_1'], results.loc[:, 'x_xmap_y_conf_2'],
                        color='#006EBC', alpha=0.1, label='Confidence Interval', lw=0)
        p2, = ax.plot(lib_sizes, results.loc[:, 'y_xmap_x_mean'], c='#D00000', lw=2,
                      label=enviNames[env] + ' $\rightarrow$ ' + hm)
        ax.fill_between(lib_sizes, results.loc[:, 'y_xmap_x_conf_1'], results.loc[:, 'y_xmap_x_conf_2'],
                        color='#D00000', alpha=0.1, label='Confidence Interval', lw=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

        legend = ax.legend([p2, p1],
                           [enviNames[env] + r' $\rightarrow$ ' + hm, hm + r' $\rightarrow$ ' + enviNames[env]],
                           scatterpoints=1,
                           numpoints=1, markerscale=1., handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)},
                           handlelength=0.5, loc='upper left', frameon=False)

        ax.set_ylim(0, 1)
        # ax.set_title(r'$\textbf a)$')
        ax.set_xlabel('L')
        ax.set_ylabel(r'$\rho$')
        plt.tight_layout()

        plt.savefig('rTrend/'+enviNames[env]+'_'+hm+'.png', bbox_inches='tight')
        results.to_csv('rTrend/'+enviNames[env]+'_'+hm+'.csv', index=False)
