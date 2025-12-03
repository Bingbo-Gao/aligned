import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import pearsonr

from basic_gao import significance
from diffusion import remove_linear_signals, run_sim
from optimalEmbedding_sampling import run_optEmbedding_sampling
from GCCM_sampling import run_GCCM_sampling
import random




def simulate():
    T = 30
    a1 = 2.8e-4  # 2.8e-5
    a2 = 2.8e-3
    size = 100  # size of the 2D grid
    dx = 2. / size  # space step

    cors=np.zeros(100)
    p_values=np.zeros(100)
    significanceNum=0
    maxCor=0
    for i in range(100):
        count = 0
        s = 0
        c = 0
        X_rand = np.random.rand(size, size)
        Y_rand = np.random.rand(size, size)
        X, Y = run_sim(X_rand, Y_rand, T=T, c=c, a1=a1, a2=a2, plot=False)
        cors[i], p_values[i] = pearsonr(X.flatten(), Y.flatten())

    return cors, p_values




if __name__ == "__main__":
    cors, p_values=simulate()

    print(f" meanCor ={np.mean(np.abs(cors)):.2f} ; maxCor ={np.max(np.abs(cors)):.2f} ; "
          f"minCor={np.min(np.abs(cors)):.2f} ；the standard deviation= ={np.std(np.abs(cors)):.2f}")
    print(f" There  are {np.sum(p_values < 0.01)} pairs that have significant linear correlation")

    print(cors)
    print(p_values)