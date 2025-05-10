import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import pearsonr
from diffusion import remove_linear_signals, run_sim
from optimalEmbedding_sampling import run_optEmbedding_sampling
from GCCM_sampling import run_GCCM_sampling






#%%
T = 30
a1 = 2.8e-4  # 2.8e-5
a2 = 2.8e-3
size = 100  # size of the 2D grid
dx = 2. / size  # space step
dims = np.arange(1, 9)
lib_sizes = np.arange(10, 101, 30)
lib_size = 100
#%%


#%%
cors=np.zeros(100)
p_values=np.zeros(100)
for i in range(100):
    count = 0
    s = 0
    c = 0.15
    X_rand = np.random.rand(size, size)
    Y_rand = np.random.rand(size, size)
    X, Y = run_sim(X_rand, Y_rand, T=T, c=c, a1=a1, a2=a2, plot=False)
    cors[i], p_values[i] = pearsonr(X.flatten(), Y.flatten())
print(cors)
print(p_values)

#%%
count = 0
s = 0
c=0.00
T=1000
X_rand = np.random.rand(size, size)
Y_rand = np.random.rand(size, size)
X, Y = run_sim(X_rand, Y_rand, T=T, c=c, a1=a1, a2=a2, plot=False)
correlation_coefficient, p_value = pearsonr(X.flatten(), Y.flatten())
#%%



#%%
plt.matshow(X, cmap=plt.cm.Reds) # 这里设置颜色为红色
plt.title("matrix X")
plt.show()
#%%

#%%
plt.matshow(Y,cmap=plt.cm.Reds) # 这里设置颜色为红色
plt.title("matrix Y")
plt.show()
#%%

#%%
lib_sizes = np.arange(10,101,30)
conv = run_GCCM_sampling(X, Y, lib_sizes, E=5, cores=6)

#x_xmap_y_all, x_xmap_y_results = GCCM(X, Y, lib_sizes, E=5, cores=cores)
#print('y_xmap_x')
#y_xmap_x_all, y_xmap_x_results = GCCM(Y, X, lib_sizes, E=5, cores=cores)

print(conv)
#%%