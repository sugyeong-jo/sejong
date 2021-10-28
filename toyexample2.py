#%%
import chart_studio.plotly 
import GPy
GPy.plotting.change_plotting_library('plotly_offline')
import numpy as np
from IPython.display import display
from sklearn.datasets import load_iris

#%%
X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05
Y
#%%
X, Y = load_iris(return_X_y=True)
Y
#%%
Y_fin = []
for i in Y:
    Y_fin.append([i])
Y_fin
 
#%%
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
#%%
m = GPy.models.GPRegression(X,Y_fin,kernel)
display(m)
#%%
fig = m.plot()
GPy.plotting.show(fig, filename='basic_gp_regression_notebook')
#%%
m.optimize(messages=True)
#%%
m.optimize_restarts(num_restarts = 10)
display(m)
fig = m.plot()
GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized')
#%%
display(m)
fig = m.plot(plot_density=True)
GPy.plotting.show(fig, filename='basic_gp_regression_density_notebook_optimized')
# %%
