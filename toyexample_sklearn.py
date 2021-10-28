#%%
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
X, y = load_iris(return_X_y=True)
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X, y)
#%%
gpc.score(X, y)

gpc.predict_proba(X[:2,:])

gpc.fit(X, y)
# %%
y_pred, MSE = gp.predict(X, return_std=True)

# %%
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
Xp, Yp = np.meshgrid(x1, x2)
Zp = np.reshape(y_pred,50)

surf = ax.plot_surface(Xp, Yp, Zp, rstride=1, cstride=1, cmap=cm.jet,
linewidth=0, antialiased=False)
pl.show()