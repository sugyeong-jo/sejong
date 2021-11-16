#%%
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.gaussian_process.kernels import RationalQuadratic, Exponentiation
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import ExpSineSquared

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np 

df_org = pd.read_csv('dataset_20211108.csv')
df_org = df_org[df_org.columns.difference(['Location'])]
df = df_org.dropna()

#df = df[:-2]

print(df)
#df = df[['A11', 'A12', 'A21', 'A22', 'D1', 'D2', 'vonMises']]
X = np.array(df[['입력하중', 'PIPE 길이', 'PIPE 직경', '용접길이', 'A11', 'A12', 'A21', 'A22', 'D1', 'D2']])
y = np.array(df[['vonMises']])
#kernel = DotProduct()
#kernel = ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2)
#kernel = RBF() + ConstantKernel(constant_value=1)
#kernel = Exponentiation(RationalQuadratic(), exponent=2)
#kernel = PairwiseKernel(metric='rbf')
#kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
#kernel = 1.0 * Matern(length_scale=1.0, nu=1.77)
#kernel = 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0),)
kernel = DotProduct() + Exponentiation(RationalQuadratic(), exponent=2)
#kernel = ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2) + Exponentiation(RationalQuadratic(), exponent=2)

gpr = GaussianProcessRegressor(kernel=kernel,  random_state=0).fit(X, y)
print(f'평가지표(the coefficient of determination of the prediction, r^2, 1 is best): {gpr.score(X, y)}')

test = df_org[-1:]
test_true = test['vonMises']
test =  np.array(test[['입력하중', 'PIPE 길이', 'PIPE 직경', '용접길이', 'A11', 'A12', 'A21', 'A22', 'D1', 'D2']])
test_pred, test_std = gpr.predict(test, return_std=True)
print(test)
print(f'true value: {test_true.values[0]}, predicted value: {test_pred[0][0]}, std: {test_std}')





## test set만 따로 이렇게 만들어서 하시는게 더 편하실 것 같아요!

test = pd.read_csv('dataset_test_20211108.csv') # 
test = test[-2:] 
pred_list = []
std_list = []
acc_list = []
for i in range(len(test)):    
    test_x =  np.array(test.iloc[i][['입력하중', 'PIPE 길이', 'PIPE 직경', '용접길이', 'A11', 'A12', 'A21', 'A22', 'D1', 'D2']])
    test_true = test.iloc[i]['vonMises']

    test_pred, test_std = gpr.predict([test_x], return_std=True)
    test_pred = test_pred[0][0]
    test_std = test_std[0]
    print(f'true value: {test_true}, predicted value: {test_pred}, std: {test_std}')
    pred_list.append(test_pred)
    std_list.append(test_std)


test = test[['입력하중', 'PIPE 길이', 'PIPE 직경', '용접길이', 'A11', 'A12', 'A21', 'A22', 'D1', 'D2', 'vonMises']]
test['predict'] = pred_list
test['std'] = std_list
test['accuracy'] = abs(test['predict'] - test['vonMises'])/test['predict']*100

# R^2
y_bar = np.mean(test['vonMises'])
sst = sum((test['vonMises'] - y_bar).apply(lambda x: x**2 if x==x else 0 ))
sse = sum((test['predict'] - y_bar).apply(lambda x: x**2 if x==x else 0))
ssr = sum((test['predict'] - test['vonMises']).apply(lambda x: x**2 if x==x else 0))
r_score = sse/sst
print(f'the r square value of test is {r_score}')
test.to_csv('20211108_test_result.csv', index=False)
test


#%%
#%%
# example of bayesian optimization for a 1d function from scratch
from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
import random
# objective function
def objective(x, noise=0.1):
    noise = normal(loc=0, scale=noise)
	#y = (x**2 * sin(5 * pi * x)**6.0) 
    y = gpr.predict(np.array([x]), return_std=True)[0][0]+noise
	#y = -(x**4-x**2)+noise
	#print(f"'x': {x} | 'y': {y}")
    return y

 
# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict([X], return_std=True)
 
# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	best_list = []
	for X_elm in X:
		yhat, _ = surrogate(model, X_elm)
		best_list.append(yhat)
	best = max(best_list)
	# calculate mean and stdev via surrogate function
	# mu, std = surrogate(model, Xsamples)
	# mu = mu[:, 0]

	mu = []
	for sample in Xsamples:
		m, std = surrogate(model, sample)
		#print(m[0])
		mu.append(m[0])
	
	# calculate the probability of improvement
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs
 
# optimize the acquisition function
def opt_acquisition(X, y, model):
	# random search, generate random samples
	# Xsamples = random(100, 10)
	# Xsamples = Xsamples.reshape(len(Xsamples), 1)
	Xsamples = []
	for i in range(10):
		samples = []
		samples.append(100)
		samples.append(300)
		samples.append(45)
		samples.append(25)
		samples.append(random.randint(0, 50))
		samples.append(random.randint(0, 50))
		samples.append(-random.randint(0, 50))
		samples.append(-random.randint(0, 50))
		samples.append(random.randint(20, 50))
		samples.append(random.randint(50, 80))
		Xsamples.append(samples)
	Xsamples
	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	# locate the index of the largest scores
	ix = argmax(scores)

	return Xsamples[ix]
 
# plot real observations vs surrogate function
def plot(X, y, model):
	# scatter plot of inputs and real objective function
	pyplot.scatter(X, y)
	# line plot of surrogate function across domain
	Xsamples = asarray(arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	pyplot.plot(Xsamples, ysamples)
	# show the plot
	pyplot.show()
 

df = pd.read_csv('dataset_sejong.csv')
df = df.dropna()
df = df.iloc[[0, 1, 2], :]
df
# sample the domain sparsely with noise
#X = random(100)*100
#y = asarray([objective(x) for x in X])
df = pd.read_csv('dataset_sejong.csv')
df = df.dropna()
df = df.iloc[[0], :]
X = np.array(df[['입력하중', 'PIPE 길이', 'PIPE 직경', '용접길이', 'A11', 'A12', 'A21', 'A22', 'D1', 'D2']])
y = asarray([objective(x)[0] for x in X])
y = y.reshape((len(y), 1))
# reshape into rows and cols
#X = X.reshape(len(X), 1)
#y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot before hand
#plot(X, y, model)
# perform the optimization process
for i in range(300):
	# select the next point to sample
	x = opt_acquisition(X, y, model)
	# sample the point
	actual = objective(x)[0]
	# summarize the finding
	est, _ = surrogate(model, x)
	print(f'>x={x}, f()={est[0]}, actual={actual}')
	# add the data to the dataset
	X = vstack((X, np.array([x])))
	#print(y)
	y = vstack((y, np.array(actual))) # vstack((y, actual))
	#print(X)
	#print(y)
	# update the model
	model.fit(X, y)
 
# plot all samples and the final surrogate function
# plot(X, y, model)
# best result
ix = argmax(y)
#%%
print(f'Best Result: index = {ix}, x={X[ix]}, y={y[ix]}')
# %%
