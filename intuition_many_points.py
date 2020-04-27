import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 6)
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse, Rectangle
import itertools

np.random.seed(4)

log_sigma2 = np.log(1.)
log_ls = np.log(0.8)
num_samples = 50
resolution = 200
num_frames = 100
num_points = 10

def k(X, log_sigma2, log_ls):
    X = X / np.exp(log_ls)
    X_sq = np.sum(np.square(X), 1)[:, None]
    sq_dist = X_sq + X_sq.T -2*np.matmul(X, X.T)
    return np.exp(log_sigma2 - 0.5 * sq_dist) + np.eye(X.shape[0]) * 1e-6

xx = np.linspace(-1, 1, resolution)[:, None]
K = k(xx, log_sigma2=log_sigma2, log_ls=log_ls)

X = np.sort(np.random.rand(num_points) * 2 - 1)

L = np.linalg.cholesky(K)
samples = np.dot(L, np.random.RandomState(0).randn(resolution, num_samples))
funcs = [interp1d(xx.flatten(), s) for s in samples.T]
fig = plt.figure()
main_ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
main_ax.set_xlim(-1, 1)
main_ax.set_ylim(-3, 2)
little_ax = fig.add_axes([0.12, 0.13, 0.3 * 0.6, 0.3])
main_ax.set_xticks(X)
main_ax.set_xticklabels(['x_{}'.format(i+1) for i in range(num_points)])
main_ax.set_yticks([])


little_ax.set_xticks(np.arange(num_points) + 0.5)
little_ax.set_yticks(np.arange(num_points) + 0.5)
little_ax.set_xticklabels(['f(x_{})'.format(i+1) for i in range(num_points)])
little_ax.set_yticklabels(['f(x_{})'.format(i+1) for i in range(num_points)])
little_ax.set_xlim(0, num_points)
little_ax.set_ylim(0, num_points)
cov_image = little_ax.imshow(k(X[:, None], log_sigma2, log_ls), cmap=plt.cm.gray, extent=[0, num_points, 0, num_points], origin='lower', interpolation='nearest', vmin=0, vmax=1.)
lines = main_ax.plot(xx, samples, 'C0', lw=.6)
points = [main_ax.plot([Xi] * num_samples, [f(Xi) for f in funcs], 'C1o', ms=5) for Xi in X]

cov_ax = fig.add_axes([0.72, 0.13, 0.3 * 0.6, 0.3])
cov_ax.imshow(K, cmap=plt.cm.gray, extent=[-1, 1, -1, 1], origin='lower')
cov_ax.set_xlim(-1, 1)
cov_ax.set_ylim(-1, 1)
cov_ax.set_xticks(X)
cov_ax.set_xticklabels(['x_{}'.format(i+1) for i in range(num_points)])
cov_ax.set_yticks(X)
cov_ax.set_yticklabels(['x_{}'.format(i+1) for i in range(num_points)])
boxes = [Rectangle((xi, xj), width=0.08, height=0.08, fill=False) for xi, xj in itertools.product(X, X)]
[b.set_edgecolor('C1') for b in boxes]
[cov_ax.add_artist(b) for b in boxes]

plt.savefig('intuition_many_points.png')
