import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 6)
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse, Rectangle

log_sigma2 = np.log(1.)
log_ls = np.log(0.8)
num_samples = 50
two_sided = True
resolution = 200
num_frames = 100
show_cov = True

def k(X, log_sigma2, log_ls):
    X = X / np.exp(log_ls)
    X_sq = np.sum(np.square(X), 1)[:, None]
    sq_dist = X_sq + X_sq.T -2*np.matmul(X, X.T)
    return np.exp(log_sigma2 - 0.5 * sq_dist) + np.eye(X.shape[0]) * 1e-6

xx = np.linspace(-1, 1, resolution)[:, None]
K = k(xx, log_sigma2=log_sigma2, log_ls=log_ls)
L = np.linalg.cholesky(K)
samples = np.dot(L, np.random.RandomState(0).randn(resolution, num_samples))
funcs = [interp1d(xx.flatten(), s) for s in samples.T]
fig = plt.figure()
main_ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
main_ax.set_xlim(-1, 1)
main_ax.set_ylim(-3, 2)
little_ax = fig.add_axes([0.12, 0.13, 0.3 * 0.6, 0.3])
main_ax.set_xticks([0, 0])
main_ax.set_xticklabels(['x_1', 'x_2'])
main_ax.set_yticks([])
little_ax.set_xlabel('f(x_1)')
little_ax.set_ylabel('f(x_2)')
little_ax.set_xlim(-2, 2)
little_ax.set_ylim(-2, 2)
lines = main_ax.plot(xx, samples, 'C0', lw=.6)
points, = main_ax.plot(0, 0, 'C1o')
points2, = main_ax.plot(0, 0, 'C1o')
points3, = little_ax.plot(0, 0, 'C1o', ms=3)
ell = Ellipse(xy=(0, 0), width=0., height=0., angle=0.)
ell.set_facecolor('none')
ell.set_edgecolor('k')
little_ax.add_artist(ell)
if show_cov:
    cov_ax = fig.add_axes([0.72, 0.13, 0.3 * 0.6, 0.3])
    cov_ax.imshow(K, cmap=plt.cm.gray, extent=[-1, 1, -1, 1], origin='lower')
    cov_ax.set_xlim(-1, 1)
    cov_ax.set_ylim(-1, 1)
    cov_ax.set_xticks([0, 0])
    cov_ax.set_xticklabels(['x_1', 'x_2'])
    cov_ax.set_yticks([0, 0])
    cov_ax.set_yticklabels(['x_1', 'x_2'])
    boxes = [Rectangle((0, 0), width=0.08, height=0.08, fill=False) for _ in range(4)]
    [b.set_edgecolor('C1') for b in boxes]
    [cov_ax.add_artist(b) for b in boxes]


 

def interpolate(x_start, x_end, i, num_iter, loop=False):
    fac = 2 * np.pi if loop else np.pi
    alpha = np.cos(fac * float(i) / float(num_iter-1)) / 2 + .5
    return x_start * alpha + (1-alpha) * x_end

def init():
    return lines + [points, points2, points3, ell] + ([] if not show_cov else boxes)

def animate(i):
    # get the x1, x2 locations:
    if two_sided:
        x1 = interpolate(-1, 0, i, num_frames, loop=True)
        x2 = -x1
    else:
        x1 = interpolate(-1, 1, i, num_frames, loop=True)
        x2 = 0.2
    fx1 = [fi(x1) for fi in funcs]
    fx2 = [fi(x2) for fi in funcs]
    main_ax.set_xticks([x1, x2])
    points.set_xdata([x1] * num_samples)
    points.set_ydata(fx1)
    points2.set_xdata([x2] * num_samples)
    points2.set_ydata(fx2)
    points3.set_xdata(fx1)
    points3.set_ydata(fx2)

    K = k(np.array([x1, x2])[:, None], log_sigma2=log_sigma2, log_ls=log_ls)
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    ell.width = np.sqrt(eigenvalues[0])*2
    ell.height = np.sqrt(eigenvalues[1])*2
    ell.angle = np.rad2deg(np.arccos(eigenvectors[0, 0]))
    ell.stale = True

    if show_cov:
        boxes[0].set_xy([x1-0.04, x1-0.04])
        boxes[1].set_xy([x1-0.04, x2-0.04])
        boxes[2].set_xy([x2-0.04, x1-0.04])
        boxes[3].set_xy([x2-0.04, x2-0.04])
        cov_ax.set_xticks([x1, x2])
        cov_ax.set_yticks([x1, x2])



    return lines + [points, points2, points3, ell] + ([] if not show_cov else boxes)

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=10, blit=False, save_count=num_frames)

w = animation.ImageMagickWriter(fps=25)
ani.save('prior_samples_twopoints{}{}.gif'.format('' if two_sided else '_onesided', '_with_cov' if show_cov else ''), writer=w)
