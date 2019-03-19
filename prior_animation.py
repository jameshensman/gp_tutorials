import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 6)


def k(X, log_sigma2, log_ls):
    X = X / np.exp(log_ls)
    X_sq = np.sum(np.square(X), 1)[:, None]
    sq_dist = X_sq + X_sq.T -2*np.matmul(X, X.T)
    return np.exp(log_sigma2 - 0.5 * sq_dist) + np.eye(X.shape[0]) * 1e-6

# make xx in a random order because cholesky is not symmetric.
xx = np.linspace(-1, 1, 200)[:, None]
order = np.random.permutation(200)
xx = xx[order]
unorder = np.argsort(xx.flatten())

white_samples = np.random.randn(200, 50)
fig = plt.figure()
main_ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
main_ax.set_xlim(-1, 1)
little_ax = fig.add_axes([0.12, 0.13, 0.2, 0.2])
main_ax.set_xticks([])
main_ax.set_yticks([])
little_ax.set_xlabel('log(sigma2)')
little_ax.set_ylabel('log(lengthscale)')
little_ax.set_xlim(-2, 0)
little_ax.set_ylim(-3, -1)
lines = main_ax.plot(xx[unorder], white_samples, 'C0', lw=.6)
point = little_ax.plot(0, 0, 'C0o')

def init():
    return lines + point

def animate(i):
    # move on a circle:
    log_sigma2 = np.cos(i * 2 * np.pi / 300 - 0.5 * np.pi) - 1
    log_ls = np.sin(i * 2 * np.pi / 300 - 0.5 * np.pi) - 2
    point[0].set_xdata(log_sigma2)
    point[0].set_ydata(log_ls)
    K = k(xx, log_sigma2, log_ls)
    L = np.linalg.cholesky(K)
    samples = np.dot(L, white_samples)
    [l.set_ydata(s[unorder]) for l, s in zip(lines, samples.T)]
    return lines + point


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=10, blit=True, save_count=300)

w = animation.ImageMagickWriter(fps=25)
ani.save('prior_samples.gif', writer=w)
